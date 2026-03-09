"""Evaluate solar irradiance forecasting with Dual-Phase Online Adaptation.

Runs :class:`~cirr.tta.DualPhaseAdapter` in an online day-by-day loop over a
prepared dataset. Includes Dynamic Regime-Aware Ensembling using the
Clear Sky Index (CSI).
"""

from collections import deque
from pathlib import Path
import sys
from typing import Any, Literal

import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cirr.datamodules.loading import load_dataset
from cirr.evaluation.printing import print_metrics_header, print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.tta import DualPhaseAdapter
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Online calibration
# ---------------------------------------------------------------------------


class OnlineAffineCalibrator:
    """Online affine calibrator for absolute predictions: y' = a*y + b.

    Parameters
    ----------
    lr : float
        SGD-style learning rate for online updates.
    l2 : float
        L2 regularization strength toward identity calibration
        (a -> 1, b -> 0).
    """

    def __init__(self, lr: float = 1e-3, l2: float = 1e-4) -> None:
        self.lr = lr
        self.l2 = l2
        self.a = 1.0
        self.b = 0.0

    def apply(self, pred: torch.Tensor) -> torch.Tensor:
        """Apply calibration to a tensor prediction."""
        calibrated = self.a * pred + self.b
        return torch.clamp(calibrated, min=0.0)

    def update(self, pred_raw: float, target: float) -> None:
        """Update calibrator params using a delayed-label sample."""
        pred_cal = self.a * pred_raw + self.b
        error = pred_cal - target

        grad_a = 2.0 * error * pred_raw + 2.0 * self.l2 * (self.a - 1.0)
        grad_b = 2.0 * error + 2.0 * self.l2 * self.b

        self.a -= self.lr * grad_a
        self.b -= self.lr * grad_b

        # Keep calibration stable and physically sensible.
        self.a = float(np.clip(self.a, 0.5, 1.5))
        self.b = float(np.clip(self.b, -0.5, 0.5))


def _load_clear_sky_lookup(dataset_path: Path) -> dict[str, float]:
    """Load clear-sky irradiance indexed by ``YYYYMMDD_HHMMSS`` timestamp.

    Returns values scaled to ``kW/m^2`` to match dataset target scaling.
    """
    clear_sky_files = sorted(dataset_path.glob("*clear_sky.csv"))
    if not clear_sky_files:
        return {}

    clear_sky_path = clear_sky_files[0]
    frame = pd.read_csv(clear_sky_path)
    if "datetime" not in frame.columns or "ghi_clear" not in frame.columns:
        raise ValueError(f"Invalid clear-sky file format: {clear_sky_path}")

    dt = pd.to_datetime(frame["datetime"], errors="coerce")
    ghi_kw = frame["ghi_clear"].astype(float) / 1000.0
    valid = ~dt.isna() & ghi_kw.notna()

    keys = dt.loc[valid].dt.strftime("%Y%m%d_%H%M%S")
    values = ghi_kw.loc[valid].astype(float)
    return dict(zip(keys.tolist(), values.tolist(), strict=False))


def _resolve_clear_sky_value(clear_sky_lookup: dict[str, float], timestamp: str) -> float | None:
    """Resolve clear-sky value for a sample timestamp.

    Image timestamps are second-level; clear-sky tables are often 1-minute.
    Tries exact key first, then falls back to ``...SS=00`` for the same minute.
    """
    value = clear_sky_lookup.get(timestamp)
    if value is not None:
        return value

    if len(timestamp) >= 15:
        minute_key = f"{timestamp[:-2]}00"
        return clear_sky_lookup.get(minute_key)

    return None


def _is_in_learning_zone(error_value: float, low: float, high: float) -> bool:
    """Return ``True`` when error lies inside the configured update band."""
    return low <= error_value <= high


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    adapter: DualPhaseAdapter,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    horizon_steps: int,
    forecasting_horizon: int,
    ogd_gating: Literal["none", "raw", "csi"],
    ogd_gating_low: float,
    ogd_gating_high: float,
    clear_sky_lookup: dict[str, float],
    csi_min_clear_sky: float,
    use_online_calibrator: bool,
    cal_lr: float,
    cal_l2: float,
    warmup_samples: int,
    lr_decay_rate: float,
    lr_decay_steps: int,
    use_dynamic_ensemble: bool,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    all_predictions: list[float] = []
    all_targets: list[float] = []
    predictions_data: list[dict[str, Any]] = []
    per_day_metrics: dict[str, dict[str, float]] = {}
    causal_queue: deque[dict[str, Any]] = deque()
    calibrator = OnlineAffineCalibrator(lr=cal_lr, l2=cal_l2) if use_online_calibrator else None
    update_stats = {
        "considered": 0,
        "updated": 0,
        "skipped_below_low": 0,
        "skipped_above_high": 0,
        "skipped_no_clear_sky": 0,
        "skipped_low_clear_sky": 0,
        "skipped_warmup": 0,
    }
    total_samples_processed = 0
    ogd_lr_current = adapter.lr_ogd

    print_metrics_header()

    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])
        if not day_indices:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []
        day_regime1_count = 0
        day_regime2_count = 0

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False, file=sys.stderr):
            image_input, irradiance_history, target_irradiance, image_name = dataset[idx]

            images = image_input.unsqueeze(0).to(device)  # [1, C, H, W]
            irr_history = irradiance_history.unsqueeze(0).to(device)  # [1, hist]
            target = target_irradiance.unsqueeze(0).to(device)  # [1, 1]

            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            causal_queue.append(
                {
                    "image": image_input,
                    "history": irradiance_history,
                    "gt": target_irradiance.unsqueeze(0),
                    "raw_pred": None,
                    "timestamp": image_name,
                }
            )

            if len(causal_queue) > horizon_steps:
                oldest = causal_queue.popleft()
                should_update = True
                update_stats["considered"] += 1
                total_samples_processed += 1

                if warmup_samples > 0 and total_samples_processed <= warmup_samples:
                    should_update = False
                    update_stats["skipped_warmup"] += 1

                if ogd_gating != "none" and oldest["raw_pred"] is not None:
                    gt_value = float(oldest["gt"].view(-1)[0].item())
                    raw_error = abs(float(oldest["raw_pred"]) - gt_value)

                    if ogd_gating == "raw":
                        error_value = raw_error
                    else:
                        clear_sky = _resolve_clear_sky_value(clear_sky_lookup, str(oldest["timestamp"]))
                        if clear_sky is None:
                            should_update = False
                            update_stats["skipped_no_clear_sky"] += 1
                            error_value = None
                        elif clear_sky < csi_min_clear_sky:
                            should_update = False
                            update_stats["skipped_low_clear_sky"] += 1
                            error_value = None
                        else:
                            error_value = raw_error / clear_sky

                    if error_value is not None and not _is_in_learning_zone(error_value, ogd_gating_low, ogd_gating_high):
                        should_update = False
                        if error_value < ogd_gating_low:
                            update_stats["skipped_below_low"] += 1
                        else:
                            update_stats["skipped_above_high"] += 1

                if should_update:
                    if lr_decay_rate > 0 and lr_decay_steps > 0 and total_samples_processed % lr_decay_steps == 0:
                        ogd_lr_current *= 1.0 - lr_decay_rate
                        for param_group in adapter._ogd_optimizer.param_groups:
                            param_group["lr"] = ogd_lr_current

                    adapter.update(
                        oldest["image"].to(device),
                        oldest["history"].to(device),
                        oldest["gt"].to(device),
                    )
                    update_stats["updated"] += 1

                if calibrator is not None and oldest["raw_pred"] is not None:
                    delayed_target = float(oldest["gt"].view(-1)[0].item())
                    calibrator.update(float(oldest["raw_pred"]), delayed_target)

            # C+D: TTA-adapt a deepcopy and predict with it.
            mean_raw, _ = adapter.predict(images, irr_history)
            mean = calibrator.apply(mean_raw) if calibrator is not None else mean_raw

            # -----------------------------------------------------------------
            # DYNAMIC REGIME-AWARE ENSEMBLE (CSI-BASED)
            # -----------------------------------------------------------------
            if use_dynamic_ensemble:
                current_irr_val = irr_history[0, -1].item()
                timestamp_str = Path(str(image_name)).stem
                current_dt = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S", errors="coerce")

                target_dt = current_dt + pd.Timedelta(minutes=forecasting_horizon)

                current_cs = _resolve_clear_sky_value(clear_sky_lookup, current_dt.strftime("%Y%m%d_%H%M%S"))
                target_cs = _resolve_clear_sky_value(clear_sky_lookup, target_dt.strftime("%Y%m%d_%H%M%S"))

                # 1. Calculate Smart Persistence Prediction
                if current_cs is not None and target_cs is not None and current_cs > 0:
                    kc_last = current_irr_val / current_cs
                    kc_last = max(0.0, min(kc_last, 2.0))  # Clamp to [0, 2]
                    smart_persistence_val = kc_last * target_cs
                else:
                    smart_persistence_val = current_irr_val

                smart_persistence_pred = torch.tensor([[smart_persistence_val]], dtype=mean.dtype, device=device)

                # 2. Calculate CSI Variance to determine the Weather Regime
                if current_cs is not None and current_cs > 0:
                    cs_history_vals = []
                    # Assuming 4 history steps are [t-15, t-10, t-5, t]
                    for offset in [15, 10, 5, 0]:
                        hist_dt = current_dt - pd.Timedelta(minutes=offset)
                        hist_cs = _resolve_clear_sky_value(clear_sky_lookup, hist_dt.strftime("%Y%m%d_%H%M%S"))
                        # Fallback to current_cs if a history clear sky value is missing
                        cs_history_vals.append(hist_cs if hist_cs and hist_cs > 0 else current_cs)

                    cs_history_tensor = torch.tensor(cs_history_vals, dtype=mean.dtype, device=device)
                    recent_csi = irr_history[0, -4:] / cs_history_tensor
                    csi_var = torch.var(recent_csi).item() if recent_csi.numel() > 1 else 0.0
                else:
                    csi_var = torch.var(irr_history[0, -4:]).item() if irr_history[0, -4:].numel() > 1 else 0.0

                # 3. Dynamic Router (Blend)
                csi_flat_threshold = 0.015

                if csi_var < csi_flat_threshold:
                    # REGIME 1: Stable Weather. Anchor heavily to Smart Persistence.
                    dyn_weight = 1.0
                    day_regime1_count += 1
                else:
                    # REGIME 2: Chaotic Weather / Broken Clouds. Trust the Vision Network.
                    dyn_weight = 0.0
                    day_regime2_count += 1

                mean = (1.0 - dyn_weight) * mean + dyn_weight * smart_persistence_pred
            else:
                # Dynamic routing is disabled, so all predictions come from the model path.
                day_regime2_count += 1
            # -----------------------------------------------------------------

            # Store raw prediction for delayed-label calibrator updates.
            causal_queue[-1]["raw_pred"] = float(mean_raw.detach().cpu().view(-1)[0].item())

            pred_np = mean.detach().cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()

            day_preds.extend(pred_np)
            day_targets.extend(target_np)
            all_predictions.extend(pred_np)
            all_targets.extend(target_np)

            predictions_data.append(
                {
                    "timestamp": image_name if isinstance(image_name, str) else None,
                    "day": day_key,
                    "prediction": pred_np[0] if len(pred_np) == 1 else pred_np,
                    "target": target_np[0] if len(target_np) == 1 else target_np,
                    "error": float(pred_np[0] - target_np[0]),
                    "absolute_error": float(abs(pred_np[0] - target_np[0])),
                }
            )

        if day_preds:
            day_metrics = compute_metrics(np.array(day_preds), np.array(day_targets))
            per_day_metrics[day_key] = day_metrics
            samples_with_regimes = f"{day_metrics['count']} (R1:{day_regime1_count}; R2:{day_regime2_count})"
            print(
                f"{day_key:<6} {day_metrics['mae']:<10.4f} {day_metrics['rmse']:<10.4f} "
                f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% {samples_with_regimes:<16}",
                flush=True,
            )

        # Flush the causal queue at day boundaries to prevent overnight
        # ground-truth leakage.
        causal_queue.clear()

    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    if ogd_gating == "none" and warmup_samples == 0:
        print("\nOGD gating: disabled (all causally-available samples were updated).", flush=True)
    else:
        print(
            "\nOGD gating stats: "
            f"considered={update_stats['considered']}  "
            f"updated={update_stats['updated']}  "
            f"warmup={update_stats['skipped_warmup']}  "
            f"below_low={update_stats['skipped_below_low']}  "
            f"above_high={update_stats['skipped_above_high']}  "
            f"no_clear_sky={update_stats['skipped_no_clear_sky']}  "
            f"low_clear_sky={update_stats['skipped_low_clear_sky']}",
            flush=True,
        )
    if lr_decay_rate > 0:
        print(f"Final OGD learning rate: {ogd_lr_current:.2e}", flush=True)
    return overall_metrics, per_day_metrics, predictions_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint file.",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prepared dataset directory (e.g. data/alpnach_prepared).",
)
@click.option(
    "--tta-method",
    type=click.Choice(["memo", "note", "none"], case_sensitive=False),
    default="memo",
    show_default=True,
    help="Unsupervised TTA variant. 'none' = OGD-only (no ephemeral TTA).",
)
@click.option(
    "--horizon-steps",
    type=int,
    default=15,
    show_default=True,
    help="Causal queue depth: time steps between prediction and label arrival.",
)
@click.option(
    "--forecasting-horizon",
    type=int,
    default=15,
    show_default=True,
    help="Forecasting horizon in minutes, used by the clear-sky persistence ensemble.",
)
@click.option(
    "--ogd-lr",
    type=float,
    default=1e-5,
    show_default=True,
    help="Learning rate for the persistent supervised OGD optimizer.",
)
@click.option(
    "--ogd-scope",
    type=click.Choice(["head_only", "head_and_encoder"], case_sensitive=False),
    default="head_only",
    show_default=True,
    help="Parameters updated by OGD.",
)
@click.option(
    "--ogd-loss",
    type=click.Choice(["berhu", "huber", "mse"], case_sensitive=False),
    default="berhu",
    show_default=True,
    help="Supervised loss function used in the OGD phase.",
)
@click.option(
    "--ogd-ramp-penalty",
    type=float,
    default=0.0,
    show_default=True,
    help="Multiplier for OGD loss based on true irradiance delta.",
)
@click.option(
    "--lr-tta",
    type=float,
    default=1e-5,
    show_default=True,
    help="Learning rate for the ephemeral TTA AdamW optimizer.",
)
@click.option(
    "--n-augmentations",
    type=int,
    default=8,
    show_default=True,
    help="(MEMO) Number of augmented image views used for variance minimisation.",
)
@click.option(
    "--niter",
    type=int,
    default=1,
    show_default=True,
    help="Number of TTA gradient steps per time step.",
)
@click.option(
    "--note-buffer-size",
    type=int,
    default=64,
    show_default=True,
    help="(NOTE) Maximum number of recent samples retained in the persistent buffer.",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    show_default=True,
    help="Number of days to evaluate. -1 means all available days.",
)
@click.option(
    "--start-day",
    type=str,
    default="0101",
    show_default=True,
    help="Starting day in MMDD format (e.g. 0101).",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    show_default=True,
    help="Device to run on ('cuda' or 'cpu').",
)
@click.option(
    "--save-results",
    is_flag=True,
    default=False,
    help="Save per-day metrics report and predictions CSV to disk.",
)
@click.option(
    "--use-online-calibrator/--no-online-calibrator",
    default=True,
    show_default=True,
    help="Apply delayed-label affine calibration: y' = a*y + b.",
)
@click.option(
    "--cal-lr",
    type=float,
    default=1e-3,
    show_default=True,
    help="Learning rate for online affine calibrator updates.",
)
@click.option(
    "--cal-l2",
    type=float,
    default=1e-4,
    show_default=True,
    help="L2 regularization of calibrator toward identity (a=1, b=0).",
)
@click.option(
    "--ogd-gating",
    type=click.Choice(["none", "raw", "csi"], case_sensitive=False),
    default="none",
    show_default=True,
    help="Selective OGD updates by delayed error thresholds.",
)
@click.option(
    "--ogd-gating-low",
    type=float,
    default=0.0,
    show_default=True,
    help="Lower threshold for the OGD learning zone (inclusive).",
)
@click.option(
    "--ogd-gating-high",
    type=float,
    default=1e9,
    show_default=True,
    help="Upper threshold for the OGD learning zone (inclusive).",
)
@click.option(
    "--csi-min-clear-sky",
    type=float,
    default=0.01,
    show_default=True,
    help="For csi gating: skip updates when clear sky is below this value (kW/m^2).",
)
@click.option(
    "--warmup-samples",
    type=int,
    default=0,
    show_default=True,
    help="Number of initial samples to skip OGD updates (warm-up period).",
)
@click.option(
    "--lr-decay-rate",
    type=float,
    default=0.0,
    show_default=True,
    help="Learning rate decay rate per decay-steps (0.0 = no decay).",
)
@click.option(
    "--lr-decay-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Number of OGD updates between learning rate decay applications.",
)
@click.option(
    "--use-dynamic-ensemble/--no-dynamic-ensemble",
    default=True,
    show_default=True,
    help="Use CSI-based variance to dynamically switch between Smart Persistence and Model.",
)
def main(
    checkpoint: Path,
    dataset_path: Path,
    tta_method: str,
    horizon_steps: int,
    forecasting_horizon: int,
    ogd_lr: float,
    ogd_scope: str,
    ogd_loss: str,
    ogd_ramp_penalty: float,
    lr_tta: float,
    n_augmentations: int,
    niter: int,
    note_buffer_size: int,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
    use_online_calibrator: bool,
    cal_lr: float,
    cal_l2: float,
    ogd_gating: str,
    ogd_gating_low: float,
    ogd_gating_high: float,
    csi_min_clear_sky: float,
    warmup_samples: int,
    lr_decay_rate: float,
    lr_decay_steps: int,
    use_dynamic_ensemble: bool,
) -> None:
    """Evaluate solar irradiance forecasting with Dual-Phase Online Adaptation."""

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    dataset_name = dataset_path.name.split("_")[0]
    cal_tag = "cal" if use_online_calibrator else "nocal"
    gating_tag = f"gate_{ogd_gating}"
    method_tag = f"dual_{tta_method}_{ogd_scope}_{cal_tag}_{gating_tag}"

    if ogd_gating_low > ogd_gating_high:
        raise ValueError(f"Invalid OGD gating range: low={ogd_gating_low} > high={ogd_gating_high}.")

    # Always load clear sky data (required by the CSI dynamic router)
    clear_sky_lookup = _load_clear_sky_lookup(dataset_path)
    if not clear_sky_lookup:
        print("Warning: No '*clear_sky.csv' file was found. The CSI Router and Gating might fail.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {checkpoint}...")
    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    if hasattr(model.image_encoder, "fc") and not isinstance(model.image_encoder.fc, torch.nn.Identity):
        model.image_encoder.fc = torch.nn.Identity()

    # ------------------------------------------------------------------
    # Build adapter
    # ------------------------------------------------------------------
    print(
        f"Building DualPhaseAdapter ("
        f"tta_method={tta_method!r}, horizon_steps={horizon_steps}, "
        f"ogd_lr={ogd_lr}, ogd_scope={ogd_scope!r}, ogd_loss={ogd_loss!r}, "
        f"ogd_ramp_penalty={ogd_ramp_penalty}, lr_tta={lr_tta}, "
        f"n_augmentations={n_augmentations}, niter={niter}, "
        f"note_buffer_size={note_buffer_size})..."
    )
    adapter = DualPhaseAdapter(
        base_model=model,
        lr_ogd=ogd_lr,
        lr_tta=lr_tta,
        tta_method=tta_method,
        ogd_scope=ogd_scope,
        ogd_loss=ogd_loss,
        ogd_ramp_penalty=ogd_ramp_penalty,
        tta_weight_decay=0.0,
        n_augmentations=n_augmentations,
        niter_tta=niter,
        note_buffer_size=note_buffer_size,
    ).to(device)

    # ------------------------------------------------------------------
    # Load dataset and determine days to evaluate
    # ------------------------------------------------------------------
    print(f"Loading dataset from {dataset_path}...")
    test_dataset = load_dataset(dataset_path, split="test")
    day_index_map = group_indices_by_day(test_dataset._periods)
    print(f"Test set size: {len(test_dataset)} samples")
    print(f"Number of unique days: {len(day_index_map)}\n")

    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day: {start_day!r}")
    start_idx = all_days.index(start_day)

    days_to_eval = all_days[start_idx : start_idx + num_days] if num_days > 0 else all_days[start_idx:]
    days_to_eval = [d for d in days_to_eval if d in day_index_map]

    total_samples = sum(len(day_index_map[d]) for d in days_to_eval)
    print(f"Evaluating {len(days_to_eval)} days starting from {start_day}")
    print(f"Total samples: {total_samples}\n")
    if ogd_gating == "none":
        print("OGD gating: disabled (all delayed samples update base model).\n")
    elif ogd_gating == "raw":
        print(f"OGD gating: raw absolute error band [{ogd_gating_low}, {ogd_gating_high}] (inclusive).\n")
    else:
        print(
            "OGD gating: CSI-normalized error band "
            f"[{ogd_gating_low}, {ogd_gating_high}] (inclusive), "
            f"min clear sky={csi_min_clear_sky} kW/m^2.\n"
        )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print("Running Dual-Phase Online Adaptation evaluation...\n")

    overall_metrics, per_day_metrics, predictions_data = evaluate(
        adapter=adapter,
        dataset=test_dataset,
        day_index_map=day_index_map,
        days_to_eval=days_to_eval,
        device=device,
        horizon_steps=horizon_steps,
        forecasting_horizon=forecasting_horizon,
        ogd_gating=ogd_gating,
        ogd_gating_low=ogd_gating_low,
        ogd_gating_high=ogd_gating_high,
        clear_sky_lookup=clear_sky_lookup,
        csi_min_clear_sky=csi_min_clear_sky,
        use_online_calibrator=use_online_calibrator,
        cal_lr=cal_lr,
        cal_l2=cal_l2,
        warmup_samples=warmup_samples,
        lr_decay_rate=lr_decay_rate,
        lr_decay_steps=lr_decay_steps,
        use_dynamic_ensemble=use_dynamic_ensemble,
    )

    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    if save_results:
        csv_path = f"{dataset_name}_{method_tag}_predictions.csv"
        txt_path = f"{dataset_name}_{method_tag}_results.txt"
        save_predictions_csv(predictions_data, csv_path)
        print(f"\nPredictions saved to {csv_path}")
        extra = [
            f"TTA: DualPhaseAdapter  tta_method={tta_method}"
            f"  horizon_steps={horizon_steps}  ogd_lr={ogd_lr}  ogd_scope={ogd_scope}"
            f"  ogd_loss={ogd_loss}  ogd_ramp_penalty={ogd_ramp_penalty}"
            f"  lr_tta={lr_tta}  n_aug={n_augmentations}"
            f"  niter={niter}  note_buffer_size={note_buffer_size}"
            f"  use_online_calibrator={use_online_calibrator}"
            f"  cal_lr={cal_lr}  cal_l2={cal_l2}"
            f"  ogd_gating={ogd_gating}"
            f"  ogd_gating_low={ogd_gating_low}  ogd_gating_high={ogd_gating_high}"
            f"  csi_min_clear_sky={csi_min_clear_sky}"
            f"  warmup_samples={warmup_samples}"
            f"  lr_decay_rate={lr_decay_rate}  lr_decay_steps={lr_decay_steps}"
            f"  use_dynamic_ensemble={use_dynamic_ensemble}"
        ]
        save_metrics_report(
            tag=method_tag,
            dataset_name=dataset_name,
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            overall_metrics=overall_metrics,
            per_day_metrics=per_day_metrics,
            path=txt_path,
            extra_lines=extra,
        )
        print(f"Results saved to {txt_path}")


if __name__ == "__main__":
    main()
