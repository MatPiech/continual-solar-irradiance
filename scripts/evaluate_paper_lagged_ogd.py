"""Evaluate forecasting with the paper-method lagged OGD continual adaptation.

Implements Algorithm 1 from the paper directly:

    For each timestep t:
      1. Predict y_hat_t with the current model.
      2. Enqueue sample in a causal delay buffer of depth ``horizon_steps``.
      3. When the delayed ground-truth arrives, compute the absolute error
         against the prediction that was made ``horizon_steps`` ago:
           - error < T1  -> skip update ("none")
           - T1 <= error < T2 -> update head only ("head")
           - error >= T2 -> update whole network ("whole")

Default thresholds per the paper: T1 = 0.01 kW/m2, T2 = 0.1 kW/m2.

No DualPhaseAdapter, no AR residual filter, no k-NN memory, no dynamic router.
"""

from collections import deque
from pathlib import Path
import sys
from typing import Any, Literal

import click
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cirr.datamodules.loading import load_dataset
from cirr.evaluation.printing import print_metrics_header, print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.losses import BerHuLoss
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# OGD helpers
# ---------------------------------------------------------------------------


def _make_optimizer(
    model: Forecaster,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer covering all model parameters."""
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def _algorithm1_scope(
    delayed_abs_error: float,
    t1: float,
    t2: float,
) -> Literal["none", "head_only", "head_and_encoder"]:
    """Return Algorithm 1 update scope based on delayed prediction error."""
    if delayed_abs_error < t1:
        return "none"
    if delayed_abs_error < t2:
        return "head_only"
    return "head_and_encoder"


def _set_trainable(model: Forecaster, scope: Literal["head_only", "head_and_encoder"]) -> None:
    """Freeze all parameters then unfreeze those matching *scope*."""
    for param in model.parameters():
        param.requires_grad_(False)

    for param in model.network_head.parameters():
        param.requires_grad_(True)

    if scope == "head_and_encoder":
        encoder_children = list(model.image_encoder.children())
        if encoder_children:
            for child in encoder_children[-2:]:
                for param in child.parameters():
                    param.requires_grad_(True)


@torch.enable_grad()
def _ogd_step(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    scope: Literal["head_only", "head_and_encoder"],
    loss_fn: nn.Module,
    image: torch.Tensor,  # [C, H, W]  (CPU tensor from the causal queue)
    history: torch.Tensor,  # [hist_len]
    gt: torch.Tensor,  # [1, 1]
    device: str,
) -> None:
    """One supervised OGD gradient step updating only the parameters in *scope*."""
    _set_trainable(model, scope)
    model.eval()  # keep BN running stats frozen during supervised update

    imgs = image.unsqueeze(0).to(device)
    hist = history.unsqueeze(0).to(device)
    target = gt.to(device)

    optimizer.zero_grad()
    output = model(imgs, hist)
    pred = output[0] if isinstance(output, tuple) else output
    loss = loss_fn(pred, target.view_as(pred))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Re-freeze everything after the update step.
    for param in model.parameters():
        param.requires_grad_(False)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    horizon_steps: int,
    paper_t1: float,
    paper_t2: float,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    all_predictions: list[float] = []
    all_targets: list[float] = []
    predictions_data: list[dict[str, Any]] = []
    per_day_metrics: dict[str, dict[str, float]] = {}
    per_day_adaptation: dict[str, dict[str, int]] = {}
    causal_queue: deque[dict[str, Any]] = deque()

    update_stats: dict[str, int] = {
        "considered": 0,
        "none": 0,
        "head": 0,
        "whole": 0,
    }

    print_metrics_header()

    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])
        if not day_indices:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []
        day_update_stats: dict[str, int] = {
            "considered": 0,
            "none": 0,
            "head": 0,
            "whole": 0,
        }

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False, file=sys.stderr):
            image_input, irradiance_history, target_irradiance, image_name = dataset[idx]

            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            images = image_input.unsqueeze(0).to(device)
            irr_history = irradiance_history.unsqueeze(0).to(device)
            target = target_irradiance.unsqueeze(0).to(device)

            # --- Inference first (current model state, no-gradient) ---
            model.eval()
            with torch.no_grad():
                output = model(images, irr_history)
                mean_raw = output[0] if isinstance(output, tuple) else output
            mean = torch.clamp(mean_raw, min=0.0)

            pred_np = mean.detach().cpu().numpy().flatten()
            target_np = target.detach().cpu().numpy().flatten()

            day_preds.extend(pred_np)
            day_targets.extend(target_np)
            all_predictions.extend(pred_np)
            all_targets.extend(target_np)

            pred_row_idx = len(predictions_data)
            predictions_data.append(
                {
                    "timestamp": image_name if isinstance(image_name, str) else None,
                    "day": day_key,
                    "prediction": pred_np[0] if len(pred_np) == 1 else pred_np,
                    "target": target_np[0] if len(target_np) == 1 else target_np,
                    "error": float(pred_np[0] - target_np[0]),
                    "absolute_error": float(abs(pred_np[0] - target_np[0])),
                    "update_considered": False,
                    "adapted": False,
                    "adaptation_scope": "not_available",
                    "delayed_abs_error": np.nan,
                }
            )

            # --- Enqueue (store raw prediction for delayed error computation) ---
            causal_queue.append(
                {
                    "image": image_input,
                    "history": irradiance_history,
                    "gt": target_irradiance.unsqueeze(0),
                    "raw_pred": float(mean_raw.detach().cpu().view(-1)[0].item()),
                    "pred_row_idx": pred_row_idx,
                }
            )

            # --- Dequeue oldest and apply Algorithm 1 update ---
            if len(causal_queue) > horizon_steps:
                oldest = causal_queue.popleft()
                update_stats["considered"] += 1
                day_update_stats["considered"] += 1

                delayed_gt = float(oldest["gt"].view(-1)[0].item())
                delayed_abs_error = abs(float(oldest["raw_pred"]) - delayed_gt)
                scope = _algorithm1_scope(delayed_abs_error, paper_t1, paper_t2)
                oldest_row = predictions_data[oldest["pred_row_idx"]]
                oldest_row["update_considered"] = True
                oldest_row["delayed_abs_error"] = delayed_abs_error

                if scope == "none":
                    update_stats["none"] += 1
                    day_update_stats["none"] += 1
                    oldest_row["adaptation_scope"] = "none"
                else:
                    _ogd_step(
                        model=model,
                        optimizer=optimizer,
                        scope=scope,
                        loss_fn=loss_fn,
                        image=oldest["image"],
                        history=oldest["history"],
                        gt=oldest["gt"],
                        device=device,
                    )
                    if scope == "head_only":
                        update_stats["head"] += 1
                        day_update_stats["head"] += 1
                        oldest_row["adapted"] = True
                        oldest_row["adaptation_scope"] = "head"
                    else:
                        update_stats["whole"] += 1
                        day_update_stats["whole"] += 1
                        oldest_row["adapted"] = True
                        oldest_row["adaptation_scope"] = "whole"

        if day_preds:
            day_metrics = compute_metrics(np.array(day_preds), np.array(day_targets))
            per_day_metrics[day_key] = day_metrics
            per_day_adaptation[day_key] = day_update_stats
            print(
                f"{day_key:<6} {day_metrics['mae']:<10.4f} {day_metrics['rmse']:<10.4f} "
                f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% {day_metrics['count']:<16}",
                flush=True,
            )
            print(
                f"       adapt: considered={day_update_stats['considered']} "
                f"none={day_update_stats['none']} "
                f"head={day_update_stats['head']} "
                f"whole={day_update_stats['whole']}",
                flush=True,
            )

        # Prevent overnight ground-truth leakage across day boundaries.
        causal_queue.clear()

    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    print(
        f"\nAlgorithm 1 update stats:"
        f"  considered={update_stats['considered']}"
        f"  none={update_stats['none']}"
        f"  head={update_stats['head']}"
        f"  whole={update_stats['whole']}",
        flush=True,
    )

    return overall_metrics, per_day_metrics, predictions_data, per_day_adaptation, update_stats


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
    help="Path to prepared dataset directory (e.g. data/poznan_prepared).",
)
@click.option(
    "--horizon-steps",
    type=click.IntRange(min=1),
    default=15,
    show_default=True,
    help="Causal delay in samples between prediction and label availability.",
)
@click.option(
    "--t1",
    type=float,
    default=0.01,
    show_default=True,
    help="Algorithm 1 lower threshold (kW/m2): error < T1 -> skip update.",
)
@click.option(
    "--t2",
    type=float,
    default=0.1,
    show_default=True,
    help="Algorithm 1 upper threshold (kW/m2): error >= T2 -> update whole network.",
)
@click.option(
    "--ogd-lr",
    type=float,
    default=1e-5,
    show_default=True,
    help="SGD learning rate for OGD updates.",
)
@click.option(
    "--ogd-beta1",
    type=float,
    default=0.5,
    show_default=True,
    help="AdamW beta1.",
)
@click.option(
    "--ogd-beta2",
    type=float,
    default=0.975,
    show_default=True,
    help="AdamW beta2.",
)
@click.option(
    "--ogd-weight-decay",
    type=float,
    default=1e-4,
    show_default=True,
    help="SGD weight decay.",
)
@click.option(
    "--ogd-loss",
    type=click.Choice(["berhu", "huber", "mse"], case_sensitive=False),
    default="mse",
    show_default=True,
    help="Supervised loss used in OGD updates.",
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
def main(
    checkpoint: Path,
    dataset_path: Path,
    horizon_steps: int,
    t1: float,
    t2: float,
    ogd_lr: float,
    ogd_beta1: float,
    ogd_beta2: float,
    ogd_weight_decay: float,
    ogd_loss: str,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
) -> None:
    """Evaluate paper-style Algorithm 1 lagged OGD continual adaptation."""

    if t1 >= t2:
        raise click.UsageError(f"--t1 ({t1}) must be strictly less than --t2 ({t2}).")

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    dataset_name = dataset_path.name.split("_")[0]
    method_tag = "paper_alg1_ogd"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {checkpoint}...")
    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    if hasattr(model.image_encoder, "fc") and not isinstance(model.image_encoder.fc, torch.nn.Identity):
        model.image_encoder.fc = torch.nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Build loss and optimizers (one per Algorithm 1 scope)
    # ------------------------------------------------------------------
    if ogd_loss == "berhu":
        loss_fn: nn.Module = BerHuLoss(threshold=30, scale=1000)
    elif ogd_loss == "huber":
        loss_fn = nn.HuberLoss()
    else:
        loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)

    optimizer = _make_optimizer(model, ogd_lr, (ogd_beta1, ogd_beta2), ogd_weight_decay)

    print(
        f"Algorithm 1:  T1={t1}  T2={t2}  lr={ogd_lr}  betas=({ogd_beta1}, {ogd_beta2})  loss={ogd_loss!r}"
        f"  horizon_steps={horizon_steps}"
    )

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

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print("Running paper Algorithm 1 lagged OGD evaluation...\n")
    overall_metrics, per_day_metrics, predictions_data, per_day_adaptation, update_stats = evaluate(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataset=test_dataset,
        day_index_map=day_index_map,
        days_to_eval=days_to_eval,
        device=device,
        horizon_steps=horizon_steps,
        paper_t1=t1,
        paper_t2=t2,
    )

    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    if save_results:
        csv_path = f"{dataset_name}_{method_tag}_predictions.csv"
        txt_path = f"{dataset_name}_{method_tag}_results.txt"
        save_predictions_csv(predictions_data, csv_path)
        print(f"\nPredictions saved to {csv_path}")
        extra = [
            "Method: paper Algorithm 1 lagged OGD",
            f"horizon_steps={horizon_steps}",
            f"t1={t1}",
            f"t2={t2}",
            f"ogd_lr={ogd_lr}",
            f"ogd_beta1={ogd_beta1}",
            f"ogd_beta2={ogd_beta2}",
            f"ogd_weight_decay={ogd_weight_decay}",
            f"ogd_loss={ogd_loss}",
            "Adaptation summary:",
            f"considered={update_stats['considered']}",
            f"none={update_stats['none']}",
            f"head={update_stats['head']}",
            f"whole={update_stats['whole']}",
            "Per-day adaptation:",
        ]
        for day_key in sorted(per_day_adaptation.keys()):
            day_stats = per_day_adaptation[day_key]
            extra.append(
                f"{day_key}: considered={day_stats['considered']} none={day_stats['none']} "
                f"head={day_stats['head']} whole={day_stats['whole']}"
            )
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
