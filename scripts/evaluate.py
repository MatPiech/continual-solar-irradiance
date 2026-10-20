"""Evaluate solar irradiance forecasting model on test datasets.

Supports plain baseline inference and two test-time adaptation (TTA) methods.
For the Dual-Phase OGD+TTA method see ``scripts/evaluate_dual_phase.py``.

MEMO-Reg  (``--method memo``)
    Per-sample episodic adaptation: minimises prediction variance across *N*
    augmented views of each sky image before predicting on the original.

NOTE-Reg  (``--method note``)
    Online buffer-based adaptation: collects recent samples in a FIFO deque
    and updates BN (or full-model) params via prediction variance loss every
    ``--update-every-x`` samples.

Usage examples
--------------
# Baseline (no TTA):
python scripts/evaluate.py \\
    --checkpoint checkpoints/epoch_024.ckpt \\
    --dataset-path data/poznan_prepared

# MEMO-Reg, 3-day smoke test:
python scripts/evaluate.py \\
    --checkpoint checkpoints/epoch_024.ckpt \\
    --dataset-path data/poznan_prepared \\
    --method memo --n-augmentations 4 --niter 1 --num-days 3

# NOTE-Reg, BN-only, buffer of 32:
python scripts/evaluate.py \\
    --checkpoint checkpoints/epoch_024.ckpt \\
    --dataset-path data/poznan_prepared \\
    --method note --adapt-mode bn_only --update-every-x 32
"""

from pathlib import Path
import random
import sys
import time
from typing import Any

import click
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cirr.evaluation.campaign import SOURCE_CHECKPOINT, CampaignData, load_campaign_days, write_campaign
from cirr.evaluation.online import OnlineEvalResult
from cirr.evaluation.printing import print_metrics_header, print_overall_metrics
from cirr.models.forecaster import Forecaster
from cirr.models.loading import checkpoint_physics, load_forecaster_from_checkpoint
from cirr.physics.clear_sky import DEFAULT_MIN_GHI_CLEAR
from cirr.training.replay import build_folsom_replay_buffer
from cirr.tta import MEMOReg, NOTEReg
from cirr.utils.metrics import compute_metrics
from cirr.utils.provenance import set_determinism

# ---------------------------------------------------------------------------
# Shared evaluation loop (day-by-day, works for baseline and all TTA methods)
# ---------------------------------------------------------------------------


def evaluate(
    tta_model: MEMOReg | NOTEReg | None,
    base_model: Forecaster,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    reset_per_day: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Run the full day-by-day evaluation loop with optional TTA.

    Parameters
    ----------
    tta_model : MEMOReg | NOTEReg | None
        Wrapped TTA model.  Pass ``None`` for plain baseline inference.
    base_model : Forecaster
        Original model, used directly when ``tta_model`` is ``None``.
    dataset : SkyImageDataset
        Test dataset.
    day_index_map : dict[str, list[int]]
        Maps MMDD keys to lists of dataset indices belonging to that day.
    days_to_eval : list[str]
        Ordered list of MMDD keys to evaluate.
    device : str
        Target device string (``'cuda'`` or ``'cpu'``).
    reset_per_day : bool, optional
        If ``True`` and a TTA model is active, reset its weights to the
        initial checkpoint state after each evaluated day.

    Returns
    -------
    overall_metrics, per_day_metrics, predictions_data
    """
    if tta_model is None:
        base_model.eval()
        base_model.to(device)

    all_predictions: list[float] = []
    all_targets: list[float] = []
    predictions_data: list[dict[str, Any]] = []
    per_day_metrics: dict[str, dict[str, float]] = {}
    events_seen = 0

    print_metrics_header()

    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])
        if not day_indices:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False, file=sys.stderr):
            image_input, irradiance_history, ghi_clear, target_irradiance, image_name = dataset[idx]

            images = image_input.unsqueeze(0).to(device)
            irr_history = irradiance_history.unsqueeze(0).to(device)
            target = target_irradiance.unsqueeze(0).to(device)

            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            if tta_model is not None:
                mean, _ = tta_model(images, irr_history)
            else:
                with torch.no_grad():
                    output = base_model(images, irr_history)
                    mean = output[0] if isinstance(output, tuple) else output

            pred_np = mean.detach().cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()

            day_preds.extend(pred_np)
            day_targets.extend(target_np)
            all_predictions.extend(pred_np)
            all_targets.extend(target_np)

            # TTA wrappers count their own update events, so the campaign
            # action mix is what the method actually did, not what it declares.
            events_now = getattr(tta_model, "adapt_events", 0) if tta_model is not None else 0
            adapted = events_now > events_seen
            events_seen = events_now

            timestamp = image_name if isinstance(image_name, str) else None
            predictions_data.append(
                {
                    "timestamp": timestamp,
                    "day": day_key,
                    "prediction": pred_np[0] if len(pred_np) == 1 else pred_np,
                    "target": target_np[0] if len(target_np) == 1 else target_np,
                    "error": float(pred_np[0] - target_np[0]),
                    "absolute_error": float(abs(pred_np[0] - target_np[0])),
                    "update_considered": tta_model is not None,
                    "adapted": adapted,
                    # TTA backpropagates into the encoder in both adapt modes,
                    # so its update events report as the full-model action.
                    "adaptation_scope": "whole" if adapted else "none",
                }
            )

        if day_preds:
            day_metrics = compute_metrics(np.array(day_preds), np.array(day_targets))
            per_day_metrics[day_key] = day_metrics
            print(
                f"{day_key:<6} {day_metrics['mae']:<10.4f} {day_metrics['rmse']:<10.4f} "
                f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% {day_metrics['count']:<8}",
                flush=True,
            )

        if reset_per_day and tta_model is not None:
            tta_model.reset()

    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    return overall_metrics, per_day_metrics, predictions_data


def _predict_single(model: Forecaster, images: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Return model mean prediction for a batch."""
    output = model(images, history)
    return output[0] if isinstance(output, tuple) else output


def _fewshot_compute_loss(model: Forecaster, pred_mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute supervised adaptation loss for few-shot fitting."""
    if hasattr(model, "loss_fn"):
        try:
            return model.loss_fn(pred_mean, target)
        except TypeError:
            pass
    return F.mse_loss(pred_mean, target)


class _PairDataset(torch.utils.data.Dataset):
    """Indexes a list of ``(dataset, index)`` pairs as one dataset.

    The few-shot fit pools the deployment site with a Folsom replay buffer, so its
    samples come from two datasets.  Wrapping the pairs lets a DataLoader decode
    and transform them in worker processes; before this the fit indexed the
    datasets inline and spent about 1.5 s per batch of 64 with the GPU idle.
    """

    def __init__(self, pairs: list[tuple[Any, int]]):
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds, idx = self._pairs[i]
        image, history, _ghi_clear, target, _name = ds[idx]
        return image, history, target


def _worker_init(_worker_id: int) -> None:
    # Each worker decodes with OpenCV; leaving its thread pool enabled makes the
    # workers fight each other for the same cores.
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _fewshot_loader(pairs: list[tuple[Any, int]], batch_size: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        _PairDataset(pairs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=workers > 0,
        persistent_workers=workers > 0,
        worker_init_fn=_worker_init if workers > 0 else None,
        generator=generator,
    )


@torch.enable_grad()
def _fewshot_fit(
    model: Forecaster,
    dataset: Any,
    adaptation_indices: list[int],
    replay_samples: list[tuple[Any, int]],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    val_ratio: float,
    seed: int,
    device: str,
    workers: int = 0,
) -> dict[str, float]:
    """Fit model once on adaptation data with early stopping."""
    pooled_samples: list[tuple[Any, int]] = [(dataset, idx) for idx in adaptation_indices] + replay_samples
    if not pooled_samples:
        return {
            "epochs_trained": 0.0,
            "best_monitor": np.nan,
            "train_samples": 0.0,
            "val_samples": 0.0,
            "replay_samples": float(len(replay_samples)),
        }

    rng = random.Random(seed)
    rng.shuffle(pooled_samples)

    n_total = len(pooled_samples)
    if n_total < 2:
        val_size = 0
    else:
        val_size = int(round(n_total * val_ratio))
        val_size = max(1, min(val_size, n_total - 1))

    val_samples = pooled_samples[:val_size]
    train_samples = pooled_samples[val_size:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Built once: persistent workers survive the epoch loop, and the seeded
    # generator keeps the shuffle reproducible, which S3 asserts.
    train_loader = _fewshot_loader(train_samples, batch_size, workers, shuffle=True, seed=seed)
    val_loader = _fewshot_loader(val_samples, batch_size, workers, shuffle=False, seed=seed) if val_samples else None

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_monitor = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epochs_trained = 0

    print(
        f"Few-shot training: train_samples={len(train_samples)} val_samples={len(val_samples)} "
        f"replay_samples={len(replay_samples)} lr={lr} wd={weight_decay} "
        f"max_epochs={epochs} patience={patience} min_delta={min_delta}",
        flush=True,
    )

    for epoch_idx in range(epochs):
        model.train()
        train_losses: list[float] = []

        with tqdm(
            train_loader,
            desc=f"Few-shot epoch {epoch_idx + 1}/{epochs}",
            unit="batch",
            leave=False,
            file=sys.stderr,
        ) as batch_bar:
            for images, histories, targets in batch_bar:
                images = images.to(device, non_blocking=True)
                histories = histories.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad()
                pred_mean = _predict_single(model, images, histories)
                loss = _fewshot_compute_loss(model, pred_mean, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

                running_train = float(np.mean(train_losses)) if train_losses else float("nan")
                batch_bar.set_postfix(train=f"{running_train:.4f}")

        model.eval()
        if val_loader is not None:
            with torch.no_grad():
                val_losses: list[float] = []
                for images, histories, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    histories = histories.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    pred_mean = _predict_single(model, images, histories)
                    val_loss = _fewshot_compute_loss(model, pred_mean, targets)
                    val_losses.append(float(val_loss.item()))
                monitor = float(np.mean(val_losses)) if val_losses else float("inf")
        else:
            monitor = float(np.mean(train_losses)) if train_losses else float("inf")
        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mean = monitor if val_loader is not None else float("nan")

        epochs_trained += 1
        if monitor + min_delta < best_monitor:
            best_monitor = monitor
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch_idx + 1
            stale_epochs = 0
            improved = "yes"
        else:
            stale_epochs += 1
            improved = "no"

        monitor_name = "val" if val_loader is not None else "train"
        print(
            f"Few-shot epoch {epoch_idx + 1:>3}/{epochs}: "
            f"train_loss={train_mean:.6f} "
            f"val_loss={val_mean:.6f} "
            f"monitor({monitor_name})={monitor:.6f} "
            f"best={best_monitor:.6f}@{best_epoch} "
            f"improved={improved} stale={stale_epochs}/{patience}",
            flush=True,
        )

        if stale_epochs >= patience:
            print(
                f"Few-shot early stopping at epoch {epoch_idx + 1} (best_epoch={best_epoch}, best_monitor={best_monitor:.6f})",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    model.eval()

    return {
        "epochs_trained": float(epochs_trained),
        "best_monitor": best_monitor,
        "best_epoch": float(best_epoch),
        "train_samples": float(len(train_samples)),
        "val_samples": float(len(val_samples)),
        "replay_samples": float(len(replay_samples)),
    }


def evaluate_fewshot(
    model: Forecaster,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    fewshot_days: int,
    fewshot_epochs: int,
    fewshot_batch_size: int,
    fewshot_lr: float,
    fewshot_weight_decay: float,
    fewshot_patience: int,
    fewshot_min_delta: float,
    fewshot_val_ratio: float,
    fewshot_seed: int,
    fewshot_workers: int,
    fewshot_replay_ratio: float,
    fewshot_replay_buffer_size: int,
    folsom_path: Path,
    device: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]], dict[str, float]]:
    """Evaluate with a single few-shot adaptation after the first N days."""
    if fewshot_days <= 0:
        raise ValueError("--fewshot-days must be >= 1 for method='fewshot'.")

    fit_days = days_to_eval[:fewshot_days]
    predict_days = days_to_eval[fewshot_days:]
    fit_indices = [idx for day in fit_days for idx in day_index_map.get(day, [])]

    replay_samples: list[tuple[Any, int]] = []
    desired_replay_count = int(round(len(fit_indices) * fewshot_replay_ratio))
    if desired_replay_count > 0 and fewshot_replay_buffer_size > 0:
        effective_replay_count = min(desired_replay_count, fewshot_replay_buffer_size)
        if effective_replay_count < desired_replay_count:
            print(
                f"Few-shot replay capped by buffer size: requested={desired_replay_count} cap={fewshot_replay_buffer_size}",
                flush=True,
            )
        try:
            # Replay is used once in few-shot adaptation, so we directly build
            # the final replay set with Folsom-distribution-aware stratified sampling.
            replay_samples, _ = build_folsom_replay_buffer(folsom_path, effective_replay_count)
        except FileNotFoundError:
            print(
                f"Few-shot replay skipped: could not load Folsom training split from {folsom_path}",
                flush=True,
            )

    print(
        f"Few-shot setup: fit_days={len(fit_days)} predict_days={len(predict_days)} "
        f"fit_samples={len(fit_indices)} replay_samples={len(replay_samples)} "
        f"epochs={fewshot_epochs} batch_size={fewshot_batch_size}",
        flush=True,
    )

    print("Few-shot phase 1: baseline inference on adaptation days.\n", flush=True)
    base_overall, base_per_day, base_predictions = evaluate(
        tta_model=None,
        base_model=model,
        dataset=dataset,
        day_index_map=day_index_map,
        days_to_eval=fit_days,
        device=device,
        reset_per_day=False,
    )

    adaptation_stats = _fewshot_fit(
        model=model,
        dataset=dataset,
        adaptation_indices=fit_indices,
        replay_samples=replay_samples,
        epochs=fewshot_epochs,
        batch_size=fewshot_batch_size,
        lr=fewshot_lr,
        weight_decay=fewshot_weight_decay,
        patience=fewshot_patience,
        min_delta=fewshot_min_delta,
        val_ratio=fewshot_val_ratio,
        seed=fewshot_seed,
        workers=fewshot_workers,
        device=device,
    )

    print(
        f"Few-shot training finished: epochs_trained={int(adaptation_stats['epochs_trained'])} "
        f"best_epoch={int(adaptation_stats['best_epoch'])} "
        f"best_monitor={adaptation_stats['best_monitor']:.6f}",
        flush=True,
    )

    print("\nFew-shot phase 2: adapted inference on remaining days.\n", flush=True)
    adapted_overall, adapted_per_day, adapted_predictions = evaluate(
        tta_model=None,
        base_model=model,
        dataset=dataset,
        day_index_map=day_index_map,
        days_to_eval=predict_days,
        device=device,
        reset_per_day=False,
    )
    _ = base_overall
    _ = adapted_overall

    predictions_data = base_predictions + adapted_predictions
    per_day_metrics = {**base_per_day, **adapted_per_day}

    if predictions_data:
        all_predictions = np.array([row["prediction"] for row in predictions_data], dtype=float)
        all_targets = np.array([row["target"] for row in predictions_data], dtype=float)
        overall_metrics = compute_metrics(all_predictions, all_targets)
    else:
        overall_metrics = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "wape": float("nan"),
            "smape": float("nan"),
            "count": 0,
        }

    return overall_metrics, per_day_metrics, predictions_data, adaptation_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=str(SOURCE_CHECKPOINT),
    show_default=True,
    help="Source model. The default is the checkpoint Chapter 5 uses.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Run folder. Defaults to results/ch4/<site>/<method>/seed<k>.",
)
@click.option(
    "--min-ghi-clear",
    type=float,
    default=DEFAULT_MIN_GHI_CLEAR,
    show_default=True,
    help="Daylight filter (kW/m^2), shared with the federated benchmark. 0 disables it.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--fewshot-workers",
    type=click.IntRange(min=0),
    default=8,
    show_default=True,
    help="DataLoader workers for the few-shot fit. The fit decodes and transforms every image, "
    "so at 0 it is data-loading bound with the GPU idle.",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prepared dataset directory (e.g. data/poznan_prepared).",
)
@click.option(
    "--method",
    type=click.Choice(["none", "memo", "note", "fewshot"], case_sensitive=False),
    default="none",
    show_default=True,
    help="Evaluation method: 'none'=baseline, 'memo'=MEMO-Reg, 'note'=NOTE-Reg, 'fewshot'=offline N-day adaptation.",
)
@click.option(
    "--adapt-mode",
    type=click.Choice(["all", "bn_only"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Parameters to update: 'all' or 'bn_only' (BatchNorm affine only).",
)
@click.option(
    "--n-augmentations",
    type=int,
    default=8,
    show_default=True,
    help="(MEMO) Number of augmented views per test image.",
)
@click.option(
    "--update-every-x",
    type=int,
    default=64,
    show_default=True,
    help="(NOTE) FIFO buffer capacity and adaptation trigger interval (samples).",
)
@click.option(
    "--use-learned-stats",
    is_flag=True,
    default=False,
    help="(NOTE) Use BN running statistics with momentum instead of per-batch stats.",
)
@click.option(
    "--bn-momentum",
    type=float,
    default=0.1,
    show_default=True,
    help="(NOTE) Momentum for BN running stats when --use-learned-stats is set.",
)
@click.option(
    "--niter",
    type=int,
    default=1,
    show_default=True,
    help="Number of gradient steps per adaptation trigger.",
)
@click.option(
    "--lr",
    type=float,
    default=1e-5,
    show_default=True,
    help="Adaptation learning rate.",
)
@click.option(
    "--weight-decay",
    type=float,
    default=0.01,
    show_default=True,
    help="Optimizer weight decay.",
)
@click.option(
    "--fewshot-days",
    type=click.IntRange(min=1),
    default=7,
    show_default=True,
    help="(Fewshot) Number of initial days used for one-time adaptation.",
)
@click.option(
    "--fewshot-epochs",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="(Fewshot) Maximum adaptation epochs.",
)
@click.option(
    "--fewshot-batch-size",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="(Fewshot) Mini-batch size for adaptation.",
)
@click.option(
    "--fewshot-lr",
    type=float,
    default=1e-5,
    show_default=True,
    help="(Fewshot) Learning rate for offline adaptation.",
)
@click.option(
    "--fewshot-weight-decay",
    type=float,
    default=1e-4,
    show_default=True,
    help="(Fewshot) Weight decay for offline adaptation optimizer.",
)
@click.option(
    "--fewshot-patience",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="(Fewshot) Early-stopping patience in epochs.",
)
@click.option(
    "--fewshot-min-delta",
    type=float,
    default=1e-5,
    show_default=True,
    help="(Fewshot) Minimum monitor improvement to reset patience.",
)
@click.option(
    "--fewshot-val-ratio",
    type=click.FloatRange(min=0.0, max=0.8),
    default=0.1,
    show_default=True,
    help="(Fewshot) Validation split ratio used for early stopping.",
)
@click.option(
    "--fewshot-seed",
    type=int,
    default=42,
    show_default=True,
    help="(Fewshot) Random seed for sample split and replay sampling.",
)
@click.option(
    "--fewshot-replay-ratio",
    type=click.FloatRange(min=0.0),
    default=0.0,
    show_default=True,
    help="(Fewshot) Ratio of replay samples relative to target adaptation samples.",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    show_default=True,
    help="Number of days to evaluate. -1 means all.",
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
    help="Device to use ('cuda' or 'cpu').",
)
@click.option(
    "--horizon-steps",
    type=click.IntRange(min=1),
    default=15,
    show_default=True,
    help="Supervision lag, recorded with the run and used for the buffer accounting.",
)
@click.option(
    "--reset-per-day",
    is_flag=True,
    default=False,
    help="(TTA) Reset model to initial checkpoint weights after each evaluated day.",
)
def main(
    checkpoint: Path,
    dataset_path: Path,
    method: str,
    adapt_mode: str,
    n_augmentations: int,
    update_every_x: int,
    use_learned_stats: bool,
    bn_momentum: float,
    niter: int,
    lr: float,
    weight_decay: float,
    fewshot_days: int,
    fewshot_epochs: int,
    fewshot_batch_size: int,
    fewshot_lr: float,
    fewshot_weight_decay: float,
    fewshot_patience: int,
    fewshot_min_delta: float,
    fewshot_val_ratio: float,
    fewshot_seed: int,
    fewshot_workers: int,
    fewshot_replay_ratio: float,
    num_days: int,
    start_day: str,
    device: str,
    horizon_steps: int,
    reset_per_day: bool,
    out_dir: Path | None,
    min_ghi_clear: float,
    seed: int,
) -> None:
    """Evaluate solar irradiance forecasting (baseline or TTA)."""

    set_determinism(seed)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    checkpoint = Path(checkpoint)
    dataset_name = dataset_path.name.split("_")[0]
    if method == "none":
        method_tag = "baseline"
    elif method == "fewshot":
        method_tag = f"fewshot_d{fewshot_days}"
    else:
        method_tag = f"{method}_reg_{adapt_mode}"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {checkpoint}...")
    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    if hasattr(model.image_encoder, "fc") and not isinstance(model.image_encoder.fc, torch.nn.Identity):
        model.image_encoder.fc = torch.nn.Identity()

    # ------------------------------------------------------------------
    # Wrap with TTA (if requested)
    # ------------------------------------------------------------------
    tta_model: MEMOReg | NOTEReg | None = None
    if method == "memo":
        print(
            f"Wrapping model with MEMOReg ("
            f"adapt_mode={adapt_mode!r}, n_augmentations={n_augmentations}, "
            f"niter={niter}, lr={lr})..."
        )
        tta_model = MEMOReg(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            n_augmentations=n_augmentations,
            niter=niter,
            episodic=True,
            adapt_mode=adapt_mode,
        ).to(device)
    elif method == "note":
        print(
            f"Wrapping model with NOTEReg ("
            f"adapt_mode={adapt_mode!r}, update_every_x={update_every_x}, "
            f"niter={niter}, lr={lr}, use_learned_stats={use_learned_stats})..."
        )
        tta_model = NOTEReg(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            update_every_x=update_every_x,
            niter=niter,
            adapt_mode=adapt_mode,
            use_learned_stats=use_learned_stats,
            bn_momentum=bn_momentum,
        ).to(device)
    elif method == "fewshot":
        print(
            f"Few-shot mode (fit_days={fewshot_days}, epochs={fewshot_epochs}, "
            f"patience={fewshot_patience}, replay_ratio={fewshot_replay_ratio})..."
        )
    else:
        print("No TTA — running baseline inference.")

    # ------------------------------------------------------------------
    # Load dataset and determine days to evaluate
    # ------------------------------------------------------------------
    print(f"Loading dataset from {dataset_path}...")
    data: CampaignData = load_campaign_days(
        dataset_path, start_day, num_days, min_ghi_clear, physics=checkpoint_physics(checkpoint)
    )
    test_dataset, day_index_map, days_to_eval = data.dataset, data.day_map, data.days
    print(
        f"{data.site}: {len(data.days)} days, {data.n_samples} lit samples "
        f"({data.dropped_total} dark samples dropped at min_ghi_clear={min_ghi_clear})\n"
    )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    method_labels = {
        "none": "baseline",
        "memo": "MEMO-Reg TTA",
        "note": "NOTE-Reg TTA",
        "fewshot": "few-shot offline adaptation",
    }
    print(f"Running {method_labels[method]} evaluation...\n")

    adaptation_stats: dict[str, float] | None = None
    started = time.perf_counter()
    if method == "fewshot":
        folsom_path = dataset_path.parent / "folsom_prepared"
        computed_replay_buffer_size = max(0, int(round(fewshot_days * fewshot_replay_ratio * 1000)))
        print(
            f"Few-shot replay buffer size computed as {computed_replay_buffer_size} "
            f"(fewshot_days * fewshot_replay_ratio * 1000)",
            flush=True,
        )
        overall_metrics, per_day_metrics, predictions_data, adaptation_stats = evaluate_fewshot(
            model=model,
            dataset=test_dataset,
            day_index_map=day_index_map,
            days_to_eval=days_to_eval,
            fewshot_days=fewshot_days,
            fewshot_epochs=fewshot_epochs,
            fewshot_batch_size=fewshot_batch_size,
            fewshot_lr=fewshot_lr,
            fewshot_weight_decay=fewshot_weight_decay,
            fewshot_patience=fewshot_patience,
            fewshot_min_delta=fewshot_min_delta,
            fewshot_val_ratio=fewshot_val_ratio,
            fewshot_seed=fewshot_seed,
            fewshot_workers=fewshot_workers,
            fewshot_replay_ratio=fewshot_replay_ratio,
            fewshot_replay_buffer_size=computed_replay_buffer_size,
            folsom_path=folsom_path,
            device=device,
        )
    else:
        overall_metrics, per_day_metrics, predictions_data = evaluate(
            tta_model=tta_model,
            base_model=model,
            dataset=test_dataset,
            day_index_map=day_index_map,
            days_to_eval=days_to_eval,
            device=device,
            reset_per_day=reset_per_day,
        )

    wall_time_s = time.perf_counter() - started
    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    config: dict[str, Any] = {
        "method": method_tag,
        "seed": seed,
        "start_day": start_day,
        "num_days": num_days,
        "reset_per_day": reset_per_day,
        "dataset_path": str(dataset_path),
    }
    if method in ("memo", "note"):
        config.update(adapt_mode=adapt_mode, niter=niter, lr=lr, weight_decay=weight_decay)
        config["adapt_events"] = int(getattr(tta_model, "adapt_events", 0))
    if method == "memo":
        config["n_augmentations"] = n_augmentations
    if method == "note":
        config.update(update_every_x=update_every_x, use_learned_stats=use_learned_stats, bn_momentum=bn_momentum)
    if method == "fewshot":
        config.update(
            fewshot_days=fewshot_days,
            fewshot_epochs=fewshot_epochs,
            fewshot_batch_size=fewshot_batch_size,
            fewshot_lr=fewshot_lr,
            fewshot_weight_decay=fewshot_weight_decay,
            fewshot_patience=fewshot_patience,
            fewshot_val_ratio=fewshot_val_ratio,
            fewshot_seed=fewshot_seed,
            fewshot_workers=fewshot_workers,
            fewshot_replay_ratio=fewshot_replay_ratio,
        )
        if adaptation_stats is not None:
            config["fewshot_result"] = {k: float(v) for k, v in adaptation_stats.items()}

    out_dir = Path(out_dir) if out_dir else Path("results/ch4") / data.site / method_tag / f"seed{seed}"
    write_campaign(
        out_dir,
        result=OnlineEvalResult(overall_metrics, per_day_metrics, predictions_data, {}, {}),
        data=data,
        config=config,
        checkpoint=checkpoint,
        horizon_steps=horizon_steps,
        min_ghi_clear=min_ghi_clear,
        wall_time_s=wall_time_s,
        device=device,
    )
    print(f"\nRun written to {out_dir}")


if __name__ == "__main__":
    main()
