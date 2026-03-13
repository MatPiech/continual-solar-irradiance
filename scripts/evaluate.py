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
from typing import Any

import click
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cirr.datamodules.loading import load_dataset
from cirr.evaluation.printing import print_metrics_header, print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.training.replay import build_folsom_replay_buffer
from cirr.tta import MEMOReg, NOTEReg
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

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

    print_metrics_header()

    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])
        if not day_indices:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False, file=sys.stderr):
            image_input, irradiance_history, target_irradiance, image_name = dataset[idx]

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

            timestamp = image_name if isinstance(image_name, str) else None
            predictions_data.append(
                {
                    "timestamp": timestamp,
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
        rng.shuffle(train_samples)
        train_losses: list[float] = []

        train_starts = list(range(0, len(train_samples), batch_size))
        with tqdm(
            train_starts,
            desc=f"Few-shot epoch {epoch_idx + 1}/{epochs}",
            unit="batch",
            leave=False,
            file=sys.stderr,
        ) as batch_bar:
            for start in batch_bar:
                batch = train_samples[start : start + batch_size]
                images = torch.stack([sample_ds[sample_idx][0] for sample_ds, sample_idx in batch]).to(device)
                histories = torch.stack([sample_ds[sample_idx][1] for sample_ds, sample_idx in batch]).to(device)
                targets = torch.stack([sample_ds[sample_idx][2] for sample_ds, sample_idx in batch]).to(device)

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
        if val_samples:
            with torch.no_grad():
                val_losses: list[float] = []
                for start in range(0, len(val_samples), batch_size):
                    batch = val_samples[start : start + batch_size]
                    images = torch.stack([sample_ds[sample_idx][0] for sample_ds, sample_idx in batch]).to(device)
                    histories = torch.stack([sample_ds[sample_idx][1] for sample_ds, sample_idx in batch]).to(device)
                    targets = torch.stack([sample_ds[sample_idx][2] for sample_ds, sample_idx in batch]).to(device)
                    pred_mean = _predict_single(model, images, histories)
                    val_loss = _fewshot_compute_loss(model, pred_mean, targets)
                    val_losses.append(float(val_loss.item()))
                monitor = float(np.mean(val_losses)) if val_losses else float("inf")
        else:
            monitor = float(np.mean(train_losses)) if train_losses else float("inf")
        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mean = monitor if val_samples else float("nan")

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

        monitor_name = "val" if val_samples else "train"
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
                f"Few-shot early stopping at epoch {epoch_idx + 1} "
                f"(best_epoch={best_epoch}, best_monitor={best_monitor:.6f})",
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
                f"Few-shot replay capped by buffer size: requested={desired_replay_count} "
                f"cap={fewshot_replay_buffer_size}",
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
    "--save-results",
    "--save-result",
    is_flag=True,
    default=False,
    help="Save evaluation results and predictions CSV to disk.",
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
    fewshot_replay_ratio: float,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
    reset_per_day: bool,
) -> None:
    """Evaluate solar irradiance forecasting (baseline or TTA)."""

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

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
    method_labels = {
        "none": "baseline",
        "memo": "MEMO-Reg TTA",
        "note": "NOTE-Reg TTA",
        "fewshot": "few-shot offline adaptation",
    }
    print(f"Running {method_labels[method]} evaluation...\n")

    adaptation_stats: dict[str, float] | None = None
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

    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    if save_results:
        csv_path = f"{dataset_name}_{method_tag}_predictions.csv"
        txt_path = f"{dataset_name}_{method_tag}_results.txt"
        save_predictions_csv(predictions_data, csv_path)
        print(f"\nPredictions saved to {csv_path}")
        extra: list[str] = []
        if method == "memo":
            extra.append(
                f"TTA: MEMOReg  adapt_mode={adapt_mode}  n_aug={n_augmentations}"
                f"  niter={niter}  lr={lr}  reset_per_day={reset_per_day}"
            )
        elif method == "note":
            extra.append(
                f"TTA: NOTEReg  adapt_mode={adapt_mode}  update_every_x={update_every_x}"
                f"  niter={niter}  lr={lr}  use_learned_stats={use_learned_stats}"
                f"  bn_momentum={bn_momentum}  reset_per_day={reset_per_day}"
            )
        elif method == "fewshot" and adaptation_stats is not None:
            extra.append(
                f"Fewshot: fit_days={fewshot_days}  epochs={fewshot_epochs} "
                f"batch_size={fewshot_batch_size}  lr={fewshot_lr}  wd={fewshot_weight_decay}"
            )
            extra.append(
                f"EarlyStopping: patience={fewshot_patience}  min_delta={fewshot_min_delta} " f"val_ratio={fewshot_val_ratio}"
            )
            extra.append(
                f"Replay: ratio={fewshot_replay_ratio} "
                f"buffer_size={max(0, int(round(fewshot_days * fewshot_replay_ratio * 1000)))} "
                f"samples={int(adaptation_stats['replay_samples'])}"
            )
            extra.append(
                f"Fewshot result: epochs_trained={int(adaptation_stats['epochs_trained'])} "
                f"best_monitor={adaptation_stats['best_monitor']:.6f}"
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
