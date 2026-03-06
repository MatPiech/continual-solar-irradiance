"""Evaluate solar irradiance forecasting model on test datasets.

Supports plain baseline inference and two test-time adaptation (TTA) methods:

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
import sys
from typing import Any

import click
import numpy as np
import torch
from tqdm import tqdm

from cirr.datamodules.loading import load_dataset
from cirr.evaluation.printing import print_metrics_header, print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.tta import DualPhaseAdapter, MEMOReg, NOTEReg
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Shared evaluation loop (day-by-day, works for baseline and all TTA methods)
# ---------------------------------------------------------------------------


def evaluate(
    tta_model: MEMOReg | NOTEReg | DualPhaseAdapter | None,
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
                # DualPhaseAdapter consumes the ground truth for lagged OGD
                if isinstance(tta_model, DualPhaseAdapter):
                    mean, _ = tta_model(images, irr_history, ground_truth=target)
                else:
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
    type=click.Choice(["none", "memo", "note", "dual"], case_sensitive=False),
    default="none",
    show_default=True,
    help="TTA method: 'none'=baseline, 'memo'=MEMO-Reg, 'note'=NOTE-Reg, 'dual'=Dual-Phase OGD+TTA.",
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
    "--cache-size",
    type=int,
    default=3,
    show_default=True,
    help="(Dual) FIFO cache depth in time steps (15 min / 5 min interval = 3).",
)
@click.option(
    "--ogd-lr",
    type=float,
    default=1e-4,
    show_default=True,
    help="(Dual) Learning rate for the supervised OGD optimizer.",
)
@click.option(
    "--ogd-scope",
    type=click.Choice(["head_only", "head_and_encoder"], case_sensitive=False),
    default="head_only",
    show_default=True,
    help="(Dual) OGD update scope: 'head_only' or 'head_and_encoder' (last encoder block).",
)
@click.option(
    "--ogd-loss",
    type=click.Choice(["berhu", "huber"], case_sensitive=False),
    default="berhu",
    show_default=True,
    help="(Dual) Supervised loss used in the OGD phase.",
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
    cache_size: int,
    ogd_lr: float,
    ogd_scope: str,
    ogd_loss: str,
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
    method_tag = "baseline" if method == "none" else f"{method}_reg_{adapt_mode}"

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
    tta_model: MEMOReg | NOTEReg | DualPhaseAdapter | None = None
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
    elif method == "dual":
        print(
            f"Wrapping model with DualPhaseAdapter ("
            f"cache_size={cache_size}, "
            f"ogd_lr={ogd_lr}, ogd_scope={ogd_scope!r}, ogd_loss={ogd_loss!r}, "
            f"tta_lr={lr}, n_augmentations={n_augmentations}, niter_tta={niter})..."
        )
        tta_model = DualPhaseAdapter(
            model=model,
            cache_size=cache_size,
            ogd_lr=ogd_lr,
            ogd_scope=ogd_scope,
            ogd_loss=ogd_loss,
            tta_lr=lr,
            tta_weight_decay=weight_decay,
            n_augmentations=n_augmentations,
            niter_tta=niter,
        ).to(device)
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
    method_labels = {"none": "baseline", "memo": "MEMO-Reg TTA", "note": "NOTE-Reg TTA", "dual": "Dual-Phase TTA"}
    print(f"Running {method_labels[method]} evaluation...\n")

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
        elif method == "dual":
            extra.append(
                f"TTA: DualPhaseAdapter  cache_size={cache_size}"
                f"  ogd_lr={ogd_lr}  ogd_scope={ogd_scope}  ogd_loss={ogd_loss}"
                f"  tta_lr={lr}  n_aug={n_augmentations}  niter_tta={niter}"
                f"  reset_per_day={reset_per_day}"
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
