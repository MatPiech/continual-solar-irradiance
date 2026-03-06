"""Evaluate solar irradiance forecasting with MEMO-Reg test-time adaptation.

MEMO-Reg (Prediction Variance Minimisation) adapts the model at inference time
for each sky image by minimising the variance of predictions across N augmented
views, then returning the prediction on the original unaugmented image.

Usage examples
--------------
# Basic TTA evaluation on Poznan dataset:
python scripts/evaluate_tta.py \\
    --checkpoint checkpoints/epoch_012.ckpt \\
    --dataset-path data/poznan_prepared

# BN-only adaptation, more augmentations:
python scripts/evaluate_tta.py \\
    --checkpoint checkpoints/epoch_012.ckpt \\
    --dataset-path data/poznan_prepared \\
    --adapt-mode bn_only --n-augmentations 16

# Smoke test (3 days, fast):
python scripts/evaluate_tta.py \\
    --checkpoint checkpoints/epoch_012.ckpt \\
    --dataset-path data/poznan_prepared \\
    --num-days 3 --n-augmentations 4 --niter 1

# Disable TTA for baseline comparison:
python scripts/evaluate_tta.py \\
    --checkpoint checkpoints/epoch_012.ckpt \\
    --dataset-path data/poznan_prepared \\
    --no-tta
"""

from pathlib import Path
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
from cirr.tta import MEMOReg
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_with_tta(
    tta_model: MEMOReg | None,
    base_model: Forecaster,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Run the full evaluation loop with (or without) TTA, processing day by day.

    Parameters
    ----------
    tta_model : MEMOReg or None
        Wrapped TTA model.  Pass ``None`` to run baseline inference.
    base_model : Forecaster
        Original model (used for baseline when ``tta_model`` is ``None``).
    dataset : SkyImageDataset
        Test dataset containing periods.
    day_index_map : dict[str, list[int]]
        Dictionary mapping MMDD to list of period indices.
    days_to_eval : list[str]
        List of day keys (MMDD) to evaluate.
    device : str
        Target device.

    Returns
    -------
    overall_metrics, per_day_metrics, predictions_data
    """
    if tta_model is None:
        base_model.eval()
        base_model.to(device)
    # tta_model keeps its own internal state; no explicit .eval() needed.

    all_predictions: list[float] = []
    all_targets: list[float] = []
    predictions_data: list[dict[str, Any]] = []
    per_day_metrics: dict[str, dict[str, float]] = {}

    print_metrics_header()

    # Process each day sequentially
    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])

        if len(day_indices) == 0:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False):
            # Get sample from dataset
            image_input, irradiance_history, target_irradiance, image_name = dataset[idx]

            # Move to device and add batch dimension
            images = image_input.unsqueeze(0).to(device)  # [1, C, H, W]
            irr_history = irradiance_history.unsqueeze(0).to(device)  # [1, history_len]
            target = target_irradiance.unsqueeze(0).to(device)  # [1, 1] or [1]

            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            # Forward (with or without TTA)
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

            # Extract timestamp
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

        # Compute and print metrics for this day
        if len(day_preds) > 0:
            day_metrics = compute_metrics(np.array(day_preds), np.array(day_targets))
            per_day_metrics[day_key] = day_metrics
            print(
                f"{day_key:<6} {day_metrics['mae']:<10.4f} {day_metrics['rmse']:<10.4f} "
                f"{day_metrics['mape']:<16.2%} {day_metrics['wape']:<10.2f}% "
                f"{day_metrics['smape']:<10.2f}% {day_metrics['count']:<8}",
                flush=True,
            )

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
    "--no-tta",
    is_flag=True,
    default=False,
    help="Disable TTA and run plain inference (baseline comparison).",
)
@click.option(
    "--adapt-mode",
    type=click.Choice(["all", "bn_only"], case_sensitive=False),
    default="all",
    show_default=True,
    help=(
        "Which parameters to update during adaptation. "
        "'all' updates the full model; 'bn_only' updates only BatchNorm affine params."
    ),
)
@click.option(
    "--n-augmentations",
    type=int,
    default=8,
    show_default=True,
    help="Number of augmented views per test image.",
)
@click.option(
    "--niter",
    type=int,
    default=1,
    show_default=True,
    help="Number of gradient steps per test image.",
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
    is_flag=True,
    default=False,
    help="Save evaluation results and predictions CSV to disk.",
)
def main(
    checkpoint: Path,
    dataset_path: Path,
    no_tta: bool,
    adapt_mode: str,
    n_augmentations: int,
    niter: int,
    lr: float,
    weight_decay: float,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
) -> None:
    """Evaluate with MEMO-Reg Test-Time Adaptation on a sky-image dataset."""

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    dataset_name = dataset_path.name.split("_")[0]
    method_tag = "baseline" if no_tta else f"memo_reg_{adapt_mode}"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {checkpoint}...")
    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    # For ResNet checkpoints built with an older timm where fc was not replaced
    # with Identity, correct it now so the encoder outputs features, not logits.
    if hasattr(model.image_encoder, "fc") and not isinstance(model.image_encoder.fc, torch.nn.Identity):
        model.image_encoder.fc = torch.nn.Identity()

    # ------------------------------------------------------------------
    # Wrap with TTA
    # ------------------------------------------------------------------
    tta_model: MEMOReg | None = None
    if not no_tta:
        print(
            f"Wrapping model with MEMOReg ("
            f"adapt_mode={adapt_mode!r}, "
            f"n_augmentations={n_augmentations}, "
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
    else:
        print("TTA disabled — running baseline inference.")

    # ------------------------------------------------------------------
    # Load dataset and determine days to evaluate
    # ------------------------------------------------------------------
    print(f"Loading dataset from {dataset_path}...")
    test_dataset = load_dataset(dataset_path, split="test")
    day_index_map = group_indices_by_day(test_dataset._periods)
    print(f"Test set size: {len(test_dataset)} samples")
    print(f"Number of unique days: {len(day_index_map)}\n")

    # Generate list of days to evaluate
    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day: {start_day}")
    start_idx = all_days.index(start_day)

    days_to_eval = all_days[start_idx : start_idx + num_days] if num_days > 0 else all_days[start_idx:]

    # Filter to only days that exist in the dataset
    days_to_eval = [day for day in days_to_eval if day in day_index_map]

    total_samples = sum(len(day_index_map[day]) for day in days_to_eval)
    print(f"Evaluating {len(days_to_eval)} days starting from {start_day}")
    print(f"Total samples: {total_samples}\n")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print(f"Running {'MEMO-Reg TTA' if not no_tta else 'baseline'} evaluation...\n")
    overall_metrics, per_day_metrics, predictions_data = evaluate_with_tta(
        tta_model=tta_model,
        base_model=model,
        dataset=test_dataset,
        day_index_map=day_index_map,
        days_to_eval=days_to_eval,
        device=device,
    )

    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    if save_results:
        csv_path = f"{dataset_name}_{method_tag}_predictions.csv"
        txt_path = f"{dataset_name}_{method_tag}_results.txt"
        save_predictions_csv(predictions_data, csv_path)
        print(f"\nPredictions saved to {csv_path}")
        extra: list[str] = []
        if not no_tta:
            extra.append(f"TTA: MEMOReg  adapt_mode={adapt_mode}  n_aug={n_augmentations}  niter={niter}  lr={lr}")
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
