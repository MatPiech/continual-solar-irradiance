"""Evaluate solar irradiance forecasting model on test datasets."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from cirr.datamodules.loading import load_dataset
from cirr.evaluation.printing import print_day_metrics, print_metrics_header, print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.utils.dataset_utils import build_subset_loader, group_indices_by_day
from cirr.utils.metrics import compute_metrics


def evaluate_dataset(
    model: Forecaster,
    test_loader: DataLoader,
    periods: list[dict[str, Any]],
    indices: list[int] | None = None,
    device: str = "cuda",
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Evaluate model on dataset and compute per-day metrics.

    Parameters
    ----------
    model : Forecaster
        PyTorch Lightning model to evaluate.
    test_loader : DataLoader
        DataLoader for test set.
    periods : list[dict[str, Any]]
        List of period dictionaries containing metadata.
    indices : list[int] or None, optional
        Optional list of dataset indices corresponding to each batch item.
    device : str, optional
        Device to use for evaluation, by default "cuda".

    Returns
    -------
    overall_metrics : dict[str, float]
        Overall metrics (MAE, RMSE, MAPE, count).
    per_day_metrics : dict[str, dict[str, float]]
        Per-day metrics with MMDD as keys.
    predictions_data : list[dict[str, Any]]
        List of dictionaries with predictions, targets, and timestamps.
    """
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []
    predictions_data = []
    day_predictions = defaultdict(list)
    day_targets = defaultdict(list)
    current_day = None
    current_day_preds = []
    current_day_targets = []

    print_metrics_header()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Handle batch from DataLoader (collated from dataset items)
            # Dataset returns: (image_input, source_irradiances, target_irradiance, image_name)
            if isinstance(batch, list | tuple) and len(batch) == 4:
                images = batch[0].to(device)  # [1, 3, H, W]
                irradiance_history = batch[1].to(device)  # [1, history_length]
                target = batch[2].to(device)  # [1, 1]
                image_name = batch[3][0] if isinstance(batch[3], list) else batch[3]
            else:
                print(f"Unexpected batch format at index {batch_idx}, skipping...")
                continue

            # Forward pass
            output = model(images, irradiance_history)

            # Handle GNLL output (mean, variance) or direct predictions
            pred_mean = output[0] if isinstance(output, tuple) else output

            # Move to CPU and convert to numpy
            pred_np = pred_mean.cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()

            all_predictions.extend(pred_np)
            all_targets.extend(target_np)

            # Extract day information from image name
            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            day_key: str | None = None
            timestamp: str | None = None
            if isinstance(image_name, str) and len(image_name) >= 8:
                day_key = image_name[4:8]
                timestamp = image_name
            elif indices is not None and batch_idx < len(indices):
                period_image = periods[indices[batch_idx]]["history"][-1]["image_name"]
                day_key = period_image[4:8]
                timestamp = period_image

            # Store prediction data with timestamp
            predictions_data.append(
                {
                    "timestamp": timestamp,
                    "day": day_key,
                    "prediction": pred_np[0] if len(pred_np) == 1 else pred_np,
                    "target": target_np[0] if len(target_np) == 1 else target_np,
                    "error": pred_np - target_np,
                    "absolute_error": np.abs(pred_np - target_np),
                }
            )

            if day_key is not None:
                day_predictions[day_key].extend(pred_np)
                day_targets[day_key].extend(target_np)

                if current_day is None:
                    current_day = day_key

                if day_key != current_day:
                    print_day_metrics(current_day, current_day_preds, current_day_targets)
                    current_day = day_key
                    current_day_preds = []
                    current_day_targets = []

                current_day_preds.extend(pred_np)
                current_day_targets.extend(target_np)

    if current_day is not None and current_day_preds:
        print_day_metrics(current_day, current_day_preds, current_day_targets)

    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    per_day_metrics = {
        day_key: compute_metrics(np.array(day_predictions[day_key]), np.array(day_targets[day_key]))
        for day_key in sorted(day_predictions)
    }
    return overall_metrics, per_day_metrics, predictions_data


@click.command()
@click.option(
    "--checkpoint", type=click.Path(exists=True, path_type=Path), required=True, help="Path to model checkpoint file"
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prepared dataset directory",
)
@click.option(
    "--save-results",
    "--save-result",
    is_flag=True,
    default=False,
    help="Save evaluation results to files with dataset-based names",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    help="Number of days to evaluate. -1 means all days.",
)
@click.option(
    "--start-day",
    type=str,
    default="0101",
    help="Starting day in MMDD format (e.g., 0101)",
)
def main(
    checkpoint: Path,
    dataset_path: Path,
    save_results: bool,
    num_days: int,
    start_day: str,
) -> None:
    """Evaluate solar irradiance forecasting model."""
    dataset_name = dataset_path.name.split("_")[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = load_forecaster_from_checkpoint(checkpoint, device=device, strict=False)

    print(f"Loading dataset from {dataset_path}...")
    test_dataset = load_dataset(dataset_path, split="test")
    day_index_map = group_indices_by_day(test_dataset._periods)
    print(f"Test set size: {len(test_dataset)} samples")
    print(f"Number of unique days: {len(day_index_map)}\n")

    test_loader, indices = build_subset_loader(test_dataset, day_index_map, num_days, start_day)
    if indices:
        print(f"Evaluating {num_days} days starting from {start_day}")
        print(f"Total samples: {len(indices)}\n")
    else:
        print("Evaluating all days\n")

    print("Evaluating model...")
    overall_metrics, per_day_metrics, predictions_data = evaluate_dataset(
        model, test_loader, test_dataset._periods, indices=indices or None, device=device
    )

    print_overall_metrics(overall_metrics, tag=dataset_name)

    if save_results:
        csv_path = f"{dataset_name}_predictions.csv"
        txt_path = f"{dataset_name}_results.txt"
        save_predictions_csv(predictions_data, csv_path)
        print(f"Predictions saved to {csv_path}")
        save_metrics_report(
            tag=dataset_name,
            dataset_name=dataset_name,
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            overall_metrics=overall_metrics,
            per_day_metrics=per_day_metrics,
            path=txt_path,
        )
        print(f"Results saved to {txt_path}")


if __name__ == "__main__":
    main()
