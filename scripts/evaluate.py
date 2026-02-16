"""Evaluate solar irradiance forecasting model on test datasets."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from cirr.datamodules.forecasting import ForecastingDataModule
from cirr.models.forecaster import Forecaster
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
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

    print("Day    MAE        RMSE       MAPE       WAPE       SMAPE      Samples ")
    print("------------------------------------------------------------------------")

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

            day_key = None
            timestamp = None
            if isinstance(image_name, str) and len(image_name) >= 8:
                day_key = image_name[4:8]  # MMDD format
                timestamp = image_name  # Use full image name as timestamp
            elif indices is not None and batch_idx < len(indices):
                period = periods[indices[batch_idx]]
                period_image = period["history"][-1]["image_name"]
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
                    day_metrics = compute_metrics(
                        np.array(current_day_preds),
                        np.array(current_day_targets),
                    )
                    print(
                        f"{current_day:<6} {day_metrics['mae']:<10.4f} "
                        f"{day_metrics['rmse']:<10.4f} {day_metrics['mape']:<8.2%} "
                        f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% "
                        f"{day_metrics['count']:<8}"
                    )
                    current_day = day_key
                    current_day_preds = []
                    current_day_targets = []

                current_day_preds.extend(pred_np)
                current_day_targets.extend(target_np)

    if current_day is not None and current_day_preds:
        day_metrics = compute_metrics(
            np.array(current_day_preds),
            np.array(current_day_targets),
        )
        print(
            f"{current_day:<6} {day_metrics['mae']:<10.4f} "
            f"{day_metrics['rmse']:<10.4f} {day_metrics['mape']:<8.2%} "
            f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% "
            f"{day_metrics['count']:<8}"
        )

    # Compute overall metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    overall_metrics = compute_metrics(all_predictions, all_targets)

    # Compute per-day metrics
    per_day_metrics = {}
    for day_key in sorted(day_predictions.keys()):
        day_preds = np.array(day_predictions[day_key])
        day_targets_arr = np.array(day_targets[day_key])
        per_day_metrics[day_key] = compute_metrics(day_preds, day_targets_arr)

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
    checkpoint: str,
    dataset_path: Path,
    save_results: bool,
    num_days: int,
    start_day: str,
) -> None:
    """Evaluate solar irradiance forecasting model."""
    dataset_name = dataset_path.name.split("_")[0]
    # Load model
    print("Loading model...")
    model = Forecaster(
        model_name="timm-resnet50",
        pretrained=True,
        input_channels=3,
        loss_function="GNLL",
        lr=0.001,
        lr_patience=5,
    )
    checkpoint_data = torch.load(checkpoint)
    model.load_state_dict(checkpoint_data["state_dict"])

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    datamodule = ForecastingDataModule(
        root_data_path=dataset_path,
        periods_path=dataset_path / "periods.pickle",
        augment=False,
        image_size=[128, 128],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        batch_size=1,
        workers=0,
        seed=42,
    )
    datamodule.setup(None)
    test_dataset = datamodule._test_dataset

    print(f"Test set size: {len(test_dataset)} samples\n")

    # Group indices by day
    day_index_map = group_indices_by_day(test_dataset._periods)
    print(f"Number of unique days: {len(day_index_map)}\n")

    # Determine which days to evaluate
    indices = None
    if num_days > 0:
        all_days = generate_year_days()
        if start_day not in all_days:
            raise ValueError(f"Invalid start_day '{start_day}'. Must be in MMDD format (0101-1231)")
        start_idx = all_days.index(start_day)
        selected_days = all_days[start_idx : start_idx + num_days]

        # Collect indices for selected days
        indices = []
        for day_key in selected_days:
            if day_key in day_index_map:
                indices.extend(day_index_map[day_key])

        print(f"Evaluating {len(selected_days)} days starting from {start_day}")
        print(f"Total samples: {len(indices)}\n")
    else:
        print("Evaluating all days\n")

    # Create test loader
    if indices is not None:
        test_loader = DataLoader(
            Subset(test_dataset, indices),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    else:
        test_loader = datamodule.test_dataloader()

    # Evaluate
    print("Evaluating model...")
    overall_metrics, per_day_metrics, predictions_data = evaluate_dataset(
        model,
        test_loader,
        test_dataset._periods,
        indices=indices,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {dataset_name}")
    print(f"{'='*60}\n")

    print("Overall Metrics:")
    print(f"  MAE:   {overall_metrics['mae']:.4f}")
    print(f"  RMSE:  {overall_metrics['rmse']:.4f}")
    print(f"  MAPE:  {overall_metrics['mape']:.2%}")
    print(f"  WAPE:  {overall_metrics['wape']:.2f}%")
    print(f"  SMAPE: {overall_metrics['smape']:.2f}%")
    print(f"  Samples: {overall_metrics['count']}")

    # Save results if requested
    if save_results:
        results_file = f"{dataset_name}_results.txt"
        predictions_file = f"{dataset_name}_predictions.csv"
        df_predictions = pd.DataFrame(predictions_data)
        df_predictions.to_csv(predictions_file, index=False)
        print(f"Predictions saved to {predictions_file}")

        with Path(results_file).open("w") as f:
            f.write(f"Evaluation Results: {dataset_name}\n")
            f.write(f"Checkpoint: {checkpoint}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write(f"  MAE:   {overall_metrics['mae']:.4f}\n")
            f.write(f"  RMSE:  {overall_metrics['rmse']:.4f}\n")
            f.write(f"  MAPE:  {overall_metrics['mape']:.4f}%\n")
            f.write(f"  WAPE:  {overall_metrics['wape']:.2f}%\n")
            f.write(f"  SMAPE: {overall_metrics['smape']:.2f}%\n")
            f.write(f"  Samples: {overall_metrics['count']}\n\n")
            f.write("Per-Day Metrics:\n")
            f.write(f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'WAPE':<10} {'SMAPE':<10} {'Samples':<8}\n")
            f.write("-" * 78 + "\n")
            for day_key, metrics in per_day_metrics.items():
                f.write(
                    f"{day_key:<6} {metrics['mae']:<10.4f} "
                    f"{metrics['rmse']:<10.4f} {metrics['mape']:<10.2%} "
                    f"{metrics['wape']:<10.2f}% {metrics['smape']:<10.2f}% "
                    f"{metrics['count']:<8}\n"
                )

        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
