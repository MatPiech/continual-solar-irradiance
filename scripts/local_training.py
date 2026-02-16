"""Local training script for continual adaptation of solar irradiance forecasting model.

This script demonstrates continual learning where the model:
1. Makes predictions on samples from day N
2. Retrains on all data from day N
3. Uses updated weights to predict on day N+1

Prevents overfitting through early stopping and gradient clipping.
"""

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


def load_model_and_data(
    checkpoint_path: Path,
    dataset_path: Path,
    device: str = "cuda",
) -> tuple[Forecaster, dict[str, list[int]], Any]:
    """Load pretrained model and dataset.

    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint file.
    dataset_path : Path
        Path to prepared dataset directory containing periods.pickle.
    device : str, optional
        Device to use ('cuda' or 'cpu'), by default "cuda".

    Returns
    -------
    model : Forecaster
        Loaded model on specified device.
    day_index_map : dict[str, list[int]]
        Dictionary mapping MMDD to list of period indices.
    dataset : SkyImageDataset
        Test dataset object containing periods.
    """
    # Load model
    model = Forecaster(
        model_name="timm-resnet50",
        pretrained=True,
        input_channels=3,
        loss_function="GNLL",
        lr=0.001,
        lr_patience=5,
    )
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_data["state_dict"])
    model = model.to(device)

    # Load dataset
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
    day_index_map = group_indices_by_day(test_dataset._periods)

    return model, day_index_map, test_dataset


def predict_on_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions for all samples on a given day.

    Parameters
    ----------
    model : Forecaster
        Model to use for predictions.
    dataset : SkyImageDataset
        Dataset containing the periods.
    day_indices : list[int]
        Indices of samples to predict on.
    device : str, optional
        Device to use for inference, by default "cuda".

    Returns
    -------
    predictions : np.ndarray
        Model predictions, shape (n_samples,).
    targets : np.ndarray
        Ground truth values, shape (n_samples,).
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for idx in day_indices:
            images, irradiance_history, target, _ = dataset[idx]
            images = images.unsqueeze(0).to(device)
            irradiance_history = irradiance_history.unsqueeze(0).to(device)
            target = target.to(device)

            # Forward pass
            pred_mean, _ = model(images, irradiance_history)
            predictions.append(pred_mean.cpu().item())
            targets.append(target.cpu().item())

    return np.array(predictions), np.array(targets)


def train_on_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    num_epochs: int = 1,
    learning_rate: float = 0.0001,
    weight_decay: float = 0.001,
    max_grad_norm: float = 1.0,
    l2sp_lambda: float = 1e-4,
    device: str = "cuda",
) -> float:
    """Retrain model on all samples from a given day.

    Implements early stopping and gradient clipping to prevent overfitting.

    Parameters
    ----------
    model : Forecaster
        Model to train.
    dataset : SkyImageDataset
        Dataset containing the periods.
    day_indices : list[int]
        Indices of samples to train on.
    num_epochs : int, optional
        Maximum number of epochs, by default 5.
    learning_rate : float, optional
        Learning rate for optimizer, by default 0.0001.
    weight_decay : float, optional
        AdamW weight decay, by default 0.001.
    max_grad_norm : float, optional
        Gradient clipping max norm, by default 1.0.
    l2sp_lambda : float, optional
        L2-SP anti-drift regularization coefficient, by default 1e-4.
    device : str, optional
        Device to use for training, by default "cuda".

    Returns
    -------
    float
        Final training loss.
    """
    if len(day_indices) == 0:
        return 0.0

    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Snapshot trainable parameters for L2-SP anti-drift regularization
    reference_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    # Create dataloader
    day_loader = DataLoader(
        Subset(dataset, day_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    best_loss = float("inf")
    patience_counter = 0
    patience = 2  # Early stopping patience

    for __ in range(num_epochs):
        total_loss = 0.0
        batch_count = 0

        for batch in day_loader:
            images, irradiance_history, target, _ = batch
            images = images.to(device)
            irradiance_history = irradiance_history.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_mean, pred_var = model(images, irradiance_history)

            # Compute base loss
            base_loss = (
                model.loss_fn(pred_mean, target, pred_var) if pred_var is not None else model.loss_fn(pred_mean, target)
            )

            # L2-SP anti-drift penalty (distance to pre-adaptation weights)
            l2sp_terms = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                reference_parameter = reference_parameters[name]
                l2sp_terms.append(torch.mean((parameter - reference_parameter) ** 2))

            l2sp_penalty = torch.stack(l2sp_terms).mean() if l2sp_terms else torch.tensor(0.0, device=device)
            loss = base_loss + l2sp_lambda * l2sp_penalty

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_loss


def run_local_training(
    checkpoint_path: Path,
    dataset_path: Path,
    num_days: int = 10,
    start_day: str = "0101",
    device: str = "cuda",
) -> dict[str, Any]:
    """Run continual local training simulation.

    For each day:
    1. Make predictions on day N using current model state
    2. Train on day N data to adapt the model
    3. Move to next day with adapted model

    Each sample is used exactly once for inference and once for training.

    Parameters
    ----------
    checkpoint_path : Path
        Path to initial model checkpoint.
    dataset_path : Path
        Path to prepared dataset directory.
    num_days : int, optional
        Number of days to process, by default -1 (all days).
    start_day : str, optional
        Starting day in MMDD format, by default "0101".
    device : str, optional
        Device to use, by default "cuda".

    Returns
    -------
    results : list[dict[str, Any]]
        List of dictionaries containing daily training results.
    predictions_data : list[dict[str, Any]]
        List of dictionaries with predictions, targets, and timestamps.
    overall_metrics : dict[str, float]
        Overall metrics (MAE, RMSE, MAPE, WAPE, SMAPE, count).
    per_day_metrics : dict[str, dict[str, float]]
        Per-day metrics with MMDD as keys.
    """
    print(f"Loading model from {checkpoint_path}...")
    model, day_index_map, dataset = load_model_and_data(checkpoint_path, dataset_path, device=device)

    print(f"Dataset size: {len(dataset._periods)} samples")
    print(f"Number of unique days: {len(day_index_map)}\n")

    # Generate training days
    all_days = generate_year_days()

    # Apply start_day and num_days filters
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day '{start_day}'. Must be in MMDD format (0101-1231)")
    start_idx = all_days.index(start_day)

    if num_days > 0:
        all_days = all_days[start_idx : start_idx + num_days]
        print(f"Processing {len(all_days)} days starting from {all_days[0]}\n")
    else:
        all_days = all_days[start_idx:]
        print(f"Processing all days starting from {all_days[0]} ({len(all_days)} days)\n")
    print(f"{'Day':<6} {'Samples':<10} {'MAE':<10} {'RMSE':<10} {'Train Loss':<12}")
    print("-" * 60)

    results = []
    all_predictions_data = []

    for day_key in all_days:
        day_indices = day_index_map.get(day_key, [])

        if len(day_indices) == 0:
            continue

        n_samples = len(day_indices)

        # Step 1: Predict on current day using current model state
        predictions, targets = predict_on_day(model, dataset, day_indices, device=device)
        metrics = compute_metrics(predictions, targets)

        # Collect predictions data
        for idx, (pred, target) in enumerate(zip(predictions, targets, strict=False)):
            sample_idx = day_indices[idx]
            image_name = dataset._periods[sample_idx]["history"][-1]["image_name"]
            all_predictions_data.append(
                {
                    "timestamp": image_name,
                    "day": day_key,
                    "prediction": float(pred),
                    "target": float(target),
                    "error": float(pred - target),
                    "absolute_error": float(np.abs(pred - target)),
                }
            )

        # Step 2: Train on current day to adapt the model
        train_loss = train_on_day(model, dataset, day_indices, num_epochs=1, learning_rate=1e-6, device=device)

        results.append(
            {
                "day": day_key,
                "n_samples": n_samples,
                "metrics": metrics,
                "training_loss": float(train_loss),
            }
        )

        print(f"{day_key:<6} {n_samples:<10} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} " f"{train_loss:<12.6f}")

    # Compute overall metrics
    all_predictions = np.array([p["prediction"] for p in all_predictions_data])
    all_targets = np.array([p["target"] for p in all_predictions_data])
    overall_metrics = compute_metrics(all_predictions, all_targets)

    # Compute per-day metrics
    per_day_metrics = defaultdict(lambda: {"predictions": [], "targets": []})
    for pred_data in all_predictions_data:
        day = pred_data["day"]
        per_day_metrics[day]["predictions"].append(pred_data["prediction"])
        per_day_metrics[day]["targets"].append(pred_data["target"])

    per_day_metrics_computed = {}
    for day_key in sorted(per_day_metrics.keys()):
        preds = np.array(per_day_metrics[day_key]["predictions"])
        targets = np.array(per_day_metrics[day_key]["targets"])
        per_day_metrics_computed[day_key] = compute_metrics(preds, targets)

    return results, all_predictions_data, overall_metrics, per_day_metrics_computed


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint file",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prepared dataset directory",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    help="Number of days to process. -1 means all days.",
)
@click.option(
    "--start-day",
    type=str,
    default="0101",
    help="Starting day in MMDD format (e.g., 0101)",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to use (cuda or cpu)",
)
@click.option(
    "--save-results",
    is_flag=True,
    default=False,
    help="Save results to files with dataset-based names",
)
def main(
    checkpoint: Path,
    dataset_path: Path,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
) -> None:
    """Run local continual training simulation for solar irradiance forecasting."""
    dataset_name = dataset_path.name.split("_")[0]

    results, predictions_data, overall_metrics, per_day_metrics = run_local_training(
        checkpoint_path=checkpoint,
        dataset_path=dataset_path,
        num_days=num_days,
        start_day=start_day,
        device=device,
    )

    print(f"\n{'='*78}")
    print(f"Completed {len(results)} days of training")
    print(f"{'='*78}\n")

    print("Overall Metrics:")
    print(f"  MAE:   {overall_metrics['mae']:.4f}")
    print(f"  RMSE:  {overall_metrics['rmse']:.4f}")
    print(f"  MAPE:  {overall_metrics['mape']:.2%}")
    print(f"  WAPE:  {overall_metrics['wape']:.2f}%")
    print(f"  SMAPE: {overall_metrics['smape']:.2f}%")
    print(f"  Samples: {overall_metrics['count']}\n")

    print("Per-Day Metrics:")
    print(f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'WAPE':<10} {'SMAPE':<10} {'Samples':<8}")
    print("-" * 78)
    for day_key, metrics in per_day_metrics.items():
        print(
            f"{day_key:<6} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
            f"{metrics['mape']:<10.2%} {metrics['wape']:<10.2f} {metrics['smape']:<10.2f} "
            f"{metrics['count']:<8}"
        )

    # Save results if requested
    if save_results:
        results_file = f"{dataset_name}_local_training_results.txt"
        predictions_file = f"{dataset_name}_local_training_predictions.csv"

        # Save predictions to CSV
        df_predictions = pd.DataFrame(predictions_data)
        df_predictions.to_csv(predictions_file, index=False)
        print(f"\nPredictions saved to {predictions_file}")

        # Save results to text file
        with Path(results_file).open("w") as f:
            f.write(f"Local Training Results: {dataset_name}\n")
            f.write(f"Checkpoint: {checkpoint}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write("=" * 78 + "\n\n")

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
                    f"{day_key:<6} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
                    f"{metrics['mape']:<10.2%} {metrics['wape']:<10.2f} {metrics['smape']:<10.2f} "
                    f"{metrics['count']:<8}\n"
                )

            f.write("\n" + "=" * 78 + "\n")
            f.write("Per-Day Training Results:\n")
            f.write(f"{'Day':<6} {'Samples':<10} {'MAE':<12} {'RMSE':<12} {'Train Loss':<14}\n")
            f.write("-" * 60 + "\n")
            for result in results:
                mae = result["metrics"]["mae"]
                rmse = result["metrics"]["rmse"]
                train_loss = result["training_loss"]

                f.write(f"{result['day']:<6} {result['n_samples']:<10} {mae:<12.4f} " f"{rmse:<12.4f} {train_loss:<14.6f}\n")

        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
