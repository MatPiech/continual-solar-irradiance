"""Local training script for continual adaptation of solar irradiance forecasting model.

This script demonstrates continual learning where the model:
1. Makes predictions on samples from day N
2. Retrains on all data from day N
3. Uses updated weights to predict on day N+1

Prevents overfitting through early stopping and gradient clipping.
"""

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import click
import numpy as np
import pandas as pd
import torch

from cirr.datamodules.forecasting import ForecastingDataModule
from cirr.models.forecaster import Forecaster
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics


def _period_target_value(period: dict[str, Any]) -> float:
    """Get normalized target value directly from period metadata."""
    return float(period["target_irradiance"]) / 1000.0


def _load_dataset(dataset_path: Path, split: str = "test") -> Any:
    """Load selected dataset split from prepared directory path."""
    datamodule = ForecastingDataModule(
        root_data_path=dataset_path,
        periods_path=dataset_path / "periods.pickle" if split == "test" else dataset_path / "train_periods.pickle",
        augment=False,
        image_size=[128, 128],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        batch_size=1,
        workers=0,
        seed=42,
    )
    datamodule.setup(None)
    if split == "test":
        return datamodule._test_dataset
    if split == "training":
        return datamodule._train_dataset
    raise ValueError(f"Invalid split '{split}'. Expected 'test' or 'training'.")


def _build_initial_replay_buffer(
    dataset_path: Path,
    replay_size: int,
    nonzero_threshold: float = 0.001,
) -> tuple[list[tuple[Any, int]], dict[str, float]]:
    """Create initial replay samples from Folsom prepared dataset using normal distribution.

    Parameters
    ----------
    dataset_path : Path
        Path to dataset directory.
    replay_size : int
        Number of samples to include in replay buffer.
    nonzero_threshold : float, optional
        Minimum target value for replay eligibility, by default 0.001.

    Returns
    -------
    replay_buffer : list[tuple[Any, int]]
        List of (dataset, sample_index) tuples for replay.
    distribution_stats : dict[str, float]
        Dictionary with 'mean' and 'std' computed from Folsom irradiance values.
    """
    if replay_size <= 0:
        return [], {"mean": 0.0, "std": 0.0}

    folsom_dataset_path = dataset_path.parent / "folsom_prepared"
    if not folsom_dataset_path.exists():
        return [], {"mean": 0.0, "std": 0.0}

    folsom_dataset = _load_dataset(folsom_dataset_path, split="training")

    # Collect eligible samples and their irradiance values
    irradiance_values = []
    eligible_samples: list[tuple[Any, int]] = []

    for idx, period in enumerate(folsom_dataset._periods):
        target_value = _period_target_value(period)
        if target_value < nonzero_threshold:
            continue
        irradiance_values.append(target_value)
        eligible_samples.append((folsom_dataset, idx))

    if len(eligible_samples) == 0:
        return [], {"mean": 0.0, "std": 0.0}

    # Compute distribution statistics from Folsom
    irradiance_array = np.array(irradiance_values)
    mean_irr = float(np.mean(irradiance_array))
    std_irr = float(np.std(irradiance_array))
    max_irr = float(np.max(irradiance_array))

    # Create irradiance bins (e.g., 10 bins) for stratified sampling
    # to match normal distribution while covering the full range
    n_bins = min(10, max(2, replay_size // 20))
    bin_edges = np.linspace(0, max_irr, n_bins + 1)
    bins: dict[int, list[tuple[Any, int]]] = {i: [] for i in range(n_bins)}

    for sample, irr_val in zip(eligible_samples, irradiance_values, strict=False):
        bin_idx = min(n_bins - 1, int(np.digitize(irr_val, bin_edges)) - 1)
        bins[bin_idx].append(sample)

    # Compute normal distribution weights for each bin center
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    normal_weights = np.exp(-0.5 * ((bin_centers - mean_irr) / (std_irr + 1e-6)) ** 2)
    normal_weights /= np.sum(normal_weights)  # Normalize to sum to 1

    # Allocate replay samples proportionally to normal distribution
    replay_buffer: list[tuple[Any, int]] = []
    for bin_idx in range(n_bins):
        quota = max(0, int(round(replay_size * normal_weights[bin_idx])))
        if quota > 0 and bins[bin_idx]:
            replay_buffer.extend(bins[bin_idx][:quota])

    # Fill remaining slots with any available samples
    if len(replay_buffer) < replay_size:
        all_remaining = [
            sample
            for bin_idx in range(n_bins)
            for sample in bins[bin_idx][len([s for s in replay_buffer if s in bins[bin_idx]]) :]
        ]
        replay_buffer.extend(all_remaining[: replay_size - len(replay_buffer)])

    random.Random(42).shuffle(replay_buffer)
    distribution_stats = {"mean": mean_irr, "std": std_irr, "max": max_irr}
    return replay_buffer[:replay_size], distribution_stats


def load_model_and_data(
    checkpoint_path: Path,
    dataset_path: Path,
    device: str,
) -> tuple[Forecaster, dict[str, list[int]], Any]:
    """Load pretrained model and dataset.

    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint file.
    dataset_path : Path
        Path to prepared dataset directory containing periods.pickle.
    device : str
        Device to use ('cuda' or 'cpu').

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
    test_dataset = _load_dataset(dataset_path, split="test")
    day_index_map = group_indices_by_day(test_dataset._periods)

    return model, day_index_map, test_dataset


def predict_on_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    device: str,
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
    device : str
        Device to use for inference.

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
    replay_buffer: list[tuple[Any, int]],
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    l2sp_lambda: float,
    batch_size: int,
    replay_samples: int,
    device: str,
) -> float:
    """Retrain model on all samples from a given day.

    Combines current-day samples with distribution-matched replay samples.

    Parameters
    ----------
    model : Forecaster
        Model to train.
    dataset : SkyImageDataset
        Dataset containing the periods.
    day_indices : list[int]
        Indices of samples to train on.
    replay_buffer : list[tuple[Any, int]]
        Candidate replay samples as (dataset_ref, sample_index) pairs.
    learning_rate : float
        Learning rate for optimizer.
    weight_decay : float
        AdamW weight decay.
    max_grad_norm : float
        Gradient clipping max norm.
    l2sp_lambda : float
        L2-SP anti-drift regularization coefficient.
    batch_size : int
        Mini-batch size for adaptation.
    replay_samples : int
        Number of replay samples to draw per day.
    device : str
        Device to use for training.

    Returns
    -------
    loss : float
        Final training loss.
    replay_stats : dict[str, Any]
        Statistics about replay samples used (count, mean, std irradiance).
    """
    if len(day_indices) == 0:
        return 0.0, {"count": 0, "mean_irradiance": 0.0, "std_irradiance": 0.0}

    # Collect current day samples
    current_samples: list[tuple[Any, int]] = [(dataset, idx) for idx in day_indices]

    # Select replay samples to maintain distribution
    if replay_samples > 0 and len(replay_buffer) > 0:
        # Compute irradiance distribution from replay buffer
        replay_irradiances = [_period_target_value(rb_dataset._periods[rb_idx]) for rb_dataset, rb_idx in replay_buffer]
        replay_array = np.array(replay_irradiances)

        # Create bins for distribution matching
        n_bins = min(10, max(2, len(replay_buffer) // 20))
        bin_edges = np.linspace(0, np.max(replay_array) + 1e-6, n_bins + 1)

        # Partition replay buffer into bins
        replay_bins: dict[int, list[tuple[Any, int]]] = {i: [] for i in range(n_bins)}
        for sample, irr_val in zip(replay_buffer, replay_irradiances, strict=False):
            bin_idx = min(n_bins - 1, int(np.digitize(irr_val, bin_edges)) - 1)
            replay_bins[bin_idx].append(sample)

        # Allocate replay samples according to replay buffer bin distribution
        selected_replay: list[tuple[Any, int]] = []
        total_replay_buffer = len(replay_buffer)
        for bin_idx in range(n_bins):
            bin_proportion = len(replay_bins[bin_idx]) / total_replay_buffer if total_replay_buffer > 0 else 0
            target_quota = max(0, int(round(replay_samples * bin_proportion)))
            if target_quota > 0 and replay_bins[bin_idx]:
                selected_replay.extend(replay_bins[bin_idx][:target_quota])

        # Fill remaining slots if needed
        if len(selected_replay) < replay_samples:
            remaining = [
                s
                for b_idx in range(n_bins)
                for s in replay_bins[b_idx][len([x for x in selected_replay if x in replay_bins[b_idx]]) :]
            ]
            selected_replay.extend(remaining[: replay_samples - len(selected_replay)])

        pooled_samples = current_samples + selected_replay[:replay_samples]
    else:
        pooled_samples = current_samples
        selected_replay = []

    if len(pooled_samples) == 0:
        return 0.0, {"count": 0, "mean_irradiance": 0.0, "std_irradiance": 0.0}

    random.shuffle(pooled_samples)
    mini_batches = [pooled_samples[idx : idx + batch_size] for idx in range(0, len(pooled_samples), batch_size)]

    if len(mini_batches) == 0:
        return 0.0, {"count": 0, "mean_irradiance": 0.0, "std_irradiance": 0.0}

    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Snapshot trainable parameters for L2-SP anti-drift regularization
    reference_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    total_loss = 0.0
    batch_count = 0

    # One pass over stratified mini-batches
    for batch_indices in mini_batches:
        samples = [sample_dataset[sample_idx] for sample_dataset, sample_idx in batch_indices]
        images = torch.stack([sample[0] for sample in samples]).to(device)
        irradiance_history = torch.stack([sample[1] for sample in samples]).to(device)
        target = torch.stack([sample[2] for sample in samples]).to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_mean, pred_var = model(images, irradiance_history)

        # Compute base loss
        base_loss = model.loss_fn(pred_mean, target, pred_var) if pred_var is not None else model.loss_fn(pred_mean, target)

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

    # Compute replay sample distribution statistics
    replay_stats: dict[str, Any] = {"count": len(selected_replay)}
    if len(selected_replay) > 0:
        replay_irradiances = [_period_target_value(rb_dataset._periods[rb_idx]) for rb_dataset, rb_idx in selected_replay]
        replay_irr_array = np.array(replay_irradiances)
        replay_stats["mean_irradiance"] = float(np.mean(replay_irr_array))
        replay_stats["std_irradiance"] = float(np.std(replay_irr_array))
        replay_stats["min_irradiance"] = float(np.min(replay_irr_array))
        replay_stats["max_irradiance"] = float(np.max(replay_irr_array))
    else:
        replay_stats["mean_irradiance"] = 0.0
        replay_stats["std_irradiance"] = 0.0
        replay_stats["min_irradiance"] = 0.0
        replay_stats["max_irradiance"] = 0.0

    return total_loss / batch_count if batch_count > 0 else 0.0, replay_stats


def run_local_training(
    checkpoint_path: Path,
    dataset_path: Path,
    num_days: int,
    start_day: str,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_l2sp_lambda: float,
    adapt_batch_size: int,
    adapt_replay_size: int,
    adapt_replay_samples: int,
    device: str,
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
    num_days : int
        Number of days to process.
    start_day : str
        Starting day in MMDD format.
    adapt_lr : float
        Learning rate for day adaptation.
    adapt_weight_decay : float
        Weight decay for day adaptation optimizer.
    adapt_max_grad_norm : float
        Gradient clipping max norm.
    adapt_l2sp_lambda : float
        L2-SP anti-drift regularization coefficient.
    adapt_batch_size : int
        Adaptation mini-batch size.
    adapt_replay_size : int
        Maximum replay buffer size.
    adapt_replay_samples : int
        Replay samples drawn per day adaptation.
    device : str
        Device to use.

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
        print(f"Processing all days starting from {all_days[0]}\n")

    results = []
    all_predictions_data = []
    replay_buffer, replay_dist_stats = _build_initial_replay_buffer(
        dataset_path=dataset_path,
        replay_size=adapt_replay_size,
    )
    if len(replay_buffer) > 0:
        print(f"Initialized replay buffer from Folsom with {len(replay_buffer)} samples")
        print(
            f"  Irradiance distribution: μ={replay_dist_stats['mean']:.2f}, σ={replay_dist_stats['std']:.2f}, max={replay_dist_stats['max']:.2f}\n"  # noqa: RUF001
        )

    print(f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'MAPE':<16} {'WAPE':<4} {'SMAPE':<4} {'Samples':<8} {'Train Loss':<12}")
    print("-" * 100)

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
        train_loss, replay_stats = train_on_day(
            model,
            dataset,
            day_indices,
            replay_buffer=replay_buffer,
            learning_rate=adapt_lr,
            weight_decay=adapt_weight_decay,
            max_grad_norm=adapt_max_grad_norm,
            l2sp_lambda=adapt_l2sp_lambda,
            batch_size=adapt_batch_size,
            replay_samples=adapt_replay_samples,
            device=device,
        )

        results.append(
            {
                "day": day_key,
                "n_samples": n_samples,
                "metrics": metrics,
                "training_loss": float(train_loss),
                "replay_stats": replay_stats,
            }
        )

        print(
            f"{day_key:<6} {metrics['mae']:<10.4f} "
            f"{metrics['rmse']:<10.4f} {metrics['mape']:<16.2%} "
            f"{metrics['wape']:<10.2f}% {metrics['smape']:<10.2f}% "
            f"{metrics['count']:<8} {train_loss:.4f}"
        )
        print(
            f"  Replay: n={replay_stats['count']:<4} "
            f"μ_irr={replay_stats['mean_irradiance']:<7.2f} "
            f"σ_irr={replay_stats['std_irradiance']:<7.2f} "  # noqa: RUF001
            f"[{replay_stats['min_irradiance']:.2f}, {replay_stats['max_irradiance']:.2f}]"
        )

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
@click.option("--adapt-lr", type=float, default=1e-6, help="Learning rate for day adaptation")
@click.option("--adapt-weight-decay", type=float, default=0.001, help="Weight decay for day adaptation")
@click.option("--adapt-max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm")
@click.option("--adapt-l2sp-lambda", type=float, default=1e-2, help="L2-SP anti-drift regularization coefficient")
@click.option("--adapt-batch-size", type=int, default=32, help="Adaptation mini-batch size")
@click.option("--adapt-replay-size", type=int, default=7000, help="Maximum replay buffer size")
@click.option("--adapt-replay-samples", type=int, default=200, help="Replay samples drawn per day")
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
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_l2sp_lambda: float,
    adapt_batch_size: int,
    adapt_replay_size: int,
    adapt_replay_samples: int,
    save_results: bool,
) -> None:
    """Run local continual training simulation for solar irradiance forecasting."""
    dataset_name = dataset_path.name.split("_")[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results, predictions_data, overall_metrics, per_day_metrics = run_local_training(
        checkpoint_path=checkpoint,
        dataset_path=dataset_path,
        num_days=num_days,
        start_day=start_day,
        adapt_lr=adapt_lr,
        adapt_weight_decay=adapt_weight_decay,
        adapt_max_grad_norm=adapt_max_grad_norm,
        adapt_l2sp_lambda=adapt_l2sp_lambda,
        adapt_batch_size=adapt_batch_size,
        adapt_replay_size=adapt_replay_size,
        adapt_replay_samples=adapt_replay_samples,
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
    print(f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'MAPE':<16} {'WAPE':<10} {'SMAPE':<10} {'Samples':<8}")
    print("-" * 78)
    for day_key, metrics in per_day_metrics.items():
        print(
            f"{day_key:<6} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
            f"{metrics['mape']:<16.2%} {metrics['wape']:<10.2f} {metrics['smape']:<10.2f} "
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
            f.write(f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'MAPE':<16} {'WAPE':<10} {'SMAPE':<10} {'Samples':<8}\n")
            f.write("-" * 78 + "\n")
            for day_key, metrics in per_day_metrics.items():
                f.write(
                    f"{day_key:<6} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
                    f"{metrics['mape']:<16.2%} {metrics['wape']:<10.2f} {metrics['smape']:<10.2f} "
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
