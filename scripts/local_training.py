"""Local training script for continual adaptation of solar irradiance forecasting model.

This script demonstrates continual learning where the model:
1. Makes predictions on samples from day N
2. Retrains on all data from day N
3. Uses updated weights to predict on day N+1

Prevents overfitting through early stopping and gradient clipping.
"""

from collections import defaultdict
from collections.abc import Callable
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


def _sample_replay_for_day(
    replay_buffer: list[tuple[Any, int]],
    num_samples: int,
    seed: int | None = None,
) -> list[tuple[Any, int]]:
    """Sample replay instances for a single day from the replay buffer.

    Maintains the distribution of the replay buffer while providing different
    samples for each day.

    Parameters
    ----------
    replay_buffer : list[tuple[Any, int]]
        Full replay buffer containing all candidate samples.
    num_samples : int
        Number of samples to draw for this day.
    seed : int, optional
        Random seed for reproducibility. If None, uses system randomness.

    Returns
    -------
    sampled_replay : list[tuple[Any, int]]
        Subset of replay samples for this day.
    """
    if num_samples <= 0 or len(replay_buffer) == 0:
        return []

    # Sample without replacement to ensure diversity across days
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    selected_indices = rng.choice(len(replay_buffer), size=min(num_samples, len(replay_buffer)), replace=False)
    return [replay_buffer[idx] for idx in selected_indices]


def _compute_sample_distribution(samples: list[tuple[Any, int]]) -> dict[str, float]:
    """Compute irradiance distribution statistics for a set of samples.

    Parameters
    ----------
    samples : list[tuple[Any, int]]
        List of (dataset_ref, sample_index) tuples.

    Returns
    -------
    stats : dict[str, float]
        Dictionary with mean, std, min, max irradiance values.
    """
    if len(samples) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    irradiances = [_period_target_value(dataset._periods[idx]) for dataset, idx in samples]
    irr_array = np.array(irradiances)

    return {
        "mean": float(np.mean(irr_array)),
        "std": float(np.std(irr_array)),
        "min": float(np.min(irr_array)),
        "max": float(np.max(irr_array)),
    }


def load_model_and_data(
    checkpoint_path: Path,
    dataset_path: Path,
    device: str,
    use_lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
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
    use_lora : bool, optional
        Whether to apply LoRA to the model, by default False.
    lora_rank : int, optional
        Rank for LoRA decomposition, by default 4.
    lora_alpha : float, optional
        LoRA scaling factor, by default 1.0.

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

    # Apply LoRA if requested
    if use_lora:
        print(f"\nApplying LoRA with rank={lora_rank}, alpha={lora_alpha}")
        model.apply_lora_to_head(rank=lora_rank, alpha=lora_alpha)
        model.freeze_all_except_lora()
        print("Frozen entire network except LoRA parameters\n")

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
    replay_samples: list[tuple[Any, int]],
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    l2sp_lambda: float,
    batch_size: int,
    epochs: int,
    device: str,
    freeze_image_encoder: bool = False,
    use_lora: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Retrain model on all samples from a given day.

    Combines current-day samples with pre-sampled replay samples.

    Parameters
    ----------
    model : Forecaster
        Model to train.
    dataset : SkyImageDataset
        Dataset containing the periods.
    day_indices : list[int]
        Indices of samples to train on.
    replay_samples : list[tuple[Any, int]]
        Pre-sampled replay samples as (dataset_ref, sample_index) pairs.
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
    epochs : int
        Number of training epochs to perform on the day's data.
    device : str
        Device to use for training.
    freeze_image_encoder : bool, optional
        If True, freeze the image encoder and only train the network head, by default False.
        Note: Ignored when use_lora=True, as LoRA freezes the entire network.
    use_lora : bool, optional
        If True, only LoRA parameters are trainable (extreme parameter efficiency), by default False.

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

    # Combine current day samples with pre-sampled replay samples
    pooled_samples = current_samples + replay_samples

    if len(pooled_samples) == 0:
        return 0.0, {"count": 0, "mean_irradiance": 0.0, "std_irradiance": 0.0}

    random.shuffle(pooled_samples)
    mini_batches = [pooled_samples[idx : idx + batch_size] for idx in range(0, len(pooled_samples), batch_size)]

    if len(mini_batches) == 0:
        return 0.0, {"count": 0, "mean_irradiance": 0.0, "std_irradiance": 0.0}

    model.train()

    # Apply freezing strategy based on use_lora
    if use_lora:
        # LoRA: entire network is frozen except LoRA parameters
        # This is already done in load_model_and_data via freeze_all_except_lora()
        pass
    elif freeze_image_encoder:
        # Legacy mode: freeze only the image encoder
        model.freeze_image_encoder()

    # Create optimizer (only trainable parameters will be optimized)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay
    )

    # Snapshot trainable parameters for L2-SP anti-drift regularization
    reference_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    total_loss = 0.0
    batch_count = 0

    # Training loop over epochs
    for __ in range(epochs):
        # One pass over stratified mini-batches per epoch
        for batch_indices in mini_batches:
            samples = [sample_dataset[sample_idx] for sample_dataset, sample_idx in batch_indices]
            images = torch.stack([sample[0] for sample in samples]).to(device)
            irradiance_history = torch.stack([sample[1] for sample in samples]).to(device)
            target = torch.stack([sample[2] for sample in samples]).to(device)

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
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

    # Compute replay sample distribution statistics
    replay_stats: dict[str, Any] = {"count": len(replay_samples)}
    if len(replay_samples) > 0:
        replay_irradiances = [_period_target_value(rb_dataset._periods[rb_idx]) for rb_dataset, rb_idx in replay_samples]
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
    adapt_epochs: int,
    adapt_replay_buffer_size: int,
    adapt_replay_ratio: float,
    device: str,
    seed: int,
    freeze_image_encoder: bool = False,
    use_lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
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
    adapt_epochs : int
        Number of epochs to train on each day.
    adapt_replay_buffer_size : int
        Maximum replay buffer size.
    adapt_replay_ratio : float
        Ratio of replay samples to current day samples (e.g., 0.1 means 10% of current day samples will be added from the replay buffer).
    device : str
        Device to use.
    seed : int
        Random seed for reproducibility.
    freeze_image_encoder : bool, optional
        If True, freeze the image encoder and only train the network head, by default False.
        Note: Ignored when use_lora=True.
    use_lora : bool, optional
        If True, use LoRA for extreme parameter efficiency, by default False.
    lora_rank : int, optional
        Rank for LoRA decomposition (only used if use_lora=True), by default 4.
    lora_alpha : float, optional
        LoRA scaling factor (only used if use_lora=True), by default 1.0.

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
    model, day_index_map, dataset = load_model_and_data(
        checkpoint_path,
        dataset_path,
        device=device,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

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
        replay_size=adapt_replay_buffer_size,
    )
    if len(replay_buffer) > 0:
        print(f"Initialized replay buffer from Folsom with {len(replay_buffer)} samples")
        print(
            f"  Irradiance distribution: μ={replay_dist_stats['mean']:.2f}, σ={replay_dist_stats['std']:.2f}, max={replay_dist_stats['max']:.2f}\n"  # noqa: RUF001
        )
        # Compute full replay buffer distribution for later comparison
        full_buffer_dist = _compute_sample_distribution(replay_buffer)
    else:
        full_buffer_dist = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    print(
        f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'MAPE':<16} {'WAPE':<10} {'SMAPE':<10} {'Samples':<8} {'Train Loss':<12} {'Replay Dist Match':<18}"
    )
    print("-" * 130)

    for day_idx, day_key in enumerate(all_days):
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

        # Step 2: Sample replay for this day and train on current day to adapt the model
        # Use a day-specific seed based on global seed and day index
        day_seed = seed + day_idx if seed is not None else None

        # Calculate number of replay samples based on ratio of current day samples
        num_replay_samples = max(0, int(round(len(day_indices) * adapt_replay_ratio)))

        day_replay_samples = _sample_replay_for_day(
            replay_buffer,
            num_replay_samples,
            seed=day_seed,
        )

        # Compute sampled replay distribution
        sampled_replay_dist = _compute_sample_distribution(day_replay_samples)

        train_loss, replay_stats = train_on_day(
            model,
            dataset,
            day_indices,
            replay_samples=day_replay_samples,
            learning_rate=adapt_lr,
            weight_decay=adapt_weight_decay,
            max_grad_norm=adapt_max_grad_norm,
            l2sp_lambda=adapt_l2sp_lambda,
            batch_size=adapt_batch_size,
            epochs=adapt_epochs,
            device=device,
            freeze_image_encoder=freeze_image_encoder,
            use_lora=use_lora,
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

        # Calculate distribution match metrics
        if len(full_buffer_dist) > 0 and full_buffer_dist["std"] > 0:
            mean_diff_pct = (
                100 * abs(sampled_replay_dist["mean"] - full_buffer_dist["mean"]) / (full_buffer_dist["mean"] + 1e-6)
            )
            std_diff_pct = 100 * abs(sampled_replay_dist["std"] - full_buffer_dist["std"]) / (full_buffer_dist["std"] + 1e-6)
            dist_match_str = f"Δμ={mean_diff_pct:.1f}% Δσ={std_diff_pct:.1f}%"
        else:
            dist_match_str = "N/A"

        print(
            f"{day_key:<6} {metrics['mae']:<10.4f} "
            f"{metrics['rmse']:<10.4f} {metrics['mape']:<16.2%} "
            f"{metrics['wape']:<10.2f}% {metrics['smape']:<10.2f}% "
            f"{metrics['count']:<8} {train_loss:<12.4f}  {dist_match_str:<16}"
        )
        print(
            f"  Replay: n={replay_stats['count']:<4} "
            f"μ={replay_stats['mean_irradiance']:<7.2f} "
            f"σ={replay_stats['std_irradiance']:<7.2f} "  # noqa: RUF001
            f"[{replay_stats['min_irradiance']:.2f}, {replay_stats['max_irradiance']:.2f}]"
        )

        if progress_callback is not None:
            current_predictions = np.array([p["prediction"] for p in all_predictions_data])
            current_targets = np.array([p["target"] for p in all_predictions_data])
            current_metrics = compute_metrics(current_predictions, current_targets)
            progress_callback(day_idx, current_metrics)

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
@click.option("--adapt-max-grad-norm", type=float, default=-1.0, help="Gradient clipping max norm, -1 means no clipping")
@click.option("--adapt-l2sp-lambda", type=float, default=1e-2, help="L2-SP anti-drift regularization coefficient")
@click.option("--adapt-batch-size", type=int, default=32, help="Adaptation mini-batch size")
@click.option("--adapt-epochs", type=int, default=1, help="Number of epochs to train on each day")
@click.option("--adapt-replay-buffer-size", type=int, default=7000, help="Maximum replay buffer size")
@click.option("--adapt-replay-ratio", type=float, default=0.1, help="Ratio of replay samples relative to current day samples")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option(
    "--freeze-image-encoder",
    is_flag=True,
    default=False,
    help="Freeze image encoder and only train network head (ignored if --use-lora is set)",
)
@click.option(
    "--use-lora",
    is_flag=True,
    default=False,
    help="Use LoRA for extreme parameter efficiency (freezes entire network except LoRA params)",
)
@click.option(
    "--lora-rank",
    type=int,
    default=4,
    help="LoRA rank (lower = fewer trainable params). Typical: 1-8 for extreme efficiency, 16-64 for moderate",
)
@click.option(
    "--lora-alpha",
    type=float,
    default=1.0,
    help="LoRA scaling factor",
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
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_l2sp_lambda: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    adapt_replay_buffer_size: int,
    adapt_replay_ratio: float,
    seed: int,
    freeze_image_encoder: bool,
    use_lora: bool,
    lora_rank: int,
    lora_alpha: float,
    save_results: bool,
) -> None:
    """Run local continual training simulation for solar irradiance forecasting."""
    # Set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
        adapt_epochs=adapt_epochs,
        adapt_replay_buffer_size=adapt_replay_buffer_size,
        adapt_replay_ratio=adapt_replay_ratio,
        device=device,
        seed=seed,
        freeze_image_encoder=freeze_image_encoder,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    print(f"\n{'='*78}")
    print(f"Completed {len(results)} days of training")
    print(f"{'='*78}\n")

    print("Overall Metrics:")
    print(f"  MAE:   {overall_metrics['mae']:.7f}")
    print(f"  RMSE:  {overall_metrics['rmse']:.7f}")
    print(f"  MAPE:  {overall_metrics['mape']:.2%}")
    print(f"  WAPE:  {overall_metrics['wape']:.2f}%")
    print(f"  SMAPE: {overall_metrics['smape']:.2f}%")
    print(f"  Samples: {overall_metrics['count']}\n")

    # Save results if requested
    if save_results:
        # Add LoRA info to filename if used
        suffix = f"_lora_r{lora_rank}" if use_lora else ""
        results_file = f"{dataset_name}_local_training{suffix}_results.txt"
        predictions_file = f"{dataset_name}_local_training{suffix}_predictions.csv"

        # Save predictions to CSV
        df_predictions = pd.DataFrame(predictions_data)
        df_predictions.to_csv(predictions_file, index=False)
        print(f"\nPredictions saved to {predictions_file}")

        # Save results to text file
        with Path(results_file).open("w") as f:
            f.write(f"Local Training Results: {dataset_name}\n")
            f.write(f"Checkpoint: {checkpoint}\n")
            f.write(f"Dataset: {dataset_path}\n")
            if use_lora:
                f.write(f"LoRA: rank={lora_rank}, alpha={lora_alpha}\n")
            f.write("=" * 78 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write(f"  MAE:   {overall_metrics['mae']:.7f}\n")
            f.write(f"  RMSE:  {overall_metrics['rmse']:.7f}\n")
            f.write(f"  MAPE:  {overall_metrics['mape']:.2f}%\n")
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
