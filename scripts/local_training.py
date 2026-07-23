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
import torch

from cirr.data.loading import load_dataset
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.results import format_percentage_metric, save_metrics_report, save_predictions_csv
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.training.day_loop import predict_day, train_day
from cirr.training.replay import build_folsom_replay_buffer, period_target_value, sample_replay_for_day
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics


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

    irradiances = [period_target_value(dataset.periods[idx]) for dataset, idx in samples]
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
    use_dual_lora: bool = False,
    encoder_rank: int = 4,
    head_rank: int = 8,
    encoder_alpha: float = 16.0,
    head_alpha: float = 16.0,
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
        Whether to apply LoRA to head only, by default False.
    lora_rank : int, optional
        Rank for head-only LoRA decomposition, by default 4.
    lora_alpha : float, optional
        Scaling factor for head-only LoRA, by default 1.0.
    use_dual_lora : bool, optional
        Whether to apply dual-LoRA (encoder + head), by default False.
        Takes precedence over use_lora if both are True.
    encoder_rank : int, optional
        Rank for encoder LoRA (dual-LoRA only), by default 4.
    head_rank : int, optional
        Rank for head LoRA (dual-LoRA only), by default 8.
    encoder_alpha : float, optional
        Scaling factor for encoder LoRA (dual-LoRA only), by default 16.0.
    head_alpha : float, optional
        Scaling factor for head LoRA (dual-LoRA only), by default 16.0.

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
    model = load_forecaster_from_checkpoint(checkpoint_path, device=device, use_hparams=True)

    # Apply LoRA (dual-LoRA takes precedence)
    if use_dual_lora:
        model.apply_dual_lora(
            encoder_rank=encoder_rank,
            head_rank=head_rank,
            encoder_alpha=encoder_alpha,
            head_alpha=head_alpha,
        )
        model.freeze_all_except_lora()
    elif use_lora:
        print(f"\nApplying head-only LoRA with rank={lora_rank}, alpha={lora_alpha}")
        model.apply_lora_to_head(rank=lora_rank, alpha=lora_alpha)
        model.freeze_all_except_lora()
        print("Frozen entire network except LoRA parameters\n")

    # Load dataset
    test_dataset = load_dataset(dataset_path, split="test")
    day_index_map = group_indices_by_day(test_dataset.periods)

    return model, day_index_map, test_dataset


def predict_on_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    device: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions for all samples on a given day."""
    predictions, targets, _ = predict_day(
        model,
        dataset,
        day_indices,
        device,
        batch_size=batch_size,
    )
    return predictions, targets


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
    use_dual_lora: bool = False,
    encoder_lr: float | None = None,
    head_lr: float | None = None,
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
        Learning rate for optimizer (used for head-only LoRA or legacy modes).
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
        Note: Ignored when use_lora=True or use_dual_lora=True.
    use_lora : bool, optional
        If True, only head LoRA parameters are trainable, by default False.
    use_dual_lora : bool, optional
        If True, both encoder and head LoRA parameters are trainable, by default False.
    encoder_lr : float, optional
        Learning rate for encoder LoRA (dual-LoRA only), by default None.
    head_lr : float, optional
        Learning rate for head LoRA (dual-LoRA only), by default None.

    Returns
    -------
    loss : float
        Final training loss.
    replay_stats : dict[str, Any]
        Statistics about replay samples used (count, mean, std irradiance).
    """
    if len(day_indices) == 0:
        return 0.0, {"count": 0, "mean_irradiance": 0.0, "std_irradiance": 0.0}

    model.train()

    if use_dual_lora or use_lora:
        pass
    elif freeze_image_encoder:
        model.freeze_image_encoder()

    if use_dual_lora and encoder_lr is not None and head_lr is not None:
        param_groups = model.get_lora_param_groups(encoder_lr=encoder_lr, head_lr=head_lr, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay
        )

    reference_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    def extra_loss_fn(_base_loss: torch.Tensor, model: Forecaster) -> torch.Tensor:
        l2sp_terms = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            reference_parameter = reference_parameters[name]
            l2sp_terms.append(torch.mean((parameter - reference_parameter) ** 2))
        if not l2sp_terms:
            return torch.tensor(0.0, device=device)
        return l2sp_lambda * torch.stack(l2sp_terms).mean()

    avg_loss = train_day(
        model,
        dataset,
        day_indices,
        lr=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        replay_entries=replay_samples or None,
        optimizer=optimizer,
        extra_loss_fn=extra_loss_fn,
        shuffle_each_epoch=False,
    )

    replay_stats: dict[str, Any] = {"count": len(replay_samples)}
    if len(replay_samples) > 0:
        replay_irradiances = [period_target_value(rb_dataset.periods[rb_idx]) for rb_dataset, rb_idx in replay_samples]
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

    return avg_loss, replay_stats


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
    use_dual_lora: bool = False,
    encoder_rank: int = 4,
    head_rank: int = 8,
    encoder_alpha: float = 16.0,
    head_alpha: float = 16.0,
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-4,
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
        Learning rate for day adaptation (head-only LoRA or legacy modes).
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
        Note: Ignored when use_lora=True or use_dual_lora=True.
    use_lora : bool, optional
        If True, use head-only LoRA, by default False.
    lora_rank : int, optional
        Rank for head-only LoRA decomposition, by default 4.
    lora_alpha : float, optional
        Scaling factor for head-only LoRA, by default 1.0.
    use_dual_lora : bool, optional
        If True, use dual-LoRA (encoder + head), by default False.
        Takes precedence over use_lora.
    encoder_rank : int, optional
        Rank for encoder LoRA (dual-LoRA only), by default 4.
    head_rank : int, optional
        Rank for head LoRA (dual-LoRA only), by default 8.
    encoder_alpha : float, optional
        Scaling factor for encoder LoRA (dual-LoRA only), by default 16.0.
    head_alpha : float, optional
        Scaling factor for head LoRA (dual-LoRA only), by default 16.0.
    encoder_lr : float, optional
        Learning rate for encoder LoRA (dual-LoRA only), by default 1e-5.
    head_lr : float, optional
        Learning rate for head LoRA (dual-LoRA only), by default 1e-4.

    Returns
    -------
    results : list[dict[str, Any]]
        List of dictionaries containing daily training results.
    predictions_data : list[dict[str, Any]]
        List of dictionaries with predictions, targets, and timestamps.
    overall_metrics : dict[str, float]
        Overall metrics (MAE, RMSE, WAPE, SMAPE, count).
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
        use_dual_lora=use_dual_lora,
        encoder_rank=encoder_rank,
        head_rank=head_rank,
        encoder_alpha=encoder_alpha,
        head_alpha=head_alpha,
    )

    print(f"Dataset size: {len(dataset.periods)} samples")
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
    folsom_path = dataset_path.parent / "folsom_prepared"
    replay_buffer, replay_dist_stats = build_folsom_replay_buffer(folsom_path, adapt_replay_buffer_size)
    if len(replay_buffer) > 0:
        print(f"Initialized replay buffer from Folsom with {len(replay_buffer)} samples")
        print(
            f"  Irradiance distribution: μ={replay_dist_stats['mean']:.2f}, σ={replay_dist_stats['std']:.2f}, max={replay_dist_stats['max']:.2f}\n"
        )
        # Compute full replay buffer distribution for later comparison
        full_buffer_dist = _compute_sample_distribution(replay_buffer)
    else:
        full_buffer_dist = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    print(
        f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'WAPE':<10} {'SMAPE':<10} {'Samples':<8} {'Train Loss':<12} {'Replay Dist Match':<18}"
    )
    print("-" * 100)

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
            image_name = dataset.periods[sample_idx]["history"][-1]["image_name"]
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

        day_replay_samples = sample_replay_for_day(
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
            use_dual_lora=use_dual_lora,
            encoder_lr=encoder_lr if use_dual_lora else None,
            head_lr=head_lr if use_dual_lora else None,
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

        wape_str = format_percentage_metric(metrics["wape"])
        smape_str = format_percentage_metric(metrics["smape"])

        print(
            f"{day_key:<6} {metrics['mae']:<10.4f} "
            f"{metrics['rmse']:<10.4f} {wape_str} {smape_str} "
            f"{metrics['count']:<8} {train_loss:<12.4f}  {dist_match_str:<16}"
        )
        print(
            f"  Replay: n={replay_stats['count']:<4} "
            f"μ={replay_stats['mean_irradiance']:<7.2f} "
            f"σ={replay_stats['std_irradiance']:<7.2f} "
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
    help="LoRA scaling factor (head-only LoRA)",
)
@click.option(
    "--use-dual-lora",
    is_flag=True,
    default=False,
    help="Use Dual-LoRA (encoder + head) with asymmetric bottlenecking for handling seasonal drift",
)
@click.option(
    "--encoder-rank",
    type=int,
    default=4,
    help="Encoder LoRA rank (dual-LoRA only). Very low (2-4) prevents memorization. Default: 4",
)
@click.option(
    "--head-rank",
    type=int,
    default=8,
    help="Head LoRA rank (dual-LoRA only). Moderate (8-16) for final mapping. Default: 8",
)
@click.option(
    "--encoder-alpha",
    type=float,
    default=16.0,
    help="Encoder LoRA scaling factor (dual-LoRA only). Default: 16.0",
)
@click.option(
    "--head-alpha",
    type=float,
    default=16.0,
    help="Head LoRA scaling factor (dual-LoRA only). Default: 16.0",
)
@click.option(
    "--encoder-lr",
    type=float,
    default=1e-5,
    help="Encoder LoRA learning rate (dual-LoRA only). Recommended: 1e-5. Default: 1e-5",
)
@click.option(
    "--head-lr",
    type=float,
    default=1e-4,
    help="Head LoRA learning rate (dual-LoRA only). Recommended: 1e-4 (10x encoder_lr). Default: 1e-4",
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
    use_dual_lora: bool,
    encoder_rank: int,
    head_rank: int,
    encoder_alpha: float,
    head_alpha: float,
    encoder_lr: float,
    head_lr: float,
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
        use_dual_lora=use_dual_lora,
        encoder_rank=encoder_rank,
        head_rank=head_rank,
        encoder_alpha=encoder_alpha,
        head_alpha=head_alpha,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
    )

    print(f"\n{'='*78}")
    print(f"Completed {len(results)} days of training")
    print(f"{'='*78}\n")

    print_overall_metrics(overall_metrics, tag=dataset_name)

    if save_results:
        if use_dual_lora:
            suffix = f"_dual_lora_e{encoder_rank}_h{head_rank}"
        elif use_lora:
            suffix = f"_lora_r{lora_rank}"
        else:
            suffix = ""
        predictions_file = f"{dataset_name}_local_training{suffix}_predictions.csv"
        results_file = f"{dataset_name}_local_training{suffix}_results.txt"

        save_predictions_csv(predictions_data, predictions_file)
        print(f"\nPredictions saved to {predictions_file}")

        extra_lines: list[str] = []
        if use_dual_lora:
            extra_lines += [
                "Dual-LoRA Configuration:",
                f"  Encoder: rank={encoder_rank}, alpha={encoder_alpha}, lr={encoder_lr}",
                f"  Head:    rank={head_rank}, alpha={head_alpha}, lr={head_lr}",
            ]
        elif use_lora:
            extra_lines.append(f"LoRA: rank={lora_rank}, alpha={lora_alpha}")
        save_metrics_report(
            tag="local_training" + suffix,
            dataset_name=dataset_name,
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            overall_metrics=overall_metrics,
            per_day_metrics=per_day_metrics,
            path=results_file,
            extra_lines=extra_lines,
        )
        # Append per-day training results section (unique to this script)
        with Path(results_file).open("a") as f:
            f.write("\n" + "=" * 78 + "\n")
            f.write("Per-Day Training Results:\n")
            f.write(f"{'Day':<6} {'Samples':<10} {'MAE':<12} {'RMSE':<12} {'Train Loss':<14}\n")
            f.write("-" * 60 + "\n")
            for result in results:
                f.write(
                    f"{result['day']:<6} {result['n_samples']:<10} "
                    f"{result['metrics']['mae']:<12.4f} "
                    f"{result['metrics']['rmse']:<12.4f} "
                    f"{result['training_loss']:<14.6f}\n"
                )
        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
