"""Evaluate forecasting with sliding-window mini-batch gradient adaptation.

Simulates the spirit of the asynchronous continual-learning pipeline from the
paper in a sequential evaluation setting.  After the causal delay, the
``buffer_size`` most recently GT-confirmed samples form a mini-batch for one
epoch of supervised gradient adaptation, triggered for every new arrival.

Algorithm:

    For each timestep t:
      1. Predict y_hat_t with the current model (inference-only, no gradient).
      2. Enqueue the sample in a causal delay buffer of depth ``horizon_steps``.
      3. When the oldest item's ground-truth is revealed (queue depth exceeded),
         append it to a rolling adaptation buffer (maxlen = ``buffer_size``).
      4. Once the adaptation buffer is full, perform one mini-batch gradient
         step on all ``buffer_size`` buffered samples (one epoch, batched).

Example (horizon_steps=15, buffer_size=4):
    GT for T+18 is revealed at physical step T+18+15.
    At that point the rolling buffer contains [T+15, T+16, T+17, T+18].
    One batched gradient step is performed on those 4 samples.
    At the next step GT for T+19 is revealed; buffer becomes [T+16, T+17, T+18, T+19],
    and another gradient step is taken.

The causal delay queue and the adaptation buffer are both cleared at day
boundaries to prevent overnight ground-truth leakage.
"""

from collections import deque
from pathlib import Path
import sys
from typing import Any, Literal

import click
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cirr.datamodules.loading import load_dataset
from cirr.evaluation.printing import print_metrics_header, print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.losses import BerHuLoss
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Gradient-based helpers
# ---------------------------------------------------------------------------


def _make_optimizer(
    model: Forecaster,
    lr: float,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer covering all model parameters."""
    return torch.optim.Adam(model.parameters(), lr=lr)


def _set_trainable(model: Forecaster, scope: Literal["head_only", "head_and_encoder"]) -> None:
    """Freeze all parameters then unfreeze those in *scope*."""
    for param in model.parameters():
        param.requires_grad_(False)

    for param in model.network_head.parameters():
        param.requires_grad_(True)

    if scope == "head_and_encoder":
        encoder_children = list(model.image_encoder.children())
        if encoder_children:
            for child in encoder_children[-2:]:
                for param in child.parameters():
                    param.requires_grad_(True)


@torch.enable_grad()
def _minibatch_step(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    scope: Literal["head_only", "head_and_encoder"],
    loss_fn: nn.Module,
    buffer: list[dict[str, Any]],
    device: str,
) -> None:
    """One batched gradient step (1 epoch) on all samples in *buffer*."""
    _set_trainable(model, scope)
    model.eval()  # keep BN running stats and dropout frozen

    images = torch.stack([item["image"] for item in buffer]).to(device)  # [B, C, H, W]
    histories = torch.stack([item["history"] for item in buffer]).to(device)  # [B, hist_len]
    targets = torch.cat([item["gt"] for item in buffer], dim=0).to(device)  # [B, 1]

    optimizer.zero_grad()
    output = model(images, histories)
    pred = output[0] if isinstance(output, tuple) else output
    loss = loss_fn(pred, targets.view_as(pred))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Re-freeze everything after the update step.
    for param in model.parameters():
        param.requires_grad_(False)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    horizon_steps: int,
    buffer_size: int,
    scope: Literal["head_only", "head_and_encoder"],
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    all_predictions: list[float] = []
    all_targets: list[float] = []
    predictions_data: list[dict[str, Any]] = []
    per_day_metrics: dict[str, dict[str, float]] = {}
    per_day_adaptation: dict[str, dict[str, int]] = {}

    # Causal delay queue: holds ``horizon_steps`` pending samples before GT
    # is considered available.
    causal_queue: deque[dict[str, Any]] = deque()

    # Rolling adaptation buffer: the ``buffer_size`` most recently revealed
    # GT samples used for each mini-batch gradient step.
    adapt_buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)

    update_stats: dict[str, int] = {
        "revealed": 0,
        "adapted": 0,
    }

    print_metrics_header()

    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])
        if not day_indices:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []
        day_update_stats: dict[str, int] = {
            "revealed": 0,
            "adapted": 0,
        }

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False, file=sys.stderr):
            image_input, irradiance_history, target_irradiance, image_name = dataset[idx]

            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            images = image_input.unsqueeze(0).to(device)
            irr_history = irradiance_history.unsqueeze(0).to(device)
            target = target_irradiance.unsqueeze(0).to(device)

            # --- Inference first (current model state, no gradient) ---
            model.eval()
            with torch.no_grad():
                output = model(images, irr_history)
                mean_raw = output[0] if isinstance(output, tuple) else output
            mean = torch.clamp(mean_raw, min=0.0)

            pred_np = mean.detach().cpu().numpy().flatten()
            target_np = target.detach().cpu().numpy().flatten()

            day_preds.extend(pred_np)
            day_targets.extend(target_np)
            all_predictions.extend(pred_np)
            all_targets.extend(target_np)

            pred_row_idx = len(predictions_data)
            predictions_data.append(
                {
                    "timestamp": image_name if isinstance(image_name, str) else None,
                    "day": day_key,
                    "prediction": pred_np[0] if len(pred_np) == 1 else pred_np,
                    "target": target_np[0] if len(target_np) == 1 else target_np,
                    "error": float(pred_np[0] - target_np[0]),
                    "absolute_error": float(abs(pred_np[0] - target_np[0])),
                    "gt_revealed": False,
                    "adapted": False,
                    "adaptation_scope": scope,
                }
            )

            # --- Enqueue into causal delay queue ---
            causal_queue.append(
                {
                    "image": image_input,
                    "history": irradiance_history,
                    "gt": target_irradiance.unsqueeze(0),
                    "pred_row_idx": pred_row_idx,
                }
            )

            # --- Dequeue when GT becomes available (horizon elapsed) ---
            if len(causal_queue) > horizon_steps:
                oldest = causal_queue.popleft()
                update_stats["revealed"] += 1
                day_update_stats["revealed"] += 1

                oldest_row = predictions_data[oldest["pred_row_idx"]]
                oldest_row["gt_revealed"] = True

                # Add to rolling adaptation buffer (oldest entry auto-evicted
                # once maxlen is reached).
                adapt_buffer.append(oldest)

                # Adapt once the buffer is full (buffer_size samples available).
                if len(adapt_buffer) >= buffer_size:
                    _minibatch_step(
                        model=model,
                        optimizer=optimizer,
                        scope=scope,
                        loss_fn=loss_fn,
                        buffer=list(adapt_buffer),
                        device=device,
                    )
                    update_stats["adapted"] += 1
                    day_update_stats["adapted"] += 1
                    oldest_row["adapted"] = True

        if day_preds:
            day_metrics = compute_metrics(np.array(day_preds), np.array(day_targets))
            per_day_metrics[day_key] = day_metrics
            per_day_adaptation[day_key] = day_update_stats
            print(
                f"{day_key:<6} {day_metrics['mae']:<10.4f} {day_metrics['rmse']:<10.4f} "
                f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% {day_metrics['count']:<16}",
                flush=True,
            )
            print(
                f"       adapt: revealed={day_update_stats['revealed']} " f"adapted={day_update_stats['adapted']}",
                flush=True,
            )

        # Clear both queues at day boundaries to prevent overnight GT leakage.
        causal_queue.clear()
        adapt_buffer.clear()

    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    print(
        f"\nSliding-window buffer update stats:"
        f"  revealed={update_stats['revealed']}"
        f"  adapted={update_stats['adapted']}"
        f"  buffer_size={buffer_size}"
        f"  scope={scope}",
        flush=True,
    )

    return overall_metrics, per_day_metrics, predictions_data, per_day_adaptation, update_stats


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
    "--horizon-steps",
    type=click.IntRange(min=1),
    default=15,
    show_default=True,
    help="Causal delay in samples between prediction and label availability.",
)
@click.option(
    "--buffer-size",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of recent GT-confirmed samples in the sliding adaptation window.",
)
@click.option(
    "--scope",
    type=click.Choice(["head_only", "head_and_encoder"], case_sensitive=False),
    default="head_only",
    show_default=True,
    help="Which parameters to update: head_only or head_and_encoder.",
)
@click.option(
    "--lr",
    type=float,
    default=1e-5,
    show_default=True,
    help="AdamW learning rate.",
)
@click.option(
    "--loss",
    type=click.Choice(["berhu", "huber", "mse"], case_sensitive=False),
    default="mse",
    show_default=True,
    help="Supervised loss used during adaptation.",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    show_default=True,
    help="Number of days to evaluate. -1 means all available days.",
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
    help="Device to run on ('cuda' or 'cpu').",
)
@click.option(
    "--save-results",
    is_flag=True,
    default=False,
    help="Save per-day metrics report and predictions CSV to disk.",
)
def main(
    checkpoint: Path,
    dataset_path: Path,
    horizon_steps: int,
    buffer_size: int,
    scope: str,
    lr: float,
    loss: str,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
) -> None:
    """Evaluate forecasting with sliding-window mini-batch gradient adaptation."""

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    dataset_name = dataset_path.name.split("_")[0]
    method_tag = "sliding_window"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {checkpoint}...")
    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    if hasattr(model.image_encoder, "fc") and not isinstance(model.image_encoder.fc, torch.nn.Identity):
        model.image_encoder.fc = torch.nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Build loss and optimizer
    # ------------------------------------------------------------------
    if loss == "berhu":
        loss_fn: nn.Module = BerHuLoss(threshold=30, scale=1000)
    elif loss == "huber":
        loss_fn = nn.HuberLoss()
    else:
        loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)

    optimizer = _make_optimizer(model, lr)

    print(
        f"Sliding-window:  buffer_size={buffer_size}  scope={scope}" f"  lr={lr}  loss={loss!r}  horizon_steps={horizon_steps}"
    )

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
    print("Running sliding-window buffer adaptation evaluation...\n")
    overall_metrics, per_day_metrics, predictions_data, per_day_adaptation, update_stats = evaluate(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataset=test_dataset,
        day_index_map=day_index_map,
        days_to_eval=days_to_eval,
        device=device,
        horizon_steps=horizon_steps,
        buffer_size=buffer_size,
        scope=scope,  # type: ignore[arg-type]
    )

    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    if save_results:
        csv_path = f"{dataset_name}_{method_tag}_predictions.csv"
        txt_path = f"{dataset_name}_{method_tag}_results.txt"
        save_predictions_csv(predictions_data, csv_path)
        print(f"\nPredictions saved to {csv_path}")
        extra = [
            "Method: Sliding-window mini-batch gradient adaptation",
            f"horizon_steps={horizon_steps}",
            f"buffer_size={buffer_size}",
            f"scope={scope}",
            f"lr={lr}",
            f"loss={loss}",
            "Adaptation summary:",
            f"revealed={update_stats['revealed']}",
            f"adapted={update_stats['adapted']}",
            "Per-day adaptation:",
        ]
        for day_key in sorted(per_day_adaptation.keys()):
            s = per_day_adaptation[day_key]
            extra.append(f"{day_key}: revealed={s['revealed']} adapted={s['adapted']}")
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
