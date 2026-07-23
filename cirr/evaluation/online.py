"""Shared delayed-ground-truth online evaluation loop."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import sys
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from cirr.evaluation.printing import print_metrics_header
from cirr.training.ogd import pred_mean
from cirr.utils.metrics import compute_metrics


def infer_one(
    model: Any,
    image_input: torch.Tensor,
    irradiance_history: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass; returns ``(mean_raw, mean_clamped)``."""
    images = image_input.unsqueeze(0).to(device)
    hist = irradiance_history.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        mean_raw = pred_mean(model(images, hist))
    return mean_raw, torch.clamp(mean_raw, min=0.0)


@dataclass
class DelayedQueueItem:
    image: torch.Tensor
    history: torch.Tensor
    gt: torch.Tensor
    raw_pred: float
    pred_row_idx: int


@dataclass
class OnlineEvalContext:
    model: Any
    device: str
    day_key: str
    predictions_data: list[dict[str, Any]]
    update_stats: dict[str, int]
    day_update_stats: dict[str, int]


def default_pred_row(
    day_key: str,
    timestamp: str | None,
    prediction: float,
    target: float,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "day": day_key,
        "prediction": prediction,
        "target": target,
        "error": prediction - target,
        "absolute_error": abs(prediction - target),
        **extra,
    }


@dataclass
class OnlineEvalResult:
    overall_metrics: dict[str, float]
    per_day_metrics: dict[str, dict[str, float]]
    predictions_data: list[dict[str, Any]]
    per_day_adaptation: dict[str, dict[str, int]]
    update_stats: dict[str, int]


def run_delayed_online_eval(
    model: Any,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    horizon_steps: int,
    on_revealed: Callable[[DelayedQueueItem, OnlineEvalContext], None],
    *,
    update_stats_init: dict[str, int] | None = None,
    extra_pred_fields: dict[str, Any] | None = None,
    on_day_start: Callable[[str, OnlineEvalContext], None] | None = None,
    on_day_end: Callable[[str, OnlineEvalContext], None] | None = None,
) -> OnlineEvalResult:
    """Infer → enqueue → reveal after *horizon_steps*; delegate adaptation to *on_revealed*."""
    all_predictions: list[float] = []
    all_targets: list[float] = []
    predictions_data: list[dict[str, Any]] = []
    per_day_metrics: dict[str, dict[str, float]] = {}
    per_day_adaptation: dict[str, dict[str, int]] = {}
    causal_queue: deque[DelayedQueueItem] = deque()
    update_stats = dict(update_stats_init or {})

    print_metrics_header()

    for day_key in days_to_eval:
        day_indices = day_index_map.get(day_key, [])
        if not day_indices:
            continue

        day_preds: list[float] = []
        day_targets: list[float] = []
        day_update_stats = {k: 0 for k in update_stats}

        ctx = OnlineEvalContext(
            model=model,
            device=device,
            day_key=day_key,
            predictions_data=predictions_data,
            update_stats=update_stats,
            day_update_stats=day_update_stats,
        )
        if on_day_start:
            on_day_start(day_key, ctx)

        for idx in tqdm(day_indices, desc=f"Day {day_key}", unit="sample", leave=False, file=sys.stderr):
            image_input, irradiance_history, _ghi_clear, target_irradiance, image_name = dataset[idx]
            if isinstance(image_name, bytes):
                image_name = image_name.decode("utf-8")

            target = target_irradiance.unsqueeze(0).to(device)

            mean_raw, mean = infer_one(model, image_input, irradiance_history, device)

            pred_val = float(mean.detach().cpu().view(-1)[0].item())
            target_val = float(target.detach().cpu().view(-1)[0].item())

            day_preds.append(pred_val)
            day_targets.append(target_val)
            all_predictions.append(pred_val)
            all_targets.append(target_val)

            pred_row_idx = len(predictions_data)
            row_extra = dict(extra_pred_fields or {})
            predictions_data.append(default_pred_row(day_key, image_name, pred_val, target_val, **row_extra))

            causal_queue.append(
                DelayedQueueItem(
                    image=image_input,
                    history=irradiance_history,
                    gt=target_irradiance.unsqueeze(0),
                    raw_pred=float(mean_raw.detach().cpu().view(-1)[0].item()),
                    pred_row_idx=pred_row_idx,
                )
            )

            if len(causal_queue) > horizon_steps:
                oldest = causal_queue.popleft()
                on_revealed(oldest, ctx)

        if day_preds:
            day_metrics = compute_metrics(np.array(day_preds), np.array(day_targets))
            per_day_metrics[day_key] = day_metrics
            per_day_adaptation[day_key] = dict(day_update_stats)
            print(
                f"{day_key:<6} {day_metrics['mae']:<10.4f} {day_metrics['rmse']:<10.4f} "
                f"{day_metrics['wape']:<10.2f}% {day_metrics['smape']:<10.2f}% {day_metrics['count']:<16}",
                flush=True,
            )

        if on_day_end:
            on_day_end(day_key, ctx)

        causal_queue.clear()

    overall = compute_metrics(np.array(all_predictions), np.array(all_targets)) if all_predictions else {}
    return OnlineEvalResult(overall, per_day_metrics, predictions_data, per_day_adaptation, update_stats)
