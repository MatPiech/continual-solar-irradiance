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
from typing import Any, Literal

import click
import torch
import torch.nn as nn

from cirr.evaluation.online import DelayedQueueItem, OnlineEvalContext, run_delayed_online_eval
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.runner import load_eval_days, load_eval_model, print_eval_plan, resolve_device, save_online_eval_results
from cirr.models.forecaster import Forecaster
from cirr.training.ogd import make_regression_loss, sliding_minibatch_step


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
    adapt_buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)

    def on_day_end(_day: str, ctx: OnlineEvalContext) -> None:
        adapt_buffer.clear()
        print(
            f"       adapt: revealed={ctx.day_update_stats['revealed']} " f"adapted={ctx.day_update_stats['adapted']}",
            flush=True,
        )

    def on_revealed(oldest: DelayedQueueItem, ctx: OnlineEvalContext) -> None:
        ctx.update_stats["revealed"] += 1
        ctx.day_update_stats["revealed"] += 1
        row = ctx.predictions_data[oldest.pred_row_idx]
        row["gt_revealed"] = True
        item = {
            "image": oldest.image,
            "history": oldest.history,
            "gt": oldest.gt,
            "pred_row_idx": oldest.pred_row_idx,
        }
        adapt_buffer.append(item)
        if len(adapt_buffer) >= buffer_size:
            sliding_minibatch_step(
                model=model,
                optimizer=optimizer,
                scope=scope,
                loss_fn=loss_fn,
                buffer=list(adapt_buffer),
                device=device,
            )
            ctx.update_stats["adapted"] += 1
            ctx.day_update_stats["adapted"] += 1
            row["adapted"] = True

    result = run_delayed_online_eval(
        model,
        dataset,
        day_index_map,
        days_to_eval,
        device,
        horizon_steps,
        on_revealed,
        update_stats_init={"revealed": 0, "adapted": 0},
        extra_pred_fields={
            "gt_revealed": False,
            "adapted": False,
            "adaptation_scope": scope,
        },
        on_day_end=on_day_end,
    )
    print(
        f"\nSliding-window buffer update stats:"
        f"  revealed={result.update_stats['revealed']}"
        f"  adapted={result.update_stats['adapted']}"
        f"  buffer_size={buffer_size}"
        f"  scope={scope}",
        flush=True,
    )
    return (
        result.overall_metrics,
        result.per_day_metrics,
        result.predictions_data,
        result.per_day_adaptation,
        result.update_stats,
    )


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

    device = resolve_device(device)
    dataset_name = dataset_path.name.split("_")[0]
    method_tag = "sliding_window"

    print(f"Loading model from {checkpoint}...")
    model = load_eval_model(checkpoint, device)
    loss_fn = make_regression_loss(loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Sliding-window:  buffer_size={buffer_size}  scope={scope}  lr={lr}  loss={loss!r}  horizon_steps={horizon_steps}")

    test_dataset, day_index_map, days_to_eval, _ = load_eval_days(dataset_path, start_day, num_days)
    print_eval_plan(dataset_path, test_dataset, day_index_map, days_to_eval, start_day)

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
        save_online_eval_results(
            method_tag=method_tag,
            dataset_name=dataset_name,
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            overall_metrics=overall_metrics,
            per_day_metrics=per_day_metrics,
            predictions_data=predictions_data,
            extra_lines=extra,
        )


if __name__ == "__main__":
    main()
