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

Reference:
    Lara-Benitez, P., Carranza-Garcia, M., Luna-Romera, J., and Riquelme, J.
    Short-term solar irradiance forecasting in streaming with deep learning.
    Neurocomputing, doi: 10.1016/j.neucom.2023.126312.
"""

from collections import deque
from pathlib import Path
import time
from typing import Any, Literal

import click
import torch
import torch.nn as nn

from cirr.evaluation.campaign import SOURCE_CHECKPOINT, CampaignData, load_campaign_days, write_campaign
from cirr.evaluation.online import DelayedQueueItem, OnlineEvalContext, OnlineEvalResult, run_delayed_online_eval
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.runner import load_eval_model, resolve_device
from cirr.models.forecaster import Forecaster
from cirr.physics.clear_sky import DEFAULT_MIN_GHI_CLEAR
from cirr.training.ogd import make_regression_loss, sliding_minibatch_step
from cirr.utils.provenance import set_determinism

# Comparators report actions in the same vocabulary as the conditional policy, so
# events.csv means the same thing for every method in the campaign.
SCOPE_TO_ACTION = {"head_only": "head", "head_and_encoder": "whole"}


def evaluate(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    data: CampaignData,
    device: str,
    horizon_steps: int,
    buffer_size: int,
    scope: Literal["head_only", "head_and_encoder"],
) -> OnlineEvalResult:
    adapt_buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)

    def on_day_end(_day: str, ctx: OnlineEvalContext) -> None:
        adapt_buffer.clear()
        print(
            f"       adapt: revealed={ctx.day_update_stats['revealed']} adapted={ctx.day_update_stats['adapted']}",
            flush=True,
        )

    def on_revealed(oldest: DelayedQueueItem, ctx: OnlineEvalContext) -> None:
        ctx.update_stats["revealed"] += 1
        ctx.day_update_stats["revealed"] += 1
        row = ctx.predictions_data[oldest.pred_row_idx]
        row["gt_revealed"] = True
        row["update_considered"] = True
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
            row["adaptation_scope"] = SCOPE_TO_ACTION[scope]

    result = run_delayed_online_eval(
        model,
        data.dataset,
        data.day_map,
        data.days,
        device,
        horizon_steps,
        on_revealed,
        update_stats_init={"revealed": 0, "adapted": 0},
        extra_pred_fields={
            "gt_revealed": False,
            "update_considered": False,
            "adapted": False,
            "adaptation_scope": "none",
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
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=str(SOURCE_CHECKPOINT),
    show_default=True,
    help="Source model. The default is the checkpoint Chapter 5 uses.",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Prepared dataset directory (e.g. data/poznan_prepared).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Run folder. Defaults to results/ch4/<site>/sliding_window/seed<k>.",
)
@click.option(
    "--min-ghi-clear",
    type=float,
    default=DEFAULT_MIN_GHI_CLEAR,
    show_default=True,
    help="Daylight filter (kW/m^2), shared with the federated benchmark. 0 disables it.",
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
    help="Recent GT-confirmed samples in the sliding adaptation window.",
)
@click.option(
    "--scope",
    type=click.Choice(["head_only", "head_and_encoder"]),
    default="head_only",
    show_default=True,
    help="Which parameters to update.",
)
@click.option("--lr", type=float, default=1e-5, show_default=True)
@click.option("--loss", type=click.Choice(["berhu", "huber", "mse"]), default="mse", show_default=True)
@click.option("--num-days", type=int, default=-1, show_default=True, help="-1 means all available days.")
@click.option("--start-day", type=str, default="0101", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
def main(
    checkpoint: Path,
    dataset_path: Path,
    out_dir: Path | None,
    min_ghi_clear: float,
    horizon_steps: int,
    buffer_size: int,
    scope: str,
    lr: float,
    loss: str,
    num_days: int,
    start_day: str,
    seed: int,
    device: str,
) -> None:
    """Evaluate forecasting with sliding-window mini-batch gradient adaptation."""
    set_determinism(seed)
    device = resolve_device(device)
    checkpoint = Path(checkpoint)

    data = load_campaign_days(dataset_path, start_day, num_days, min_ghi_clear)
    print(
        f"{data.site}: {len(data.days)} days, {data.n_samples} lit samples "
        f"({data.dropped_total} dark samples dropped at min_ghi_clear={min_ghi_clear})"
    )

    model = load_eval_model(checkpoint, device)
    loss_fn = make_regression_loss(loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Sliding-window:  buffer_size={buffer_size}  scope={scope}  lr={lr}  loss={loss!r}  seed={seed}\n")

    started = time.perf_counter()
    result = evaluate(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data=data,
        device=device,
        horizon_steps=horizon_steps,
        buffer_size=buffer_size,
        scope=scope,  # type: ignore[arg-type]
    )
    wall_time_s = time.perf_counter() - started

    print_overall_metrics(result.overall_metrics, tag=f"{data.site}  [sliding_window]")

    out_dir = Path(out_dir) if out_dir else Path("results/ch4") / data.site / "sliding_window" / f"seed{seed}"
    write_campaign(
        out_dir,
        result=result,
        data=data,
        config={
            "method": "sliding_window",
            "buffer_size": buffer_size,
            "scope": scope,
            "lr": lr,
            "loss": loss,
            "seed": seed,
            "start_day": start_day,
            "num_days": num_days,
            "dataset_path": str(dataset_path),
        },
        checkpoint=checkpoint,
        horizon_steps=horizon_steps,
        min_ghi_clear=min_ghi_clear,
        wall_time_s=wall_time_s,
        device=device,
    )
    print(f"\nRun written to {out_dir}")


if __name__ == "__main__":
    main()
