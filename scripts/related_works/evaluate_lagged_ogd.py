"""Evaluate forecasting with the lagged OGD continual adaptation.

Implements Algorithm 1 from the paper directly:

    For each timestep t:
      1. Predict y_hat_t with the current model.
      2. Enqueue sample in a causal delay buffer of depth ``horizon_steps``.
      3. When the delayed ground-truth arrives, compute the absolute error
         against the prediction that was made ``horizon_steps`` ago:
           - error < T1  -> skip update ("none")
           - T1 <= error < T2 -> update head only ("head")
           - error >= T2 -> update whole network ("whole")

Default thresholds per the paper: T1 = 0.01 kW/m2, T2 = 0.1 kW/m2.

Reference:
    Piechocki, M., Kraft, M., Capotondi, A. On-device continual adaptation for reliable solar irradiance forecasting.
    In Advanced Concepts for Intelligent Vision Systems. 2026. doi: 10.1007/978-3-032-07343-3_28
"""

from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.nn as nn

from cirr.evaluation.online import DelayedQueueItem, OnlineEvalContext, run_delayed_online_eval
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.runner import load_eval_days, load_eval_model, print_eval_plan, resolve_device, save_online_eval_results
from cirr.models.forecaster import Forecaster
from cirr.training.ogd import algorithm1_scope, make_regression_loss, ogd_step


def evaluate(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    horizon_steps: int,
    paper_t1: float,
    paper_t2: float,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    def on_revealed(oldest: DelayedQueueItem, ctx: OnlineEvalContext) -> None:
        ctx.update_stats["considered"] += 1
        ctx.day_update_stats["considered"] += 1
        delayed_gt = float(oldest.gt.view(-1)[0].item())
        delayed_abs_error = abs(oldest.raw_pred - delayed_gt)
        scope = algorithm1_scope(delayed_abs_error, paper_t1, paper_t2)
        row = ctx.predictions_data[oldest.pred_row_idx]
        row["update_considered"] = True
        row["delayed_abs_error"] = delayed_abs_error
        row.setdefault("adapted", False)
        row.setdefault("adaptation_scope", "not_available")
        if scope == "none":
            ctx.update_stats["none"] += 1
            ctx.day_update_stats["none"] += 1
            row["adaptation_scope"] = "none"
            return
        ogd_step(
            model=model,
            optimizer=optimizer,
            scope=scope,
            loss_fn=loss_fn,
            image=oldest.image,
            history=oldest.history,
            gt=oldest.gt,
            device=device,
        )
        if scope == "head_only":
            ctx.update_stats["head"] += 1
            ctx.day_update_stats["head"] += 1
            row["adaptation_scope"] = "head"
        else:
            ctx.update_stats["whole"] += 1
            ctx.day_update_stats["whole"] += 1
            row["adaptation_scope"] = "whole"
        row["adapted"] = True

    result = run_delayed_online_eval(
        model,
        dataset,
        day_index_map,
        days_to_eval,
        device,
        horizon_steps,
        on_revealed,
        update_stats_init={"considered": 0, "none": 0, "head": 0, "whole": 0},
        extra_pred_fields={
            "update_considered": False,
            "adapted": False,
            "adaptation_scope": "not_available",
            "delayed_abs_error": np.nan,
        },
    )
    print(
        f"\nAlgorithm 1 update stats:"
        f"  considered={result.update_stats['considered']}"
        f"  none={result.update_stats['none']}"
        f"  head={result.update_stats['head']}"
        f"  whole={result.update_stats['whole']}",
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
    "--t1",
    type=float,
    default=0.01,
    show_default=True,
    help="Algorithm 1 lower threshold (kW/m2): error < T1 -> skip update.",
)
@click.option(
    "--t2",
    type=float,
    default=0.1,
    show_default=True,
    help="Algorithm 1 upper threshold (kW/m2): error >= T2 -> update whole network.",
)
@click.option(
    "--ogd-lr",
    type=float,
    default=1e-5,
    show_default=True,
    help="SGD learning rate for OGD updates.",
)
@click.option(
    "--ogd-beta1",
    type=float,
    default=0.5,
    show_default=True,
    help="AdamW beta1.",
)
@click.option(
    "--ogd-beta2",
    type=float,
    default=0.975,
    show_default=True,
    help="AdamW beta2.",
)
@click.option(
    "--ogd-weight-decay",
    type=float,
    default=1e-4,
    show_default=True,
    help="SGD weight decay.",
)
@click.option(
    "--ogd-loss",
    type=click.Choice(["berhu", "huber", "mse"], case_sensitive=False),
    default="mse",
    show_default=True,
    help="Supervised loss used in OGD updates.",
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
    t1: float,
    t2: float,
    ogd_lr: float,
    ogd_beta1: float,
    ogd_beta2: float,
    ogd_weight_decay: float,
    ogd_loss: str,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
) -> None:
    """Evaluate paper-style Algorithm 1 lagged OGD continual adaptation."""

    if t1 >= t2:
        raise click.UsageError(f"--t1 ({t1}) must be strictly less than --t2 ({t2}).")

    device = resolve_device(device)

    dataset_name = dataset_path.name.split("_")[0]
    method_tag = "paper_alg1_ogd"

    print(f"Loading model from {checkpoint}...")
    model = load_eval_model(checkpoint, device)

    loss_fn = make_regression_loss(ogd_loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ogd_lr, betas=(ogd_beta1, ogd_beta2), weight_decay=ogd_weight_decay)

    print(
        f"Algorithm 1:  T1={t1}  T2={t2}  lr={ogd_lr}  betas=({ogd_beta1}, {ogd_beta2})  loss={ogd_loss!r}"
        f"  horizon_steps={horizon_steps}"
    )

    test_dataset, day_index_map, days_to_eval, _ = load_eval_days(dataset_path, start_day, num_days)
    print_eval_plan(dataset_path, test_dataset, day_index_map, days_to_eval, start_day)

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print("Running paper Algorithm 1 lagged OGD evaluation...\n")
    overall_metrics, per_day_metrics, predictions_data, per_day_adaptation, update_stats = evaluate(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataset=test_dataset,
        day_index_map=day_index_map,
        days_to_eval=days_to_eval,
        device=device,
        horizon_steps=horizon_steps,
        paper_t1=t1,
        paper_t2=t2,
    )

    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    if save_results:
        extra = [
            "Method: paper Algorithm 1 lagged OGD",
            f"horizon_steps={horizon_steps}",
            f"t1={t1}",
            f"t2={t2}",
            f"ogd_lr={ogd_lr}",
            f"ogd_beta1={ogd_beta1}",
            f"ogd_beta2={ogd_beta2}",
            f"ogd_weight_decay={ogd_weight_decay}",
            f"ogd_loss={ogd_loss}",
            "Adaptation summary:",
            f"considered={update_stats['considered']}",
            f"none={update_stats['none']}",
            f"head={update_stats['head']}",
            f"whole={update_stats['whole']}",
            "Per-day adaptation:",
        ]
        for day_key in sorted(per_day_adaptation.keys()):
            day_stats = per_day_adaptation[day_key]
            extra.append(
                f"{day_key}: considered={day_stats['considered']} none={day_stats['none']} "
                f"head={day_stats['head']} whole={day_stats['whole']}"
            )
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
