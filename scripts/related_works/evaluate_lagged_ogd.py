"""Conditional on-device adaptation under lagged supervision (thesis Section 4.3).

Algorithm 1: predict at *t*, buffer, and when the label arrives ``horizon_steps``
later compare the realized absolute error against two thresholds, which select
between the two update actions of the section:

    error < T1        -> no update
    T1 <= error < T2  -> head-only update,   S = {0}^|theta_b| x R^|theta_h|
    error >= T2       -> full-model update,  S = R^P

The error is computed on the *reported* forecast, that is after the
non-negativity clamp, matching ``eq:ch4-abs-error``: irradiance is non-negative
everywhere in the data, so a negative model output is not a forecast the device
would emit.  Each run also records the action a trigger on the raw output would
have selected; as long as that count stays zero the two definitions agree, and a
nonzero count means the model has started predicting negative irradiance and the
clamp is hiding that error from the policy.

The full-model action is the whole network, as the chapter describes it.  Results
published before Phase 1 came from a code path that unfroze two parameter-free
encoder children, so their high-error action trained only the head; see
``cirr.training.ogd.set_trainable``.  Those runs are not reproducible from this
script and should not be compared against its output.

Reference:
    Piechocki, M., Kraft, M., Capotondi, A. On-device continual adaptation for
    reliable solar irradiance forecasting. In Advanced Concepts for Intelligent
    Vision Systems. 2026. doi: 10.1007/978-3-032-07343-3_28
"""

from pathlib import Path
import time
from typing import Any

import click
import numpy as np
import torch
import torch.nn as nn

from cirr.evaluation.campaign import (
    SOURCE_CHECKPOINT,
    CampaignData,
    load_campaign_days,
    score_without_adaptation,
    static_baseline,
    write_campaign,
)
from cirr.evaluation.online import DelayedQueueItem, OnlineEvalContext, run_delayed_online_eval
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.runner import load_eval_model, resolve_device
from cirr.models.forecaster import Forecaster
from cirr.physics.clear_sky import DEFAULT_MIN_GHI_CLEAR
from cirr.training.ogd import algorithm1_scope, make_regression_loss, ogd_step
from cirr.utils.provenance import set_determinism

SCOPE_TO_ACTION = {"none": "none", "head_only": "head", "head_and_encoder": "whole"}


def run_policy(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    data: CampaignData,
    device: str,
    horizon_steps: int,
    t1: float,
    t2: float,
    encoder_blocks: int | None = None,
) -> Any:
    def on_revealed(oldest: DelayedQueueItem, ctx: OnlineEvalContext) -> None:
        ctx.update_stats["considered"] += 1
        ctx.day_update_stats["considered"] += 1
        delayed_gt = float(oldest.gt.view(-1)[0].item())
        # L is the error of the forecast the device reported, so the clamp applies
        # here too; ``raw_pred`` is kept for the divergence monitor below.
        delayed_abs_error = abs(max(oldest.raw_pred, 0.0) - delayed_gt)
        scope = algorithm1_scope(delayed_abs_error, t1, t2)

        row = ctx.predictions_data[oldest.pred_row_idx]
        row["update_considered"] = True
        row["delayed_abs_error"] = delayed_abs_error
        row["delayed_abs_error_raw"] = abs(oldest.raw_pred - delayed_gt)
        # Sentinel: the action an unclamped trigger would have taken.  Any
        # disagreement means the model emitted a negative prediction.
        row["scope_if_raw"] = SCOPE_TO_ACTION[algorithm1_scope(row["delayed_abs_error_raw"], t1, t2)]
        row["adaptation_scope"] = SCOPE_TO_ACTION[scope]

        if scope == "none":
            ctx.update_stats["none"] += 1
            ctx.day_update_stats["none"] += 1
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
            encoder_blocks=encoder_blocks,
        )
        action = SCOPE_TO_ACTION[scope]
        ctx.update_stats[action] += 1
        ctx.day_update_stats[action] += 1
        row["adapted"] = True

    return run_delayed_online_eval(
        model,
        data.dataset,
        data.day_map,
        data.days,
        device,
        horizon_steps,
        on_revealed,
        update_stats_init={"considered": 0, "none": 0, "head": 0, "whole": 0},
        extra_pred_fields={
            "update_considered": False,
            "adapted": False,
            "adaptation_scope": "not_available",
            "scope_if_raw": "not_available",
            "delayed_abs_error": np.nan,
            "delayed_abs_error_raw": np.nan,
        },
    )


def retention_report(
    model: Forecaster,
    checkpoint: Path,
    source_dataset: Path | None,
    deployment: CampaignData,
    early_days: int,
    source_days: int,
    device: str,
    horizon_steps: int,
    min_ghi_clear: float,
) -> dict[str, Any]:
    """Score the adapted model where it is *not* supposed to have improved.

    Two references, because forgetting can show up in either: the held-out source
    stream the model was pre-trained on, and the early segment of the deployment
    stream the model has already moved past.
    """
    out: dict[str, Any] = {"early_days": early_days}

    early = CampaignData(
        dataset=deployment.dataset,
        day_map=deployment.day_map,
        days=deployment.days[:early_days],
        lookup=deployment.lookup,
        dropped_per_day=deployment.dropped_per_day,
        dropped_total=deployment.dropped_total,
        site=deployment.site,
    )
    out["early_segment"] = {
        "site": deployment.site,
        "adapted": score_without_adaptation(model, early, device, horizon_steps),
    }

    if source_dataset is not None:
        # A full source year is an hour of GPU time for a check that only needs to
        # detect degradation, so it is bounded and the bound is recorded.
        source = load_campaign_days(source_dataset, num_days=source_days, min_ghi_clear=min_ghi_clear)
        out["source_days"] = source_days
        out["source_stream"] = {
            "site": source.site,
            "n_days": len(source.days),
            "adapted": score_without_adaptation(model, source, device, horizon_steps),
            "frozen": static_baseline(checkpoint, source_dataset, device, horizon_steps, min_ghi_clear, num_days=source_days),
        }
    return out


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
    help="Prepared dataset directory (e.g. data/karpicko_prepared).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Run folder. Defaults to results/ch4/<site>/conditional/seed<k>.",
)
@click.option(
    "--min-ghi-clear",
    type=float,
    default=DEFAULT_MIN_GHI_CLEAR,
    show_default=True,
    help="Daylight filter (kW/m^2), shared with the federated benchmark. 0 disables it.",
)
@click.option("--horizon-steps", type=click.IntRange(min=1), default=15, show_default=True)
@click.option("--t1", type=float, default=0.01, show_default=True, help="Lower threshold (kW/m^2).")
@click.option("--t2", type=float, default=0.1, show_default=True, help="Upper threshold (kW/m^2).")
@click.option(
    "--encoder-blocks",
    type=click.IntRange(min=1),
    default=None,
    help="Ablation: the high-error action unfreezes only the last N parameterised encoder "
    "groups instead of the whole encoder. 2 = ResNet50 layer3+layer4. Default: whole encoder.",
)
@click.option("--ogd-lr", type=float, default=1e-5, show_default=True)
@click.option("--ogd-beta1", type=float, default=0.5, show_default=True)
@click.option("--ogd-beta2", type=float, default=0.975, show_default=True)
@click.option("--ogd-weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--ogd-loss", type=click.Choice(["berhu", "huber", "mse"]), default="mse", show_default=True)
@click.option("--num-days", type=int, default=-1, show_default=True, help="-1 means all available days.")
@click.option("--start-day", type=str, default="0101", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option(
    "--retention-source",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Source stream to re-score after the campaign (e.g. data/folsom_prepared).",
)
@click.option("--retention-early-days", type=int, default=30, show_default=True)
@click.option(
    "--retention-source-days",
    type=int,
    default=30,
    show_default=True,
    help="Days of the source stream to re-score. A full year costs about an hour.",
)
@click.option("--no-retention", is_flag=True, default=False, help="Skip the retention pass.")
def main(
    checkpoint: Path,
    dataset_path: Path,
    out_dir: Path | None,
    min_ghi_clear: float,
    horizon_steps: int,
    t1: float,
    t2: float,
    encoder_blocks: int | None,
    ogd_lr: float,
    ogd_beta1: float,
    ogd_beta2: float,
    ogd_weight_decay: float,
    ogd_loss: str,
    num_days: int,
    start_day: str,
    seed: int,
    device: str,
    retention_source: Path | None,
    retention_early_days: int,
    retention_source_days: int,
    no_retention: bool,
) -> None:
    """Evaluate the conditional lagged-supervision update policy."""
    if t1 >= t2:
        raise click.UsageError(f"--t1 ({t1}) must be strictly less than --t2 ({t2}).")

    set_determinism(seed)
    device = resolve_device(device)
    checkpoint = Path(checkpoint)

    data = load_campaign_days(dataset_path, start_day, num_days, min_ghi_clear)
    print(
        f"{data.site}: {len(data.days)} days, {data.n_samples} lit samples "
        f"({data.dropped_total} dark samples dropped at min_ghi_clear={min_ghi_clear})"
    )

    model = load_eval_model(checkpoint, device)
    loss_fn = make_regression_loss(ogd_loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ogd_lr, betas=(ogd_beta1, ogd_beta2), weight_decay=ogd_weight_decay)
    scope = "whole encoder" if encoder_blocks is None else f"last {encoder_blocks} encoder groups"
    print(f"T1={t1}  T2={t2}  high-error action: head + {scope}  seed={seed}\n")

    started = time.perf_counter()
    result = run_policy(model, optimizer, loss_fn, data, device, horizon_steps, t1, t2, encoder_blocks)
    wall_time_s = time.perf_counter() - started

    print_overall_metrics(result.overall_metrics, tag=f"{data.site}  [conditional]")
    print(f"\nAction mix: {dict(result.update_stats)}")

    retention = None
    if not no_retention:
        retention = retention_report(
            model,
            checkpoint,
            retention_source,
            data,
            retention_early_days,
            retention_source_days,
            device,
            horizon_steps,
            min_ghi_clear,
        )

    out_dir = Path(out_dir) if out_dir else Path("results/ch4") / data.site / "conditional" / f"seed{seed}"
    write_campaign(
        out_dir,
        result=result,
        data=data,
        config={
            "method": "conditional",
            "t1": t1,
            "t2": t2,
            "encoder_blocks": encoder_blocks,
            "ogd_lr": ogd_lr,
            "ogd_betas": [ogd_beta1, ogd_beta2],
            "ogd_weight_decay": ogd_weight_decay,
            "ogd_loss": ogd_loss,
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
        retention=retention,
    )
    print(f"\nRun written to {out_dir}")


if __name__ == "__main__":
    main()
