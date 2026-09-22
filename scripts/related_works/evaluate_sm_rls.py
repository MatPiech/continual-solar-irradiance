"""Evaluate forecasting with closed-form Sherman-Morrison RLS head adaptation.

Online, one-sample-at-a-time head update using the Sherman-Morrison rank-1
formula — no gradient computation, no optimizer state.

Algorithm (same causal delay structure as Algorithm 1 from the paper):

    For each timestep t:
      1. Predict y_hat_t with the current model (no gradient).
      2. Enqueue the sample in a causal delay buffer of depth ``horizon_steps``.
            3. When the delayed ground-truth arrives, update the model on every
                 sample using a fixed scope chosen via CLI:
                     - ``last_layer`` -> update output layer only
                     - ``full_head``  -> update both linear layers in the head

The head is adapted with a single closed-form RLS step per sample:

    P_{t+1} = (1/λ) * (P_t  -  P_t φ φ^T P_t / (λ + φ^T P_t φ))
    w_{t+1} = w_t  +  P_{t+1} φ * e_t

where φ is the feature vector entering the layer, e_t = y_t - ŷ_t is the
prediction error, and λ ∈ (0, 1] is a forgetting factor.

Reference:
    Zhang, J., Xu, Y., Xue, J., and Xiao, W.
    Real-time prediction of solar radiation based on online sequential extreme learning machine.
    In 2018 13th IEEE Conference on Industrial Electronics and Applications (ICIEA). 2018.
    doi: 10.1109/ICIEA.2018.8397688
"""

from pathlib import Path
import time
from typing import Literal

import click
import numpy as np

from cirr.evaluation.campaign import SOURCE_CHECKPOINT, CampaignData, load_campaign_days, write_campaign
from cirr.evaluation.online import DelayedQueueItem, OnlineEvalContext, OnlineEvalResult, run_delayed_online_eval
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.runner import load_eval_model, resolve_device
from cirr.models.forecaster import Forecaster
from cirr.physics.clear_sky import DEFAULT_MIN_GHI_CLEAR
from cirr.training.sm_rls import ShermanMorrisonHeadUpdater
from cirr.utils.provenance import set_determinism

# SM-RLS updates the head by a recursive least-squares step rather than a gradient
# step, but it is still a head-scope action, so it reports as one.
SM_ACTION = "head"


def evaluate(
    model: Forecaster,
    updater: ShermanMorrisonHeadUpdater,
    update_scope: Literal["last_layer", "full_head"],
    reset_covariance_per_day: bool,
    data: CampaignData,
    device: str,
    horizon_steps: int,
) -> OnlineEvalResult:
    def on_day_start(_day: str, _ctx: OnlineEvalContext) -> None:
        if reset_covariance_per_day:
            updater.reset_covariance()

    def on_day_end(_day: str, ctx: OnlineEvalContext) -> None:
        print(
            f"       adapt: considered={ctx.day_update_stats['considered']} "
            f"updated={ctx.day_update_stats['updated']} "
            f"scope={update_scope}",
            flush=True,
        )

    def on_revealed(oldest: DelayedQueueItem, ctx: OnlineEvalContext) -> None:
        ctx.update_stats["considered"] += 1
        ctx.day_update_stats["considered"] += 1
        delayed_abs_error = abs(oldest.raw_pred - float(oldest.gt.view(-1)[0].item()))
        row = ctx.predictions_data[oldest.pred_row_idx]
        row["update_considered"] = True
        row["delayed_abs_error"] = delayed_abs_error
        updater.update(
            model=model,
            image=oldest.image,
            history=oldest.history,
            gt=oldest.gt,
            device=device,
            scope=update_scope,
        )
        ctx.update_stats["updated"] += 1
        ctx.day_update_stats["updated"] += 1
        row["adapted"] = True
        row["adaptation_scope"] = SM_ACTION

    result = run_delayed_online_eval(
        model,
        data.dataset,
        data.day_map,
        data.days,
        device,
        horizon_steps,
        on_revealed,
        update_stats_init={"considered": 0, "updated": 0},
        extra_pred_fields={
            "update_considered": False,
            "adapted": False,
            "adaptation_scope": "not_available",
            "delayed_abs_error": np.nan,
        },
        on_day_start=on_day_start,
        on_day_end=on_day_end,
    )
    print(
        f"\nSM-RLS update stats:"
        f"  considered={result.update_stats['considered']}"
        f"  updated={result.update_stats['updated']}"
        f"  scope={update_scope}",
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
    help="Run folder. Defaults to results/ch4/<site>/sm_rls/seed<k>.",
)
@click.option(
    "--min-ghi-clear",
    type=float,
    default=DEFAULT_MIN_GHI_CLEAR,
    show_default=True,
    help="Daylight filter (kW/m^2), shared with the federated benchmark. 0 disables it.",
)
@click.option("--horizon-steps", type=click.IntRange(min=1), default=15, show_default=True)
@click.option("--update-scope", type=click.Choice(["last_layer", "full_head"]), default="last_layer", show_default=True)
@click.option("--lambda-init", type=float, default=1.0, show_default=True)
@click.option("--forgetting-factor", type=click.FloatRange(min=1e-6, max=1.0), default=1.0, show_default=True)
@click.option("--error-clip", type=click.FloatRange(min=1e-6), default=0.2, show_default=True)
@click.option("--max-last-layer-update-norm", type=click.FloatRange(min=1e-8), default=0.01, show_default=True)
@click.option("--max-first-layer-update-norm", type=click.FloatRange(min=1e-8), default=0.002, show_default=True)
@click.option("--anchor-decay-last", type=click.FloatRange(min=0.0, max=1.0), default=1e-4, show_default=True)
@click.option("--anchor-decay-first", type=click.FloatRange(min=0.0, max=1.0), default=5e-4, show_default=True)
@click.option("--full-head-regularization-mult", type=click.FloatRange(min=1.0), default=100.0, show_default=True)
@click.option("--reset-covariance-per-day/--keep-covariance", default=False, show_default=True)
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
    update_scope: str,
    lambda_init: float,
    forgetting_factor: float,
    error_clip: float,
    max_last_layer_update_norm: float,
    max_first_layer_update_norm: float,
    anchor_decay_last: float,
    anchor_decay_first: float,
    full_head_regularization_mult: float,
    reset_covariance_per_day: bool,
    num_days: int,
    start_day: str,
    seed: int,
    device: str,
) -> None:
    """Evaluate final-layer Sherman-Morrison recursive least-squares adaptation."""
    set_determinism(seed)
    device = resolve_device(device)
    checkpoint = Path(checkpoint)

    data = load_campaign_days(dataset_path, start_day, num_days, min_ghi_clear)
    print(
        f"{data.site}: {len(data.days)} days, {data.n_samples} lit samples "
        f"({data.dropped_total} dark samples dropped at min_ghi_clear={min_ghi_clear})"
    )

    model = load_eval_model(checkpoint, device)
    updater = ShermanMorrisonHeadUpdater(
        model=model,
        lambda_init=lambda_init,
        forgetting_factor=forgetting_factor,
        error_clip=error_clip,
        max_last_layer_update_norm=max_last_layer_update_norm,
        max_first_layer_update_norm=max_first_layer_update_norm,
        anchor_decay_last=anchor_decay_last,
        anchor_decay_first=anchor_decay_first,
        full_head_regularization_mult=full_head_regularization_mult,
        device=device,
    )
    print(f"SM-RLS:  scope={update_scope}  lambda={lambda_init}  ff={forgetting_factor}  seed={seed}\n")

    started = time.perf_counter()
    result = evaluate(
        model=model,
        updater=updater,
        update_scope=update_scope,  # type: ignore[arg-type]
        reset_covariance_per_day=reset_covariance_per_day,
        data=data,
        device=device,
        horizon_steps=horizon_steps,
    )
    wall_time_s = time.perf_counter() - started

    print_overall_metrics(result.overall_metrics, tag=f"{data.site}  [sm_rls]")

    out_dir = Path(out_dir) if out_dir else Path("results/ch4") / data.site / "sm_rls" / f"seed{seed}"
    write_campaign(
        out_dir,
        result=result,
        data=data,
        config={
            "method": "sm_rls",
            "update_scope": update_scope,
            "lambda_init": lambda_init,
            "forgetting_factor": forgetting_factor,
            "error_clip": error_clip,
            "reset_covariance_per_day": reset_covariance_per_day,
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
