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
"""

from pathlib import Path
from typing import Any, Literal

import click
import numpy as np

from cirr.evaluation.online import DelayedQueueItem, OnlineEvalContext, run_delayed_online_eval
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.runner import load_eval_days, load_eval_model, print_eval_plan, resolve_device, save_online_eval_results
from cirr.models.forecaster import Forecaster
from cirr.training.sm_rls import ShermanMorrisonHeadUpdater


def evaluate(
    model: Forecaster,
    updater: ShermanMorrisonHeadUpdater,
    update_scope: Literal["last_layer", "full_head"],
    reset_covariance_per_day: bool,
    dataset: Any,
    day_index_map: dict[str, list[int]],
    days_to_eval: list[str],
    device: str,
    horizon_steps: int,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    dict[str, dict[str, int]],
    dict[str, int],
]:
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
        row["adaptation_scope"] = update_scope

    result = run_delayed_online_eval(
        model,
        dataset,
        day_index_map,
        days_to_eval,
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
    "--update-scope",
    type=click.Choice(["last_layer", "full_head"], case_sensitive=False),
    default="last_layer",
    show_default=True,
    help="SM-RLS update scope applied to every delayed sample.",
)
@click.option(
    "--lambda-init",
    type=float,
    default=1.0,
    show_default=True,
    help="Initialises P = (1/lambda_init)*I.  Larger ⟹ stronger regularisation.",
)
@click.option(
    "--forgetting-factor",
    type=click.FloatRange(min=1e-6, max=1.0),
    default=1.0,
    show_default=True,
    help="Exponential forgetting factor λ ∈ (0, 1]. 1.0 = no forgetting (standard RLS).",
)
@click.option(
    "--full-head-regularization-mult",
    type=click.FloatRange(min=1.0),
    default=100.0,
    show_default=True,
    help="Extra regularization factor for full-head hidden-layer RLS (larger = more conservative).",
)
@click.option(
    "--error-clip",
    type=click.FloatRange(min=1e-6),
    default=0.2,
    show_default=True,
    help="Absolute clip applied to per-sample error before SM-RLS update.",
)
@click.option(
    "--max-last-layer-update-norm",
    type=click.FloatRange(min=1e-8),
    default=0.01,
    show_default=True,
    help="L2 norm cap for one-step output-layer parameter update.",
)
@click.option(
    "--max-first-layer-update-norm",
    type=click.FloatRange(min=1e-8),
    default=0.002,
    show_default=True,
    help="L2 norm cap for one-step hidden-layer parameter update.",
)
@click.option(
    "--anchor-decay-last",
    type=click.FloatRange(min=0.0, max=1.0),
    default=1e-4,
    show_default=True,
    help="Per-step pullback strength of output layer toward checkpoint weights.",
)
@click.option(
    "--anchor-decay-first",
    type=click.FloatRange(min=0.0, max=1.0),
    default=5e-4,
    show_default=True,
    help="Per-step pullback strength of hidden layer toward checkpoint weights.",
)
@click.option(
    "--reset-covariance-per-day/--no-reset-covariance-per-day",
    default=True,
    show_default=True,
    help="Reset RLS covariance matrices at day boundaries while keeping adapted weights.",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    show_default=True,
    help="Number of days to evaluate. -1 = all available days.",
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
    update_scope: str,
    lambda_init: float,
    forgetting_factor: float,
    full_head_regularization_mult: float,
    error_clip: float,
    max_last_layer_update_norm: float,
    max_first_layer_update_norm: float,
    anchor_decay_last: float,
    anchor_decay_first: float,
    reset_covariance_per_day: bool,
    num_days: int,
    start_day: str,
    device: str,
    save_results: bool,
) -> None:
    """Evaluate forecasting with closed-form Sherman-Morrison RLS head adaptation."""

    device = resolve_device(device)
    dataset_name = dataset_path.name.split("_")[0]
    method_tag = "zhang_sm_rls"

    print(f"Loading model from {checkpoint}...")
    model = load_eval_model(checkpoint, device)

    updater = ShermanMorrisonHeadUpdater(
        model=model,
        lambda_init=lambda_init,
        forgetting_factor=forgetting_factor,
        full_head_regularization_mult=full_head_regularization_mult,
        error_clip=error_clip,
        max_last_layer_update_norm=max_last_layer_update_norm,
        max_first_layer_update_norm=max_first_layer_update_norm,
        anchor_decay_last=anchor_decay_last,
        anchor_decay_first=anchor_decay_first,
        device=device,
    )

    print(
        f"SM-RLS:  update_scope={update_scope}"
        f"  lambda_init={lambda_init}  forgetting={forgetting_factor}"
        f"  full_head_reg_mult={full_head_regularization_mult}"
        f"  error_clip={error_clip}"
        f"  max_last_delta={max_last_layer_update_norm}"
        f"  max_first_delta={max_first_layer_update_norm}"
        f"  anchor_last={anchor_decay_last}"
        f"  anchor_first={anchor_decay_first}"
        f"  reset_cov_per_day={reset_covariance_per_day}"
        f"  horizon_steps={horizon_steps}"
    )

    # ------------------------------------------------------------------
    # Load dataset and determine days to evaluate
    # ------------------------------------------------------------------
    test_dataset, day_index_map, days_to_eval, _ = load_eval_days(dataset_path, start_day, num_days)
    print_eval_plan(dataset_path, test_dataset, day_index_map, days_to_eval, start_day)

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print("Running SM-RLS evaluation...\n")
    overall_metrics, per_day_metrics, predictions_data, per_day_adaptation, update_stats = evaluate(
        model=model,
        updater=updater,
        update_scope=update_scope,  # type: ignore[arg-type]
        reset_covariance_per_day=reset_covariance_per_day,
        dataset=test_dataset,
        day_index_map=day_index_map,
        days_to_eval=days_to_eval,
        device=device,
        horizon_steps=horizon_steps,
    )

    print_overall_metrics(overall_metrics, tag=f"{dataset_name}  [{method_tag}]")

    if save_results:
        extra = [
            "Method: Sherman-Morrison RLS",
            f"horizon_steps={horizon_steps}",
            f"update_scope={update_scope}",
            f"lambda_init={lambda_init}",
            f"forgetting_factor={forgetting_factor}",
            f"full_head_regularization_mult={full_head_regularization_mult}",
            f"error_clip={error_clip}",
            f"max_last_layer_update_norm={max_last_layer_update_norm}",
            f"max_first_layer_update_norm={max_first_layer_update_norm}",
            f"anchor_decay_last={anchor_decay_last}",
            f"anchor_decay_first={anchor_decay_first}",
            f"reset_covariance_per_day={reset_covariance_per_day}",
            "Adaptation summary:",
            f"considered={update_stats['considered']}",
            f"updated={update_stats['updated']}",
            f"scope={update_scope}",
            "Per-day adaptation:",
        ]
        for day_key in sorted(per_day_adaptation.keys()):
            s = per_day_adaptation[day_key]
            extra.append(f"{day_key}: considered={s['considered']} updated={s['updated']} scope={update_scope}")
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
