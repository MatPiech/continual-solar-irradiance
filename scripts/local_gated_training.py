"""Local continual adaptation using prototype-based delayed gating.

This script reuses the same gating building blocks as the federated benchmark:
`build_gated_thresholds` and `eval_gated_online` from ``cirr.continual.gating``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import click
import numpy as np
import torch

from cirr.continual import build_gated_thresholds, eval_gated_online
from cirr.data.loading import load_dataset
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.losses import BerHuLoss
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics


def _select_days(day_index_map: dict[str, list[int]], start_day: str, num_days: int) -> list[str]:
    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day '{start_day}'. Must be MMDD format.")
    selected = all_days[all_days.index(start_day) :]
    if num_days > 0:
        selected = selected[:num_days]
    return [day for day in selected if day in day_index_map]


def _build_predictions_table(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        timestamp = str(row["timestamp"])
        day_key = timestamp[4:8] if len(timestamp) >= 8 else "----"
        prediction = float(row["prediction"])
        target = float(row["target"])
        out.append(
            {
                "timestamp": timestamp,
                "day": day_key,
                "prediction": prediction,
                "target": target,
                "error": prediction - target,
                "absolute_error": abs(prediction - target),
                "gate_zone": row.get("gate_zone", "pending"),
            }
        )
    return out


def _compute_per_day_metrics(predictions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"preds": [], "tgts": []})
    for row in predictions:
        grouped[row["day"]]["preds"].append(float(row["prediction"]))
        grouped[row["day"]]["tgts"].append(float(row["target"]))

    out: dict[str, dict[str, float]] = {}
    for day_key, vals in grouped.items():
        out[day_key] = compute_metrics(np.array(vals["preds"]), np.array(vals["tgts"]))
    return out


def run_local_gated_training(
    checkpoint: Path,
    dataset_path: Path,
    num_days: int,
    start_day: str,
    device: str,
    gate_calibration_samples: int,
    gate_horizon_steps: int,
    gate_epsilon: float,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_loss_threshold: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]], dict[str, int]]:
    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    dataset = load_dataset(dataset_path, split="test")
    day_index_map = group_indices_by_day(dataset.periods)
    days = _select_days(day_index_map, start_day=start_day, num_days=num_days)

    print(f"Dataset: {dataset_path.name}  samples={len(dataset.periods)}  days={len(days)}")
    print("Building gated prototypes/thresholds...")
    prototypes, thresholds = build_gated_thresholds(model, dataset, device, n_samples=gate_calibration_samples)

    optimizer = torch.optim.AdamW(model.network_head.parameters(), lr=adapt_lr, weight_decay=adapt_weight_decay)
    loss_fn = BerHuLoss(threshold=adapt_loss_threshold, scale=1000)

    preds, tgts, rows, gate_stats = eval_gated_online(
        model=model,
        dataset=dataset,
        day_map=day_index_map,
        days=days,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_fn,
        prototypes=prototypes,
        thresholds=thresholds,
        horizon_steps=gate_horizon_steps,
        epsilon=gate_epsilon,
    )

    overall = compute_metrics(np.array(preds), np.array(tgts))
    predictions = _build_predictions_table(rows)
    per_day = _compute_per_day_metrics(predictions)
    return overall, per_day, predictions, gate_stats


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--dataset-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--num-days", type=int, default=-1, show_default=True, help="Number of days to simulate (-1 = all).")
@click.option("--start-day", type=str, default="0101", show_default=True, help="Start day in MMDD format.")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda", show_default=True)
@click.option(
    "--gate-calibration-samples",
    type=int,
    default=300,
    show_default=True,
    help="Number of initial samples to estimate gating prototypes/thresholds.",
)
@click.option("--gate-horizon-steps", type=int, default=3, show_default=True, help="Delayed feedback horizon.")
@click.option("--gate-epsilon", type=float, default=0.05, show_default=True, help="Relative clear-sky error threshold.")
@click.option("--adapt-lr", type=float, default=1e-4, show_default=True, help="Gated head optimizer learning rate.")
@click.option("--adapt-weight-decay", type=float, default=1e-3, show_default=True, help="Gated head optimizer weight decay.")
@click.option("--adapt-loss-threshold", type=float, default=30.0, show_default=True, help="BerHu threshold.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--save-results", is_flag=True, default=False)
def main(
    checkpoint: Path,
    dataset_path: Path,
    num_days: int,
    start_day: str,
    device: str,
    gate_calibration_samples: int,
    gate_horizon_steps: int,
    gate_epsilon: float,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_loss_threshold: float,
    seed: int,
    save_results: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    run_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    dataset_name = dataset_path.name.split("_")[0]

    overall, per_day, predictions, gate_stats = run_local_gated_training(
        checkpoint=checkpoint,
        dataset_path=dataset_path,
        num_days=num_days,
        start_day=start_day,
        device=run_device,
        gate_calibration_samples=gate_calibration_samples,
        gate_horizon_steps=gate_horizon_steps,
        gate_epsilon=gate_epsilon,
        adapt_lr=adapt_lr,
        adapt_weight_decay=adapt_weight_decay,
        adapt_loss_threshold=adapt_loss_threshold,
    )

    print_overall_metrics(overall, tag=f"{dataset_name} gated-local")
    print("Gate stats:", gate_stats)

    if save_results:
        pred_file = Path(f"{dataset_name}_local_gated_predictions.csv")
        report_file = Path(f"{dataset_name}_local_gated_results.txt")
        save_predictions_csv(predictions, pred_file)
        save_metrics_report(
            tag="local_gated",
            dataset_name=dataset_name,
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            overall_metrics=overall,
            per_day_metrics=per_day,
            path=report_file,
            extra_lines=[
                f"gate_calibration_samples={gate_calibration_samples}",
                f"gate_horizon_steps={gate_horizon_steps}",
                f"gate_epsilon={gate_epsilon}",
                f"adapt_lr={adapt_lr}",
                f"adapt_weight_decay={adapt_weight_decay}",
                f"gate_stats={gate_stats}",
            ],
        )
        print(f"Saved predictions: {pred_file}")
        print(f"Saved report: {report_file}")


if __name__ == "__main__":
    main()
