"""Gated continual adaptation evaluation script."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch

from cirr.continual import build_gated_thresholds, eval_gated_online
from cirr.evaluation.printing import print_overall_metrics
from cirr.evaluation.results import save_metrics_report, save_predictions_csv
from cirr.evaluation.runner import load_eval_days, load_eval_model, resolve_device
from cirr.losses import BerHuLoss
from cirr.physics.thresholds import load_thresholds, save_thresholds
from cirr.utils.metrics import compute_metrics


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--dataset-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--thresholds-path", type=click.Path(path_type=Path), default=None)
@click.option("--fit-thresholds", is_flag=True, default=False)
@click.option("--horizon-steps", type=int, default=3, show_default=True)
@click.option("--epsilon", type=float, default=0.05, show_default=True)
@click.option("--ogd-lr", type=float, default=1e-4, show_default=True)
@click.option("--num-days", type=int, default=30, show_default=True)
@click.option("--start-day", type=str, default="0101", show_default=True)
@click.option("--random-gate", is_flag=True, default=False)
@click.option("--distance-unaware", is_flag=True, default=False)
@click.option("--device", type=str, default="cuda")
@click.option("--save-results", is_flag=True, default=False)
def main(
    checkpoint: Path,
    dataset_path: Path,
    thresholds_path: Path | None,
    fit_thresholds: bool,
    horizon_steps: int,
    epsilon: float,
    ogd_lr: float,
    num_days: int,
    start_day: str,
    random_gate: bool,
    distance_unaware: bool,
    device: str,
    save_results: bool,
) -> None:
    device = resolve_device(device)
    model = load_eval_model(checkpoint, device)
    loss_fn = BerHuLoss(threshold=30, scale=1000).to(device)
    optimizer = torch.optim.AdamW(model.network_head.parameters(), lr=ogd_lr)

    dataset, day_map, days, dataset_name = load_eval_days(dataset_path, start_day, num_days)
    prototypes, thresholds = build_gated_thresholds(model, dataset, device)

    if fit_thresholds or thresholds_path is None:
        if thresholds_path:
            save_thresholds(thresholds, thresholds_path)
    elif thresholds_path and thresholds_path.exists():
        thresholds = load_thresholds(thresholds_path)

    all_preds, all_tgts, predictions, gate_stats = eval_gated_online(
        model,
        dataset,
        day_map,
        days,
        device,
        optimizer,
        loss_fn,
        prototypes,
        thresholds,
        horizon_steps=horizon_steps,
        epsilon=epsilon,
        random_gate=random_gate,
        distance_unaware=distance_unaware,
    )

    metrics = compute_metrics(np.array(all_preds), np.array(all_tgts))
    backprop = gate_stats.get("backprop", 0)
    print_overall_metrics(metrics, tag=f"{dataset_name} [gated]")
    print(f"Gate stats: {gate_stats}")
    print(f"Backprop steps: {backprop} / {len(all_preds)} ({100 * backprop / max(len(all_preds), 1):.1f}%)")

    if save_results:
        save_predictions_csv(predictions, f"{dataset_name}_gated_predictions.csv")
        save_metrics_report(
            tag="gated",
            overall_metrics=metrics,
            per_day_metrics={},
            output_path=f"{dataset_name}_gated_results.txt",
            extra_lines=[f"gate_stats={gate_stats}", f"backprop={backprop}"],
        )


if __name__ == "__main__":
    main()
