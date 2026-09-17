"""Compare source checkpoints by static (no-adaptation) test-set accuracy.

Answers "which pre-trained model should the chapter deploy?" on the same
population the benchmark scores: per site, with Forecast Skill against Smart
Persistence.  Days are sampled across the calendar so the comparison is not
decided by one season.

    python scripts/compare_checkpoints.py \\
        --dataset data/folsom_prepared --dataset data/karpicko_prepared \\
        --checkpoint checkpoints/resnet50_epoch_015.ckpt \\
        --checkpoint checkpoints/epoch_010.ckpt \\
        --days-per-month 2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from cirr.data.loading import load_dataset
from cirr.evaluation.smart_persistence import predict_periods
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.physics.clear_sky import load_clear_sky_lookup
from cirr.training.day_loop import predict_day
from cirr.utils.dataset_utils import group_indices_by_day
from cirr.utils.metrics import compute_metrics


def select_days(day_map: dict[str, list[int]], days_per_month: int) -> list[str]:
    """First *days_per_month* available days of each month — deterministic, seasonal."""
    chosen: list[str] = []
    for month in range(1, 13):
        month_days = sorted(d for d in day_map if d.startswith(f"{month:02d}"))
        chosen.extend(month_days[:days_per_month])
    return chosen


@click.command()
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), multiple=True, required=True)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), multiple=True, required=True)
@click.option("--days-per-month", type=int, default=2, show_default=True)
@click.option(
    "--max-samples-per-day",
    type=int,
    default=0,
    show_default=True,
    help="Evenly stride each day down to at most this many samples (0 = use all).",
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda", show_default=True)
@click.option("--save-results", type=click.Path(path_type=Path), default=None)
def main(
    dataset: tuple[Path, ...],
    checkpoint: tuple[Path, ...],
    days_per_month: int,
    max_samples_per_day: int,
    batch_size: int,
    device: str,
    save_results: Path | None,
) -> None:
    """Static accuracy of each checkpoint on each site's test split."""
    rows: list[dict[str, Any]] = []

    for ds_path in dataset:
        site = ds_path.name.removesuffix("_prepared")
        ds = load_dataset(ds_path, split="test")
        day_map = group_indices_by_day(ds.periods)

        days = select_days(day_map, days_per_month)
        if max_samples_per_day > 0:
            # Even stride keeps the diurnal cycle intact; taking the first N would
            # score every checkpoint on mornings only.
            day_map = {d: idx[:: max(1, len(idx) // max_samples_per_day)][:max_samples_per_day] for d, idx in day_map.items()}
        indices = [i for d in days for i in day_map[d]]
        print(f"\n{site}: {len(days)} days, {len(indices)} samples")

        # Smart Persistence on exactly these samples — the FS denominator.
        sp_preds, sp_tgts, _ = predict_periods(
            [ds.periods[i] for i in indices], load_clear_sky_lookup(ds_path), period_scale=1 / 1000.0
        )
        sp_rmse = float(np.sqrt(np.mean((sp_preds - sp_tgts) ** 2)))
        sp = compute_metrics(sp_preds, sp_tgts)
        rows.append({"site": site, "model": "smart_persistence", **sp, "fs": 0.0})

        for ckpt in checkpoint:
            model = load_forecaster_from_checkpoint(ckpt, device=device, use_hparams=True)
            preds, tgts = [], []
            for day in days:
                p, t, _ = predict_day(model, ds, day_map[day], device, batch_size=batch_size, client_name=site)
                if len(p):
                    preds.append(p)
                    tgts.append(t)
            m = compute_metrics(np.concatenate(preds), np.concatenate(tgts))
            m["fs"] = 100.0 * (1.0 - m["rmse"] / sp_rmse)
            rows.append({"site": site, "model": ckpt.name, "arch": model.hparams["model_name"], **m})
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n{'site':<10} {'model':<32} {'MAE':>9} {'RMSE':>9} {'WAPE':>8} {'SMAPE':>8} {'FS':>7} {'N':>8}")
    print("-" * 96)
    for r in rows:
        print(
            f"{r['site']:<10} {r['model']:<32} {r['mae'] * 1000:>9.2f} {r['rmse'] * 1000:>9.2f} "
            f"{r['wape']:>7.2f}% {r['smape']:>7.2f}% {r['fs']:>6.2f}% {r['count']:>8}"
        )

    if save_results is not None:
        save_results.parent.mkdir(parents=True, exist_ok=True)
        save_results.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved to {save_results}")


if __name__ == "__main__":
    main()
