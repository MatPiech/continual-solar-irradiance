"""Fit and score the local-feature maintenance alert (docs/CH4_ADMISSIBILITY.md §5).

Every design choice here was fixed in the pre-registration before any synthetic
fault was scored: layer2 features of the forecaster, 3 x 3 neighbourhood pooling,
a 128-d random projection, per-cell nearest-neighbour banks of 2,000 nominal frames,
per-cell normalisation on 1,000 other frames, the worst cell as the frame score, a
99th-percentile frame threshold on held-out days, and a 30-minute causal rolling
median for the maintenance alert, thresholded on 30 whole held-out days.

    python scripts/fit_local_gate.py --out results/ch4/gate/local
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from cirr.data.loading import load_dataset
from cirr.evaluation.admissibility import (
    CorruptedStream,
    LocalGate,
    alert_statistic,
    local_features,
    nominal_split,
    stamp,
    valid_cells,
)
from cirr.evaluation.campaign import SOURCE_CHECKPOINT, load_campaign_days
from cirr.evaluation.runner import load_eval_model
from cirr.utils.provenance import git_commit, set_determinism, sha256_file


def score_stream(gate, model, data, dataset, device, workers):
    idx = [i for d in data.days for i in data.day_map[d]]
    ts = pd.Series([stamp(data.dataset.periods[i]) for i in idx])
    score, cell = gate.score(local_features(model, dataset, idx, gate.cells, device, workers), device)
    alert = alert_statistic(ts, score)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "score": score,
            "cell": cell,
            "flagged": score > gate.threshold,
            "alert_stat": alert,
            "alert": alert > gate.alert_threshold,
        }
    )


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=str(SOURCE_CHECKPOINT), show_default=True)
@click.option("--source", type=click.Path(exists=True, path_type=Path), default="data/folsom_prepared", show_default=True)
@click.option("--stream", type=click.Path(exists=True, path_type=Path), default="data/folsom_prepared", show_default=True)
@click.option(
    "--corruption-schedule",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="results/ch4/gate/folsom_corrupted/schedule.csv",
    show_default=True,
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--workers", type=click.IntRange(min=0), default=8, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
def main(
    checkpoint: Path,
    source: Path,
    stream: Path,
    corruption_schedule: Path,
    seed: int,
    workers: int,
    device: str,
    out_dir: Path,
) -> None:
    set_determinism(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_eval_model(Path(checkpoint), device)
    rng = np.random.default_rng(seed)

    # --- the same nominal frames and day split as the Isolation Forest gate --------
    train = load_dataset(source, split="training")
    fit_idx, cal_idx = nominal_split(train.periods, {"2014", "2015"}, 5, 0.2, seed)
    perm = rng.permutation(fit_idx)
    bank_idx, norm_idx = sorted(perm[:2000].tolist()), sorted(perm[2000:3000].tolist())
    thr_idx = sorted(rng.choice(cal_idx, 3000, replace=False).tolist())
    cal_days = sorted({stamp(train.periods[i])[:8] for i in cal_idx})
    alert_days = set(rng.choice(cal_days, 30, replace=False))
    alert_idx = [i for i, p in enumerate(train.periods) if stamp(p)[:8] in alert_days]  # every frame, real time
    assert not {stamp(train.periods[i])[:4] for i in bank_idx + norm_idx + thr_idx + alert_idx} - {"2014", "2015"}

    cells = valid_cells(train[bank_idx[0]][0])
    print(
        f"cells scored: {int(cells.sum())} of 256 | bank {len(bank_idx)} | norm {len(norm_idx)} | "
        f"threshold {len(thr_idx)} | alert calibration {len(alert_idx)} frames on {len(alert_days)} days"
    )

    bank = local_features(model, train, bank_idx, cells, device, workers).transpose(0, 1).contiguous()
    gate = LocalGate(bank, cells)
    gate.scale = torch.quantile(
        gate.distances(local_features(model, train, norm_idx, cells, device, workers), device).float(), 0.95, dim=0
    ).clamp_min(1e-6)
    thr_scores, _ = gate.score(local_features(model, train, thr_idx, cells, device, workers), device)
    gate.threshold = float(np.percentile(thr_scores, 99))
    a_scores, _ = gate.score(local_features(model, train, alert_idx, cells, device, workers), device)
    a_ts = pd.Series([stamp(train.periods[i]) for i in alert_idx])
    gate.alert_threshold = float(np.nanpercentile(alert_statistic(a_ts, a_scores), 99))
    print(f"frame threshold {gate.threshold:.4f} | alert threshold {gate.alert_threshold:.4f}")

    meta = dict(
        design="docs/CH4_ADMISSIBILITY.md §5",
        threshold=gate.threshold,
        alert_threshold=gate.alert_threshold,
        cells=int(cells.sum()),
        n_bank=len(bank_idx),
        n_norm=len(norm_idx),
        n_threshold=len(thr_idx),
        alert_calibration_days=len(alert_days),
        percentile=99.0,
        window_minutes=30,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        git_commit=git_commit(),
        seed=seed,
    )
    gate.save(out_dir / "local_gate.npz", **meta)

    # --- the 2016 stream, clean and with the synthetic faults ------------------------
    data = load_campaign_days(stream, min_ghi_clear=0.001)
    clean = score_stream(gate, model, data, data.dataset, device, workers)
    clean.to_csv(out_dir / "scores_clean.csv", index=False)
    sched = pd.read_csv(corruption_schedule, dtype={"timestamp": str})
    faulty = score_stream(gate, model, data, CorruptedStream(data.dataset, sched), device, workers)
    faulty.to_csv(out_dir / "scores_corrupted.csv", index=False)
    meta.update(clean_flag_rate_pct=float(100 * clean.flagged.mean()), clean_alert_rate_pct=float(100 * clean.alert.mean()))
    (out_dir / "gate.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("clean_flag_rate_pct", "clean_alert_rate_pct")}, indent=2))


if __name__ == "__main__":
    main()
