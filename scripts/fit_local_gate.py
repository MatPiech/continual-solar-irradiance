"""Fit and score the local-feature maintenance alert (docs/CH4_ADMISSIBILITY.md §5).

Every design choice here was fixed in the pre-registration before any synthetic
fault was scored: layer2 features of the forecaster, 3 x 3 neighbourhood pooling,
a 128-d random projection, per-cell nearest-neighbour banks of 2,000 nominal frames,
per-cell normalisation on 1,000 other frames, the worst cell as the frame score, a
99th-percentile frame threshold on held-out days, and a 30-minute causal rolling
median for the maintenance alert, thresholded on 30 whole held-out days.

    python scripts/fit_local_gate.py --out results/adaptation/gate/local
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
    local_scores,
    nominal_split,
    stamp,
    valid_cells,
)
from cirr.evaluation.campaign import SOURCE_CHECKPOINT, load_campaign_days
from cirr.evaluation.runner import load_eval_model
from cirr.utils.provenance import git_commit, set_determinism, sha256_file


def score_table(gate: LocalGate, ts: pd.Series, z: np.ndarray) -> pd.DataFrame:
    """The §5 columns (score, cell, flagged, alert) and their §6 interior/edge counterparts."""
    s = gate.split(z)
    alert, i_alert = alert_statistic(ts, s["all"]), alert_statistic(ts, s["interior"])
    return pd.DataFrame(
        {
            "timestamp": ts,
            "score": s["all"],
            "cell": s["all_cell"],
            "flagged": s["all"] > gate.threshold,
            "alert_stat": alert,
            "alert": alert > gate.alert_threshold,
            "interior": s["interior"],
            "interior_cell": s["interior_cell"],
            "interior_flagged": s["interior"] > gate.thresholds["interior"],
            "interior_alert_stat": i_alert,
            "interior_alert": i_alert > gate.thresholds["interior_alert"],
            "edge": s["edge"],
            "edge_cell": s["edge_cell"],
            "edge_flagged": s["edge"] > gate.thresholds["edge"],
        }
    )


def score_stream(gate, model, data, dataset, device, workers, cells_path: Path):
    idx = [i for d in data.days for i in data.day_map[d]]
    ts = pd.Series([stamp(data.dataset.periods[i]) for i in idx])
    z = local_scores(model, dataset, idx, gate, device, workers)
    np.save(cells_path, z)  # per-cell matrix, so any other split is scoreable without a GPU
    return score_table(gate, ts, z)


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=str(SOURCE_CHECKPOINT), show_default=True)
@click.option("--source", type=click.Path(exists=True, path_type=Path), default="data/folsom_prepared", show_default=True)
@click.option("--stream", type=click.Path(exists=True, path_type=Path), default="data/folsom_prepared", show_default=True)
@click.option(
    "--corruption-schedule",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="results/adaptation/gate/folsom_corrupted/schedule.csv",
    show_default=True,
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--workers", type=click.IntRange(min=0), default=8, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Reuse the bank and normalisation of local_gate.npz in --out; thresholds and streams are recomputed.",
)
def main(
    checkpoint: Path,
    source: Path,
    stream: Path,
    corruption_schedule: Path,
    seed: int,
    workers: int,
    device: str,
    out_dir: Path,
    resume: bool,
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

    if resume and (out_dir / "local_gate.npz").exists():
        gate = LocalGate.load(out_dir / "local_gate.npz")
        assert torch.equal(gate.cells, cells)
        print("resumed bank and normalisation from local_gate.npz")
    else:
        bank = local_features(model, train, bank_idx, cells, device, workers).transpose(0, 1).contiguous()
        gate = LocalGate(bank, cells)
        gate.scale = torch.quantile(
            gate.distances(local_features(model, train, norm_idx, cells, device, workers), device).float(), 0.95, dim=0
        ).clamp_min(1e-6)

    # thresholds: the §5 frame/alert pair, and the §6 pair per split, from the same calibration frames
    thr = gate.split(gate.normalised(local_features(model, train, thr_idx, cells, device, workers), device))
    a_z = gate.normalised(local_features(model, train, alert_idx, cells, device, workers), device)
    a_ts = pd.Series([stamp(train.periods[i]) for i in alert_idx])
    a_split = gate.split(a_z)
    for name in LocalGate.SPLITS:
        gate.thresholds[name] = float(np.percentile(thr[name], 99))
        gate.thresholds[f"{name}_alert"] = float(np.nanpercentile(alert_statistic(a_ts, a_split[name]), 99))
    gate.threshold, gate.alert_threshold = gate.thresholds["all"], gate.thresholds["all_alert"]
    print(f"frame threshold {gate.threshold:.4f} | alert threshold {gate.alert_threshold:.4f} (§5: 1.7527 / 1.6356)")
    print("§6 thresholds: " + ", ".join(f"{k} {v:.4f}" for k, v in gate.thresholds.items()))

    meta = dict(
        design="docs/CH4_ADMISSIBILITY.md §5 and §6",
        threshold=gate.threshold,
        alert_threshold=gate.alert_threshold,
        thresholds=gate.thresholds,
        cells=int(cells.sum()),
        edge_cells=int(gate.edge.sum()),
        n_bank=len(bank_idx),
        n_norm=len(norm_idx),
        n_threshold=len(thr_idx),
        alert_calibration_days=len(alert_days),
        percentile=99.0,
        window_minutes=30,
        bank_bytes_fp32=int(gate.bank.numel() * 4),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        git_commit=git_commit(),
        seed=seed,
    )
    gate.save(out_dir / "local_gate.npz", **meta)

    # --- the 2016 stream, clean and with the synthetic faults ------------------------
    data = load_campaign_days(stream, min_ghi_clear=0.001)
    clean = score_stream(gate, model, data, data.dataset, device, workers, out_dir / "cells_clean.npy")
    clean.to_csv(out_dir / "scores_clean.csv", index=False)
    sched = pd.read_csv(corruption_schedule, dtype={"timestamp": str})
    faulty = score_stream(
        gate, model, data, CorruptedStream(data.dataset, sched), device, workers, out_dir / "cells_corrupted.npy"
    )
    faulty.to_csv(out_dir / "scores_corrupted.csv", index=False)
    for col in ("flagged", "alert", "interior_flagged", "interior_alert", "edge_flagged"):
        meta[f"clean_{col}_rate_pct"] = float(100 * clean[col].mean())
    (out_dir / "gate.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k.startswith("clean_")}, indent=2))


if __name__ == "__main__":
    main()
