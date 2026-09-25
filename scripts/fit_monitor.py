"""Calibrate and score the two-channel monitor (docs/CH4_ADMISSIBILITY.md §7).

Reuses the §5/§6 local detector unchanged (bank, normalisation, calibration frames and
days) and the clean 2016 per-cell matrix that ``fit_local_gate.py`` saved, so only the
calibration frames and the §7 corrupted stream need a GPU pass.

* global G = median per-cell score, core C = max over cells at depth >= 3;
* exclusion flag E = [G > t_G] or [C > t_C], t = 99.5th percentile on the calibration frames;
* maintenance alert M = causal 60-minute rolling median of G, 99th percentile on the 30 calibration days.

It also writes E in the format ``evaluate_lagged_ogd.py --anomaly-gate`` reads
(``gate.json`` + ``scores.csv``, score = max(G / t_G, C / t_C), threshold 1), so the
policy can withhold the flagged frames from its updates.

    python scripts/fit_monitor.py --gate results/adaptation/gate/local --out results/adaptation/gate/monitor
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from cirr.data.loading import load_dataset
from cirr.evaluation.admissibility import (
    CorruptedStream,
    LocalGate,
    alert_statistic,
    build_monitor_schedule,
    local_features,
    local_scores,
    nominal_split,
    stamp,
)
from cirr.evaluation.campaign import SOURCE_CHECKPOINT, load_campaign_days
from cirr.evaluation.runner import load_eval_model
from cirr.utils.provenance import git_commit, set_determinism

EXCLUDE_PCT, ALERT_PCT, WINDOW_MIN = 99.5, 99.0, 60


def monitor_table(gate: LocalGate, ts: pd.Series, z: np.ndarray, thr: dict[str, float]) -> pd.DataFrame:
    m = gate.monitor(z)
    alert = alert_statistic(ts, m["global"], minutes=WINDOW_MIN)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "global": m["global"],
            "core": m["core"],
            "core_cell": m["core_cell"],
            "excluded": (m["global"] > thr["global"]) | (m["core"] > thr["core"]),
            "maint_stat": alert,
            "maint_alert": alert > thr["maint"],
        }
    )


def write_gate(out: Path, table: pd.DataFrame, thr: dict[str, float]) -> None:
    """E as an --anomaly-gate directory: a frame is rejected when its score exceeds 1."""
    out.mkdir(parents=True, exist_ok=True)
    score = np.maximum(table["global"] / thr["global"], table["core"] / thr["core"])
    pd.DataFrame({"timestamp": table.timestamp, "score": score, "flagged": score > 1.0}).drop_duplicates("timestamp").to_csv(
        out / "scores.csv", index=False
    )
    (out / "gate.json").write_text(
        json.dumps(
            dict(
                threshold=1.0,
                percentile=EXCLUDE_PCT,
                score="max(G / t_G, C / t_C), docs/CH4_ADMISSIBILITY.md §7",
                t_global=thr["global"],
                t_core=thr["core"],
                flag_rate_pct=float(100 * (score > 1.0).mean()),
            ),
            indent=2,
        )
    )


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=str(SOURCE_CHECKPOINT), show_default=True)
@click.option("--source", type=click.Path(exists=True, path_type=Path), default="data/folsom_prepared", show_default=True)
@click.option("--stream", type=click.Path(exists=True, path_type=Path), default="data/folsom_prepared", show_default=True)
@click.option("--gate", "gate_dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--seed", type=int, default=0, show_default=True, help="Calibration split seed (as fit_local_gate.py).")
@click.option("--schedule-seed", type=int, default=1, show_default=True)
@click.option("--workers", type=click.IntRange(min=0), default=4, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
def main(checkpoint, source, stream, gate_dir, seed, schedule_seed, workers, device, out_dir) -> None:
    set_determinism(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_eval_model(Path(checkpoint), device)
    gate = LocalGate.load(gate_dir / "local_gate.npz")

    # the same calibration frames and days as fit_local_gate.py (same rng, same draw order)
    rng = np.random.default_rng(seed)
    train = load_dataset(source, split="training")
    fit_idx, cal_idx = nominal_split(train.periods, {"2014", "2015"}, 5, 0.2, seed)
    rng.permutation(fit_idx)  # the bank / normalisation draw, consumed to keep the stream aligned
    thr_idx = sorted(rng.choice(cal_idx, 3000, replace=False).tolist())
    cal_days = sorted({stamp(train.periods[i])[:8] for i in cal_idx})
    alert_days = set(rng.choice(cal_days, 30, replace=False))
    alert_idx = [i for i, p in enumerate(train.periods) if stamp(p)[:8] in alert_days]

    t_m = gate.monitor(gate.normalised(local_features(model, train, thr_idx, gate.cells, device, workers), device))
    a_m = gate.monitor(gate.normalised(local_features(model, train, alert_idx, gate.cells, device, workers), device))
    a_ts = pd.Series([stamp(train.periods[i]) for i in alert_idx])
    thr = {
        "global": float(np.percentile(t_m["global"], EXCLUDE_PCT)),
        "core": float(np.percentile(t_m["core"], EXCLUDE_PCT)),
        "maint": float(np.nanpercentile(alert_statistic(a_ts, a_m["global"], minutes=WINDOW_MIN), ALERT_PCT)),
    }
    cal_excl = float(100 * ((t_m["global"] > thr["global"]) | (t_m["core"] > thr["core"])).mean())
    print(f"§7 thresholds: {thr} | calibration exclusion rate {cal_excl:.2f} %")

    # clean 2016 stream from the saved per-cell matrix; §7 corrupted stream on the GPU
    data = load_campaign_days(stream, min_ghi_clear=0.001)
    idx = [i for d in data.days for i in data.day_map[d]]
    ts = pd.Series([stamp(data.dataset.periods[i]) for i in idx])
    z_clean = np.load(gate_dir / "cells_clean.npy")
    assert len(z_clean) == len(ts), "cells_clean.npy does not match the stream; re-run fit_local_gate.py"
    clean = monitor_table(gate, ts, z_clean, thr)
    clean.to_csv(out_dir / "scores_clean.csv", index=False)

    sched = build_monitor_schedule(ts.tolist(), seed=schedule_seed)
    sched.to_csv(out_dir / "schedule.csv", index=False)
    print("schedule:", sched.groupby("kind").agg(frames=("timestamp", "size"), episodes=("episode", "nunique")).to_dict())
    z_cor = local_scores(model, CorruptedStream(data.dataset, sched), idx, gate, device, workers)
    np.save(out_dir / "cells_corrupted.npy", z_cor)
    faulty = monitor_table(gate, ts, z_cor, thr)
    faulty.to_csv(out_dir / "scores_corrupted.csv", index=False)

    write_gate(out_dir / "exclusion_clean", clean, thr)
    write_gate(out_dir / "exclusion_corrupted", faulty, thr)
    meta = dict(
        design="docs/CH4_ADMISSIBILITY.md §7",
        thresholds=thr,
        exclude_percentile=EXCLUDE_PCT,
        alert_percentile=ALERT_PCT,
        window_minutes=WINDOW_MIN,
        core_cells=int(gate.core.sum()),
        calibration_exclusion_rate_pct=cal_excl,
        clean_exclusion_rate_pct=float(100 * clean.excluded.mean()),
        clean_maint_alert_rate_pct=float(100 * clean.maint_alert.mean()),
        schedule_seed=schedule_seed,
        git_commit=git_commit(),
    )
    (out_dir / "monitor.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
