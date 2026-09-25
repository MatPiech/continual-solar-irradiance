"""Write the §8 fault streams (docs/CH4_ADMISSIBILITY.md): three doses and two continuous faults.

Deterministic: the same command on every server yields byte-identical schedules; the
printed SHA-256 lets you check.

    python scripts/make_dose_schedules.py --out results/adaptation/gate/dose
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click

from cirr.evaluation.admissibility import build_continuous_schedule, build_dose_schedule, stamp
from cirr.evaluation.campaign import load_campaign_days

STREAMS = {
    "r01": ("dose", 1.0),
    "r05": ("dose", 5.0),
    "r15": ("dose", 15.0),
    "cont_soiling": ("continuous", ("soiling", "20160701", "20160815")),
    "cont_occluder": ("continuous", ("occlusion", "20160901", "20161231")),
}


@click.command()
@click.option("--stream", "stream_path", type=click.Path(exists=True, path_type=Path), default="data/folsom_prepared")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default="results/adaptation/gate/dose")
@click.option("--only", multiple=True, help="Subset of stream names; default all.")
def main(stream_path: Path, out_dir: Path, only: tuple[str, ...]) -> None:
    data = load_campaign_days(stream_path, min_ghi_clear=0.001)
    ts = [stamp(data.dataset.periods[i]) for d in data.days for i in data.day_map[d]]
    for name, (mode, arg) in STREAMS.items():
        if only and name not in only:
            continue
        if mode == "dose":
            sched = build_dose_schedule(ts, arg, seed=int(round(arg)))
        else:
            sched = build_continuous_schedule(ts, *arg)
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        sched.to_csv(d / "schedule.csv", index=False)
        stats = dict(
            stream=name,
            frames=len(ts),
            faulty=int(len(sched)),
            faulty_pct=100 * len(sched) / len(ts),
            by_kind={k: dict(frames=int(len(g)), episodes=int(g.episode.nunique())) for k, g in sched.groupby("kind")},
            sha256=hashlib.sha256((d / "schedule.csv").read_bytes()).hexdigest(),
        )
        (d / "stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats))


if __name__ == "__main__":
    main()
