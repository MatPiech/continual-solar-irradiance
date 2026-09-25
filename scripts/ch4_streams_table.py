"""The policy under faults: four streams x three arms, the central table of Chapter 4's admissibility section.

Streams: clean 2016; transient faults (docs/CH4_ADMISSIBILITY.md §7 stream); continuous
soiling and permanent occluder (§8).  Arms: static model (policy with T1 = T2 = inf),
damped policy, damped policy gated by the exclusion flag E.  Metrics on fault-free and on
faulty frames, plus the gated-vs-ungated and ungated-vs-static contrasts (day-block bootstrap).

    python scripts/ch4_streams_table.py      # -> tables/ch4_admissibility_streams*.csv, tables/ch4_streams.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admissibility_study as S  # noqa: E402
from cirr.fl.analysis import contrast  # noqa: E402

ROOT, RUNS, GATE, TABLES, W = S.ROOT, S.RUNS, S.GATE, S.TABLES, S.W

STREAMS = {
    "clean": (
        None,
        {
            "static": S.STATIC["folsom"],
            "ungated": ROOT / "results/adaptation/ablation/folsom/conditional_last2_lowlr/seed0",
            "gated": RUNS / "monitor_clean/last2_lowlr_excl/seed0",
        },
    ),
    "transient": (
        GATE / "monitor/schedule.csv",
        {
            "static": RUNS / "monitor_corrupted/static/seed0",
            "ungated": RUNS / "monitor_corrupted/last2_lowlr/seed0",
            "gated": RUNS / "monitor_corrupted/last2_lowlr_excl/seed0",
        },
    ),
    "cont_soiling": (
        GATE / "dose/cont_soiling/schedule.csv",
        {a: RUNS / "dose_cont_soiling" / n / "seed0" for a, n in S.DOSE_ARMS.items()},
    ),
    "cont_occluder": (
        GATE / "dose/cont_occluder/schedule.csv",
        {a: RUNS / "dose_cont_occluder" / n / "seed0" for a, n in S.DOSE_ARMS.items()},
    ),
}
LABEL = {
    "clean": "clean",
    "transient": "mixed episodes",
    "cont_soiling": "continuous soiling",
    "cont_occluder": "permanent occluder",
}
ARM = {"static": "static", "ungated": "policy", "gated": "policy, gated by $E$"}


def main() -> None:
    base = S.frames("folsom")[["timestamp", "sp"]]
    rows, cons = [], []
    for stream, (sched_path, runs) in STREAMS.items():
        sched = pd.read_csv(sched_path, dtype={"timestamp": str}) if sched_path else None
        P = {}
        for arm, path in runs.items():
            if not (path / "predictions.csv").exists():
                print(f"{stream}/{arm}: missing ({path})")
                continue
            p = pd.read_csv(path / "predictions.csv", dtype={"timestamp": str, "day": str}).merge(base, on="timestamp")
            p["fault"] = (
                p.timestamp.map(dict(zip(sched.timestamp, sched.kind, strict=True))).fillna("none")
                if sched is not None
                else "none"
            )
            p["gated_frame"] = p.get("gated", pd.Series(False, index=p.index)).astype(bool)
            P[arm] = p
            ff, fa = p[p.fault == "none"], p[p.fault != "none"]
            upd = p[p.adapted == True]  # noqa: E712
            rows.append(
                dict(
                    stream=stream,
                    arm=arm,
                    faulty_pct=100 * len(fa) / len(p),
                    mae_fault_free=S._mae(ff),
                    fs_fault_free=S._fs(ff),
                    mae_faulty=S._mae(fa) if len(fa) else np.nan,
                    fs_faulty=S._fs(fa) if len(fa) else np.nan,
                    updates=len(upd),
                    updates_on_faulty=int((upd.fault != "none").sum()),
                    withheld_pct=100 * float(p.gated_frame.mean()),
                    withheld_faulty_pct=100 * float(p.gated_frame[p.fault != "none"].mean()) if len(fa) else np.nan,
                )
            )
        for a, b, frames in (
            ("ungated", "static", "faulty"),
            ("gated", "ungated", "faulty"),
            ("gated", "ungated", "fault-free"),
            ("ungated", "static", "fault-free"),
        ):
            if a not in P or b not in P:
                continue
            ma = (P[a].fault != "none") if frames == "faulty" else (P[a].fault == "none")
            mb = (P[b].fault != "none") if frames == "faulty" else (P[b].fault == "none")
            if not ma.any():
                continue
            c = contrast(S._daily_on(P[a][ma]), S._daily_on(P[b][mb]), tag=f"streams-{stream}-{a}-{b}-{frames}")
            cons.append(
                dict(
                    stream=stream,
                    a=a,
                    b=b,
                    frames=frames,
                    **{k: c[k] for k in ("dfs", "dfs_lo", "dfs_hi", "rel", "rel_lo", "rel_hi", "p", "n_days")},
                )
            )
    t, c = pd.DataFrame(rows), pd.DataFrame(cons)
    t.to_csv(TABLES / "ch4_admissibility_streams.csv", index=False)
    c.to_csv(TABLES / "ch4_admissibility_streams_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(c.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # LaTeX body: one block per stream
    lines = []
    for stream in STREAMS:
        g = t[t.stream == stream]
        if g.empty:
            continue
        share = g.faulty_pct.iloc[0]
        head = LABEL[stream] + ("" if stream == "clean" else f" ({share:.1f}\\,\\% of frames)")
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\itshape {head}}}\\\\")
        for arm in ("static", "ungated", "gated"):
            r = g[g.arm == arm]
            if r.empty:
                continue
            r = r.iloc[0]
            gv = c[(c.stream == stream) & (c.a == "gated") & (c.b == "ungated") & (c.frames == "faulty")]
            us = c[(c.stream == stream) & (c.a == "ungated") & (c.b == "static") & (c.frames == "faulty")]
            d = ""
            if arm == "ungated" and not us.empty:
                d = f"{us.dfs.iloc[0]:+.2f} [{us.dfs_lo.iloc[0]:+.2f}, {us.dfs_hi.iloc[0]:+.2f}]"
            if arm == "gated" and not gv.empty:
                d = f"{gv.dfs.iloc[0]:+.2f} [{gv.dfs_lo.iloc[0]:+.2f}, {gv.dfs_hi.iloc[0]:+.2f}]"
            fa = "--" if np.isnan(r.mae_faulty) else f"{r.mae_faulty:.2f}"
            fsa = "--" if np.isnan(r.fs_faulty) else f"{r.fs_faulty:.2f}"
            wh = (
                "--"
                if arm != "gated"
                else (f"{r.withheld_faulty_pct:.0f}" if not np.isnan(r.withheld_faulty_pct) else f"{r.withheld_pct:.1f}")
            )
            uf = "--" if arm == "static" or stream == "clean" else f"{r.updates_on_faulty}"
            lines.append(
                f"\\quad {ARM[arm]} & {r.mae_fault_free:.2f} & {r.fs_fault_free:.2f} & {fa} & {fsa} & {r.updates} & {uf} & {wh} & {d}\\\\"
            )
    (TABLES / "ch4_streams.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TABLES / "ch4_streams.tex")


if __name__ == "__main__":
    main()
