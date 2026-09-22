"""S1-S11 acceptance checks for the Chapter 4 smoke benchmark (plan section 5.4).

Reads only results/ch4/smoke/ and the window manifest, and prints a markdown
pass/fail table followed by the supporting detail.

    python scripts/check_ch4_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "results" / "ch4" / "smoke"
LOGS = ROOT / "logs" / "ch4" / "smoke"
MANIFEST = yaml.safe_load((ROOT / "configs" / "ch4_smoke_window.yaml").read_text())

METHODS = ["baseline", "note_reg_all", "memo_reg_all", "fewshot_d1", "fewshot_d7", "sliding_window", "sm_rls", "conditional"]
# Poznan carries one verification run rather than the full method set (author
# decision 2026-09-22), so every per-site check reads this map instead of
# assuming each site has every method.
SITE_METHODS = {"folsom": METHODS, "karpicko": METHODS, "poznan": ["conditional"]}
SITES = list(SITE_METHODS)
COMPARISON_SITES = [s for s, ms in SITE_METHODS.items() if len(ms) > 1]
EXTRA = [
    ("karpicko", "conditional", "seed0_repeat"),
    ("karpicko", "memo_reg_all", "seed1"),
    ("karpicko", "regression_legacy", "seed0"),
]
ALL_RUNS = [(site, m, "seed0") for site, ms in SITE_METHODS.items() for m in ms] + EXTRA

# fewshot_d7 runs on the manifest's exception window, so it is excluded wherever
# a check compares populations against the 3-day smoke window.
WINDOW_RUNS = [(s, m, k) for s, m, k in ALL_RUNS if m != "fewshot_d7" and m != "regression_legacy"]

REQUIRED = {
    "metrics_daily.csv": ["date", "n_samples", "n_dropped_dark", "mae", "rmse", "wape", "smape", "sp_rmse"],
    "events.csv": [
        "date",
        "considered",
        "queue_flushed",
        "none",
        "head",
        "whole",
        "lit_none",
        "lit_head",
        "lit_whole",
        "dark_none",
        "dark_head",
        "dark_whole",
    ],
    "predictions.csv": ["timestamp", "day", "prediction", "target", "absolute_error", "ghi_clear", "lit"],
}
OTHER = ["config.yaml", "summary.json"]

results: list[tuple[str, str, str, str]] = []


def record(cid: str, check: str, status: str, detail: str) -> None:
    results.append((cid, check, status, detail))


def rundir(site: str, method: str, key: str = "seed0") -> Path:
    return SMOKE / site / method / key


def summary(site: str, method: str, key: str = "seed0") -> dict:
    return json.loads((rundir(site, method, key) / "summary.json").read_text())


def load(site: str, method: str, fname: str, key: str = "seed0") -> pd.DataFrame:
    return pd.read_csv(rundir(site, method, key) / fname, dtype={"date": str, "day": str})


def config(site: str, method: str, key: str = "seed0") -> dict:
    return yaml.safe_load((rundir(site, method, key) / "config.yaml").read_text())


def present() -> list[tuple[str, str, str]]:
    return [r for r in ALL_RUNS if (rundir(*r) / "summary.json").exists()]


def s1() -> None:
    missing = [f"{s}/{m}/{k}" for s, m, k in ALL_RUNS if not (rundir(s, m, k) / "summary.json").exists()]
    fail_log = LOGS / "failures.log"
    ftext = fail_log.read_text().strip() if fail_log.exists() else ""
    if missing:
        record("S1", "Completion", "FAIL", f"{len(missing)} of {len(ALL_RUNS)} runs missing:\n  " + "\n  ".join(missing))
    elif ftext:
        record("S1", "Completion", "FAIL", f"failures.log is non-empty:\n{ftext}")
    else:
        record("S1", "Completion", "PASS", f"all {len(ALL_RUNS)} runs produced summary.json; failures.log empty")


def s2() -> None:
    bad, worst = [], (0.0, "")
    for s, m, k in present():
        d = load(s, m, "metrics_daily.csv", k)
        if not np.isfinite(d[["mae", "rmse"]].to_numpy()).all():
            bad.append(f"{s}/{m}: non-finite MAE or RMSE")
        hi = d.mae.max() * 1000
        if hi > worst[0]:
            worst = (hi, f"{s}/{m}")
        if hi > 300:
            bad.append(f"{s}/{m}: daily MAE {hi:.1f} W/m^2 above 300")
    detail = f"worst daily MAE {worst[0]:.1f} W/m^2 ({worst[1]}); limit 300"
    record("S2", "Sanity", "FAIL" if bad else "PASS", "\n".join(bad) if bad else detail)


def s3() -> None:
    lines, ok = [], True
    a = load("karpicko", "conditional", "metrics_daily.csv", "seed0")
    b = load("karpicko", "conditional", "metrics_daily.csv", "seed0_repeat")
    same = a.drop(columns=["date"]).equals(b.drop(columns=["date"])) and list(a.date) == list(b.date)
    lines.append(f"conditional seed 0 twice: {'identical' if same else 'DIFFER'}")
    ok &= same
    try:
        m0 = summary("karpicko", "memo_reg_all", "seed0")["pooled"]["mae"]
        m1 = summary("karpicko", "memo_reg_all", "seed1")["pooled"]["mae"]
        differs = abs(m0 - m1) > 1e-12
        lines.append(f"MEMO seed 0 vs 1 pooled MAE: {m0:.8f} vs {m1:.8f} — {'differ' if differs else 'IDENTICAL'}")
        ok &= differs
    except FileNotFoundError:
        lines.append("MEMO seed 1 missing")
        ok = False
    record("S3", "Determinism", "PASS" if ok else "FAIL", "\n".join(lines))


def s4() -> None:
    bad, lines = [], []
    floor = MANIFEST["min_ghi_clear"]
    for s, m, k in present():
        if m == "regression_legacy":
            continue
        p = load(s, m, "predictions.csv", k)
        dark = int((p.ghi_clear < floor).sum())
        if dark:
            bad.append(f"{s}/{m}: {dark} scored samples below the filter")
    for site in SITES:
        exp = MANIFEST["coverage"][site]["samples"]
        got = {
            m: summary(site, m)["n_samples"]
            for s2_, m, k in WINDOW_RUNS
            if s2_ == site and (rundir(site, m, k) / "summary.json").exists()
        }
        agree = set(got.values())
        lines.append(f"{site}: manifest {exp}, runs {sorted(agree)}")
        if agree != {exp}:
            bad.append(f"{site}: run population {sorted(agree)} != manifest {exp}")
    record(
        "S4",
        "Daylight filter",
        "FAIL" if bad else "PASS",
        "\n".join(bad) if bad else "every run scores the manifest population\n" + "\n".join(lines),
    )


def s5() -> None:
    lines, bad = [], []
    for site in SITES:
        e = load(site, "conditional", "events.csv")
        tot = {a: int(e[a].sum()) for a in ("none", "head", "whole")}
        lines.append(f"{site} conditional: {tot}")
        if tot["whole"] == 0:
            bad.append(f"{site}: the full-model action never fired, so the smoke does not exercise it")
    for site in SITES:
        for m in ("memo_reg_all", "note_reg_all"):
            if not (rundir(site, m) / "summary.json").exists():
                continue
            declared = config(site, m).get("adapt_events")
            logged = int(load(site, m, "events.csv")["whole"].sum())
            lines.append(f"{site} {m}: adapt_events={declared} events.csv whole={logged}")
            if declared != logged:
                bad.append(f"{site}/{m}: adapt_events {declared} != events.csv {logged}")
    record(
        "S5",
        "Action accounting across methods",
        "FAIL" if bad else "PASS",
        ("\n".join(bad) + "\n\n" if bad else "") + "\n".join(lines),
    )


def s6() -> None:
    bad, sentinel = [], []
    for s, m, k in present():
        e = load(s, m, "events.csv", k)
        p = load(s, m, "predictions.csv", k)
        c = p[p.update_considered] if "update_considered" in p else p.iloc[0:0]
        for a in ("none", "head", "whole"):
            got = int(e[a].sum())
            exp = int((c.adaptation_scope == a).sum()) if "adaptation_scope" in c else 0
            if got != exp:
                bad.append(f"{s}/{m}: events {a}={got} but predictions {a}={exp}")
            if not (e[a] == e[f"lit_{a}"] + e[f"dark_{a}"]).all():
                bad.append(f"{s}/{m}: lit+dark != {a}")
        if "scope_changed_if_raw" in e:
            n = int(e["scope_changed_if_raw"].sum())
            sentinel.append(f"{s}/{m}: scope_changed_if_raw={n}")
            if n:
                bad.append(f"{s}/{m}: the clamp bound on {n} samples")
    record(
        "S6",
        "Action accounting",
        "FAIL" if bad else "PASS",
        "\n".join(bad) if bad else "events.csv reconciles with predictions.csv everywhere\n" + "\n".join(sentinel),
    )


def s7() -> None:
    lines, bad = [], []
    for site in COMPARISON_SITES:
        base = summary(site, "baseline")["pooled"]["mae"]
        e = load(site, "conditional", "events.csv")
        # bracket access throughout: the "head" action column shadows DataFrame.head
        updates = int(e["head"].sum() + e["whole"].sum())
        lines.append(f"{site}: static pooled MAE {base * 1000:.4f} W/m^2, conditional updates {updates}")
        if not np.isfinite(base):
            bad.append(f"{site}: static MAE not finite")
    record("S7", "Static reference", "FAIL" if bad else "PASS", "\n".join(bad) if bad else "\n".join(lines))


def s8() -> None:
    bad, lines = [], []
    for site in SITES:
        frames = {
            m: load(site, m, "metrics_daily.csv")
            for s_, m, k in WINDOW_RUNS
            if s_ == site and (rundir(site, m, k) / "summary.json").exists()
        }
        if len(frames) < 2:
            lines.append(f"{site}: single run, no cross-method comparison available")
            continue
        ref = None
        for m, d in frames.items():
            sp = d.set_index("date").sp_rmse
            if ref is None:
                ref, refname = sp, m
            elif not np.allclose(sp.reindex(ref.index), ref, equal_nan=True):
                bad.append(f"{site}: Smart Persistence differs between {refname} and {m}")
        if ref is not None:
            lines.append(
                f"{site}: sp_rmse identical across {len(frames)} methods, pooled {np.sqrt((ref**2).mean()) * 1000:.3f} W/m^2"
            )
    record("S8", "Smart Persistence", "FAIL" if bad else "PASS", "\n".join(bad) if bad else "\n".join(lines))


def s9() -> None:
    """Population regression against the published run — metrics cannot match (plan R13)."""
    legacy = ROOT / "results" / "karpicko" / "karpicko_paper_alg1_ogd_results.txt"
    run = rundir("karpicko", "regression_legacy")
    if not (run / "summary.json").exists() or not legacy.exists():
        record("S9", "Population regression", "FAIL", "regression run or published artifact missing")
        return
    pub = {}
    for line in legacy.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0].isdigit() and len(parts[0]) == 4:
            pub[parts[0]] = int(parts[-1])
    d = load("karpicko", "regression_legacy", "metrics_daily.csv")
    lines, bad = [], []
    for day, n in zip(d.date, d.n_samples, strict=True):
        p = pub.get(day)
        lines.append(f"  {day}: published {p}, rerun {n}")
        if p != n:
            bad.append(f"{day}: published {p} != rerun {n}")
    cfg = config("karpicko", "regression_legacy")
    lines.append(f"checkpoint {Path(cfg['checkpoint']).name}, min_ghi_clear {cfg['min_ghi_clear']}")
    lines.append(
        "Metrics are deliberately not compared: the published run adapted from 0101, and its "
        "high-error action was a head update (plan R13)."
    )
    record(
        "S9",
        "Population regression",
        "FAIL" if bad else "PASS",
        ("\n".join(bad) + "\n" if bad else "per-day sample counts reproduce the published run\n") + "\n".join(lines),
    )


def s10() -> None:
    bad = []
    for s, m, k in present():
        d = rundir(s, m, k)
        for fname, cols in REQUIRED.items():
            if not (d / fname).exists():
                bad.append(f"{s}/{m}/{k}: {fname} missing")
                continue
            have = set(pd.read_csv(d / fname, nrows=1).columns)
            miss = [c for c in cols if c not in have]
            if miss:
                bad.append(f"{s}/{m}/{k}: {fname} missing columns {miss}")
        for fname in OTHER:
            if not (d / fname).exists():
                bad.append(f"{s}/{m}/{k}: {fname} missing")
    n = len(present())
    record(
        "S10",
        "Logs",
        "FAIL" if bad else "PASS",
        "\n".join(bad) if bad else f"all {len(REQUIRED) + len(OTHER)} files and every required column present in {n} runs",
    )


# Methods that fit once, offline, before the stream.  Their cost does not scale
# with campaign length the way a per-sample online method's does, so a flat
# ms/sample extrapolation would multiply a fixed fit by the whole campaign.
OFFLINE_FIT = ("fewshot_d1", "fewshot_d7")
CAMPAIGN_DAYS = 300
CAMPAIGN_SAMPLES = CAMPAIGN_DAYS * 800


def s11() -> None:
    rows = []
    for site, method, key in present():
        d = summary(site, method, key)
        rows.append(
            (
                f"{site}/{method}/{key}",
                site,
                method,
                d["wall_time_s"],
                d["n_samples"],
                d["wall_time_s"] / d["n_samples"] * 1000,
                d["peak_rss_mb"],
            )
        )
    rows.sort(key=lambda r: -r[3])

    # the frozen run is the per-sample inference rate an offline-fit method pays
    # once its fitting is done
    infer_rate = {}
    for site in COMPARISON_SITES:
        d = summary(site, "baseline")
        infer_rate[site] = d["wall_time_s"] / d["n_samples"]

    lines = [
        f"total wall time {sum(r[3] for r in rows) / 60:.1f} min over {len(rows)} runs",
        "",
        "| run | wall (s) | samples | ms/sample | peak RSS (MB) |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {n} | {w:.0f} | {ns:d} | {ms:.1f} | {rss:.0f} |" for n, _s, _m, w, ns, ms, rss in rows]

    lines += [
        "",
        f"Extrapolated to one {CAMPAIGN_DAYS}-day site (~800 lit samples/day):",
        "",
        "| configuration | model | extrapolated |",
        "|---|---|---|",
    ]
    worst = (0.0, "")
    for name, site, method, wall, ns, ms, _rss in rows:
        if method in OFFLINE_FIT and site in infer_rate:
            rate = infer_rate[site]
            fixed = max(wall - ns * rate, 0.0)  # everything that is not inference
            hours = (fixed + CAMPAIGN_SAMPLES * rate) / 3600
            model = f"fixed fit {fixed / 60:.0f} min + inference at {rate * 1000:.1f} ms/sample"
        else:
            hours = ms / 1000 * CAMPAIGN_SAMPLES / 3600
            model = f"per-sample at {ms:.1f} ms"
        lines.append(f"| {name} | {model} | {hours:.1f} h |")
        if hours > worst[0]:
            worst = (hours, name)

    lines += ["", f"worst configuration: {worst[1]} at {worst[0]:.1f} h per site"]
    record("S11", "Runtime", "PASS" if worst[0] < 4 else "WARN", "\n".join(lines))


def main() -> None:
    for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11):
        try:
            fn()
        except Exception as exc:  # a broken check must not hide the other ten
            record(fn.__name__.upper(), "", "ERROR", f"{type(exc).__name__}: {exc}")

    print("| ID | Check | Result | Summary |")
    print("|---|---|---|---|")
    for cid, check, status, detail in results:
        first = detail.splitlines()[0] if detail else ""
        print(f"| {cid} | {check} | **{status}** | {first[:160]} |")
    print()
    for cid, check, status, detail in results:
        print(f"### {cid} — {check}: {status}\n\n{detail}\n")


if __name__ == "__main__":
    main()
