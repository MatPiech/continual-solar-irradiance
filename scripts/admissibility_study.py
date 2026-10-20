"""Chapter 4 input-admissibility study: every table the section reports.

Subcommands, in the order the reproduction guide runs them
(docs/CH4_ADMISSIBILITY.md):

    schedule     synthetic-fault schedule over the Folsom 2016 stream
    reliability  the anomaly score as a label-free forecast-error signal
    detection    detection of synthetic faults: Isolation Forest vs baselines
    crosssite    flag rate at the deployment sites under the Folsom-fitted gate
    adaptation   the policy on the corrupted stream, with and without the gate

Each writes CSVs under tables/ that scripts/plot_admissibility.py turns
into figures; nothing is edited by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

from cirr.data.loading import load_dataset
from cirr.evaluation.admissibility import (
    _MEAN,
    _STD,
    EDGE_KINDS,
    KINDS,
    MONITOR_KINDS,
    PERSISTENT,
    SHORT_TERM,
    LocalGate,
    anomaly_score,
    build_schedule,
    corrupt,
    embed,
    load_gate,
    local_features,
    nominal_split,
    stamp,
)
from cirr.evaluation.campaign import PERIOD_SCALE, SOURCE_CHECKPOINT, load_campaign_days
from cirr.evaluation.runner import load_eval_model
from cirr.evaluation.smart_persistence import predict_periods
from cirr.fl.analysis import contrast
from cirr.physics.clear_sky import resolve_clear_sky_value

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "results" / "adaptation" / "gate"
RUNS = GATE / "runs"
TABLES = ROOT / "tables"
W = 1000.0
SITES = ("folsom", "karpicko", "poznan")
PROBE_KINDS = KINDS + EDGE_KINDS + MONITOR_KINDS  # the synthetic faults every detector is probed with
# detection-table row per split of the local detector (docs/CH4_ADMISSIBILITY.md §5 and §6)
LOCAL_ROWS = {"all": "local (layer2, per cell)", "interior": "local, interior cells", "edge": "local, edge cells"}
# the policy each site's forecasts come from: head-only, the stable policy of Section 4.3
POLICY = {s: ROOT / "results" / "adaptation" / "ablation" / s / "conditional_headonly" / "seed0" for s in SITES}
STATIC = {s: ROOT / "results" / "adaptation" / "tier1" / s / "baseline" / "seed0" for s in SITES}
COVERAGE = list(range(50, 101))


def frames(site: str) -> pd.DataFrame:
    """Per-frame, label-free signals available at prediction time, plus Smart Persistence."""
    data = load_campaign_days(ROOT / "data" / f"{site}_prepared", min_ghi_clear=0.001)
    rows = []
    for day in data.days:
        per = [data.dataset.periods[j] for j in data.day_map[day]]
        sp, _, _ = predict_periods(per, data.lookup, period_scale=PERIOD_SCALE)
        for q, v in zip(per, sp, strict=True):
            h = np.array([e["irradiance"] for e in q["history"]], float)
            cs = resolve_clear_sky_value(data.lookup, stamp(q)) or np.nan
            rows.append((stamp(q), v, h.std(), abs(1 - min(max(h[-1] / W / cs, 0), 1.2)) if cs else np.nan))
    f = pd.DataFrame(rows, columns=["timestamp", "sp", "hist_std", "kc_dev"]).drop_duplicates("timestamp")
    # the Polish streams hold occasional duplicate frames within one minute; one score per
    # timestamp keeps the merge from multiplying those rows
    sc = pd.read_csv(GATE / site / "scores.csv", dtype={"timestamp": str}).drop_duplicates("timestamp")
    pr = pd.read_csv(POLICY[site] / "predictions.csv", dtype={"timestamp": str, "day": str})
    x = pr[["timestamp", "day", "prediction", "target"]].merge(sc, on="timestamp").merge(f, on="timestamp")
    x["err"] = (x.prediction - x.target).abs() * W
    x["err_sp"] = (x.sp - x.target).abs() * W
    # label-free combination: mean of the two signals' ranks
    x["combined"] = (x.score.rank() + x.hist_std.rank()) / 2
    return x


def risk_coverage(x: pd.DataFrame, signal: str) -> list[float]:
    """MAE of the forecasts kept when the (100 - c) % with the highest ``signal`` are withheld."""
    order = x.sort_values(signal).err.to_numpy()
    return [float(order[: max(1, round(len(order) * c / 100))].mean()) for c in COVERAGE]


def rank_r2(y: np.ndarray, cols: list[np.ndarray]) -> float:
    """Share of the rank variance of ``y`` explained by the ranks of ``cols`` (least squares)."""

    def ranks(v: np.ndarray) -> np.ndarray:
        return pd.Series(v).rank().to_numpy()

    design = np.column_stack([np.ones(len(y))] + [ranks(c) for c in cols])
    yr = ranks(y)
    beta = np.linalg.lstsq(design, yr, rcond=None)[0]
    return float(1 - ((yr - design @ beta) ** 2).sum() / ((yr - yr.mean()) ** 2).sum())


@click.group()
def cli() -> None:
    TABLES.mkdir(exist_ok=True)


@cli.command()
@click.option("--stream", type=click.Path(exists=True, path_type=Path), default="data/folsom_prepared", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--out", type=click.Path(path_type=Path), default=str(GATE / "folsom_corrupted" / "schedule.csv"), show_default=True
)
def schedule(stream: Path, seed: int, out: Path) -> None:
    """Fault episodes over the daylight stream: soiling (days), droplets (hours), occlusion (minutes)."""
    data = load_campaign_days(stream, min_ghi_clear=0.001)
    ts = [stamp(data.dataset.periods[i]) for d in data.days for i in data.day_map[d]]
    sched = build_schedule(ts, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    sched.to_csv(out, index=False)
    summary = sched.groupby("kind").agg(frames=("timestamp", "size"), episodes=("episode", "nunique"))
    summary["pct_of_stream"] = 100 * summary.frames / len(ts)
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"total: {len(sched)} of {len(ts)} frames ({100 * len(sched) / len(ts):.2f} %)")


@cli.command()
def reliability() -> None:
    """The anomaly score as a label-free forecast-error signal, at every site."""
    sig, curves, groups = [], [], []
    for site in SITES:
        if not (GATE / site / "scores.csv").exists():
            print(f"{site}: no scores yet, skipped")
            continue
        x = frames(site)
        for k, lab in [
            ("score", "anomaly score"),
            ("hist_std", "irradiance-history std"),
            ("kc_dev", "|1 - clear-sky index|"),
            ("combined", "score + history std"),
        ]:
            sig.append({"site": site, "signal": lab, "spearman": float(spearmanr(x[k], x.err, nan_policy="omit").statistic)})
        sig.append(
            {
                "site": site,
                "signal": "rank R^2: history + clear-sky index",
                "spearman": rank_r2(x.err.to_numpy(), [x.hist_std.to_numpy(), x.kc_dev.fillna(0).to_numpy()]),
            }
        )
        sig.append(
            {
                "site": site,
                "signal": "rank R^2: + anomaly score",
                "spearman": rank_r2(
                    x.err.to_numpy(), [x.hist_std.to_numpy(), x.kc_dev.fillna(0).to_numpy(), x.score.to_numpy()]
                ),
            }
        )
        for k in ("score", "hist_std", "combined", "err"):
            for c, m in zip(COVERAGE, risk_coverage(x, k), strict=True):
                curves.append({"site": site, "signal": k, "coverage": c, "mae": m})
        for flag, g in x.groupby("flagged"):
            groups.append(
                {
                    "site": site,
                    "group": "flagged" if flag else "admitted",
                    "n": len(g),
                    "share_pct": 100 * len(g) / len(x),
                    "model_mae": g.err.mean(),
                    "smart_persistence_mae": g.err_sp.mean(),
                }
            )
        hybrid = np.where(x.flagged, x.err_sp, x.err).mean()
        groups.append(
            {
                "site": site,
                "group": "hybrid: Smart Persistence on flagged",
                "n": len(x),
                "share_pct": 100.0,
                "model_mae": hybrid,
                "smart_persistence_mae": np.nan,
            }
        )
    pd.DataFrame(sig).to_csv(TABLES / "ch4_admissibility_signals.csv", index=False)
    pd.DataFrame(curves).to_csv(TABLES / "ch4_admissibility_risk_coverage.csv", index=False)
    pd.DataFrame(groups).to_csv(TABLES / "ch4_admissibility_flagged.csv", index=False)
    print(pd.DataFrame(sig).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(groups).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


class _Faulted(torch.utils.data.Dataset):
    """``(frame index, kind or None, episode)`` triples of a stream, corrupted on the fly."""

    def __init__(self, base, items):
        self.base, self.items = base, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        j, kind, ep = self.items[i]
        image, history, clear, target, name = self.base[j]
        if kind is not None:
            image = corrupt(image, kind, ep, str(name)[:15])
        return image, history, clear, target, name


def _embed_items(model, base, items, device, workers):
    return embed(model, _Faulted(base, items), list(range(len(items))), device, workers)


def _pixel_stats(dataset: torch.utils.data.Dataset, workers: int) -> np.ndarray:
    """Model-free image statistics per frame: dark-pixel fraction and blur.

    Section 4.2 notes that saturation or blur proxies are cheaper than any learned
    score but only see integrity violations visible in the raw frame.  They are the
    baseline that separates "the fault is invisible" from "the embedding ignores it".
    """
    lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    out = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=workers)
    for image, *_ in tqdm(loader, desc="pixels", unit="batch"):
        fov = image.abs().sum(1, keepdim=True) > 0
        pix = (image * _STD + _MEAN).clamp(0, 1)
        luma = 0.299 * pix[:, 0:1] + 0.587 * pix[:, 1:2] + 0.114 * pix[:, 2:3]
        dark = ((luma < 0.15) & fov).flatten(1).sum(1) / fov.flatten(1).sum(1)
        edges = torch.nn.functional.conv2d(luma, lap, padding=1)
        inner = torch.nn.functional.max_pool2d((~fov).float(), 5, 1, 2) == 0  # away from the crop edge
        sharp = torch.stack([e[m].var() for e, m in zip(edges, inner, strict=True)])
        out.append(torch.stack([dark, -sharp], 1).numpy())  # both: larger = more suspicious
    return np.concatenate(out)


@cli.command()
@click.option("--every", type=int, default=80, show_default=True, help="Probe every n-th frame of the 2016 stream.")
@click.option("--knn", type=int, default=10, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--workers", type=int, default=8, show_default=True)
def detection(every: int, knn: int, device: str, workers: int) -> None:
    """Synthetic-fault detection: Isolation Forest against Mahalanobis and kNN on the same embeddings."""
    forest, meta = load_gate(GATE / "folsom")
    model = load_eval_model(Path(SOURCE_CHECKPOINT), device)
    cache = GATE / "folsom" / "embeddings.npz"
    if cache.exists():
        z = np.load(cache)
        x_fit, x_cal = z["fit"].astype(np.float32), z["cal"].astype(np.float32)
    else:
        train = load_dataset(ROOT / "data" / "folsom_prepared", split="training")
        f_idx, c_idx = nominal_split(train.periods, set(meta["fit_years"]), meta["stride"], 0.2, meta["seed"])
        x_fit, x_cal = embed(model, train, f_idx, device, workers), embed(model, train, c_idx, device, workers)
        np.savez_compressed(cache, fit=x_fit.astype(np.float16), cal=x_cal.astype(np.float16))

    data = load_campaign_days(ROOT / "data" / "folsom_prepared", min_ghi_clear=0.001)
    probe = [i for d in data.days for i in data.day_map[d]][::every]
    items = [(j, None, 0) for j in probe] + [(j, k, n) for k in PROBE_KINDS for n, j in enumerate(probe)]
    x_probe = _embed_items(model, data.dataset, items, device, workers)
    kind = np.array(["clean"] * len(probe) + [k for k in PROBE_KINDS for _ in probe])

    maha = LedoitWolf().fit(x_fit)
    ref = torch.from_numpy(x_fit[np.random.default_rng(0).choice(len(x_fit), min(20000, len(x_fit)), replace=False)]).to(
        device
    )

    def knn_score(x):
        out = []
        for chunk in torch.from_numpy(x).split(2048):
            d = torch.cdist(chunk.to(device), ref)
            out.append(d.topk(knn, largest=False).values.mean(1).cpu().numpy())
        return np.concatenate(out)

    detectors = {
        "Isolation Forest": lambda x: anomaly_score(forest, x),
        "Mahalanobis": maha.mahalanobis,
        f"kNN (k={knn})": knn_score,
    }
    # model-free baselines on the same frames, calibrated the same way
    train = load_dataset(ROOT / "data" / "folsom_prepared", split="training")
    _, c_idx = nominal_split(train.periods, set(meta["fit_years"]), meta["stride"], 0.2, meta["seed"])
    px_cal = _pixel_stats(torch.utils.data.Subset(train, c_idx), workers)
    px_probe = _pixel_stats(_Faulted(data.dataset, items), workers)
    pixel = {"dark-pixel fraction": 0, "blur (low Laplacian variance)": 1}
    rows = []
    for name, fn in detectors.items():
        thr = float(np.percentile(fn(x_cal), meta["percentile"]))
        s = fn(x_probe)
        clean = s[kind == "clean"]
        rows.append(
            {
                "detector": name,
                "fault": "none (clean 2016)",
                "n": int(clean.size),
                "flagged_pct": 100 * float((clean > thr).mean()),
                "auroc": np.nan,
            }
        )
        for k in PROBE_KINDS:
            f = s[kind == k]
            rows.append(
                {
                    "detector": name,
                    "fault": k,
                    "n": int(f.size),
                    "flagged_pct": 100 * float((f > thr).mean()),
                    "auroc": float(roc_auc_score(np.r_[np.zeros(clean.size), np.ones(f.size)], np.r_[clean, f])),
                }
            )
    for name, col in pixel.items():
        thr = float(np.percentile(px_cal[:, col], meta["percentile"]))
        sc = px_probe[:, col]
        clean = sc[kind == "clean"]
        rows.append(
            {
                "detector": name,
                "fault": "none (clean 2016)",
                "n": int(clean.size),
                "flagged_pct": 100 * float((clean > thr).mean()),
                "auroc": np.nan,
            }
        )
        for k in PROBE_KINDS:
            f = sc[kind == k]
            rows.append(
                {
                    "detector": name,
                    "fault": k,
                    "n": int(f.size),
                    "flagged_pct": 100 * float((f > thr).mean()),
                    "auroc": float(roc_auc_score(np.r_[np.zeros(clean.size), np.ones(f.size)], np.r_[clean, f])),
                }
            )
    local = GATE / "local" / "local_gate.npz"
    if local.exists():  # the pre-registered maintenance detector (§5) and its spatial split (§6)
        gate = LocalGate.load(local)
        z = gate.normalised(
            local_features(model, _Faulted(data.dataset, items), list(range(len(items))), gate.cells, device, workers), device
        )
        s = gate.split(z)
        for split, name in LOCAL_ROWS.items():
            thr = gate.threshold if split == "all" else gate.thresholds.get(split)
            if thr is None:  # a §5 gate file has no split thresholds
                continue
            ls = s[split]
            clean = ls[kind == "clean"]
            rows.append(
                {
                    "detector": name,
                    "fault": "none (clean 2016)",
                    "n": int(clean.size),
                    "flagged_pct": 100 * float((clean > thr).mean()),
                    "auroc": np.nan,
                }
            )
            for k in PROBE_KINDS:
                f = ls[kind == k]
                rows.append(
                    {
                        "detector": name,
                        "fault": k,
                        "n": int(f.size),
                        "flagged_pct": 100 * float((f > thr).mean()),
                        "auroc": float(roc_auc_score(np.r_[np.zeros(clean.size), np.ones(f.size)], np.r_[clean, f])),
                    }
                )
        mon = GATE / "monitor" / "monitor.json"
        if mon.exists():  # §7: global median, core max, and the exclusion flag they make
            thr = json.loads(mon.read_text())["thresholds"]
            m = gate.monitor(z)
            flags = {
                "local, global median": (m["global"], m["global"] > thr["global"]),
                "local, core max": (m["core"], m["core"] > thr["core"]),
                "exclusion flag (global or core)": (
                    np.maximum(m["global"] / thr["global"], m["core"] / thr["core"]),
                    (m["global"] > thr["global"]) | (m["core"] > thr["core"]),
                ),
            }
            for name, (score, flag) in flags.items():
                clean = score[kind == "clean"]
                rows.append(
                    {
                        "detector": name,
                        "fault": "none (clean 2016)",
                        "n": int(clean.size),
                        "flagged_pct": 100 * float(flag[kind == "clean"].mean()),
                        "auroc": np.nan,
                    }
                )
                for k in PROBE_KINDS:
                    f = score[kind == k]
                    rows.append(
                        {
                            "detector": name,
                            "fault": k,
                            "n": int(f.size),
                            "flagged_pct": 100 * float(flag[kind == k].mean()),
                            "auroc": float(roc_auc_score(np.r_[np.zeros(clean.size), np.ones(f.size)], np.r_[clean, f])),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "ch4_admissibility_detection.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@cli.command()
def crosssite() -> None:
    """Flag rate at each site under the Folsom-fitted gate, and what it says about the frozen model."""
    rows, monthly, daily = [], [], []
    for site in SITES:
        f = GATE / site / "scores.csv"
        if not f.exists():
            print(f"{site}: no scores yet, skipped")
            continue
        sc = pd.read_csv(f, dtype={"timestamp": str})
        sc["day"], sc["month"] = sc.timestamp.str[4:8], sc.timestamp.str[4:6]
        st = pd.read_csv(STATIC[site] / "metrics_daily.csv", dtype={"date": str})
        po = pd.read_csv(POLICY[site] / "metrics_daily.csv", dtype={"date": str})
        d = (
            sc.groupby("day")
            .flagged.mean()
            .mul(100)
            .rename("flag_pct")
            .reset_index()
            .merge(
                st[["date", "mae", "sp_rmse", "rmse"]].rename(
                    columns={"date": "day", "mae": "static_mae", "rmse": "static_rmse"}
                ),
                on="day",
            )
            .merge(po[["date", "mae"]].rename(columns={"date": "day", "mae": "policy_mae"}), on="day")
        )
        d["static_mae"] *= W
        d["policy_mae"] *= W
        d["adaptation_gain"] = d.static_mae - d.policy_mae
        d["site"] = site
        daily.append(d)
        rows.append(
            {
                "site": site,
                "frames": len(sc),
                "flag_rate_pct": 100 * sc.flagged.mean(),
                "median_score": sc.score.median(),
                "spearman_daily_flag_vs_static_mae": float(spearmanr(d.flag_pct, d.static_mae).statistic),
                "spearman_daily_flag_vs_adaptation_gain": float(spearmanr(d.flag_pct, d.adaptation_gain).statistic),
            }
        )
        for m, g in sc.groupby("month"):
            monthly.append({"site": site, "month": m, "flag_rate_pct": 100 * g.flagged.mean()})
    pd.DataFrame(rows).to_csv(TABLES / "ch4_admissibility_crosssite.csv", index=False)
    pd.DataFrame(monthly).to_csv(TABLES / "ch4_admissibility_crosssite_monthly.csv", index=False)
    if daily:
        pd.concat(daily).to_csv(TABLES / "ch4_admissibility_crosssite_daily.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _daily_on(x: pd.DataFrame) -> pd.DataFrame:
    """Per-day metrics over a frame subset, in the shape cirr.fl.analysis.contrast expects."""
    g = x.groupby("day")
    out = (
        pd.DataFrame(
            {
                "n_samples": g.size(),
                "mae": g.apply(lambda d: (d.prediction - d.target).abs().mean(), include_groups=False),
                "mse": g.apply(lambda d: ((d.prediction - d.target) ** 2).mean(), include_groups=False),
                "sp_mse": g.apply(lambda d: ((d.sp - d.target) ** 2).mean(), include_groups=False),
            }
        )
        .reset_index()
        .rename(columns={"day": "date"})
    )
    return out


@cli.command()
def adaptation() -> None:
    """The policy on the corrupted stream: does gating out faulty frames protect what it learns?"""
    sched = pd.read_csv(GATE / "folsom_corrupted" / "schedule.csv", dtype={"timestamp": str})
    faulty = dict(zip(sched.timestamp, sched.kind, strict=True))
    base = frames("folsom")[["timestamp", "sp"]]
    runs = {
        "head only": ("headonly", "headonly_gated", POLICY["folsom"]),
        "layer3+4, enc. LR 1e-6": (
            "last2_lowlr",
            "last2_lowlr_gated",
            ROOT / "results" / "adaptation" / "ablation" / "folsom" / "conditional_last2_lowlr" / "seed0",
        ),
    }
    rows, contrasts = [], []
    for policy, (ungated, gated, clean_ref) in runs.items():
        arms = {
            "clean stream": clean_ref,
            "corrupted, no gate": RUNS / "folsom_corrupted" / ungated / "seed0",
            "corrupted, gated": RUNS / "folsom_corrupted" / gated / "seed0",
        }
        if not all((a / "summary.json").exists() for a in arms.values()):
            print(f"{policy}: runs incomplete, skipped")
            continue
        table = {}
        for arm, path in arms.items():
            p = pd.read_csv(path / "predictions.csv", dtype={"timestamp": str, "day": str}).merge(base, on="timestamp")
            p["fault"] = p.timestamp.map(faulty).fillna("none")
            upd = p[p.get("adapted", False) == True]  # noqa: E712 -- pandas column
            clean = p[p.fault == "none"]
            table[arm] = clean
            rows.append(
                {
                    "policy": policy,
                    "arm": arm,
                    "mae_all": W * (p.prediction - p.target).abs().mean(),
                    "mae_clean_frames": W * (clean.prediction - clean.target).abs().mean(),
                    "mae_faulty_frames": W * (p[p.fault != "none"].prediction - p[p.fault != "none"].target).abs().mean(),
                    "updates": len(upd),
                    "updates_on_faulty_frames": int((upd.fault != "none").sum()),
                    "gated": int(p.get("gated", pd.Series(False, index=p.index)).sum()),
                }
            )
        # the effect that matters: did learning from faulty frames damage forecasts on clean frames?
        for a, b in [("corrupted, no gate", "clean stream"), ("corrupted, gated", "corrupted, no gate")]:
            c = contrast(_daily_on(table[a]), _daily_on(table[b]), tag=f"adm-{policy}-{a}-{b}")
            contrasts.append(
                {
                    "policy": policy,
                    "comparison": f"{a} vs {b}",
                    "frames": "clean frames only",
                    "rel_mae_pct": c["rel"],
                    "ci90_lo": c["rel_lo"],
                    "ci90_hi": c["rel_hi"],
                    "dfs": c["dfs"],
                    "p": c["p"],
                }
            )
    pd.DataFrame(rows).to_csv(TABLES / "ch4_admissibility_adaptation.csv", index=False)
    pd.DataFrame(contrasts).to_csv(TABLES / "ch4_admissibility_adaptation_contrasts.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(pd.DataFrame(contrasts).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _split_columns(split: str) -> tuple[str, str, str, str | None]:
    """(score, cell, flagged, alert) columns of scores_*.csv for a split; the edge split has no alert."""
    if split == "all":
        return "score", "cell", "flagged", "alert"
    return split, f"{split}_cell", f"{split}_flagged", "interior_alert" if split == "interior" else None


def _maintenance_for(split: str, cor: pd.DataFrame, cln: pd.DataFrame, det: pd.DataFrame, x: pd.DataFrame) -> None:
    """Rates, episodes, clean-stream summary, top frames and (where an alert exists) the verdict, for one split."""
    sc, ce, fl, al = _split_columns(split)
    sfx = "" if split == "all" else f"_{split}"
    agg = {"frames": ("timestamp", "size"), "flagged_pct": (fl, "mean")}
    if al:
        agg["alert_pct"] = (al, "mean")
    rates = cor.groupby("kind").agg(**agg)
    rates[[c for c in rates if c.endswith("_pct")]] *= 100
    rates = rates.reset_index()
    rates.to_csv(TABLES / f"ch4_admissibility_maintenance_rates{sfx}.csv", index=False)

    # every episode: is it flagged, does an alert fire, and how long after the fault appears?
    eps = []
    for (kind, ep), g in cor[cor.kind != "none"].groupby(["kind", "episode"]):
        g = g.sort_values("t")
        row = {
            "kind": kind,
            "episode": int(ep),
            "start": g.timestamp.iloc[0],
            "frames": len(g),
            "hours": (g.t.iloc[-1] - g.t.iloc[0]).total_seconds() / 3600,
            "frames_flagged_pct": 100 * g[fl].mean(),
        }
        if al:
            hit = g[g[al]]
            row.update(
                alerted=bool(len(hit)),
                first_alert_delay_min=(hit.t.iloc[0] - g.t.iloc[0]).total_seconds() / 60 if len(hit) else np.nan,
                frames_alerted_pct=100 * g[al].mean(),
            )
        eps.append(row)
    eps = pd.DataFrame(eps)
    eps.to_csv(TABLES / f"ch4_admissibility_maintenance_episodes{sfx}.csv", index=False)

    # the clean stream: how often would the operator be called out for nothing?
    clean_summary = {
        "split": split,
        "frames": len(cln),
        "flagged_pct": 100 * cln[fl].mean(),
        "spearman_local_vs_error": float(spearmanr(x[sc], x.err).statistic),
        "spearman_iforest_vs_error": float(spearmanr(x.if_score, x.err).statistic),
    }
    if al:
        alert_days = cln.groupby("day")[al].any()
        clean_summary.update(alert_pct=100 * cln[al].mean(), days=int(alert_days.size), days_with_alert=int(alert_days.sum()))
    (TABLES / f"ch4_admissibility_maintenance_clean{sfx}.json").write_text(json.dumps(clean_summary, indent=2))
    top = cln.sort_values(sc, ascending=False).drop_duplicates("day").head(12)
    top[["timestamp", sc, ce]].set_axis(["timestamp", "score", "cell"], axis=1).to_csv(
        TABLES / f"ch4_admissibility_local_top{sfx}.csv", index=False
    )

    print(f"--- {split} cells ---")
    print(rates.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if not al:
        return

    # the pre-registered verdict (§5 for "all", §6 for "interior")
    loc = det[det.detector == LOCAL_ROWS[split]].set_index("fault").flagged_pct
    dark = det[det.detector == "dark-pixel fraction"].set_index("fault").flagged_pct
    soil = eps[eps.kind == "soiling"]
    fault_free_alert = float(rates.set_index("kind").loc["none", "alert_pct"])
    c1 = all(loc[k] >= 50 and loc[k] > dark[k] for k in ("soiling", "occlusion"))
    c2 = bool((soil.first_alert_delay_min <= 120).all()) and fault_free_alert <= 1.0
    verdict = {
        "split": split,
        "criterion_1_frame_level": "PASS" if c1 else "FAIL",
        "soiling_flagged_pct": float(loc["soiling"]),
        "occlusion_flagged_pct": float(loc["occlusion"]),
        "dark_pixel_soiling_pct": float(dark["soiling"]),
        "dark_pixel_occlusion_pct": float(dark["occlusion"]),
        "criterion_2_maintenance": "PASS" if c2 else "FAIL",
        "soiling_episodes_alerted_within_2h": int((soil.first_alert_delay_min <= 120).sum()),
        "soiling_episodes": len(soil),
        "fault_free_alert_pct": fault_free_alert,
    }
    for k in EDGE_KINDS:  # the price of the split, reported beside the verdict
        if k in loc:
            verdict[f"{k}_flagged_pct"] = float(loc[k])
    (TABLES / f"ch4_admissibility_maintenance_verdict{sfx}.json").write_text(json.dumps(verdict, indent=2))
    print(soil.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    for k in ("droplets", "occlusion"):
        e = eps[eps.kind == k]
        print(
            f"{k}: {int(e.alerted.sum())} of {len(e)} episodes alerted; median frames flagged {e.frames_flagged_pct.median():.1f} %"
        )
    print(json.dumps(clean_summary, indent=2))
    print(json.dumps(verdict, indent=2))


@cli.command()
def maintenance() -> None:
    """The pre-registered maintenance alert (§5) and its spatial split (§6): episodes, false alerts, top frames."""
    d = GATE / "local"
    sched = pd.read_csv(GATE / "folsom_corrupted" / "schedule.csv", dtype={"timestamp": str})
    cor = pd.read_csv(d / "scores_corrupted.csv", dtype={"timestamp": str}).drop_duplicates("timestamp")
    cln = pd.read_csv(d / "scores_clean.csv", dtype={"timestamp": str}).drop_duplicates("timestamp")
    cor = cor.merge(sched, on="timestamp", how="left")
    cor["kind"] = cor.kind.fillna("none")
    cor["t"] = pd.to_datetime(cor.timestamp, format="%Y%m%d_%H%M%S")
    cln["day"] = cln.timestamp.str[:8]
    det = pd.read_csv(TABLES / "ch4_admissibility_detection.csv")
    x = frames("folsom")[["timestamp", "err", "score"]].rename(columns={"score": "if_score"}).merge(cln, on="timestamp")
    for split in LocalGate.SPLITS:
        if _split_columns(split)[0] in cln:  # §5 score files carry no split columns
            _maintenance_for(split, cor, cln, det, x)


@cli.command()
def geometry() -> None:
    """§6 geometric hypothesis: solar elevation and distance to the sun for every clean-stream flag, by split."""
    from cirr.physics.sun import SITES as SUN_SITES
    from cirr.physics.sun import project, solar_angles

    d = GATE / "local"
    gate = LocalGate.load(d / "local_gate.npz")
    rc = np.argwhere(gate.cells.numpy())  # scoring order -> (row, col) on the 16 x 16 grid
    cln = pd.read_csv(d / "scores_clean.csv", dtype={"timestamp": str}).drop_duplicates("timestamp").reset_index(drop=True)
    ang = solar_angles("folsom", cln.timestamp.tolist())
    zen = np.array([ang[t][0] for t in cln.timestamp])
    az = np.array([ang[t][1] for t in cln.timestamp])
    cfg = SUN_SITES["folsom"]
    u, v = project(zen, az, cfg["offset"], cfg["focal"], cfg["mirror"], cfg["cx"], cfg["cy"], cfg["k3"])
    sun_row, sun_col = np.where(zen < 90, v * 16, np.nan), np.where(zen < 90, u * 16, np.nan)
    elevation = 90 - zen
    hour = cln.timestamp.str[9:11].astype(int)

    rows, summary = [], {}
    base = {"frames": len(cln), "low_sun_pct": 100 * float((elevation < 20).mean())}
    for split in LocalGate.SPLITS:
        sc, ce, fl, _ = _split_columns(split)
        if fl not in cln:
            continue
        f = cln[cln[fl]]
        r, c = rc[f[ce].to_numpy()].T
        dist = np.sqrt((r + 0.5 - sun_row[f.index]) ** 2 + (c + 0.5 - sun_col[f.index]) ** 2)
        el = elevation[f.index]
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "timestamp": f.timestamp.to_numpy(),
                    "score": f[sc].to_numpy(),
                    "cell": f[ce].to_numpy(),
                    "on_edge": gate.edge.numpy()[f[ce].to_numpy()],
                    "hour": hour[f.index].to_numpy(),
                    "elevation_deg": el,
                    "sun_distance_cells": dist,
                }
            )
        )
        low, near = el < 20, dist <= 2
        summary[split] = {
            "flags": int(len(f)),
            "on_edge_pct": 100 * float(gate.edge.numpy()[f[ce].to_numpy()].mean()),
            "median_elevation_deg": float(np.median(el)),
            "low_sun_pct": 100 * float(low.mean()),
            "near_sun_pct": 100 * float(np.nan_to_num(near, nan=False).mean()),
            "illumination_pct": 100 * float((low | np.nan_to_num(near, nan=False)).mean()),
        }
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(TABLES / "ch4_admissibility_geometry.csv", index=False)
    summary["all_frames"] = base
    (TABLES / "ch4_admissibility_geometry.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


@cli.command()
def monitor() -> None:
    """§7: exclusion flag and maintenance alert against their pre-registered criteria, and E in series with the policy."""
    d = GATE / "monitor"
    meta = json.loads((d / "monitor.json").read_text())
    sched = pd.read_csv(d / "schedule.csv", dtype={"timestamp": str})
    cor = pd.read_csv(d / "scores_corrupted.csv", dtype={"timestamp": str}).drop_duplicates("timestamp")
    cln = pd.read_csv(d / "scores_clean.csv", dtype={"timestamp": str}).drop_duplicates("timestamp")
    cor = cor.merge(sched, on="timestamp", how="left")
    cor["kind"] = cor.kind.fillna("none")
    cor["t"] = pd.to_datetime(cor.timestamp, format="%Y%m%d_%H%M%S")
    cor["day"] = cor.timestamp.str[:8]
    cln["day"] = cln.timestamp.str[:8]

    rates = cor.groupby("kind").agg(
        frames=("timestamp", "size"), excluded_pct=("excluded", "mean"), maint_alert_pct=("maint_alert", "mean")
    )
    rates[["excluded_pct", "maint_alert_pct"]] *= 100
    rates = rates.reset_index()
    rates.to_csv(TABLES / "ch4_admissibility_monitor_rates.csv", index=False)

    # every episode: excluded share, and whether / when the maintenance alert fires
    eps = []
    for (kind, ep), g in cor[cor.kind != "none"].groupby(["kind", "episode"]):
        g = g.sort_values("t")
        onset = g.t.iloc[0]
        day = cor[(cor.day == g.day.iloc[0])]
        before = day[(day.t < onset) & (day.t >= onset - pd.Timedelta(minutes=30))]
        hit = g[g.maint_alert]
        eps.append(
            {
                "kind": kind,
                "episode": int(ep),
                "start": g.timestamp.iloc[0],
                "frames": len(g),
                "minutes": (g.t.iloc[-1] - onset).total_seconds() / 60 + 1,
                "excluded_pct": 100 * g.excluded.mean(),
                "alert_already_on": bool(before.maint_alert.any()),
                "alerted": bool(len(hit)),
                "first_alert_delay_min": (hit.t.iloc[0] - onset).total_seconds() / 60 if len(hit) else np.nan,
            }
        )
    eps = pd.DataFrame(eps)
    eps.to_csv(TABLES / "ch4_admissibility_monitor_episodes.csv", index=False)

    det = pd.read_csv(TABLES / "ch4_admissibility_detection.csv")
    ex = det[det.detector == "exclusion flag (global or core)"].set_index("fault").flagged_pct
    c1 = ex["banding"] >= 50 and ex["occlusion"] >= 50 and ex["none (clean 2016)"] <= 2
    limits = {"wet_dome": 90, "soiling": 120}
    detected = {
        k: bool(
            ((eps.kind == k) & ~eps.alert_already_on & (eps.first_alert_delay_min <= limits[k])).sum() == (eps.kind == k).sum()
        )
        for k in PERSISTENT
    }
    fault_free = float(rates.set_index("kind").loc["none", "maint_alert_pct"])
    short = eps[eps.kind.isin(SHORT_TERM)]
    short_alerted = 100 * float(short.alerted.mean())
    c2 = all(detected.values()) and fault_free <= 1.0 and short_alerted <= 10.0
    verdict = {
        "criterion_1_exclusion": "PASS" if c1 else "FAIL",
        "excluded_banding_pct": float(ex["banding"]),
        "excluded_occlusion_pct": float(ex["occlusion"]),
        "excluded_insect_pct": float(ex["insect"]),
        "excluded_clean_pct": float(ex["none (clean 2016)"]),
        "criterion_2_maintenance": "PASS" if c2 else "FAIL",
        "wet_dome_episodes_detected": int(
            ((eps.kind == "wet_dome") & ~eps.alert_already_on & (eps.first_alert_delay_min <= 90)).sum()
        ),
        "wet_dome_episodes": int((eps.kind == "wet_dome").sum()),
        "soiling_episodes_detected": int(
            ((eps.kind == "soiling") & ~eps.alert_already_on & (eps.first_alert_delay_min <= 120)).sum()
        ),
        "soiling_episodes": int((eps.kind == "soiling").sum()),
        "maint_alert_fault_free_pct": fault_free,
        "short_term_episodes_with_alert_pct": short_alerted,
        "clean_stream_exclusion_pct": float(100 * cln.excluded.mean()),
        "clean_stream_maint_alert_pct": float(100 * cln.maint_alert.mean()),
        "clean_days_with_maint_alert": int(cln.groupby("day").maint_alert.any().sum()),
        "thresholds": meta["thresholds"],
    }
    (TABLES / "ch4_admissibility_monitor_verdict.json").write_text(json.dumps(verdict, indent=2))

    for col, name in (("global", "global"), ("core", "core"), ("maint_stat", "maint")):
        top = cln.sort_values(col, ascending=False).drop_duplicates("day").head(12)
        cell = top.core_cell if col == "core" else pd.Series(-1, index=top.index)
        pd.DataFrame({"timestamp": top.timestamp, "score": top[col], "cell": cell}).to_csv(
            TABLES / f"ch4_admissibility_monitor_top_{name}.csv", index=False
        )

    # E in series with the damped policy
    faulty = dict(zip(sched.timestamp, sched.kind, strict=True))
    base = frames("folsom")[["timestamp", "sp"]]
    arms = {
        "clean stream, no exclusion": ROOT
        / "results"
        / "adaptation"
        / "ablation"
        / "folsom"
        / "conditional_last2_lowlr"
        / "seed0",
        "clean stream, exclusion": RUNS / "monitor_clean" / "last2_lowlr_excl" / "seed0",
        "corrupted, no exclusion": RUNS / "monitor_corrupted" / "last2_lowlr" / "seed0",
        "corrupted, exclusion": RUNS / "monitor_corrupted" / "last2_lowlr_excl" / "seed0",
    }
    rows, contrasts, table = [], [], {}
    for arm, path in arms.items():
        if not (path / "predictions.csv").exists():
            print(f"{arm}: run missing ({path}), skipped")
            continue
        p = pd.read_csv(path / "predictions.csv", dtype={"timestamp": str, "day": str}).merge(base, on="timestamp")
        p["fault"] = p.timestamp.map(faulty).fillna("none") if arm.startswith("corrupted") else "none"
        upd = p[p.adapted == True]  # noqa: E712 -- pandas column
        table[arm] = p[p.fault == "none"]
        rows.append(
            {
                "arm": arm,
                "mae_all": W * (p.prediction - p.target).abs().mean(),
                "mae_fault_free_frames": W * (table[arm].prediction - table[arm].target).abs().mean(),
                "updates": len(upd),
                "updates_on_faulty_frames": int((upd.fault != "none").sum()),
                "withheld": int(p.get("gated", pd.Series(False, index=p.index)).sum()),
            }
        )
    for a, b in [
        ("clean stream, exclusion", "clean stream, no exclusion"),
        ("corrupted, no exclusion", "clean stream, no exclusion"),
        ("corrupted, exclusion", "corrupted, no exclusion"),
    ]:
        if a in table and b in table:
            c = contrast(_daily_on(table[a]), _daily_on(table[b]), tag=f"mon-{a}-{b}")
            contrasts.append(dict(comparison=f"{a} vs {b}", frames="fault-free frames", **c))
    pd.DataFrame(rows).to_csv(TABLES / "ch4_admissibility_monitor_adaptation.csv", index=False)
    pd.DataFrame(contrasts).to_csv(TABLES / "ch4_admissibility_monitor_contrasts.csv", index=False)

    print(rates.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(eps[eps.kind.isin(PERSISTENT)].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(pd.DataFrame(contrasts).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(verdict, indent=2))


DOSE_STREAMS = ("r01", "r05", "r15", "cont_soiling", "cont_occluder")
DOSE_ARMS = {"static": "static", "ungated": "last2_lowlr", "gated": "last2_lowlr_excl"}
CONT_WINDOW = {"cont_soiling": ("20160701", "20160815"), "cont_occluder": ("20160901", "20161231")}


def _fs(p: pd.DataFrame) -> float:
    return 100 * (1 - np.sqrt(((p.prediction - p.target) ** 2).mean()) / np.sqrt(((p.sp - p.target) ** 2).mean()))


def _mae(p: pd.DataFrame) -> float:
    return W * (p.prediction - p.target).abs().mean()


@cli.command()
def dose() -> None:
    """§8: fault dose and duration — static, ungated and gated policy on each fault stream."""
    base = frames("folsom")[["timestamp", "sp"]]
    clean_runs = {
        "static": STATIC["folsom"],
        "ungated": ROOT / "results" / "adaptation" / "ablation" / "folsom" / "conditional_last2_lowlr" / "seed0",
        "gated": RUNS / "monitor_clean" / "last2_lowlr_excl" / "seed0",
    }

    def load(path, sched):
        p = pd.read_csv(path / "predictions.csv", dtype={"timestamp": str, "day": str}).merge(base, on="timestamp")
        p["fault"] = p.timestamp.map(dict(zip(sched.timestamp, sched.kind, strict=True))).fillna("none")
        p["gated_frame"] = p.get("gated", pd.Series(False, index=p.index)).astype(bool)
        return p

    rows, contrasts, weekly, alerts, phases = [], [], [], [], []
    for stream in DOSE_STREAMS:
        d = GATE / "dose" / stream
        if not (d / "schedule.csv").exists():
            print(f"{stream}: no schedule, skipped")
            continue
        sched = pd.read_csv(d / "schedule.csv", dtype={"timestamp": str})
        runs = {arm: RUNS / f"dose_{stream}" / name / "seed0" for arm, name in DOSE_ARMS.items()}
        missing = [a for a, r in runs.items() if not (r / "predictions.csv").exists()]
        if missing:
            print(f"{stream}: runs missing {missing}, skipped")
            continue
        preds = {arm: load(r, sched) for arm, r in runs.items()}
        clean_preds = {arm: load(r, sched) for arm, r in clean_runs.items() if (r / "predictions.csv").exists()}
        for arm, p in preds.items():
            ff, fa = p[p.fault == "none"], p[p.fault != "none"]
            upd = p[p.adapted == True]  # noqa: E712 -- pandas column
            rows.append(
                {
                    "stream": stream,
                    "arm": arm,
                    "faulty_pct": 100 * len(fa) / len(p),
                    "mae_fault_free": _mae(ff),
                    "fs_fault_free": _fs(ff),
                    "mae_faulty": _mae(fa),
                    "fs_faulty": _fs(fa),
                    "withheld": int(p.gated_frame.sum()),
                    "withheld_faulty_pct": 100 * float(p.gated_frame[p.fault != "none"].mean()) if len(fa) else np.nan,
                    "updates": len(upd),
                    "updates_on_faulty": int((upd.fault != "none").sum()),
                }
            )
        pairs = [
            ("ungated", "static", "faulty"),
            ("gated", "static", "faulty"),
            ("gated", "ungated", "faulty"),
            ("gated", "ungated", "fault-free"),
        ]
        for a, b, frames_ in pairs:
            pa, pb = preds[a], preds[b]
            m = (pa.fault != "none") if frames_ == "faulty" else (pa.fault == "none")
            mb = (pb.fault != "none") if frames_ == "faulty" else (pb.fault == "none")
            c = contrast(_daily_on(pa[m]), _daily_on(pb[mb]), tag=f"dose-{stream}-{a}-{b}-{frames_}")
            contrasts.append(
                dict(
                    stream=stream,
                    a=a,
                    b=b,
                    frames=frames_,
                    **{k: c[k] for k in ("dfs", "dfs_lo", "dfs_hi", "rel", "rel_lo", "rel_hi", "p", "n_days")},
                )
            )
        if "ungated" in clean_preds:  # contamination: ungated on this stream vs the clean-stream damped run, same frames
            pa, pb = preds["ungated"], clean_preds["ungated"]
            c = contrast(
                _daily_on(pa[pa.fault == "none"]), _daily_on(pb[pb.fault == "none"]), tag=f"dose-{stream}-contamination"
            )
            contrasts.append(
                dict(
                    stream=stream,
                    a="ungated",
                    b="clean-stream ungated",
                    frames="fault-free",
                    **{k: c[k] for k in ("dfs", "dfs_lo", "dfs_hi", "rel", "rel_lo", "rel_hi", "p", "n_days")},
                )
            )

        if stream in CONT_WINDOW:
            lo, hi = CONT_WINDOW[stream]
            sc = pd.read_csv(d / "scores_corrupted.csv", dtype={"timestamp": str}).drop_duplicates("timestamp")
            sc["day"] = sc.timestamp.str[:8]
            t = pd.to_datetime(sc.day, format="%Y%m%d")
            lo_t, hi_t = pd.to_datetime(lo), pd.to_datetime(hi)
            phase = np.where(
                t < lo_t - pd.Timedelta(days=14),
                "earlier",
                np.where(
                    t < lo_t,
                    "14 d before",
                    np.where(t <= hi_t, "during", np.where(t <= hi_t + pd.Timedelta(days=14), "14 d after", "later")),
                ),
            )
            for ph, g in sc.groupby(phase):
                alerts.append(
                    {
                        "stream": stream,
                        "phase": ph,
                        "frames": len(g),
                        "excluded_pct": 100 * g.excluded.mean(),
                        "maint_alert_pct": 100 * g.maint_alert.mean(),
                    }
                )
            for arm, p in preds.items():
                fa = p[p.fault != "none"].copy()
                if fa.empty:
                    continue
                fa["date"] = pd.to_datetime(fa.timestamp.str[:8], format="%Y%m%d")
                fa["week"] = fa.date.dt.isocalendar().week.astype(int)
                for wk, g in fa.groupby("week"):
                    weekly.append(
                        {
                            "stream": stream,
                            "arm": arm,
                            "week": wk,
                            "frames": len(g),
                            "mae_faulty": _mae(g),
                            "fs_faulty": _fs(g),
                            "withheld_pct": 100 * float(g.gated_frame.mean()),
                        }
                    )
                # first vs last two weeks of the fault, and the clean-stream run on the same dates
                days = sorted(fa.timestamp.str[:8].unique())
                first, last = fa[fa.timestamp.str[:8].isin(days[:14])], fa[fa.timestamp.str[:8].isin(days[-14:])]
                cl = clean_preds.get(arm)
                ref_first = ref_last = np.nan
                if cl is not None:
                    ref_first = _mae(cl[cl.timestamp.str[:8].isin(days[:14])])
                    ref_last = _mae(cl[cl.timestamp.str[:8].isin(days[-14:])])
                phases.append(
                    {
                        "stream": stream,
                        "arm": arm,
                        "mae_first_2wk": _mae(first),
                        "mae_last_2wk": _mae(last),
                        "clean_ref_first_2wk": ref_first,
                        "clean_ref_last_2wk": ref_last,
                        "withheld_first_2wk_pct": 100 * float(first.gated_frame.mean()),
                        "withheld_last_2wk_pct": 100 * float(last.gated_frame.mean()),
                    }
                )

    pd.DataFrame(rows).to_csv(TABLES / "ch4_admissibility_dose.csv", index=False)
    pd.DataFrame(contrasts).to_csv(TABLES / "ch4_admissibility_dose_contrasts.csv", index=False)
    pd.DataFrame(weekly).to_csv(TABLES / "ch4_admissibility_dose_weekly.csv", index=False)
    pd.DataFrame(alerts).to_csv(TABLES / "ch4_admissibility_dose_alerts.csv", index=False)
    pd.DataFrame(phases).to_csv(TABLES / "ch4_admissibility_dose_phases.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(pd.DataFrame(contrasts).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(alerts).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(pd.DataFrame(phases).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    cli()
