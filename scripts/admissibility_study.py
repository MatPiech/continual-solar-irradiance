"""Chapter 4 input-admissibility study: every table the section reports.

Subcommands, in the order the reproduction guide runs them
(docs/CH4_ADMISSIBILITY.md):

    schedule     synthetic-fault schedule over the Folsom 2016 stream
    reliability  the anomaly score as a label-free forecast-error signal
    detection    detection of synthetic faults: Isolation Forest vs baselines
    crosssite    flag rate at the deployment sites under the Folsom-fitted gate
    adaptation   the policy on the corrupted stream, with and without the gate

Each writes CSVs under tables/ that scripts/plot_ch4_admissibility.py turns
into figures; nothing is edited by hand.
"""

from __future__ import annotations

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
    KINDS,
    anomaly_score,
    build_schedule,
    corrupt,
    embed,
    load_gate,
    nominal_split,
    stamp,
)
from cirr.evaluation.campaign import PERIOD_SCALE, SOURCE_CHECKPOINT, load_campaign_days
from cirr.evaluation.runner import load_eval_model
from cirr.evaluation.smart_persistence import predict_periods
from cirr.fl.analysis import contrast
from cirr.physics.clear_sky import resolve_clear_sky_value

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "results" / "ch4" / "gate"
RUNS = GATE / "runs"
TABLES = ROOT / "tables"
W = 1000.0
SITES = ("folsom", "karpicko", "poznan")
# the policy each site's forecasts come from: head-only, the stable policy of Section 4.3
POLICY = {s: ROOT / "results" / "ch4" / "ablation" / s / "conditional_headonly" / "seed0" for s in SITES}
STATIC = {s: ROOT / "results" / "ch4" / "tier1" / s / "baseline" / "seed0" for s in SITES}
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
            sig.append(dict(site=site, signal=lab, spearman=float(spearmanr(x[k], x.err, nan_policy="omit").statistic)))
        sig.append(
            dict(
                site=site,
                signal="rank R^2: history + clear-sky index",
                spearman=rank_r2(x.err.to_numpy(), [x.hist_std.to_numpy(), x.kc_dev.fillna(0).to_numpy()]),
            )
        )
        sig.append(
            dict(
                site=site,
                signal="rank R^2: + anomaly score",
                spearman=rank_r2(x.err.to_numpy(), [x.hist_std.to_numpy(), x.kc_dev.fillna(0).to_numpy(), x.score.to_numpy()]),
            )
        )
        for k in ("score", "hist_std", "combined", "err"):
            for c, m in zip(COVERAGE, risk_coverage(x, k), strict=True):
                curves.append(dict(site=site, signal=k, coverage=c, mae=m))
        for flag, g in x.groupby("flagged"):
            groups.append(
                dict(
                    site=site,
                    group="flagged" if flag else "admitted",
                    n=len(g),
                    share_pct=100 * len(g) / len(x),
                    model_mae=g.err.mean(),
                    smart_persistence_mae=g.err_sp.mean(),
                )
            )
        hybrid = np.where(x.flagged, x.err_sp, x.err).mean()
        groups.append(
            dict(
                site=site,
                group="hybrid: Smart Persistence on flagged",
                n=len(x),
                share_pct=100.0,
                model_mae=hybrid,
                smart_persistence_mae=np.nan,
            )
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
    items = [(j, None, 0) for j in probe] + [(j, k, n) for k in KINDS for n, j in enumerate(probe)]
    x_probe = _embed_items(model, data.dataset, items, device, workers)
    kind = np.array(["clean"] * len(probe) + [k for k in KINDS for _ in probe])

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
            dict(
                detector=name,
                fault="none (clean 2016)",
                n=int(clean.size),
                flagged_pct=100 * float((clean > thr).mean()),
                auroc=np.nan,
            )
        )
        for k in KINDS:
            f = s[kind == k]
            rows.append(
                dict(
                    detector=name,
                    fault=k,
                    n=int(f.size),
                    flagged_pct=100 * float((f > thr).mean()),
                    auroc=float(roc_auc_score(np.r_[np.zeros(clean.size), np.ones(f.size)], np.r_[clean, f])),
                )
            )
    for name, col in pixel.items():
        thr = float(np.percentile(px_cal[:, col], meta["percentile"]))
        sc = px_probe[:, col]
        clean = sc[kind == "clean"]
        rows.append(
            dict(
                detector=name,
                fault="none (clean 2016)",
                n=int(clean.size),
                flagged_pct=100 * float((clean > thr).mean()),
                auroc=np.nan,
            )
        )
        for k in KINDS:
            f = sc[kind == k]
            rows.append(
                dict(
                    detector=name,
                    fault=k,
                    n=int(f.size),
                    flagged_pct=100 * float((f > thr).mean()),
                    auroc=float(roc_auc_score(np.r_[np.zeros(clean.size), np.ones(f.size)], np.r_[clean, f])),
                )
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
            dict(
                site=site,
                frames=len(sc),
                flag_rate_pct=100 * sc.flagged.mean(),
                median_score=sc.score.median(),
                spearman_daily_flag_vs_static_mae=float(spearmanr(d.flag_pct, d.static_mae).statistic),
                spearman_daily_flag_vs_adaptation_gain=float(spearmanr(d.flag_pct, d.adaptation_gain).statistic),
            )
        )
        for m, g in sc.groupby("month"):
            monthly.append(dict(site=site, month=m, flag_rate_pct=100 * g.flagged.mean()))
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
            ROOT / "results" / "ch4" / "ablation" / "folsom" / "conditional_last2_lowlr" / "seed0",
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
                dict(
                    policy=policy,
                    arm=arm,
                    mae_all=W * (p.prediction - p.target).abs().mean(),
                    mae_clean_frames=W * (clean.prediction - clean.target).abs().mean(),
                    mae_faulty_frames=W * (p[p.fault != "none"].prediction - p[p.fault != "none"].target).abs().mean(),
                    updates=len(upd),
                    updates_on_faulty_frames=int((upd.fault != "none").sum()),
                    gated=int(p.get("gated", pd.Series(False, index=p.index)).sum()),
                )
            )
        # the effect that matters: did learning from faulty frames damage forecasts on clean frames?
        for a, b in [("corrupted, no gate", "clean stream"), ("corrupted, gated", "corrupted, no gate")]:
            c = contrast(_daily_on(table[a]), _daily_on(table[b]), tag=f"adm-{policy}-{a}-{b}")
            contrasts.append(
                dict(
                    policy=policy,
                    comparison=f"{a} vs {b}",
                    frames="clean frames only",
                    rel_mae_pct=c["rel"],
                    ci90_lo=c["rel_lo"],
                    ci90_hi=c["rel_hi"],
                    dfs=c["dfs"],
                    p=c["p"],
                )
            )
    pd.DataFrame(rows).to_csv(TABLES / "ch4_admissibility_adaptation.csv", index=False)
    pd.DataFrame(contrasts).to_csv(TABLES / "ch4_admissibility_adaptation_contrasts.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(pd.DataFrame(contrasts).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    cli()
