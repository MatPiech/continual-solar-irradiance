"""Figures for the Chapter 4 input-admissibility study.

Reads only the CSVs written by ``scripts/admissibility_study.py`` (and, for the
example-frame figure, the fitted gate), and writes PDF + PNG to figures/ch4/.
A figure whose table does not exist yet is skipped, so this can be re-run as the
study fills in.

Style follows scripts/make_ch5_figures.py (same rcParams, same site colours) so
the two chapters' figures sit together.  Categorical colours were checked with the
dataviz palette validator: the site palette passes every check; in the five-slot
signal / detector palette three slots are below 3:1 contrast, so every bar carries a
direct value label, and the model-free detectors are hatched as a second encoding.

    python scripts/plot_ch4_admissibility.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "tables"
OUT = ROOT / "figures" / "ch4"
SITE = {"folsom": "#3a6fb0", "karpicko": "#d1670b", "poznan": "#1f8a70"}
SITE_NAME = {"folsom": "Folsom", "karpicko": "Karpicko", "poznan": "Poznań"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order (validated, 6 slots)
GREY, INK, MUTED = "#8c8c8c", "#222222", "#5a5a5a"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
    }
)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def table(name: str) -> pd.DataFrame | None:
    f = TABLES / f"{name}.csv"
    return pd.read_csv(f, dtype={"month": str}) if f.exists() else None


def risk_coverage() -> None:
    t = table("ch4_admissibility_risk_coverage")
    if t is None:
        return
    sites = [s for s in SITE if s in set(t.site)]
    fig, axes = plt.subplots(1, len(sites), figsize=(2.4 * len(sites) + 0.4, 2.3), squeeze=False)
    lines = [
        ("score", "anomaly score", SERIES[0], "-"),
        ("hist_std", "history std", SERIES[1], "-"),
        ("combined", "score + history std", SERIES[2], "-"),
        ("err", "oracle (true error)", GREY, "--"),
    ]
    for ax, site in zip(axes[0], sites, strict=True):
        d = t[t.site == site]
        for key, label, color, ls in lines:
            g = d[d.signal == key]
            ax.plot(g.coverage, g.mae, color=color, ls=ls, lw=1.6 if ls == "-" else 1.0, label=label)
        ax.set_title(SITE_NAME[site], fontsize=8, color=INK)
        ax.set_xlabel("coverage (% of forecasts kept)")
        ax.set_xlim(100, 50)
        ax.grid(axis="y", lw=0.4, color="#e6e6e6")
    axes[0][0].set_ylabel("MAE of kept forecasts (W/m²)")
    axes[0][-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Withholding the forecasts the signal ranks least reliable", fontsize=8, y=1.02)
    save(fig, "ch4_admissibility_risk_coverage")


def detection() -> None:
    t = table("ch4_admissibility_detection")
    if t is None:
        return
    # the §6 split rows have their own figure below; the palette has six slots and this one needs six
    t = t[~t.detector.str.startswith("local,")]
    detectors = list(dict.fromkeys(t.detector))
    faults = ["soiling", "droplets", "occlusion"]
    t = t.assign(fault=t.fault.where(~t.fault.str.startswith("none"), "clean\n(false alarms)"))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [4, 3]})
    width = 0.84 / len(detectors)
    panels = [
        ("flagged_pct", ["clean\n(false alarms)"] + faults, "frames flagged at the 1 % threshold (%)", 30),
        ("auroc", faults, "AUROC, faulty vs clean frames", 1.08),
    ]
    for panel, (col, groups, ylabel, ymax) in zip(axes, panels, strict=True):
        for i, det in enumerate(detectors):
            d = t[t.detector == det].set_index("fault")
            vals = [float(d.loc[f, col]) for f in groups]
            x = np.arange(len(groups)) + (i - (len(detectors) - 1) / 2) * width
            model_free = det.startswith(("dark", "blur"))
            bars = panel.bar(
                x,
                vals,
                width * 0.92,
                color=SERIES[i],
                label=det,
                hatch="////" if model_free else None,
                edgecolor="white",
                linewidth=0,
            )
            for b, v in zip(bars, vals, strict=True):
                panel.text(
                    b.get_x() + b.get_width() / 2,
                    v + ymax * 0.01,
                    f"{v:.1f}" if col == "flagged_pct" else f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    color=INK,
                    rotation=90,
                )
        panel.set_xticks(range(len(groups)), groups)
        panel.set_ylabel(ylabel)
        panel.set_ylim(0, ymax)
        panel.grid(axis="y", lw=0.4, color="#e6e6e6")
    axes[0].axhline(1.0, color=GREY, lw=0.8, ls="--")
    axes[1].axhline(0.5, color=GREY, lw=0.8, ls="--")
    axes[1].text(-0.5, 0.51, "chance", fontsize=6, color=MUTED, va="bottom", ha="left")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(detectors), fontsize=6.5, bbox_to_anchor=(0.5, 1.08)
    )
    save(fig, "ch4_admissibility_detection")


def detection_split() -> None:
    """§6: the local detector over all, interior and edge cells against the dark-pixel baseline, five faults."""
    t = table("ch4_admissibility_detection")
    if t is None or not t.detector.str.startswith("local,").any():
        return
    keep = ["dark-pixel fraction", "local (layer2, per cell)", "local, interior cells", "local, edge cells"]
    t = t[t.detector.isin(keep)]
    labels = dict(
        zip(keep, ["dark-pixel fraction", "local, all cells", "local, interior cells", "local, edge cells"], strict=True)
    )
    faults = ["soiling", "droplets", "occlusion", "soiling_edge", "occlusion_edge"]
    pretty = {"soiling_edge": "soiling\n(at rim)", "occlusion_edge": "occlusion\n(at rim)", "clean": "clean\n(false alarms)"}
    t = t.assign(fault=t.fault.where(~t.fault.str.startswith("none"), "clean"))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [6, 5]})
    width = 0.84 / len(keep)
    panels = [
        ("flagged_pct", ["clean"] + faults, "frames flagged at the 1 % threshold (%)", 70),
        ("auroc", faults, "AUROC, faulty vs clean frames", 1.08),
    ]
    for panel, (col, groups, ylabel, ymax) in zip(axes, panels, strict=True):
        for i, det in enumerate(keep):
            d = t[t.detector == det].set_index("fault")
            vals = [float(d.loc[f, col]) if f in d.index else np.nan for f in groups]
            x = np.arange(len(groups)) + (i - (len(keep) - 1) / 2) * width
            bars = panel.bar(
                x,
                vals,
                width * 0.92,
                color=SERIES[i],
                label=labels[det],
                hatch="////" if det.startswith("dark") else None,
                edgecolor="white",
                linewidth=0,
            )
            for b, v in zip(bars, vals, strict=True):
                if np.isfinite(v):
                    panel.text(
                        b.get_x() + b.get_width() / 2,
                        v + ymax * 0.01,
                        f"{v:.1f}" if col == "flagged_pct" else f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color=INK,
                        rotation=90,
                    )
        panel.set_xticks(range(len(groups)), [pretty.get(g, g) for g in groups], fontsize=6.5)
        panel.set_ylabel(ylabel)
        panel.set_ylim(0, ymax)
        panel.grid(axis="y", lw=0.4, color="#e6e6e6")
    axes[0].axhline(1.0, color=GREY, lw=0.8, ls="--")
    axes[0].axhline(50.0, color=GREY, lw=0.8, ls=":")
    axes[0].text(len(panels[0][1]) - 0.5, 50.8, "criterion 1", fontsize=6, color=MUTED, va="bottom", ha="right")
    axes[1].axhline(0.5, color=GREY, lw=0.8, ls="--")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(keep), fontsize=6.5, bbox_to_anchor=(0.5, 1.08)
    )
    save(fig, "ch4_admissibility_detection_split")


def geometry() -> None:
    """§6: solar elevation and distance to the sun of the clean-stream flags, interior against edge."""
    t = table("ch4_admissibility_geometry")
    if t is None:
        return
    splits = [("edge", SERIES[1]), ("interior", SERIES[2])]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.3))
    bins_el, bins_d, bins_h = np.arange(0, 80, 5), np.arange(0, 12.5, 0.5), np.arange(5, 21)
    for split, color in splits:
        g = t[t.split == split]
        label = f"{split} flags (n = {len(g)})"
        axes[0].hist(g.elevation_deg, bins=bins_el, density=True, histtype="step", color=color, lw=1.4, label=label)
        axes[1].hist(g.sun_distance_cells.dropna(), bins=bins_d, density=True, histtype="step", color=color, lw=1.4)
        axes[2].hist(g.hour, bins=bins_h, density=True, histtype="step", color=color, lw=1.4)
    axes[0].axvline(20, color=GREY, lw=0.8, ls="--")
    axes[1].axvline(2, color=GREY, lw=0.8, ls="--")
    axes[0].set_xlabel("solar elevation (°)")
    axes[1].set_xlabel("blamed cell to sun (cells)")
    axes[2].set_xlabel("hour of day")
    axes[0].set_ylabel("density")
    for ax in axes:
        ax.grid(axis="y", lw=0.4, color="#e6e6e6")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "ch4_admissibility_geometry")


def crosssite() -> None:
    m, d = table("ch4_admissibility_crosssite_monthly"), table("ch4_admissibility_crosssite_daily")
    if m is None or d is None:
        return
    bins = [-0.001, 0.0, 1.0, 5.0, 100.0]
    labels = ["none", "≤ 1 %", "1–5 %", "> 5 %"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for k, (site, color) in enumerate(SITE.items()):
        g = m[m.site == site].assign(month=lambda x: x.month.astype(int)).set_index("month").reindex(range(1, 13))
        if g.flag_rate_pct.isna().all():
            continue
        axes[0].plot(g.index, g.flag_rate_pct, marker="o", ms=3, color=color, label=SITE_NAME[site])  # NaN breaks the line
        dd = d[d.site == site].assign(bin=lambda x: pd.cut(x.flag_pct, bins, labels=labels))
        mean = dd.groupby("bin", observed=False).static_mae.mean()
        x = np.arange(len(labels)) + (k - 1) * 0.08
        axes[1].plot(x, mean.to_numpy(), marker="o", ms=4, color=color, label=SITE_NAME[site])
    axes[0].axhline(1.0, color=GREY, lw=0.8, ls="--")
    axes[0].text(6.3, 1.08, "calibrated nominal rate (1 %)", fontsize=6, color=MUTED, va="bottom")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xlabel("month")
    axes[0].set_ylabel("frames flagged (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_xlabel("frames flagged that day")
    axes[1].set_ylabel("frozen-model MAE that day (W/m²)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.grid(axis="y", lw=0.4, color="#e6e6e6")
    fig.suptitle("The Folsom-fitted gate at the deployment sites", fontsize=8, y=1.02)
    save(fig, "ch4_admissibility_crosssite")


def adaptation() -> None:
    t = table("ch4_admissibility_adaptation")
    if t is None or t.empty:
        return
    arms = ["clean stream", "corrupted, no gate", "corrupted, gated"]
    policies = list(dict.fromkeys(t.policy))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    width = 0.8 / len(arms)
    for panel, (col, ylabel) in zip(
        axes,
        [
            ("mae_clean_frames", "MAE on fault-free frames (W/m²)"),
            ("updates_on_faulty_frames", "updates made on faulty frames"),
        ],
        strict=True,
    ):
        for i, arm in enumerate(arms):
            vals = [float(t[(t.policy == p) & (t.arm == arm)][col].iloc[0]) for p in policies]
            x = np.arange(len(policies)) + (i - 1) * width
            bars = panel.bar(x, vals, width * 0.92, color=SERIES[i], label=arm)
            for b, v in zip(bars, vals, strict=True):
                panel.text(
                    b.get_x() + b.get_width() / 2,
                    v,
                    f"{v:.2f}" if col.startswith("mae") else f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=INK,
                )
        panel.set_xticks(range(len(policies)), policies)
        panel.set_ylabel(ylabel)
        panel.grid(axis="y", lw=0.4, color="#e6e6e6")
    # bars start at zero: the three arms differ by hundredths of a W/m^2, and the figure should say so
    axes[0].set_ylim(0, t.mae_clean_frames.max() * 1.15)
    axes[1].set_ylabel("updates on frames that carry a fault\n(in the corrupted stream)")
    fig.subplots_adjust(wspace=0.35)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.06))
    save(fig, "ch4_admissibility_adaptation")


def faults() -> None:
    """One frame, clean and with each synthetic fault, with the gate's score."""
    import torch

    from cirr.evaluation.admissibility import _MEAN, _STD, anomaly_score, corrupt, load_gate
    from cirr.evaluation.campaign import SOURCE_CHECKPOINT, load_campaign_days
    from cirr.evaluation.runner import load_eval_model

    gate = ROOT / "results" / "adaptation" / "gate" / "folsom"
    if not (gate / "gate.json").exists():
        return
    forest, meta = load_gate(gate)
    model = load_eval_model(Path(SOURCE_CHECKPOINT), "cpu")
    data = load_campaign_days(ROOT / "data" / "folsom_prepared", "0415", 1)
    img, hist, *_, name = data.dataset[data.day_map[data.days[0]][300]]
    kinds = ("soiling", "droplets", "occlusion", "soiling_edge", "occlusion_edge", "banding", "insect", "wet_dome")
    variants = [("clean", img)] + [(k.replace("_edge", " (rim)"), corrupt(img, k, 0, name)) for k in kinds]
    fig, axes = plt.subplots(1, 9, figsize=(7.6, 1.3))
    for ax, (label, x) in zip(axes, variants, strict=True):
        with torch.no_grad():
            s = float(anomaly_score(forest, model.encode_fusion_features(x[None], hist[None]).numpy())[0])
        view = ((x * _STD + _MEAN).clamp(0, 1) * (x.abs().sum(0, keepdim=True) > 0)).permute(1, 2, 0).numpy()
        ax.imshow(view)
        ax.axis("off")
        verdict = "rejected" if s > meta["threshold"] else "admitted"
        ax.set_title(f"{label}\nscore {s:.3f} — {verdict}", fontsize=7, color=INK)
    fig.suptitle(
        f"Synthetic faults and the gate (threshold {meta['threshold']:.3f}, p{meta['percentile']:g})", fontsize=8, y=1.04
    )
    save(fig, "ch4_admissibility_faults")


def maintenance() -> None:
    """Each soiling episode: frame score and 30-minute alert statistic around it."""
    import json

    gate_dir = ROOT / "results" / "adaptation" / "gate"
    f = gate_dir / "local" / "scores_corrupted.csv"
    if not f.exists():
        return
    meta = json.loads((gate_dir / "local" / "gate.json").read_text())
    sc = pd.read_csv(f, dtype={"timestamp": str})
    sched = pd.read_csv(gate_dir / "folsom_corrupted" / "schedule.csv", dtype={"timestamp": str})
    sc = sc.merge(sched, on="timestamp", how="left")
    sc["t"] = pd.to_datetime(sc.timestamp, format="%Y%m%d_%H%M%S")
    sc["day"] = sc.timestamp.str[:8]
    days = sorted(sc.day.unique())
    soil = sched[sched.kind == "soiling"].assign(day=lambda x: x.timestamp.str[:8])
    episodes = sorted(soil.episode.unique())
    fig, axes = plt.subplots(len(episodes), 1, figsize=(7.2, 1.35 * len(episodes)), squeeze=False)
    for ax, ep in zip(axes[:, 0], episodes, strict=True):
        ep_days = sorted(soil[soil.episode == ep].day.unique())
        first = days.index(ep_days[0])
        show = days[max(0, first - 1) : days.index(ep_days[-1]) + 2]  # one clean day either side
        g = sc[sc.day.isin(show)].sort_values("t").reset_index(drop=True)
        x = np.arange(len(g))  # frame order: no overnight gaps
        faulty = (g.kind == "soiling") & (g.episode == ep)
        ax.fill_between(
            x, 0, 1, where=faulty, transform=ax.get_xaxis_transform(), color="#f0e4d7", lw=0, label="soiling present"
        )
        ax.plot(x, g.score, color=GREY, lw=0.5, label="frame score")
        ax.plot(x, g.alert_stat, color=SERIES[0], lw=1.4, label="30-min alert statistic")
        ax.axhline(meta["alert_threshold"], color=INK, lw=0.8, ls="--", label="alert threshold")
        if "interior_alert_stat" in g:  # the §6 split
            ax.plot(x, g.interior_alert_stat, color=SERIES[2], lw=1.2, label="interior alert statistic")
            ax.axhline(meta["thresholds"]["interior_alert"], color=SERIES[2], lw=0.8, ls=":", label="interior threshold")
        starts = [0] + [i for i in range(1, len(g)) if g.day[i] != g.day[i - 1]]
        ax.set_xticks(starts, [f"{g.day[i][4:6]}-{g.day[i][6:8]}" for i in starts])
        ax.set_ylabel("score")
        ax.set_ylim(0, max(g.score.quantile(0.995), meta["alert_threshold"]) * 1.2)
        ax.grid(axis="y", lw=0.4, color="#e6e6e6")
    axes[-1, 0].set_xlabel("daylight frames (day boundaries marked)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, "ch4_admissibility_maintenance")


def local_top() -> None:
    """The real 2016 frames the local gate scores highest, one per day, with the blamed cell outlined.

    One figure per split of §6 (all / interior / edge cells), from the matching top table.
    """
    tables = {s: table(f"ch4_admissibility_local_top{'' if s == 'all' else '_' + s}") for s in ("all", "interior", "edge")}
    tables = {s: t for s, t in tables.items() if t is not None}
    if not tables:
        return
    from matplotlib.patches import Rectangle

    from cirr.evaluation.admissibility import _MEAN, _STD, LocalGate
    from cirr.evaluation.campaign import load_campaign_days

    gate = LocalGate.load(ROOT / "results" / "adaptation" / "gate" / "local" / "local_gate.npz")
    rows_cols = np.argwhere(gate.cells.numpy())  # the order cells are scored in
    data = load_campaign_days(ROOT / "data" / "folsom_prepared")
    where = {data.dataset.periods[j]["history"][-1]["image_name"][:15]: j for d in data.days for j in data.day_map[d]}
    for split, t in tables.items():
        n = len(t)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(7.2, 2.9))
        for ax, r in zip(axes.flat, t.itertuples(), strict=False):
            img = data.dataset[where[r.timestamp]][0]
            view = ((img * _STD + _MEAN).clamp(0, 1) * (img.abs().sum(0, keepdim=True) > 0)).permute(1, 2, 0).numpy()
            ax.imshow(view)
            cy, cx = rows_cols[int(r.cell)]
            ax.add_patch(Rectangle((cx * 8 - 0.5, cy * 8 - 0.5), 8, 8, fill=False, ec="#e34948", lw=1.2))
            ax.set_title(
                f"{r.timestamp[4:6]}-{r.timestamp[6:8]} {r.timestamp[9:11]}:{r.timestamp[11:13]}  {r.score:.1f}",
                fontsize=6,
                color=INK,
            )
            ax.axis("off")
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        region = {"all": "any cell", "interior": "interior cells only", "edge": "edge cells only"}[split]
        fig.suptitle(
            f"Real 2016 frames the local gate scores highest over {region} (one per day); box = the cell it blames",
            fontsize=8,
            y=1.0,
        )
        save(fig, f"ch4_admissibility_local_top{'' if split == 'all' else '_' + split}")


if __name__ == "__main__":
    for fn in (risk_coverage, detection, detection_split, geometry, crosssite, adaptation, faults, maintenance, local_top):
        fn()
