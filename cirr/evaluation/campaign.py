"""Chapter 4 campaign contract: one population, one provenance record, one schema.

Every Chapter 4 adaptation run goes through here, so a table, a notebook and a
paragraph cannot disagree about which samples were scored or how they were pooled.
The contract has four parts.

**Population.**  ``load_campaign_days`` applies :func:`cirr.physics.clear_sky.filter_lit_days`
at the one place the day index map is built — the same function, with the same
default, that the Chapter 5 federated benchmark uses.  Before this, the Chapter 4
evaluation path applied no filter at all, and scored 20.5 % night samples at
Karpicko and 32.5 % at Poznan against 0 % at daylight-only Folsom, which depressed
the Polish sites' error for every method and made the cross-site table incomparable.

**Provenance.**  ``SOURCE_CHECKPOINT`` is the single source model shared with
Chapter 5.  Its SHA-256, the git commit and the seed land in ``config.yaml`` of
every run, so a result folder can always be traced back to what produced it.

**Pooling.**  ``summary.json`` carries pooled *and* day-weighted metrics plus their
difference.  Pooled is the correct denominator and is what the chapter should
report; day-weighted is how the published tables were produced.  Publishing both
makes the gap visible instead of leaving it as an open reconciliation note.

**Units.**  Everything on disk is kW/m^2, as in Chapter 5.  Multiply by 1000 at
presentation time; never store W/m^2, so the two chapters' result trees stay
directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import resource
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from cirr.data.loading import load_dataset
from cirr.evaluation.online import OnlineEvalResult, run_delayed_online_eval
from cirr.evaluation.smart_persistence import predict_periods
from cirr.physics.clear_sky import (
    DEFAULT_MIN_GHI_CLEAR,
    filter_lit_days,
    load_clear_sky_lookup,
    resolve_clear_sky_value,
)
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.provenance import git_commit, sha256_file

# The one source model Chapters 4 and 5 share.  Chapter 4's published results used
# checkpoints/epoch_010.ckpt, a different file (md5 991e1451... against 21d5b347...),
# which is part of why its Folsom Static row disagrees with Chapter 5's.
SOURCE_CHECKPOINT = Path("checkpoints/resnet50_epoch_015.ckpt")

ACTIONS = ("none", "head", "whole")


@dataclass
class CampaignData:
    """The scored population and everything needed to describe it."""

    dataset: Any
    day_map: dict[str, list[int]]
    days: list[str]
    lookup: dict[str, float]
    dropped_per_day: dict[str, int]
    dropped_total: int
    site: str

    @property
    def n_samples(self) -> int:
        return sum(len(self.day_map[d]) for d in self.days)


def load_campaign_days(
    dataset_path: Path,
    start_day: str = "0101",
    num_days: int = -1,
    min_ghi_clear: float = DEFAULT_MIN_GHI_CLEAR,
) -> CampaignData:
    """Load a site and restrict it to the lit population, once.

    ``min_ghi_clear=0`` disables the filter and reproduces the pre-Phase-1
    population, which is what the regression check in the smoke plan needs.
    """
    dataset = load_dataset(dataset_path, split="test")
    raw_map = group_indices_by_day(dataset.periods)
    lookup = load_clear_sky_lookup(dataset_path)
    if min_ghi_clear > 0.0 and not lookup:
        raise ValueError(f"{dataset_path}: daylight filter requested but no clear-sky table found")

    lit_map, dropped_total = filter_lit_days(raw_map, dataset.periods, lookup, min_ghi_clear)
    dropped_per_day = {d: len(raw_map[d]) - len(lit_map.get(d, [])) for d in raw_map}

    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day: {start_day!r}")
    start_idx = all_days.index(start_day)
    window = all_days[start_idx : start_idx + num_days] if num_days > 0 else all_days[start_idx:]
    days = [d for d in window if d in lit_map]
    if not days:
        # Calendar gaps are common (Karpicko has no data for 01-20 April), so an
        # empty window is a silent no-op waiting to happen.  Fail here instead.
        raise ValueError(
            f"{dataset_path.name}: no scorable day in [{start_day}, +{num_days}); "
            f"site covers {len(lit_map)} days starting {sorted(lit_map)[0] if lit_map else 'none'}"
        )

    return CampaignData(
        dataset=dataset,
        day_map=lit_map,
        days=days,
        lookup=lookup,
        dropped_per_day=dropped_per_day,
        dropped_total=dropped_total,
        site=dataset_path.name.split("_")[0],
    )


def buffer_bytes(dataset: Any, horizon_steps: int) -> int:
    """Bytes the causal delay queue holds at steady state.

    The queue retains the *image* (``cirr/evaluation/online.py``), not a detached
    embedding, which is why a head update recomputes features rather than reusing
    them — and why the full-model action is feasible from retained state at all.
    """
    image, history, _clear, target, _name = dataset[0]
    per_item = sum(t.element_size() * t.numel() for t in (image, history, target))
    per_item += 8 * 2  # raw_pred and the prediction row index
    return per_item * horizon_steps


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def score_without_adaptation(
    model: Any,
    data: CampaignData,
    device: str,
    horizon_steps: int,
) -> dict[str, float]:
    """Pooled metrics for *model* over *data* with the update path disabled."""
    result = run_delayed_online_eval(
        model,
        data.dataset,
        data.day_map,
        data.days,
        device,
        horizon_steps,
        on_revealed=lambda _item, _ctx: None,
    )
    return result.overall_metrics


def smart_persistence_daily(data: CampaignData, horizon_minutes: int = 15) -> dict[str, float]:
    """Per-day Smart Persistence RMSE on exactly the scored population."""
    out: dict[str, float] = {}
    for day in data.days:
        periods = [data.dataset.periods[j] for j in data.day_map[day]]
        preds, tgts, _naive = predict_periods(periods, data.lookup, horizon_minutes)
        out[day] = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    return out


def _day_weighted(daily: pd.DataFrame) -> dict[str, float]:
    """How the published tables pooled.  Exact for MAE, approximate for WAPE/SMAPE.

    WAPE and SMAPE are computed per day on the subset above 5 W/m^2 while the
    weight is the full day count; that mismatch is exactly why the published MAE
    and WAPE rankings disagree at Poznan.
    """
    w = daily["n_samples"].to_numpy(float)

    def wavg(col: str) -> float:
        v = daily[col].to_numpy(float)
        m = np.isfinite(v)
        return float(np.average(v[m], weights=w[m])) if m.any() else float("nan")

    return {
        "mae": wavg("mae"),
        "rmse": float(np.sqrt(np.average(daily["rmse"].to_numpy(float) ** 2, weights=w))),
        "wape": wavg("wape"),
        "smape": wavg("smape"),
    }


def _events_from_predictions(preds: pd.DataFrame, flushed: dict[str, int]) -> pd.DataFrame:
    """Action mix per day, split by lit and dark.

    Derived from the per-sample rows rather than from counters in each script, so
    every method gets the same table without touching its adaptation callback.
    """
    rows = []
    has_scope = "adaptation_scope" in preds.columns
    for day, g in preds.groupby("day", sort=True):
        considered = int(g["update_considered"].sum()) if "update_considered" in g else 0
        row: dict[str, Any] = {"date": day, "considered": considered, "queue_flushed": flushed.get(day, 0)}
        c = g[g["update_considered"]] if "update_considered" in g else g.iloc[0:0]
        for action in ACTIONS:
            sel = (c["adaptation_scope"] == action) if has_scope else pd.Series(False, index=c.index)
            row[action] = int(sel.sum())
            row[f"lit_{action}"] = int((sel & c["lit"]).sum())
            row[f"dark_{action}"] = int((sel & ~c["lit"]).sum())
        if "scope_if_raw" in g.columns and has_scope:
            # Stays at zero while the non-negativity clamp never binds.  A nonzero
            # count means the model predicted negative irradiance and the clamped
            # trigger saw a smaller error than the model actually made.
            row["scope_changed_if_raw"] = int((c["scope_if_raw"] != c["adaptation_scope"]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_campaign(
    out_dir: Path,
    *,
    result: OnlineEvalResult,
    data: CampaignData,
    config: dict[str, Any],
    checkpoint: Path,
    horizon_steps: int,
    min_ghi_clear: float,
    wall_time_s: float,
    device: str,
    sp_daily: dict[str, float] | None = None,
    retention: dict[str, Any] | None = None,
) -> Path:
    """Write the six-file run folder defined in the Chapter 4 plan, §5.3."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.DataFrame(result.predictions_data)
    preds["ghi_clear"] = [resolve_clear_sky_value(data.lookup, str(t)) or 0.0 for t in preds["timestamp"]]
    preds["lit"] = preds["ghi_clear"] >= max(min_ghi_clear, DEFAULT_MIN_GHI_CLEAR)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    sp_daily = sp_daily if sp_daily is not None else smart_persistence_daily(data)
    daily = pd.DataFrame(
        [
            {
                "date": day,
                "n_samples": m["count"],
                "n_dropped_dark": data.dropped_per_day.get(day, 0),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "wape": m["wape"],
                "smape": m["smape"],
                "sp_rmse": sp_daily.get(day, float("nan")),
            }
            for day, m in result.per_day_metrics.items()
        ]
    ).sort_values("date")
    daily.to_csv(out_dir / "metrics_daily.csv", index=False)

    flushed = {d: min(horizon_steps, len(data.day_map[d])) for d in data.days}
    _events_from_predictions(preds, flushed).to_csv(out_dir / "events.csv", index=False)

    pooled = dict(result.overall_metrics)
    dayw = _day_weighted(daily)
    sp_rmse_pooled = float(np.sqrt(np.average(daily["sp_rmse"].to_numpy(float) ** 2, weights=daily["n_samples"])))
    actions = {a: int((preds.get("adaptation_scope") == a).sum()) for a in ACTIONS}

    summary = {
        "site": data.site,
        "pooled": pooled,
        "day_weighted": dayw,
        # The published tables used pooled MAE; this is the size of the choice.
        "pooling_delta": {k: pooled.get(k, float("nan")) - dayw.get(k, float("nan")) for k in dayw},
        "forecast_skill_pct": 100.0 * (1.0 - pooled["rmse"] / sp_rmse_pooled) if sp_rmse_pooled else float("nan"),
        "sp_rmse_pooled": sp_rmse_pooled,
        "actions": actions,
        "update_stats": dict(result.update_stats),
        "n_days": len(daily),
        "n_samples": int(daily["n_samples"].sum()),
        "n_dropped_dark": int(data.dropped_total),
        "buffer_bytes": buffer_bytes(data.dataset, horizon_steps),
        "peak_rss_mb": peak_rss_mb(),
        "wall_time_s": wall_time_s,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "cpu",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    if retention is not None:
        (out_dir / "retention.json").write_text(json.dumps(retention, indent=2, default=float))

    resolved = {
        **config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_commit": git_commit(),
        "min_ghi_clear": min_ghi_clear,
        "horizon_steps": horizon_steps,
        "device": device,
        "units": "kW/m^2",
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    return out_dir


def static_baseline(
    checkpoint: Path,
    dataset_path: Path,
    device: str,
    horizon_steps: int,
    min_ghi_clear: float,
    num_days: int = -1,
    cache_dir: Path = Path("results/ch4/_static_cache"),
) -> dict[str, float]:
    """Frozen-model metrics for a site, cached by checkpoint and population.

    The retention check needs the frozen reference on the same population as the
    adapted model.  Scoring a full year is expensive and the answer depends only
    on the checkpoint and the filter, so it is computed once and reused.
    """
    from cirr.evaluation.runner import load_eval_model

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{dataset_path.name}_{sha256_file(checkpoint)[:12]}_{min_ghi_clear}_{horizon_steps}_{num_days}.json"
    cached = cache_dir / key
    if cached.exists():
        return json.loads(cached.read_text())

    data = load_campaign_days(dataset_path, num_days=num_days, min_ghi_clear=min_ghi_clear)
    metrics = score_without_adaptation(load_eval_model(checkpoint, device), data, device, horizon_steps)
    cached.write_text(json.dumps(metrics, indent=2, default=float))
    return metrics
