"""Input-admissibility gate for Chapter 4: Isolation Forest on encoder embeddings.

Implements the detector of Section 4.2 (and the PP-RAI paper) so that it can be
put in series with the conditional update policy:

1. Embed Folsom *training* frames from the fit years (2014-2015 by default) with the
   deployed source encoder -- the 2048-d vector the forecasting head consumes.
2. Fit an Isolation Forest (100 trees) on a day-level split of those embeddings.
3. Set the gate threshold s_anom as a high percentile of the anomaly score on the
   held-out days, which fixes the fraction of nominal frames flagged.
4. Score every frame of the evaluation stream and write one score per timestamp.

The anomaly score is ``-IsolationForest.score_samples``, so larger means more
anomalous and the gate rejects ``score > s_anom``.  The Folsom partition file
``train_periods.pickle`` also contains 2016, the evaluation year, so the fit years
are filtered explicitly and the script refuses to fit on an evaluation-year frame.

Scores depend only on the image and the source encoder.  For a policy whose
encoder stays frozen they are identical to what the forecasting pass would
produce on the device, so precomputing them costs the device nothing extra.

    python scripts/fit_anomaly_gate.py --out results/ch4/gate/folsom
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cirr.data.loading import load_dataset
from cirr.evaluation.campaign import SOURCE_CHECKPOINT, load_campaign_days
from cirr.evaluation.runner import load_eval_model
from cirr.utils.provenance import git_commit, set_determinism, sha256_file


def stamp(period: dict) -> str:
    return period["history"][-1]["image_name"][:15]


@torch.no_grad()
def embed(model, dataset, indices: list[int], device: str, workers: int, batch: int = 256) -> np.ndarray:
    """Embeddings as the forecasting head sees them, for ``dataset[indices]``."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch, num_workers=workers, pin_memory=True)
    out = []
    for image, history, *_ in tqdm(loader, desc="embed", unit="batch"):
        feats = model.encode_fusion_features(image.to(device, non_blocking=True), history.to(device, non_blocking=True))
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out)


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=str(SOURCE_CHECKPOINT), show_default=True)
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    default="data/folsom_prepared",
    show_default=True,
    help="Prepared dataset whose training partition supplies the nominal frames.",
)
@click.option("--fit-years", default="2014,2015", show_default=True, help="Years of the training partition to fit on.")
@click.option(
    "--stream",
    type=click.Path(exists=True, path_type=Path),
    default="data/folsom_prepared",
    show_default=True,
    help="Prepared dataset whose evaluation stream is scored (daylight population).",
)
@click.option(
    "--stride",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Keep every n-th training frame; consecutive minute frames are near-duplicates.",
)
@click.option(
    "--calib-frac",
    type=click.FloatRange(0.05, 0.5),
    default=0.2,
    show_default=True,
    help="Fraction of training days held out to set the threshold.",
)
@click.option(
    "--percentile",
    type=click.FloatRange(50.0, 99.99),
    default=99.0,
    show_default=True,
    help="Threshold = this percentile of the held-out anomaly scores.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--workers", type=click.IntRange(min=0), default=8, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True)
def main(
    checkpoint: Path,
    source: Path,
    fit_years: str,
    stream: Path,
    stride: int,
    calib_frac: float,
    percentile: float,
    seed: int,
    workers: int,
    device: str,
    out_dir: Path,
) -> None:
    set_determinism(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    years = {y.strip() for y in fit_years.split(",") if y.strip()}
    model = load_eval_model(Path(checkpoint), device)

    # --- nominal frames: the fit years only, thinned, split by day ----------------
    train = load_dataset(source, split="training")
    by_day: dict[str, list[int]] = {}
    for i, p in enumerate(train.periods):
        s = stamp(p)
        if s[:4] in years:
            by_day.setdefault(s[:8], []).append(i)
    days = sorted(by_day)
    rng = np.random.default_rng(seed)
    calib_days = set(rng.choice(days, size=max(1, round(len(days) * calib_frac)), replace=False))
    fit_idx = [i for d in days if d not in calib_days for i in by_day[d][::stride]]
    cal_idx = [i for d in days if d in calib_days for i in by_day[d][::stride]]

    stream_data = load_campaign_days(stream, min_ghi_clear=0.001)
    stream_years = {stamp(stream_data.dataset.periods[i])[:4] for d in stream_data.days for i in stream_data.day_map[d]}
    leak = years & stream_years if source.resolve() == stream.resolve() else set()
    assert not leak, f"fit years {sorted(leak)} overlap the evaluation stream; refusing to fit on test frames"
    print(
        f"fit: {len(fit_idx)} frames from {len(days) - len(calib_days)} days | calibration: {len(cal_idx)} frames "
        f"from {len(calib_days)} days | years {sorted(years)} | stride {stride}"
    )

    x_fit = embed(model, train, fit_idx, device, workers)
    x_cal = embed(model, train, cal_idx, device, workers)

    forest = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(x_fit)
    cal_scores = -forest.score_samples(x_cal)
    threshold = float(np.percentile(cal_scores, percentile))

    # --- the evaluation stream -----------------------------------------------------
    s_idx = [i for d in stream_data.days for i in stream_data.day_map[d]]
    x_stream = embed(model, stream_data.dataset, s_idx, device, workers)
    scores = -forest.score_samples(x_stream)
    frame = pd.DataFrame(
        {
            "timestamp": [stamp(stream_data.dataset.periods[i]) for i in s_idx],
            "score": scores,
            "flagged": scores > threshold,
        }
    )
    frame.to_csv(out_dir / "scores.csv", index=False)
    joblib.dump(forest, out_dir / "iforest.joblib")

    frame["month"] = frame.timestamp.str[4:6]
    monthly = (100 * frame.groupby("month").flagged.mean()).round(2).to_dict()
    meta = {
        "threshold": threshold,
        "percentile": percentile,
        "score": "-IsolationForest.score_samples (higher = more anomalous)",
        "n_estimators": 100,
        "fit_years": sorted(years),
        "stride": stride,
        "n_fit": len(fit_idx),
        "n_calibration": len(cal_idx),
        "calibration_flag_rate_pct": float(100 * (cal_scores > threshold).mean()),
        "stream": str(stream),
        "n_stream": len(frame),
        "stream_flag_rate_pct": float(100 * frame.flagged.mean()),
        "stream_flag_rate_by_month_pct": monthly,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_commit": git_commit(),
        "seed": seed,
    }
    (out_dir / "gate.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("threshold", "calibration_flag_rate_pct", "stream_flag_rate_pct")}, indent=2))
    print("flag rate by month (%):", monthly)


if __name__ == "__main__":
    main()
