"""Input admissibility for Chapter 4: embedding scores and synthetic sensor faults.

One module shared by ``scripts/fit_anomaly_gate.py`` (fit the detector),
``scripts/score_anomaly_gate.py`` (score any stream with a fitted detector),
``scripts/related_works/evaluate_lagged_ogd.py`` (gate the policy, optionally on a
corrupted stream) and ``scripts/admissibility_study.py`` (the analyses).

Synthetic faults
----------------
The real Folsom stream has no labelled sensor faults, so Section 4.2 proposes
injecting them.  Three kinds, following the faults the detector found in the
2016 stream (dirt on the dome, insects, water):

``soiling``   dark translucent blotches that persist for whole days
``droplets``  many small blurred, brightened discs that persist for hours
``occlusion`` one opaque dark ellipse, as an insect or object, for minutes

A fault is applied to the *normalised* tensor the dataset returns: it is mapped
back to pixel space, corrupted, renormalised and the circular crop re-applied, so
a corrupted frame goes through the model exactly like a real one.  Every fault is
a pure function of ``(kind, episode, timestamp)``, so a schedule reproduces the
same stream on any machine.  Labels are never touched -- the irradiance measured
at a corrupted frame is still true, which is precisely what makes learning from
it harmful.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF
from tqdm import tqdm

from cirr.data.loading import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD

KINDS = ("soiling", "droplets", "occlusion")
_MEAN = torch.tensor(DEFAULT_IMAGE_MEAN).view(3, 1, 1)
_STD = torch.tensor(DEFAULT_IMAGE_STD).view(3, 1, 1)


def stamp(period: dict) -> str:
    """``YYYYMMDD_HHMMSS`` of the frame a period predicts from."""
    return period["history"][-1]["image_name"][:15]


# --------------------------------------------------------------------------- #
# Embeddings and the fitted gate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed(model: Any, dataset: Any, indices: list[int], device: str, workers: int = 8, batch: int = 256) -> np.ndarray:
    """The 2048-d vector the forecasting head consumes, for ``dataset[indices]``."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch, num_workers=workers, pin_memory=True)
    out = []
    for image, history, *_ in tqdm(loader, desc="embed", unit="batch"):
        feats = model.encode_fusion_features(image.to(device, non_blocking=True), history.to(device, non_blocking=True))
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)


def load_gate(gate_dir: Path) -> tuple[Any, dict]:
    import joblib

    return joblib.load(Path(gate_dir) / "iforest.joblib"), json.loads((Path(gate_dir) / "gate.json").read_text())


def anomaly_score(forest: Any, x: np.ndarray) -> np.ndarray:
    """Larger is more anomalous; the gate rejects ``score > threshold``."""
    return -forest.score_samples(x)


# --------------------------------------------------------------------------- #
# Synthetic faults
# --------------------------------------------------------------------------- #
def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16))


def _disc(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    return (((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0).float()


def corrupt(image: torch.Tensor, kind: str, episode: int, frame: str) -> torch.Tensor:
    """Apply one synthetic fault to a normalised ``[3, H, W]`` image."""
    _, h, w = image.shape
    crop = (image.abs().sum(0, keepdim=True) > 0).float()  # the circular field of view
    pix = (image * _STD + _MEAN).clamp(0, 1)
    ep = _rng(kind, episode)  # fixed for the episode
    if kind == "soiling":
        for _ in range(int(ep.integers(3, 7))):
            cy, cx = ep.uniform(0.2, 0.8, 2) * (h, w)
            r = float(ep.uniform(0.06, 0.16) * h)
            soft = TF.gaussian_blur(_disc(h, w, cy, cx, r, r * ep.uniform(0.7, 1.3))[None], 9, 3.0)[0]
            alpha = float(ep.uniform(0.45, 0.8)) * soft
            pix = pix * (1 - alpha) + torch.tensor([0.25, 0.2, 0.15]).view(3, 1, 1) * alpha
    elif kind == "droplets":
        fr = _rng(kind, episode, frame)  # water moves between frames
        blurred = TF.gaussian_blur(pix, 7, 2.5)
        mask = torch.zeros(h, w)
        for _ in range(int(fr.integers(20, 41))):
            cy, cx = fr.uniform(0.1, 0.9, 2) * (h, w)
            r = float(fr.uniform(2, 5))
            mask = torch.maximum(mask, _disc(h, w, cy, cx, r, r))
        pix = pix * (1 - mask) + (blurred * 1.15).clamp(0, 1) * mask
    elif kind == "occlusion":
        cy, cx = ep.uniform(0.25, 0.75, 2) * (h, w)
        ry, rx = ep.uniform(0.08, 0.22, 2) * (h, w)
        alpha = 0.9 * TF.gaussian_blur(_disc(h, w, cy, cx, ry, rx)[None], 5, 1.5)[0]
        pix = pix * (1 - alpha) + 0.05 * alpha
    else:
        raise ValueError(f"unknown fault kind {kind!r}")
    return ((pix - _MEAN) / _STD) * crop


def build_schedule(timestamps: list[str], seed: int = 0) -> pd.DataFrame:
    """Fault episodes over a stream: which frame carries which fault.

    Soiling persists for 1-3 whole days (4 episodes), droplets for 1-3 hours (20),
    occlusion for 5-30 minutes (60).  Episodes of one kind never overlap and a
    frame carries at most one fault; later kinds do not overwrite earlier ones.
    """
    rng = np.random.default_rng(seed)
    ts = pd.Series(sorted(set(timestamps)))
    when = pd.to_datetime(ts, format="%Y%m%d_%H%M%S")
    days = sorted(ts.str[:8].unique())
    rows: dict[str, tuple[str, int]] = {}

    for e, start in enumerate(rng.choice(len(days) - 3, size=4, replace=False)):
        for d in days[start : start + int(rng.integers(1, 4))]:
            for t in ts[ts.str[:8] == d]:
                rows.setdefault(t, ("soiling", e))
    for kind, n, lo, hi in (("droplets", 20, 60, 180), ("occlusion", 60, 5, 30)):
        for e, i in enumerate(rng.choice(len(ts), size=n, replace=False)):
            end = when[i] + pd.Timedelta(minutes=int(rng.integers(lo, hi + 1)))
            same_day = ts.str[:8] == ts[i][:8]
            for t in ts[same_day & (when >= when[i]) & (when < end)]:
                rows.setdefault(t, (kind, e))
    out = pd.DataFrame([(t, k, e) for t, (k, e) in rows.items()], columns=["timestamp", "kind", "episode"])
    return out.sort_values("timestamp").reset_index(drop=True)


class CorruptedStream:
    """A dataset whose frames listed in ``schedule`` carry a synthetic fault.

    Everything but ``__getitem__`` is delegated to the wrapped dataset, so the
    daylight filter, Smart Persistence and the campaign writer see the same
    periods as before.
    """

    def __init__(self, base: Any, schedule: pd.DataFrame):
        self._base = base
        self._faults = {t: (k, int(e)) for t, k, e in schedule[["timestamp", "kind", "episode"]].itertuples(index=False)}

    def __len__(self) -> int:
        return len(self._base)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __getitem__(self, i: int):
        image, history, clear, target, name = self._base[i]
        fault = self._faults.get(str(name)[:15])
        if fault is not None:
            image = corrupt(image, fault[0], fault[1], str(name)[:15])
        return image, history, clear, target, name


def nominal_split(
    periods: list[dict], years: set[str], stride: int, calib_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Fit and calibration frame indices: the fit years only, thinned, split by whole days.

    Splitting by day keeps near-identical consecutive frames out of both halves,
    so the calibration percentile is not flattered by frames the forest has
    effectively seen.
    """
    by_day: dict[str, list[int]] = {}
    for i, p in enumerate(periods):
        s = stamp(p)
        if s[:4] in years:
            by_day.setdefault(s[:8], []).append(i)
    days = sorted(by_day)
    rng = np.random.default_rng(seed)
    calib = set(rng.choice(days, size=max(1, round(len(days) * calib_frac)), replace=False))
    fit = [i for d in days if d not in calib for i in by_day[d][::stride]]
    cal = [i for d in days if d in calib for i in by_day[d][::stride]]
    return fit, cal
