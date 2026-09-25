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
EDGE_KINDS = ("soiling_edge", "occlusion_edge")  # the same faults at the rim (docs/CH4_ADMISSIBILITY.md §6)
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


def _edge_centre(rng: np.random.Generator, h: int, w: int) -> tuple[float, float]:
    """A fault centre in the annulus 0.72-0.92 of the field-of-view radius (docs/CH4_ADMISSIBILITY.md §6)."""
    rho, phi = rng.uniform(0.72, 0.92) * h / 2, rng.uniform(0, 2 * np.pi)
    return h / 2 + rho * np.sin(phi), w / 2 + rho * np.cos(phi)


def corrupt(image: torch.Tensor, kind: str, episode: int, frame: str) -> torch.Tensor:
    """Apply one synthetic fault to a normalised ``[3, H, W]`` image.

    ``soiling_edge`` and ``occlusion_edge`` are the same faults with their centres
    placed at the rim of the field of view, where the §6 split discards flags.
    """
    _, h, w = image.shape
    crop = (image.abs().sum(0, keepdim=True) > 0).float()  # the circular field of view
    pix = (image * _STD + _MEAN).clamp(0, 1)
    ep = _rng(kind, episode)  # fixed for the episode
    if kind in ("soiling", "soiling_edge"):
        for _ in range(int(ep.integers(3, 7))):
            cy, cx = _edge_centre(ep, h, w) if kind == "soiling_edge" else ep.uniform(0.2, 0.8, 2) * (h, w)
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
    elif kind in ("occlusion", "occlusion_edge"):
        cy, cx = _edge_centre(ep, h, w) if kind == "occlusion_edge" else ep.uniform(0.25, 0.75, 2) * (h, w)
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


# --------------------------------------------------------------------------- #
# Local-feature maintenance alert (pre-registered in docs/CH4_ADMISSIBILITY.md §5)
# --------------------------------------------------------------------------- #
LOCAL_DIM = 128
CELL = 8  # layer2 stride at 128 x 128 input: one cell per 8 x 8 pixel block


def local_projection(channels: int = 512, dim: int = LOCAL_DIM, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(channels, dim, generator=g) / dim**0.5


def valid_cells(image: torch.Tensor) -> torch.Tensor:
    """Cells whose whole 8 x 8 pixel block lies inside the circular field of view."""
    fov = (image.abs().sum(0) > 0).float()
    return torch.nn.functional.avg_pool2d(fov[None, None], CELL)[0, 0] >= 1.0


def edge_cells(cells: torch.Tensor) -> torch.Tensor:
    """Which valid cells (in scoring order) touch the border of the field of view.

    A valid cell is *edge* when one of its 4-neighbours is outside the mask
    (docs/CH4_ADMISSIBILITY.md §6): 40 of the 172 cells at 128 x 128.
    """
    if cells.dim() != 2:  # a flat cell list has no geometry: everything is interior
        return torch.zeros(int(cells.sum()), dtype=torch.bool)
    padded = torch.nn.functional.pad(cells.float(), (1, 1, 1, 1))
    neighbours = padded[:-2, 1:-1] * padded[2:, 1:-1] * padded[1:-1, :-2] * padded[1:-1, 2:]
    return (neighbours < 1.0)[cells]


@torch.no_grad()
def local_features(
    model: Any, dataset: Any, indices: list[int], cells: torch.Tensor, device: str, workers: int = 8, batch: int = 256
) -> torch.Tensor:
    """Per-cell features ``[N, cells, 128]`` from the forecaster's layer2, taken in the forecasting pass."""
    grabbed: dict[str, torch.Tensor] = {}
    hook = model.image_encoder.layer2.register_forward_hook(lambda _m, _i, o: grabbed.__setitem__("f", o))
    proj = local_projection().to(device)
    flat = cells.flatten().to(device)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch, num_workers=workers, pin_memory=True)
    out = []
    try:
        for image, history, *_ in tqdm(loader, desc="local", unit="batch"):
            model.encode_fusion_features(image.to(device, non_blocking=True), history.to(device, non_blocking=True))
            f = torch.nn.functional.avg_pool2d(grabbed["f"].float(), 3, 1, 1)  # neighbourhood context
            f = f.flatten(2).transpose(1, 2)[:, flat]  # [B, cells, 512]
            out.append((f @ proj).half().cpu())
    finally:
        hook.remove()
    return torch.cat(out) if out else torch.zeros(0, int(flat.sum()), LOCAL_DIM, dtype=torch.half)


class LocalGate:
    """Per-position nearest-neighbour novelty on local features: the maintenance detector.

    The camera never moves, so each cell keeps its own bank of nominal features. A
    cell's score is its nearest-neighbour distance normalised by that cell's own
    nominal 95th percentile; a frame's score is its worst cell, so one soiled patch
    is enough.
    """

    # the three frame scores of §6: max over all, interior or edge cells
    SPLITS = ("all", "interior", "edge")

    def __init__(
        self,
        bank: torch.Tensor,
        cells: torch.Tensor,
        scale: torch.Tensor | None = None,
        threshold: float = float("inf"),
        alert_threshold: float = float("inf"),
        thresholds: dict[str, float] | None = None,
    ):
        self.bank = bank  # [cells, n_bank, 128]
        self.cells = cells  # [16, 16] bool
        self.edge = edge_cells(cells)  # [cells] bool, scoring order
        self.scale = scale  # [cells] or None before calibration
        self.threshold, self.alert_threshold = threshold, alert_threshold
        # §6 thresholds: "<split>" for the frame score, "<split>_alert" for the rolling median
        self.thresholds: dict[str, float] = dict(thresholds or {})

    def normalised(self, feats: torch.Tensor, device: str) -> torch.Tensor:
        """Per-cell normalised distance ``[N, cells]``: the quantity every frame score is a max of."""
        return self.distances(feats, device) / self.scale

    def split(self, z: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        """Frame score and blamed cell for each split of §6, from the per-cell matrix."""
        z = torch.as_tensor(np.asarray(z, dtype=np.float32))
        masks = {"all": torch.ones_like(self.edge), "interior": ~self.edge, "edge": self.edge}
        out: dict[str, np.ndarray] = {}
        for name, m in masks.items():
            idx = torch.nonzero(m).flatten()
            worst, where = z[:, idx].max(1)
            out[name], out[f"{name}_cell"] = worst.numpy(), idx[where].numpy()
        return out

    @torch.no_grad()
    def distances(self, feats: torch.Tensor, device: str, chunk: int = 256) -> torch.Tensor:
        """Nearest-neighbour distance of every cell of every frame to that cell's bank: ``[N, cells]``."""
        bank = self.bank.to(device).float()
        out = []
        for part in feats.split(chunk):
            x = part.to(device).float().transpose(0, 1)  # [cells, n, 128]
            out.append(torch.cdist(x, bank).amin(-1).transpose(0, 1).cpu())  # [n, cells]
        return torch.cat(out) if out else torch.zeros(0, bank.shape[0])

    def score(self, feats: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
        """Frame score (worst normalised cell) and the index of that cell."""
        z = self.distances(feats, device) / self.scale
        worst, where = z.max(1)
        return worst.numpy(), where.numpy()

    def save(self, path: Path, **meta: Any) -> None:
        np.savez_compressed(
            path,
            bank=self.bank.half().numpy(),
            cells=self.cells.numpy(),
            scale=self.scale.numpy(),
            threshold=self.threshold,
            alert_threshold=self.alert_threshold,
            thresholds=json.dumps(self.thresholds),
            meta=json.dumps(meta),
        )

    @classmethod
    def load(cls, path: Path) -> LocalGate:
        z = np.load(path)
        return cls(
            torch.from_numpy(z["bank"]),
            torch.from_numpy(z["cells"]),
            torch.from_numpy(z["scale"]),
            float(z["threshold"]),
            float(z["alert_threshold"]),
            json.loads(str(z["thresholds"])) if "thresholds" in z else None,  # §5 files predate the split
        )


@torch.no_grad()
def iter_local_features(
    model: Any, dataset: Any, indices: list[int], cells: torch.Tensor, device: str, workers: int = 8, batch: int = 256
):
    """Yield per-cell features ``[B, cells, 128]`` batch by batch, for streams too large to hold."""
    grabbed: dict[str, torch.Tensor] = {}
    hook = model.image_encoder.layer2.register_forward_hook(lambda _m, _i, o: grabbed.__setitem__("f", o))
    proj = local_projection().to(device)
    flat = cells.flatten().to(device)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch, num_workers=workers, pin_memory=True)
    try:
        for image, history, *_ in tqdm(loader, desc="local", unit="batch"):
            model.encode_fusion_features(image.to(device, non_blocking=True), history.to(device, non_blocking=True))
            f = torch.nn.functional.avg_pool2d(grabbed["f"].float(), 3, 1, 1)
            yield f.flatten(2).transpose(1, 2)[:, flat] @ proj
    finally:
        hook.remove()


def local_scores(model: Any, dataset: Any, indices: list[int], gate: LocalGate, device: str, workers: int = 8) -> np.ndarray:
    """Per-cell normalised distances ``[N, cells]`` (float16) for every frame, scored batch by batch.

    A full stream's features are ~11 GB (245k frames x 172 cells x 128); holding them
    and then forking DataLoader workers from that process got the job killed, so a
    stream is never materialised.  The per-cell matrix is 84 MB and is kept, so any
    split of the cells (§6) can be scored again without a GPU.
    """
    parts = [
        gate.normalised(feats, device).half().numpy()
        for feats in iter_local_features(model, dataset, indices, gate.cells, device, workers)
    ]
    return np.concatenate(parts) if parts else np.zeros((0, int(gate.cells.sum())), np.float16)


def alert_statistic(timestamps: pd.Series, scores: np.ndarray, minutes: int = 30) -> np.ndarray:
    """Causal rolling median of the frame score over the preceding ``minutes`` of the same day.

    Returned in the input order.  The window is ``(t - minutes, t]``, so a frame's
    statistic uses only that frame and the ones strictly within the preceding window.
    """
    t = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), format="%Y%m%d_%H%M%S")
    s = pd.Series(np.asarray(scores, float))
    out = np.empty(len(s))
    for _, pos in t.groupby(t.dt.date).groups.items():
        pos = np.asarray(sorted(pos, key=lambda k: (t[k], k)))
        rolled = pd.Series(s[pos].to_numpy(), index=t[pos].to_numpy()).rolling(f"{minutes}min").median()
        out[pos] = rolled.to_numpy()
    return out
