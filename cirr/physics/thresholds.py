"""Dynamic prototype distance threshold fitting for gating."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def fit_thresholds(
    distances_by_bin: dict[int, list[float]],
    norm_percentile: float = 95.0,
    ood_percentile: float = 99.9,
) -> dict[int, dict[str, float]]:
    """Fit tau_norm and tau_ood per CSI bin from intra-cluster distances."""
    thresholds: dict[int, dict[str, float]] = {}
    for bin_idx, distances in distances_by_bin.items():
        if not distances:
            continue
        arr = np.asarray(distances, dtype=np.float64)
        thresholds[bin_idx] = {
            "tau_norm": float(np.percentile(arr, norm_percentile)),
            "tau_ood": float(np.percentile(arr, ood_percentile)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "count": int(len(arr)),
        }
    return thresholds


def compute_intra_bin_distances(
    embeddings: torch.Tensor,
    bin_indices: list[int],
) -> dict[int, list[float]]:
    """Compute Euclidean distance of each embedding to its bin centroid."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [N, D]")

    by_bin: dict[int, list[torch.Tensor]] = {}
    for idx, b in enumerate(bin_indices):
        by_bin.setdefault(b, []).append(embeddings[idx])

    centroids = {b: torch.stack(v).mean(dim=0) for b, v in by_bin.items()}
    distances: dict[int, list[float]] = {b: [] for b in centroids}
    for idx, b in enumerate(bin_indices):
        dist = torch.norm(embeddings[idx] - centroids[b], p=2).item()
        distances[b].append(dist)
    return distances


def save_thresholds(thresholds: dict[int, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in thresholds.items()}
    path.write_text(json.dumps(serializable, indent=2))


def load_thresholds(path: Path) -> dict[int, dict[str, float]]:
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}
