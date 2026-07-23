"""Replay-buffer utilities shared by continual-learning and FL scripts.

Both ``local_training.py`` and ``fl_benchmark.py`` maintain a static Folsom
replay buffer for experience replay.  This module provides the canonical
implementations that both scripts import instead of maintaining separate copies.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np

from cirr.data.loading import load_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def period_target_value(period: dict[str, Any]) -> float:
    """Return the normalised target irradiance for a period (divide by 1000).

    Parameters
    ----------
    period : dict
        A period dictionary from a ``SkyImageDataset.periods`` list.

    Returns
    -------
    float
        ``target_irradiance / 1000``.
    """
    return float(period["target_irradiance"]) / 1000.0


def build_folsom_replay_buffer(
    folsom_path: Path,
    replay_size: int,
    nonzero_threshold: float = 0.001,
) -> tuple[list[tuple[Any, int]], dict[str, float]]:
    """Build a static replay buffer from the Folsom training dataset.

    Samples are allocated proportionally to a normal distribution fitted to
    Folsom irradiance values so that the buffer distribution mirrors the source
    training distribution.

    Parameters
    ----------
    folsom_path : Path
        Path to the ``folsom_prepared`` directory.
    replay_size : int
        Number of entries to include.  Return an empty buffer when ``<= 0``
        or when *folsom_path* does not exist.
    nonzero_threshold : float
        Minimum normalised irradiance for a sample to be eligible.

    Returns
    -------
    replay_buffer : list[tuple[dataset, int]]
        ``(dataset, index)`` pairs ready for per-day sampling / training.
    distribution_stats : dict[str, float]
        ``mean``, ``std``, ``max`` of the Folsom irradiance values used.
    """
    if replay_size <= 0 or not folsom_path.exists():
        return [], {"mean": 0.0, "std": 0.0, "max": 0.0}

    folsom_dataset = load_dataset(folsom_path, split="training")

    irradiance_values: list[float] = []
    eligible_samples: list[tuple[Any, int]] = []
    for idx, period in enumerate(folsom_dataset.periods):
        val = period_target_value(period)
        if val < nonzero_threshold:
            continue
        irradiance_values.append(val)
        eligible_samples.append((folsom_dataset, idx))

    if not eligible_samples:
        return [], {"mean": 0.0, "std": 0.0, "max": 0.0}

    irr_array = np.array(irradiance_values)
    mean_irr = float(np.mean(irr_array))
    std_irr = float(np.std(irr_array))
    max_irr = float(np.max(irr_array))

    # Stratified sampling: bin by irradiance, weight bins by fitted normal
    n_bins = min(10, max(2, replay_size // 20))
    bin_edges = np.linspace(0, max_irr, n_bins + 1)
    bins: dict[int, list[tuple[Any, int]]] = {i: [] for i in range(n_bins)}
    for sample, irr_val in zip(eligible_samples, irradiance_values, strict=False):
        bin_idx = min(n_bins - 1, int(np.digitize(irr_val, bin_edges)) - 1)
        bins[bin_idx].append(sample)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    normal_weights = np.exp(-0.5 * ((bin_centers - mean_irr) / (std_irr + 1e-6)) ** 2)
    normal_weights /= normal_weights.sum()

    replay_buffer: list[tuple[Any, int]] = []
    for bin_idx in range(n_bins):
        quota = max(0, int(round(replay_size * normal_weights[bin_idx])))
        if quota > 0 and bins[bin_idx]:
            replay_buffer.extend(bins[bin_idx][:quota])

    # Fill any remaining slots in order (avoid quadratic membership check)
    if len(replay_buffer) < replay_size:
        already = {id(s) for s in replay_buffer}
        remaining = [s for s in eligible_samples if id(s) not in already]
        replay_buffer.extend(remaining[: replay_size - len(replay_buffer)])

    random.Random(42).shuffle(replay_buffer)
    return replay_buffer[:replay_size], {"mean": mean_irr, "std": std_irr, "max": max_irr}


def sample_replay_for_day(
    replay_buffer: list[tuple[Any, int]],
    num_samples: int,
    seed: int | None = None,
) -> list[tuple[Any, int]]:
    """Draw *num_samples* entries from *replay_buffer* without replacement.

    Uses a ``numpy.random.RandomState`` for deterministic per-day sampling.

    Parameters
    ----------
    replay_buffer : list
        Full replay buffer containing all candidate ``(dataset, index)`` pairs.
    num_samples : int
        Number of samples to draw.  Returns an empty list when ``<= 0`` or
        when *replay_buffer* is empty.
    seed : int, optional
        Seed for the per-day random state.

    Returns
    -------
    list
        Subset of *replay_buffer* entries.
    """
    if num_samples <= 0 or not replay_buffer:
        return []
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    chosen = rng.choice(len(replay_buffer), size=min(num_samples, len(replay_buffer)), replace=False)
    return [replay_buffer[i] for i in chosen]


# ---------------------------------------------------------------------------
# Recent FIFO buffer
# ---------------------------------------------------------------------------


class RecentBuffer:
    """Fixed-capacity FIFO buffer retaining the *most recent* N samples.

    Unlike the static Folsom reservoir buffer, this buffer keeps only the
    latest ``capacity`` samples.  Older entries are dropped once the buffer
    is full, giving the model access to very recent context.

    Each entry is a ``(dataset, index)`` pair — same format as the Folsom
    replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of entries to retain.  ``0`` disables the buffer.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._entries: list[tuple[Any, int]] = []

    def add_day(self, dataset: Any, indices: list[int]) -> None:
        """Append today's samples; trim to *capacity* (keep most recent)."""
        if self.capacity == 0 or not indices:
            return
        for idx in indices:
            self._entries.append((dataset, idx))
        if len(self._entries) > self.capacity:
            self._entries = self._entries[-self.capacity :]

    def sample(self, n: int, seed: int | None = None) -> list[tuple[Any, int]]:
        """Return up to *n* entries drawn without replacement."""
        if not self._entries or n <= 0:
            return []
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        n = min(n, len(self._entries))
        indices = rng.choice(len(self._entries), size=n, replace=False)
        return [self._entries[i] for i in indices]

    def __len__(self) -> int:
        return len(self._entries)
