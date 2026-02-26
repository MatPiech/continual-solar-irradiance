"""FL Benchmark: Federated Learning as a Spatial Regularizer.

Methodology
-----------
This benchmark evaluates Federated Learning algorithms on two orthogonal axes:

  Source Retention  - Can FL prevent the source-site model from overfitting to
                      daily stochastic noise and *degrading* over time?
  Target Plasticity - Can FL accelerate a model's adaptation to a new geographic
                      site that it has never seen before?

Client Setup
------------
  All clients are treated identically: same adaptation config, same training
  loop.  The first dataset passed is labelled "source" in the output for
  evaluation purposes only.  Optionally, the source client can be given
  different adaptation hyperparameters (--source-adapt-lr / --source-adapt-epochs)
  to model a scenario where the source site has a different training regime.

FL Round Schedule (Decoupled from Daily Updates)
------------------------------------------------
Clients train locally for `--sync-interval` days (default: 7), then a single
FL aggregation round fires.  This strips out accumulated daily noise before it
compounds into catastrophic forgetting.

Algorithms
----------
  isolated  No federation — lower bound, proves source degrades and targets lag.
  fedavg    Standard FedAvg: average all weights every sync interval.
  fedbn     Aggregate all layers *except* BatchNorm statistics, preserving
            per-site covariate normalization.
  fedper    Aggregate encoder only; each client's network_head stays local
            (personalised regression mapping).

Usage
-----
  python scripts/fl_benchmark.py \\
      --checkpoint checkpoints/epoch_007.ckpt \\
      --source-dataset data/folsom_prepared \\
      --target-dataset data/karpicko_prepared \\
      --target-dataset data/poznan_prepared \\
      --algorithm fedbn \\
      --sync-interval 7 \\
      --num-days 90 \\
      --save-results results/fedbn_7day
"""

import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import click
import numpy as np
import pandas as pd
import torch

from cirr.datamodules.forecasting import ForecastingDataModule
from cirr.fl.server import AGGREGATION_FNS, _is_bn_key, _is_head_key
from cirr.models.forecaster import Forecaster
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_test_dataset(dataset_path: Path) -> Any:
    """Load the test split of a prepared dataset."""
    dm = ForecastingDataModule(
        root_data_path=dataset_path,
        periods_path=dataset_path / "periods.pickle",
        augment=False,
        image_size=[128, 128],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        batch_size=1,
        workers=0,
        seed=42,
    )
    dm.setup(None)
    return dm._test_dataset


def _load_checkpoint(checkpoint_path: Path, device: str) -> Forecaster:
    """Instantiate a Forecaster and load weights from *checkpoint_path*."""
    data = torch.load(checkpoint_path, map_location=device)
    hparams = data.get("hyper_parameters", {})
    model = Forecaster(
        model_name=hparams.get("model_name", "timm-resnet50"),
        pretrained=False,
        input_channels=hparams.get("input_channels", 3),
        loss_function=hparams.get("loss_function", "Berhu"),
        lr=hparams.get("lr", 1e-3),
        lr_patience=hparams.get("lr_patience", 5),
    ).to(device)
    model.load_state_dict(data["state_dict"])
    return model


# ---------------------------------------------------------------------------
# Per-day inference / training
# ---------------------------------------------------------------------------


def _predict_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference for every sample in *day_indices*.

    Returns
    -------
    predictions : np.ndarray, shape (N,)
    targets     : np.ndarray, shape (N,)
    """
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for idx in day_indices:
            images, irr_hist, target, _ = dataset[idx]
            images = images.unsqueeze(0).to(device)
            irr_hist = irr_hist.unsqueeze(0).to(device)
            mean, _ = model(images, irr_hist)
            preds.append(mean.cpu().item())
            tgts.append(target.cpu().item())
    return np.array(preds), np.array(tgts)


def _train_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    batch_size: int,
    epochs: int,
    device: str,
    replay_entries: list[tuple[Any, int]] | None = None,
) -> float:
    """Train *model* for *epochs* epochs on *day_indices*.

    *replay_entries* is a pre-sampled list of ``(dataset, index)`` pairs
    (assembled by the caller via ``_sample_replay_for_day()``) that is
    concatenated with the current-day samples before batching — matching
    the design of ``train_on_day`` in ``local_training.py``.
    The pool is reshuffled at the start of every epoch.

    Returns
    -------
    avg_loss : float
    """
    if not day_indices:
        return 0.0

    current_samples: list[tuple[Any, int]] = [(dataset, idx) for idx in day_indices]
    # Combine current-day samples with pre-sampled replay entries.
    # Sizing and seeding are the caller's responsibility (mirrors local_training.py).
    pool_base: list[tuple[Any, int]] = current_samples + (replay_entries or [])

    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_loss = 0.0
    count = 0

    for _ in range(epochs):
        # Reshuffle the fixed pool each epoch (replay+current already sized by caller)
        pool = list(pool_base)
        random.shuffle(pool)
        batches = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]

        for batch in batches:
            imgs = torch.stack([ds[idx][0] for ds, idx in batch]).to(device)
            hist = torch.stack([ds[idx][1] for ds, idx in batch]).to(device)
            tgts = torch.stack([ds[idx][2] for ds, idx in batch]).to(device)

            optimizer.zero_grad()
            mean, var = model(imgs, hist)
            if var is not None:
                loss = model.loss_fn(mean, tgts, var)
            elif model.loss_name == "Berhu":
                loss = model.loss_fn(mean * 1000, tgts * 1000)
            else:
                loss = model.loss_fn(mean, tgts)

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            count += 1

    return total_loss / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Folsom replay buffer — static buffer sampled from Folsom training distribution
# ---------------------------------------------------------------------------


def _period_target_value(period: dict[str, Any]) -> float:
    """Get normalised target value directly from period metadata."""
    return float(period["target_irradiance"]) / 1000.0


def _load_train_dataset(dataset_path: Path) -> Any:
    """Load the training split of a prepared dataset."""
    dm = ForecastingDataModule(
        root_data_path=dataset_path,
        periods_path=dataset_path / "train_periods.pickle",
        augment=False,
        image_size=[128, 128],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        batch_size=1,
        workers=0,
        seed=42,
    )
    dm.setup(None)
    return dm._train_dataset


def _build_initial_replay_buffer(
    folsom_path: Path,
    replay_size: int,
    nonzero_threshold: float = 0.001,
) -> tuple[list[tuple[Any, int]], dict[str, float]]:
    """Build a static replay buffer from the Folsom training dataset.

    Samples are allocated proportionally to a normal distribution fitted to
    Folsom irradiance values, so the buffer distribution matches the Folsom
    training distribution (mirrors ``local_training.py``).

    Parameters
    ----------
    folsom_path : Path
        Path to the ``folsom_prepared`` directory.
    replay_size : int
        Number of entries to include.  0 returns an empty buffer.
    nonzero_threshold : float
        Minimum normalised irradiance for a sample to be eligible.

    Returns
    -------
    replay_buffer : list[tuple[Any, int]]
        ``(dataset, index)`` pairs ready for ``_train_day``.
    distribution_stats : dict[str, float]
        ``mean``, ``std``, ``max`` of the Folsom irradiance values.
    """
    if replay_size <= 0 or not folsom_path.exists():
        return [], {"mean": 0.0, "std": 0.0, "max": 0.0}

    folsom_dataset = _load_train_dataset(folsom_path)

    irradiance_values: list[float] = []
    eligible_samples: list[tuple[Any, int]] = []
    for idx, period in enumerate(folsom_dataset._periods):
        val = _period_target_value(period)
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

    # Fill any remaining slots
    if len(replay_buffer) < replay_size:
        already = {id(s) for s in replay_buffer}
        remaining = [s for s in eligible_samples if id(s) not in already]
        replay_buffer.extend(remaining[: replay_size - len(replay_buffer)])

    random.Random(42).shuffle(replay_buffer)
    return replay_buffer[:replay_size], {"mean": mean_irr, "std": std_irr, "max": max_irr}


def _sample_replay_for_day(
    replay_buffer: list[tuple[Any, int]],
    num_samples: int,
    seed: int | None = None,
) -> list[tuple[Any, int]]:
    """Draw *num_samples* entries from *replay_buffer* without replacement.

    Uses ``numpy.random.RandomState`` for deterministic per-day sampling
    (mirrors ``local_training.py``).
    """
    if num_samples <= 0 or not replay_buffer:
        return []
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    chosen = rng.choice(len(replay_buffer), size=min(num_samples, len(replay_buffer)), replace=False)
    return [replay_buffer[i] for i in chosen]


class RecentBuffer:
    """Fixed-capacity FIFO buffer retaining the *most recent* N samples.

    Unlike the Folsom reservoir buffer (static, normal-weighted), this buffer
    keeps only the latest ``capacity`` samples.  Older entries are dropped when
    the buffer is full.  This gives the model access to very recent context
    without the recency bias of reservoir sampling.

    Each entry is a ``(dataset, index)`` pair — same format as
    the Folsom replay buffer and ``_train_day``.

    Parameters
    ----------
    capacity : int
        Maximum number of entries to retain.  0 disables the buffer.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._entries: list[tuple[Any, int]] = []

    def add_day(self, dataset: Any, indices: list[int]) -> None:
        """Append today's samples; trim to capacity (keep most recent)."""
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


# ---------------------------------------------------------------------------
# Per-client BN / head state tracking for FedBN and FedPer
# ---------------------------------------------------------------------------


def _extract_local_keys(state_dict: dict[str, Any], key_filter) -> dict[str, Any]:
    """Return a sub-dict of *state_dict* for keys matching *key_filter*."""
    return {k: v.clone() for k, v in state_dict.items() if key_filter(k)}


def _restore_local_keys(
    aggregated: dict[str, Any],
    local_state: dict[str, Any],
) -> dict[str, Any]:
    """Overwrite *aggregated* with *local_state* entries (in-place copy)."""
    merged = copy.deepcopy(aggregated)
    for k, v in local_state.items():
        merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Warmup cache helpers
# ---------------------------------------------------------------------------


def _compute_warmup_cache_key(
    checkpoint_path: Path,
    clients: list[tuple[str, Path]],
    warmup_days: int,
    start_day: str,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    seed: int,
    replay_buffer_size: int = 0,
    recent_buffer_size: int = 0,
) -> str:
    """Compute a short hex digest that uniquely identifies a warmup configuration.

    *algorithm* is intentionally excluded: the warmup phase is algorithm-agnostic
    (all methods do identical isolated local training), so the resulting per-client
    weights are the same regardless of which FL algorithm follows.

    *replay_buffer_size* and *recent_buffer_size* are included because the buffer
    states (which are cached alongside model weights) depend on their capacity.
    """
    config = {
        "checkpoint": str(checkpoint_path),
        "clients": [(name, str(path)) for name, path in clients],
        "warmup_days": warmup_days,
        "start_day": start_day,
        "adapt_lr": adapt_lr,
        "adapt_weight_decay": adapt_weight_decay,
        "adapt_max_grad_norm": adapt_max_grad_norm,
        "adapt_batch_size": adapt_batch_size,
        "adapt_epochs": adapt_epochs,
        "seed": seed,
        "replay_buffer_size": replay_buffer_size,
        "recent_buffer_size": recent_buffer_size,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_fl_benchmark(
    checkpoint_path: Path,
    clients: list[tuple[str, Path]],
    algorithm: str,
    sync_interval: int,
    num_days: int,
    start_day: str,
    warmup_days: int,
    warmup_cache_dir: Path | None,
    replay_buffer_size: int,
    replay_ratio: float,
    recent_buffer_size: int,
    recent_replay_ratio: float,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Run the N-client FL benchmark.

    All clients use identical adaptation config.

    Parameters
    ----------
    checkpoint_path     : Path to initial pretrained checkpoint (all clients start here).
    clients             : Ordered list of (name, dataset_path) tuples.
    algorithm           : One of "isolated", "fedavg", "fedbn", "fedper".
    sync_interval       : Days between FL aggregation rounds (e.g. 7 or 14).
    num_days            : Total days to simulate (-1 = all).
    start_day           : MMDD string (e.g. "0101").
    warmup_days         : Number of initial days where all methods act as isolated
                          (no federation).  At the end of the warmup period a single
                          forced aggregation round fires before resuming the normal
                          sync_interval schedule.  0 = no warmup.
    warmup_cache_dir    : Optional directory for caching per-client model states at
                          end of warmup.  The cache key is derived from all warmup-
                          relevant hyperparameters (checkpoint, datasets, warmup_days,
                          lr, …) but NOT the algorithm, so the same snapshot is reused
                          across different algorithm runs with identical warmup configs.
                          Pass None to disable caching.
    replay_buffer_size  : Number of samples to retain in the static Folsom replay
                          buffer.  The buffer is built once from the Folsom training
                          split using stratified normal-weighted sampling so that its
                          irradiance distribution matches the Folsom distribution.
                          0 = disabled.
    replay_ratio        : Fraction of current-day sample count to draw from the Folsom
                          replay buffer each day.  E.g. 0.3 adds samples equal to 30 %
                          of the current day's count.  Ignored when replay_buffer_size
                          is 0.
    recent_buffer_size  : Maximum number of past samples retained in each client's
                          recent FIFO buffer.  0 = disabled.  This buffer keeps only
                          the most recent samples (as opposed to the reservoir buffer
                          which samples uniformly over history).
    recent_replay_ratio : Fraction of current-day sample count to draw from the
                          recent FIFO buffer.  Ignored when recent_buffer_size is 0.
    adapt_lr            : Local learning rate (all clients).
    adapt_weight_decay  : AdamW weight decay.
    adapt_max_grad_norm : Gradient clipping (-1 = disabled).
    adapt_batch_size    : Mini-batch size.
    adapt_epochs        : Local epochs per day (all clients).
    device              : "cuda" or "cpu".
    seed                : RNG seed.

    Returns
    -------
    results : dict with keys:
        "algorithm", "sync_interval", "client_names",
        "per_day"         - list[dict] with per-day per-client metrics,
        "per_round"       - list[dict] with per-FL-round aggregation stats,
        "overall_metrics" - dict[client_name -> metric_dict].
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(clients) < 2:
        raise ValueError("At least two clients are required.")

    algorithm = algorithm.lower()
    if algorithm not in ("isolated", *AGGREGATION_FNS.keys()):
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: isolated, {', '.join(AGGREGATION_FNS)}")

    client_names = [name for name, _ in clients]

    print(f"\n{'='*70}")
    print(f"  FL Benchmark — algorithm: {algorithm.upper()}  sync_interval: {sync_interval} days  warmup: {warmup_days} days")
    print(f"{'='*70}")
    for name, path in clients:
        print(f"  {name:20s}  lr={adapt_lr}  epochs={adapt_epochs}  path={path}")
    print()

    # --- Load datasets ---
    print("Loading datasets...")
    datasets = []
    day_maps = []
    for name, path in clients:
        ds = _load_test_dataset(path)
        dm = group_indices_by_day(ds._periods)
        datasets.append(ds)
        day_maps.append(dm)
        print(f"  {name}: {len(ds._periods)} periods")

    # --- Load models (all clients start from the same checkpoint) ---
    print(f"\nLoading checkpoint: {checkpoint_path}")
    models = [_load_checkpoint(checkpoint_path, device) for _ in clients]
    print(f"  All {len(clients)} clients initialised from the same checkpoint.\n")

    # --- Day schedule ---
    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day '{start_day}'. Expected MMDD format.")
    all_days = all_days[all_days.index(start_day) :]
    if num_days > 0:
        all_days = all_days[:num_days]
    print(f"Simulating {len(all_days)} days starting {all_days[0]} → {all_days[-1]}\n")

    # --- Progress table header ---
    _col_w = 8 + 2 + 8 + 2 + 5  # MAE(8) + 2sp + RMSE(8) + 2sp + N(5) = 25
    _col_w = max(_col_w, max(len(n) for n in client_names))
    _header_names = "  ".join(f"{n:^{_col_w}s}" for n in client_names)
    _header_metrics = "  ".join(f"{'MAE':>8}  {'RMSE':>8}  {'N':>5}" for _ in client_names)
    _sep = "  ".join(["-" * _col_w] * len(client_names))
    print(f"  {'Day':<6}  {_header_names}")
    print(f"  {'':6}  {_header_metrics}")
    print(f"  {'------'}  {_sep}")

    # --- Per-client state for FedBN / FedPer ---
    local_bn: list[dict[str, Any]] = [{} for _ in clients]
    local_head: list[dict[str, Any]] = [{} for _ in clients]

    # --- Per-client recent FIFO buffers ---
    recent_buffers: list[RecentBuffer] = [RecentBuffer(recent_buffer_size) for _ in clients]

    # --- Folsom replay buffer (static, init once from Folsom training split) ---
    # Distribution matches Folsom irradiance (stratified normal-weighted sampling).
    _folsom_replay_buffer: list[tuple[Any, int]] = []
    if replay_buffer_size > 0:
        _folsom_path = clients[0][1].parent / "folsom_prepared"
        _folsom_replay_buffer, _folsom_dist = _build_initial_replay_buffer(_folsom_path, replay_buffer_size)
        if _folsom_replay_buffer:
            print(
                f"  Folsom replay buffer: {len(_folsom_replay_buffer)} samples  "
                f"μ={_folsom_dist['mean']:.3f}  σ={_folsom_dist['std']:.3f}  max={_folsom_dist['max']:.3f}"  # noqa: RUF001
            )
        else:
            print(f"  Folsom replay buffer: DISABLED (path not found: {_folsom_path})")

    per_day: list[dict[str, Any]] = []
    per_round: list[dict[str, Any]] = []
    days_since_sync = 0
    fl_round_idx = 0
    last_global_state_dict: dict[str, Any] | None = None  # updated after every FL round
    in_warmup = warmup_days > 0 and algorithm != "isolated"

    # Accumulators for final overall metrics (avoid a second inference pass)
    _all_preds: list[list[np.ndarray]] = [[] for _ in clients]
    _all_tgts: list[list[np.ndarray]] = [[] for _ in clients]

    # -----------------------------------------------------------------------
    # Warmup cache — load if available, skipping the warmup simulation
    # -----------------------------------------------------------------------
    _warmup_cache_loaded = False
    _warmup_cache_key: str | None = None
    _warmup_cache_file: Path | None = None

    _use_cache = warmup_days > 0 and algorithm != "isolated" and warmup_cache_dir is not None
    if _use_cache:
        _warmup_cache_key = _compute_warmup_cache_key(
            checkpoint_path,
            clients,
            warmup_days,
            start_day,
            adapt_lr,
            adapt_weight_decay,
            adapt_max_grad_norm,
            adapt_batch_size,
            adapt_epochs,
            seed,
            replay_buffer_size=replay_buffer_size,
            recent_buffer_size=recent_buffer_size,
        )
        _warmup_cache_file = warmup_cache_dir / f"{_warmup_cache_key}.pt"

        if _warmup_cache_file.exists():
            print(f"  Warmup cache HIT  ({_warmup_cache_file})")
            print(f"  Skipping {warmup_days} days of isolated warmup training.\n")
            _cache = torch.load(_warmup_cache_file, map_location=device, weights_only=False)
            for i, model in enumerate(models):
                model.load_state_dict(_cache["client_states"][i])
            per_day = _cache["per_day"]
            _all_preds = _cache["all_preds"]
            _all_tgts = _cache["all_tgts"]
            # Restore recent FIFO buffer states
            if "recent_states" in _cache:
                for i, buf in enumerate(recent_buffers):
                    buf._entries = _cache["recent_states"][i]["entries"]
            # Folsom replay buffer is rebuilt from Folsom at startup — no restore needed.
            # Keep in_warmup=True so that warmup_ended_today fires on the
            # first iteration (day_num == warmup_days), triggering the forced
            # aggregation round naturally — identical to the non-cached path.
            _warmup_cache_loaded = True
        else:
            print(f"  Warmup cache MISS (key={_warmup_cache_key})")
            print(f"  Will save snapshot to {_warmup_cache_file} after warmup ends.\n")

    # When cache was loaded, skip warmup days; day_num starts at warmup_days
    # so the existing warmup_ended_today logic fires on the first iteration.
    _loop_start = warmup_days if _warmup_cache_loaded else 0

    # -----------------------------------------------------------------------
    # Main loop — identical treatment for every client
    # -----------------------------------------------------------------------
    for day_num, day_key in enumerate(all_days[_loop_start:], start=_loop_start):
        # Check whether warmup just ended (transition on this day)
        warmup_ended_today = in_warmup and day_num >= warmup_days
        if warmup_ended_today:
            in_warmup = False

        day_record: dict[str, Any] = {"day": day_key, "day_num": day_num, "fl_round_after": None, "in_warmup": in_warmup}
        losses: list[float] = []

        for i, (name, model) in enumerate(zip(client_names, models, strict=False)):
            indices = day_maps[i].get(day_key, [])
            day_record[f"{name}_n"] = len(indices)

            # Step 1: Predict
            preds, tgts = _predict_day(model, datasets[i], indices, device)
            metrics = compute_metrics(preds, tgts) if len(preds) > 0 else {}
            day_record[f"{name}_mae"] = metrics.get("mae", float("nan"))
            day_record[f"{name}_rmse"] = metrics.get("rmse", float("nan"))
            if len(preds) > 0:
                _all_preds[i].append(preds)
                _all_tgts[i].append(tgts)

            # Step 2: Train — same loop for all clients
            # Sample from Folsom replay buffer and recent FIFO buffer before training.
            # n = round(len(indices) * ratio) mirrors _sample_replay_for_day in local_training.py.
            _day_client_seed = seed + day_num * len(clients) + i
            _reservoir_entries: list[tuple[Any, int]] = []
            if replay_buffer_size > 0 and replay_ratio > 0.0 and _folsom_replay_buffer:
                _n_reservoir = max(0, int(round(len(indices) * replay_ratio)))
                _reservoir_entries = _sample_replay_for_day(_folsom_replay_buffer, _n_reservoir, seed=_day_client_seed)
            _recent_entries: list[tuple[Any, int]] = []
            if recent_buffer_size > 0 and recent_replay_ratio > 0.0:
                _n_recent = max(0, int(round(len(indices) * recent_replay_ratio)))
                _recent_entries = recent_buffers[i].sample(_n_recent, seed=_day_client_seed + 1)
            _replay = (_reservoir_entries + _recent_entries) or None
            loss = _train_day(
                model,
                datasets[i],
                indices,
                lr=adapt_lr,
                weight_decay=adapt_weight_decay,
                max_grad_norm=adapt_max_grad_norm,
                batch_size=adapt_batch_size,
                epochs=adapt_epochs,
                device=device,
                replay_entries=_replay,
            )
            losses.append(loss)

            # Update recent FIFO buffer with today's samples
            if recent_buffer_size > 0:
                recent_buffers[i].add_day(datasets[i], indices)

        per_day.append(day_record)
        # Only count toward sync_interval once warmup is over
        if not in_warmup:
            days_since_sync += 1

        # Print day row (always)
        _warmup_tag = " [W]" if day_record["in_warmup"] else "    "

        def _fmt_client_col(n: str, _rec: dict = day_record) -> str:
            n_samples = _rec.get(f"{n}_n", 0)
            if n_samples == 0:
                return f"{'---':>8}  {'---':>8}  {0:>5}"
            return (
                f"{_rec.get(f'{n}_mae', float('nan')):>8.6f}"
                f"  {_rec.get(f'{n}_rmse', float('nan')):>8.6f}"
                f"  {n_samples:>5}"
            )

        _row = "  ".join(_fmt_client_col(n) for n in client_names)
        print(f"  {day_key:<6}{_warmup_tag}  {_row}")

        # ---------------------------------------------------------------
        # Step 3: FL Aggregation
        #   - forced round immediately after warmup ends
        #   - periodic round every sync_interval days (post-warmup)
        # ---------------------------------------------------------------
        _do_aggregate = algorithm != "isolated" and (warmup_ended_today or days_since_sync >= sync_interval)
        if warmup_ended_today and not in_warmup:
            # Reset counter so the next periodic sync starts fresh
            days_since_sync = 0

        if _do_aggregate:
            agg_fn = AGGREGATION_FNS[algorithm]
            state_dicts = [m.state_dict() for m in models]

            # -----------------------------------------------------------------
            # Warmup cache — save pre-aggregation client states so future runs
            # with the same warmup config can skip straight to this point.
            # Saved before aggregation: weights are purely from isolated training
            # and contain no algorithm-specific bias.
            # -----------------------------------------------------------------
            if warmup_ended_today and _use_cache and not _warmup_cache_loaded:
                assert _warmup_cache_file is not None
                _warmup_cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "cache_key": _warmup_cache_key,
                        "warmup_days": warmup_days,
                        "client_names": client_names,
                        "client_states": list(state_dicts),
                        "per_day": list(per_day),
                        "all_preds": [[a.copy() for a in preds] for preds in _all_preds],
                        "all_tgts": [[a.copy() for a in tgts] for tgts in _all_tgts],
                        "recent_states": [{"entries": list(buf._entries)} for buf in recent_buffers],
                    },
                    _warmup_cache_file,
                )
                print(f"  {'':6}      [cache] Warmup snapshot saved → {_warmup_cache_file}")

            # Capture local-only layers before aggregation
            if algorithm == "fedbn":
                local_bn = [_extract_local_keys(sd, _is_bn_key) for sd in state_dicts]
            if algorithm == "fedper":
                local_head = [_extract_local_keys(sd, _is_head_key) for sd in state_dicts]

            new_global, wc = agg_fn(state_dicts[0], state_dicts)
            last_global_state_dict = copy.deepcopy(new_global)

            # Distribute back — restore local layers per client
            for i, model in enumerate(models):
                merged = new_global
                if algorithm == "fedbn":
                    merged = _restore_local_keys(merged, local_bn[i])
                elif algorithm == "fedper":
                    merged = _restore_local_keys(merged, local_head[i])
                model.load_state_dict(merged)

            round_tag = "[WARMUP END] " if warmup_ended_today else ""
            per_day[-1]["fl_round_after"] = fl_round_idx
            per_round.append(
                {
                    "fl_round": fl_round_idx,
                    "day": day_key,
                    "day_num": day_num,
                    "weight_change": wc,
                    "days_in_round": days_since_sync if not warmup_ended_today else warmup_days,
                    "warmup_round": warmup_ended_today,
                }
            )
            if not warmup_ended_today:
                days_since_sync = 0
            fl_round_idx += 1

            _loss_str = "  ".join(
                f"{_name}={_loss:.5f}"
                for _name, _loss in zip(client_names, losses, strict=False)
                if day_record.get(f"{_name}_n", 0) > 0
            )
            print(
                f"  {'':6}      └─ {round_tag}FL Round {fl_round_idx:3d}  ({per_round[-1]['days_in_round']}d)  ΔW={wc:.5f}  losses: {_loss_str}"
            )

    # -----------------------------------------------------------------------
    # Overall metrics — built from accumulated predictions, no re-inference
    # -----------------------------------------------------------------------
    overall_metrics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(client_names):
        if _all_preds[i]:
            overall_metrics[name] = compute_metrics(np.concatenate(_all_preds[i]), np.concatenate(_all_tgts[i]))
        else:
            overall_metrics[name] = {}

    _nw = max(10, max(len(n) for n in client_names))
    print(f"\n{'='*80}")
    print(f"  Final Results — {algorithm.upper()}  (FL rounds fired: {fl_round_idx})")
    print(f"{'='*80}")
    print(f"  {'Client':<{_nw}}  {'MAE':>9}  {'RMSE':>9}  {'MAPE':>8}  {'WAPE':>8}  {'SMAPE':>8}  {'Count':>8}")
    print(f"  {'-'*_nw}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name in client_names:
        m = overall_metrics[name]
        print(
            f"  {name:<{_nw}}  "
            f"{m.get('mae', float('nan')):>9.5f}  "
            f"{m.get('rmse', float('nan')):>9.5f}  "
            f"{m.get('mape', float('nan')):>8.2%}  "
            f"{m.get('wape', float('nan')):>7.2f}%  "
            f"{m.get('smape', float('nan')):>7.2f}%  "
            f"{m.get('count', 0):>8}"
        )
    print(f"{'='*80}\n")

    return {
        "algorithm": algorithm,
        "sync_interval": sync_interval,
        "num_days": len(all_days),
        "start_day": start_day,
        "warmup_days": warmup_days,
        "warmup_cache_loaded": _warmup_cache_loaded,
        "replay_buffer_size": replay_buffer_size,
        "replay_ratio": replay_ratio,
        "recent_buffer_size": recent_buffer_size,
        "recent_replay_ratio": recent_replay_ratio,
        "client_names": client_names,
        "fl_rounds_total": fl_round_idx,
        "per_day": per_day,
        "per_round": per_round,
        "overall_metrics": overall_metrics,
        "models": models,
        "last_global_state_dict": last_global_state_dict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the pretrained model checkpoint (shared starting point for all clients).",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help=(
        "Prepared dataset path. Repeat to add clients "
        "(e.g. --dataset data/folsom_prepared --dataset data/karpicko_prepared). "
        "At least two required. Clients are created in the order given."
    ),
)
@click.option(
    "--algorithm",
    type=click.Choice(["isolated", "fedavg", "fedbn", "fedper"], case_sensitive=False),
    default="fedbn",
    show_default=True,
    help=(
        "FL aggregation algorithm.\n"
        "  isolated — no federation (lower bound).\n"
        "  fedavg   — standard weight averaging.\n"
        "  fedbn    — FedAvg excluding BatchNorm layers.\n"
        "  fedper   — aggregate backbone only, keep head local."
    ),
)
@click.option(
    "--sync-interval",
    type=int,
    default=7,
    show_default=True,
    help="Days between FL aggregation rounds (7 = weekly, 14 = bi-weekly).",
)
@click.option(
    "--num-days",
    type=int,
    default=-1,
    show_default=True,
    help="Number of days to simulate (-1 = all available days).",
)
@click.option(
    "--start-day",
    type=str,
    default="0101",
    show_default=True,
    help="Start day in MMDD format (e.g. 0101 for January 1).",
)
@click.option(
    "--warmup-days",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Number of initial days where all methods train in isolation (no federation). "
        "At the end of the warmup period a single forced aggregation round fires, "
        "then normal sync-interval scheduling resumes.  0 = disabled."
    ),
)
@click.option(
    "--warmup-cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    show_default=True,
    help=(
        "Directory for caching per-client model states at the end of the warmup phase. "
        "The cache key is derived from all warmup-relevant hyperparameters (checkpoint, "
        "datasets, warmup_days, lr, …) but NOT the algorithm, so the same snapshot is "
        "reused across all algorithm runs that share the same warmup config.  "
        "If the cache file already exists the warmup phase is skipped entirely.  "
        "Pass None (default) to disable caching."
    ),
)
@click.option(
    "--replay-buffer-size",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Number of samples to draw from the Folsom training dataset for the static "
        "replay buffer.  Samples are allocated with stratified normal-weighted sampling "
        "so the buffer distribution matches the Folsom irradiance distribution.  "
        "0 = disabled."
    ),
)
@click.option(
    "--replay-ratio",
    type=float,
    default=0.3,
    show_default=True,
    help=(
        "Ratio of Folsom replay samples relative to current-day samples.  "
        "E.g. 0.1 adds replay samples equal to 10 %% of the current day's count.  "
        "Only used when --replay-buffer-size > 0."
    ),
)
@click.option(
    "--recent-buffer-size",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Capacity of each client's recent FIFO replay buffer (number of most-recent "
        "samples retained).  0 = disabled.  Complements the reservoir buffer by giving "
        "the model access to very recent context."
    ),
)
@click.option(
    "--recent-replay-ratio",
    type=float,
    default=0.1,
    show_default=True,
    help=(
        "Ratio of recent-buffer replay samples relative to current-day samples.  "
        "E.g. 0.1 adds samples equal to 10 %% of the current day's count from the "
        "recent FIFO buffer.  Only used when --recent-buffer-size > 0."
    ),
)
@click.option("--adapt-lr", type=float, default=1e-4, show_default=True, help="Local learning rate (all clients).")
@click.option("--adapt-weight-decay", type=float, default=1e-3, show_default=True, help="AdamW weight decay.")
@click.option(
    "--adapt-max-grad-norm",
    type=float,
    default=1.0,
    show_default=True,
    help="Gradient clipping max norm (-1 = disabled).",
)
@click.option("--adapt-batch-size", type=int, default=32, show_default=True, help="Mini-batch size for local training.")
@click.option(
    "--adapt-epochs",
    type=int,
    default=1,
    show_default=True,
    help="Local training epochs per day (all clients).",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    show_default=True,
    help="Compute device.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option(
    "--save-results",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory to save results.  Creates:\n"
        "  <dir>/per_day.csv     — daily MAE/RMSE for all clients.\n"
        "  <dir>/per_round.csv   — per-FL-round statistics.\n"
        "  <dir>/summary.json    — overall metrics and config."
    ),
)
def main(
    checkpoint: Path,
    dataset: tuple[Path, ...],
    algorithm: str,
    sync_interval: int,
    num_days: int,
    start_day: str,
    warmup_days: int,
    warmup_cache_dir: Path | None,
    replay_buffer_size: int,
    replay_ratio: float,
    recent_buffer_size: int,
    recent_replay_ratio: float,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    device: str,
    seed: int,
    save_results: Path | None,
) -> None:
    """Run the FL benchmark comparing federation algorithms on solar irradiance forecasting."""
    # Build ordered client list from all --dataset flags
    # Strip common suffixes (_prepared) to get a clean site name
    clients: list[tuple[str, Path]] = []
    for ds in dataset:
        raw = ds.parent.name if ds.name == "." else ds.name
        name = raw.removesuffix("_prepared")
        clients.append((name, ds))

    results = run_fl_benchmark(
        checkpoint_path=checkpoint,
        clients=clients,
        algorithm=algorithm,
        sync_interval=sync_interval,
        num_days=num_days,
        start_day=start_day,
        warmup_days=warmup_days,
        warmup_cache_dir=warmup_cache_dir,
        replay_buffer_size=replay_buffer_size,
        replay_ratio=replay_ratio,
        recent_buffer_size=recent_buffer_size,
        recent_replay_ratio=recent_replay_ratio,
        adapt_lr=adapt_lr,
        adapt_weight_decay=adapt_weight_decay,
        adapt_max_grad_norm=adapt_max_grad_norm,
        adapt_batch_size=adapt_batch_size,
        adapt_epochs=adapt_epochs,
        device=device,
        seed=seed,
    )

    if save_results is not None:
        save_results.mkdir(parents=True, exist_ok=True)

        # Per-day CSV
        df_day = pd.DataFrame(results["per_day"])
        df_day.to_csv(save_results / "per_day.csv", index=False)

        # Per-round CSV
        if results["per_round"]:
            df_round = pd.DataFrame(results["per_round"])
            df_round.to_csv(save_results / "per_round.csv", index=False)

        # Per-client model checkpoints (last model state after simulation)
        ckpt_dir = save_results / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        _final_models: list[Forecaster] = results["models"]
        for _name, _model in zip(results["client_names"], _final_models, strict=False):
            _ckpt_path = ckpt_dir / f"{_name}_final.pt"
            torch.save({"state_dict": _model.state_dict(), "algorithm": algorithm}, _ckpt_path)
            print(f"  {_ckpt_path}")
        # Save the last global model broadcast (None for isolated — no aggregation)
        _global_sd = results["last_global_state_dict"]
        if _global_sd is not None:
            _global_path = ckpt_dir / "global_final.pt"
            torch.save({"state_dict": _global_sd, "algorithm": algorithm}, _global_path)
            print(f"  {_global_path}")

        # Summary JSON (exclude large per_day / per_round lists and non-serialisable models)
        summary = {k: v for k, v in results.items() if k not in ("per_day", "per_round", "models", "last_global_state_dict")}
        summary["config"] = {
            "checkpoint": str(checkpoint),
            "clients": [(name, str(path)) for name, path in clients],
            "warmup_days": warmup_days,
            "warmup_cache_dir": str(warmup_cache_dir) if warmup_cache_dir else None,
            "warmup_cache_loaded": results["warmup_cache_loaded"],
            "replay_buffer_size": replay_buffer_size,
            "replay_ratio": replay_ratio,
            "recent_buffer_size": recent_buffer_size,
            "recent_replay_ratio": recent_replay_ratio,
            "adapt_lr": adapt_lr,
            "adapt_epochs": adapt_epochs,
            "adapt_weight_decay": adapt_weight_decay,
            "adapt_max_grad_norm": adapt_max_grad_norm,
            "adapt_batch_size": adapt_batch_size,
            "seed": seed,
        }
        with (save_results / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"Results saved to {save_results}/")
        print(f"  per_day.csv   ({len(results['per_day'])} rows)")
        if results["per_round"]:
            print(f"  per_round.csv ({len(results['per_round'])} rows)")
        print("  summary.json")
        _n_global = 1 if results["last_global_state_dict"] is not None else 0
        print(f"  checkpoints/  ({len(_final_models)} local models{', 1 global' if _n_global else ''})")


if __name__ == "__main__":
    main()
