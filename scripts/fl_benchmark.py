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
    static    Inference only — no training, no aggregation.
  isolated  No federation — lower bound, proves source degrades and targets lag.
  fedavg    Standard FedAvg: average all weights every sync interval.
  fedbn     Aggregate all layers *except* BatchNorm statistics, preserving
            per-site covariate normalization.
  fedper    Aggregate encoder only; each client's network_head stays local
            (personalised regression mapping).
  fedprox   FedAvg + proximal term (μ/2)‖w-w_global‖² during local training.
  ditto     Dual-model: global participates in FedAvg, personalized model
            trains with proximal regularization toward global.
  fedema    FedAvg + EMA interpolation at distribution:
            w ← α·w_local + (1-α)·w_global.
  fedproto  Federated Prototype Learning: clients exchange per-class feature
            prototypes (Clear/Overcast/Transient) instead of model weights.
            Local loss is regularised toward global consensus prototypes.

Usage
-----
  python scripts/fl_benchmark.py \\
      --checkpoint checkpoints/epoch_007.ckpt \\
      --source-dataset data/folsom_prepared \\
      --target-dataset data/karpicko_prepared \\
      --target-dataset data/poznan_prepared \\
      --algorithm static \
      --sync-interval 7 \\
      --num-days 90 \\
    --save-results results/static_7day
"""

from concurrent.futures import ThreadPoolExecutor
import contextlib
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

from cirr.continual import build_gated_thresholds, train_gated_day
from cirr.data.loading import load_dataset
from cirr.fl.kc_prototypes import (
    PROTO_CLASS_NAMES,
    aggregate_global_prototypes,
    extract_day_prototypes,
)
from cirr.fl.prototypes import accumulate_prototypes, aggregate_prototypes, extract_prototypes
from cirr.fl.server import AGGREGATION_FNS, is_bn_key, is_head_key
from cirr.losses import BerHuLoss
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.training.day_loop import predict_day, train_day
from cirr.training.replay import RecentBuffer, build_folsom_replay_buffer, sample_replay_for_day
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(checkpoint_path: Path, device: str, freeze_early_backbone: bool = False) -> Forecaster:
    """Instantiate a Forecaster and load weights from *checkpoint_path*."""
    model = load_forecaster_from_checkpoint(checkpoint_path, device=device, use_hparams=True)
    if freeze_early_backbone:
        model.freeze_early_backbone()
    return model


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


def _ema_interpolate(
    local_sd: dict[str, Any],
    global_sd: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Soft interpolation: ``w = α · w_local + (1 - α) · w_global``.

    Floating-point tensors are interpolated; integer tensors (e.g. BN
    ``num_batches_tracked``) are kept from the local model.
    """
    merged = {}
    for key in global_sd:
        if global_sd[key].is_floating_point():
            merged[key] = alpha * local_sd[key] + (1.0 - alpha) * global_sd[key]
        else:
            merged[key] = local_sd[key]
    return merged


# ---------------------------------------------------------------------------
# Parallel client processing
# ---------------------------------------------------------------------------


def _process_client_day(
    client_idx: int,
    name: str,
    model: Forecaster,
    infer_model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    device: str,
    rng_seed: int,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    folsom_replay_buffer: list[tuple[Any, int]],
    replay_buffer_size: int,
    replay_ratio: float,
    recent_buffer: RecentBuffer,
    recent_buffer_size: int,
    recent_replay_ratio: float,
    global_state_dict: dict[str, Any] | None,
    prox_mu: float,
    global_prototypes: dict[int, torch.Tensor] | None = None,
    proto_lambda: float = 0.0,
    extract_prototypes: bool = False,
    train: bool = True,
    gated: bool = False,
    gate_prototypes: dict[int, torch.Tensor] | None = None,
    gate_thresholds: dict[int, dict[str, float]] | None = None,
    gate_optimizer: torch.optim.Optimizer | None = None,
    gate_loss_fn: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Process prediction + training for one client on one day.

    Thread-safe: uses per-client CUDA stream and deterministic local RNG.
    Each client operates on its own model, dataset, and buffers — no shared
    mutable state between clients.

    When *extract_prototypes* is True, per-class prototype feature vectors
    are extracted after training and included in the returned dict under
    key ``"day_prototypes"`` (for FedProto accumulation).
    """
    use_cuda = device != "cpu" and torch.cuda.is_available()
    stream = torch.cuda.Stream(device) if use_cuda else None
    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()

    with ctx:
        # Step 1: Predict (also caches per-sample tensors for training)
        preds, tgts, day_samples = predict_day(
            infer_model, dataset, day_indices, device, batch_size=adapt_batch_size, client_name=name
        )
        metrics = compute_metrics(preds, tgts) if len(preds) > 0 else {}

        loss = 0.0
        gate_stats: dict[str, int | float] = {}
        if train:
            if gated and gate_optimizer is not None and gate_prototypes is not None:
                gate_stats = train_gated_day(
                    model,
                    dataset,
                    day_indices,
                    device,
                    gate_optimizer,
                    gate_loss_fn or BerHuLoss(threshold=30, scale=1000),
                    gate_prototypes,
                    gate_thresholds or {},
                )
                loss = float(gate_stats.get("backprop", 0))
            else:
                # Step 2: Prepare replay samples
                _reservoir_entries: list[tuple[Any, int]] = []
                if replay_buffer_size > 0 and replay_ratio > 0.0 and folsom_replay_buffer:
                    _n = max(0, int(round(len(day_indices) * replay_ratio)))
                    _reservoir_entries = sample_replay_for_day(folsom_replay_buffer, _n, seed=rng_seed)
                _recent_entries: list[tuple[Any, int]] = []
                if recent_buffer_size > 0 and recent_replay_ratio > 0.0:
                    _n = max(0, int(round(len(day_indices) * recent_replay_ratio)))
                    _recent_entries = recent_buffer.sample(_n, seed=rng_seed + 1)
                _replay = (_reservoir_entries + _recent_entries) or None

                # Step 3: Train (reuses cached day_samples to skip redundant dataset loads)
                loss = train_day(
                    model,
                    dataset,
                    day_indices,
                    lr=adapt_lr,
                    weight_decay=adapt_weight_decay,
                    max_grad_norm=adapt_max_grad_norm,
                    batch_size=adapt_batch_size,
                    epochs=adapt_epochs,
                    device=device,
                    replay_entries=_replay,
                    global_state_dict=global_state_dict,
                    mu=prox_mu,
                    rng_seed=rng_seed,
                    day_samples=day_samples,
                    client_name=name,
                    global_prototypes=global_prototypes,
                    proto_lambda=proto_lambda,
                )

        # Step 4: Extract per-class prototypes (FedProto — after training
        # so features reflect the updated model)
        day_protos: dict[int, tuple[torch.Tensor, int]] = {}
        if train and extract_prototypes and day_samples:
            day_protos = extract_day_prototypes(model, day_samples, device, adapt_batch_size)

        # Step 5: Update recent FIFO buffer
        if train and recent_buffer_size > 0:
            recent_buffer.add_day(dataset, day_indices)

    if stream is not None:
        stream.synchronize()

    return {
        "client_idx": client_idx,
        "name": name,
        "n_samples": len(day_indices),
        "preds": preds,
        "tgts": tgts,
        "mae": metrics.get("mae", float("nan")),
        "rmse": metrics.get("rmse", float("nan")),
        "loss": loss,
        "day_prototypes": day_protos,
        "gate_stats": gate_stats,
        "gate_opened": bool(gate_stats.get("backprop", 0)),
    }


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
    fedprox_mu: float = 0.0,
    ditto_lambda: float = 0.0,
    fedema_alpha: float = 0.5,
    fedproto_lambda: float = 0.1,
    parallel_clients: int = 1,
    freeze_early_backbone: bool = False,
    loso_holdout: str | None = None,
) -> dict[str, Any]:
    """Run the N-client FL benchmark.

    All clients use identical adaptation config.

    Parameters
    ----------
    checkpoint_path     : Path to initial pretrained checkpoint (all clients start here).
    clients             : Ordered list of (name, dataset_path) tuples.
    algorithm           : One of "isolated", "fedavg", "fedbn", "fedper",
                          "fedprox", "ditto", "fedema", "fedproto".
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
    fedprox_mu          : Proximal regularisation strength for FedProx.
                          Loss is augmented with (μ/2) ‖w - w_global‖².
    ditto_lambda        : Proximal strength for the Ditto personalised model.
                          The global model trains with standard FedAvg; the
                          personal model's loss is augmented with
                          (λ/2) ‖v - w_global‖².
    fedema_alpha        : EMA interpolation coefficient for FedEMA.
                          w_local ← α·w_local + (1-α)·w_global after each
                          aggregation round.  Higher α retains more local
                          knowledge.
    fedproto_lambda     : Prototype regularisation relative weight for FedProto.
                          The prototype MSE term is adaptively scaled to match
                          the task-loss magnitude, so this parameter acts as a
                          relative contribution (0.1 ≈ 10 % of the task loss).
                          Higher values pull the local feature space closer to
                          the global consensus.

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
    if algorithm not in ("static", "isolated", *AGGREGATION_FNS.keys()):
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: static, isolated, {', '.join(AGGREGATION_FNS)}")

    is_static = algorithm == "static"

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
        ds = load_dataset(path, split="test")
        dm = group_indices_by_day(ds.periods)
        datasets.append(ds)
        day_maps.append(dm)
        print(f"  {name}: {len(ds.periods)} periods")

    # --- Load models (all clients start from the same checkpoint) ---
    print(f"\nLoading checkpoint: {checkpoint_path}")
    models = [_load_checkpoint(checkpoint_path, device, freeze_early_backbone) for _ in clients]
    if freeze_early_backbone:
        print(f"  All {len(clients)} clients initialised from the same checkpoint (early backbone frozen).\n")
    else:
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
        _folsom_replay_buffer, _folsom_dist = build_folsom_replay_buffer(_folsom_path, replay_buffer_size)
        if _folsom_replay_buffer:
            print(
                f"  Folsom replay buffer: {len(_folsom_replay_buffer)} samples  "
                f"μ={_folsom_dist['mean']:.3f}  σ={_folsom_dist['std']:.3f}  max={_folsom_dist['max']:.3f}"
            )
        else:
            print(f"  Folsom replay buffer: DISABLED (path not found: {_folsom_path})")

    per_day: list[dict[str, Any]] = []
    per_round: list[dict[str, Any]] = []
    days_since_sync = 0
    fl_round_idx = 0
    last_global_state_dict: dict[str, Any] | None = None  # updated after every FL round
    in_warmup = warmup_days > 0

    # Accumulators for final overall metrics (avoid a second inference pass)
    _all_preds: list[list[np.ndarray]] = [[] for _ in clients]
    _all_tgts: list[list[np.ndarray]] = [[] for _ in clients]

    # --- FedProto state: per-class prototype accumulators and per-client global prototypes ---
    _proto_accum: list[dict[int, tuple[torch.Tensor, int]]] = [{} for _ in clients]
    # Per-client personalised global prototypes (attention-weighted).
    # After the first FedProto sync this becomes a list of dicts, one per client.
    _per_client_global_protos: list[dict[int, torch.Tensor]] | None = None

    # --- Gated FL state ---
    _is_gated = algorithm == "gated"
    _gated_prototypes: list[dict[int, torch.Tensor]] = []
    _gated_thresholds: list[dict[int, dict[str, float]]] = []
    _gated_optimizers: list[torch.optim.Optimizer] = []
    _gated_loss_fn = BerHuLoss(threshold=30, scale=1000)
    _gated_proto_accum: list[dict[int, tuple[torch.Tensor, int]]] = [{} for _ in clients]
    if _is_gated:
        print("Initialising gated prototypes and thresholds per client...")
        for i in range(len(clients)):
            protos, thresholds = build_gated_thresholds(models[i], datasets[i], device)
            _gated_prototypes.append(protos)
            _gated_thresholds.append(thresholds)
            _gated_optimizers.append(torch.optim.AdamW(models[i].network_head.parameters(), lr=adapt_lr))
        if loso_holdout:
            print(f"  LOSO holdout site (excluded from federation): {loso_holdout}")
    # Warmup cache — load if available, skipping the warmup simulation
    # -----------------------------------------------------------------------
    _warmup_cache_loaded = False
    _warmup_cache_key: str | None = None
    _warmup_cache_file: Path | None = None

    _use_cache = warmup_days > 0 and warmup_cache_dir is not None
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
            print(f"  Skipping {warmup_days} days of isolated warmup training.")
            _cache = torch.load(_warmup_cache_file, map_location=device, weights_only=False)
            for i, model in enumerate(models):
                model.load_state_dict(_cache["client_states"][i])
                if freeze_early_backbone:
                    model.freeze_early_backbone()
            per_day = _cache["per_day"]
            _all_preds = _cache["all_preds"]
            _all_tgts = _cache["all_tgts"]
            # Restore recent FIFO buffer states
            # Cache stores lightweight (client_index, sample_index) pairs;
            # reconstruct (dataset, sample_index) using the current run's datasets.
            if "recent_states" in _cache:
                assert len(recent_buffers) == len(
                    _cache["recent_states"]
                ), f"Mismatch: {len(recent_buffers)} clients but {len(_cache['recent_states'])} cached buffers"
                for i, buf in enumerate(recent_buffers):
                    assert (
                        client_names[i] == _cache["client_names"][i]
                    ), f"Client name mismatch at index {i}: {client_names[i]} vs {_cache['client_names'][i]}"
                    # Reconstruct (dataset, idx) tuples from stored (client_idx, sample_idx) pairs
                    buf._entries = [(datasets[ci], si) for ci, si in _cache["recent_states"][i]["entries"]]
                _per_client_counts = [len(s["entries"]) for s in _cache["recent_states"]]
                print("  Restored recent FIFO buffers:")
                for i, (name, count) in enumerate(zip(client_names, _per_client_counts, strict=False)):
                    print(f"    [{i}] {name:15s} {count:>5d} samples")
            # Folsom replay buffer is rebuilt from Folsom at startup — no restore needed.
            # Keep in_warmup=True so that warmup_ended_today fires on the
            # first iteration (day_num == warmup_days), triggering the forced
            # aggregation round naturally — identical to the non-cached path.
            _warmup_cache_loaded = True
        else:
            print(f"  Warmup cache MISS (key={_warmup_cache_key})")
            print(f"  Will save snapshot to {_warmup_cache_file} after warmup ends.")

    # When cache was loaded, skip warmup days; day_num starts at warmup_days
    # so the existing warmup_ended_today logic fires on the first iteration.
    _loop_start = warmup_days if _warmup_cache_loaded else 0

    # --- Ditto: personalised models (deep copies of global; diverge after first aggregation) ---
    ditto_models: list[Forecaster] = []
    if algorithm == "ditto":
        ditto_models = [copy.deepcopy(m) for m in models]

    # --- Thread pool for parallel client processing ---
    _n_workers = min(parallel_clients, len(clients)) if parallel_clients > 1 else 0
    _pool: ThreadPoolExecutor | None = None
    if _n_workers > 1:
        _pool = ThreadPoolExecutor(max_workers=_n_workers)
        print(f"  Parallel client processing: {_n_workers} workers (CUDA streams)")

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

        # --- Process all clients (parallel or sequential) ---
        _day_key = day_key
        _day_num = day_num
        _infer_models = ditto_models if algorithm == "ditto" else models
        _global_sd = last_global_state_dict if algorithm == "fedprox" else None
        _prox_mu = fedprox_mu if algorithm == "fedprox" else 0.0
        _is_fedproto = algorithm == "fedproto"
        _is_gated = algorithm == "gated"
        client_kwargs = [
            {
                "client_idx": i,
                "name": client_names[i],
                "model": models[i],
                "infer_model": _infer_models[i],
                "dataset": datasets[i],
                "day_indices": day_maps[i].get(_day_key, []),
                "device": device,
                "rng_seed": seed + _day_num * len(clients) + i,
                "adapt_lr": adapt_lr,
                "adapt_weight_decay": adapt_weight_decay,
                "adapt_max_grad_norm": adapt_max_grad_norm,
                "adapt_batch_size": adapt_batch_size,
                "adapt_epochs": adapt_epochs,
                "folsom_replay_buffer": _folsom_replay_buffer,
                "replay_buffer_size": replay_buffer_size,
                "replay_ratio": replay_ratio,
                "recent_buffer": recent_buffers[i],
                "recent_buffer_size": recent_buffer_size,
                "recent_replay_ratio": recent_replay_ratio,
                "global_state_dict": _global_sd,
                "prox_mu": _prox_mu,
                "global_prototypes": _per_client_global_protos[i]
                if (_is_fedproto and _per_client_global_protos is not None)
                else None,
                "proto_lambda": fedproto_lambda if _is_fedproto else 0.0,
                "extract_prototypes": _is_fedproto and not in_warmup,
                "train": not is_static,
                "gated": _is_gated,
                "gate_prototypes": _gated_prototypes[i] if _is_gated else None,
                "gate_thresholds": _gated_thresholds[i] if _is_gated else None,
                "gate_optimizer": _gated_optimizers[i] if _is_gated else None,
                "gate_loss_fn": _gated_loss_fn if _is_gated else None,
            }
            for i in range(len(clients))
        ]

        if _pool is not None:
            futures = [_pool.submit(_process_client_day, **kwargs) for kwargs in client_kwargs]
            client_results = [f.result() for f in futures]
        else:
            client_results = [_process_client_day(**kwargs) for kwargs in client_kwargs]

        for result in client_results:
            i = result["client_idx"]
            name = result["name"]
            day_record[f"{name}_n"] = result["n_samples"]
            day_record[f"{name}_mae"] = result["mae"]
            day_record[f"{name}_rmse"] = result["rmse"]
            if len(result["preds"]) > 0:
                _all_preds[i].append(result["preds"])
                _all_tgts[i].append(result["tgts"])
            losses.append(result["loss"])

            # FedProto: accumulate per-class prototypes for this client
            if _is_fedproto and result.get("day_prototypes"):
                accumulate_prototypes(_proto_accum[i], result["day_prototypes"])

            if _is_gated and result.get("gate_opened"):
                day_gate_samples = [datasets[i][idx] for idx in day_maps[i].get(_day_key, [])[:64]]
                if day_gate_samples:
                    day_gate_protos = extract_prototypes(models[i], day_gate_samples, device, adapt_batch_size)
                    accumulate_prototypes(_gated_proto_accum[i], day_gate_protos)

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
        print(f"  {day_key:<6}{_warmup_tag}  {_row}", flush=True)

        # ---------------------------------------------------------------
        # Warmup cache snapshot
        #   - saved at warmup end even for isolated (no aggregation)
        # ---------------------------------------------------------------
        _state_dicts: list[dict[str, Any]] | None = None
        if warmup_ended_today and _use_cache and not _warmup_cache_loaded:
            assert _warmup_cache_file is not None
            _warmup_cache_file.parent.mkdir(parents=True, exist_ok=True)
            _state_dicts = [m.state_dict() for m in models]
            # Save only (client_index, sample_index) pairs — NOT dataset objects.
            # The datasets list order matches clients, so we map each dataset
            # reference back to its client index for compact serialisation.
            _ds_id_to_client = {id(ds): ci for ci, ds in enumerate(datasets)}
            _recent_states = [
                {"entries": [(_ds_id_to_client[id(ds)], idx) for ds, idx in buf._entries]} for buf in recent_buffers
            ]
            torch.save(
                {
                    "cache_key": _warmup_cache_key,
                    "warmup_days": warmup_days,
                    "client_names": client_names,
                    "client_states": list(_state_dicts),
                    "per_day": list(per_day),
                    "all_preds": [[a.copy() for a in preds] for preds in _all_preds],
                    "all_tgts": [[a.copy() for a in tgts] for tgts in _all_tgts],
                    "recent_states": _recent_states,
                },
                _warmup_cache_file,
            )
            print(f"  {'':6}      [cache] Warmup snapshot saved → {_warmup_cache_file}")
            print(f"  {'':6}              Recent FIFO buffers per client:")
            for i, (name, buf_state) in enumerate(zip(client_names, _recent_states, strict=False)):
                print(f"  {'':6}                [{i}] {name:15s} {len(buf_state['entries']):>5d} samples")

        # ---------------------------------------------------------------
        # Step 3: FL Aggregation
        #   - forced round immediately after warmup ends
        #   - periodic round every sync_interval days (post-warmup)
        # ---------------------------------------------------------------
        _do_aggregate = not is_static and algorithm != "isolated" and (warmup_ended_today or days_since_sync >= sync_interval)
        if warmup_ended_today and not in_warmup:
            # Reset counter so the next periodic sync starts fresh
            days_since_sync = 0

        if _do_aggregate:
            if algorithm == "fedproto":
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FedProto: aggregate prototypes, NOT model weights.
                # Each client keeps its own model entirely local.
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                _per_client_global_protos, _attn_weights = aggregate_global_prototypes(_proto_accum)
                wc = 0.0  # no weight-level change

                # Log per-client prototype norms and attention weights
                _all_classes = sorted({c for p in _per_client_global_protos for c in p})
                _n_classes = len(_all_classes)
                print(f"  {'':6}      [proto] Aggregated {_n_classes} class prototypes (attention-weighted):")
                # Per-client sample counts
                for ci, cname in enumerate(client_names):
                    _counts = {PROTO_CLASS_NAMES[c]: cnt for c, (_, cnt) in _proto_accum[ci].items()}
                    _norms = {
                        PROTO_CLASS_NAMES[c]: _per_client_global_protos[ci][c].norm().item()
                        for c in _per_client_global_protos[ci]
                    }
                    if _counts:
                        _cnt_str = "  ".join(f"{k}={v}" for k, v in _counts.items())
                        _nrm_str = "  ".join(f"{k}={v:.3f}" for k, v in _norms.items())
                        print(f"  {'':6}              {cname}: samples=[{_cnt_str}]  norms=[{_nrm_str}]")
                # Attention matrix per class
                for cls in _all_classes:
                    if cls in _attn_weights:
                        cname = PROTO_CLASS_NAMES[cls]
                        present = [
                            i
                            for i in range(len(client_names))
                            if any(_attn_weights[cls][i][j] > 0 for j in range(len(client_names)))
                        ]
                        if len(present) > 1:
                            print(f"  {'':6}              attn[{cname}]:")
                            for i in present:
                                _w_str = "  ".join(f"{client_names[j]}={_attn_weights[cls][i][j]:.3f}" for j in present)
                                print(f"  {'':6}                {client_names[i]:>12s} → {_w_str}")

                # Reset accumulators for next sync interval
                _proto_accum = [{} for _ in clients]

            elif algorithm == "gated":
                gated_indices = [
                    i
                    for i, result in enumerate(client_results)
                    if result.get("gate_opened") and (loso_holdout is None or client_names[i] != loso_holdout)
                ]
                if gated_indices:
                    gated_states = [models[i].state_dict() for i in gated_indices]
                    new_global, wc = AGGREGATION_FNS["gated"](
                        last_global_state_dict or models[0].state_dict(),
                        gated_states,
                    )
                    last_global_state_dict = copy.deepcopy(new_global)
                    for i, model in enumerate(models):
                        if loso_holdout and client_names[i] == loso_holdout:
                            continue
                        merged = copy.deepcopy(model.state_dict())
                        for key in merged:
                            if is_head_key(key):
                                merged[key] = new_global[key]
                        model.load_state_dict(merged)

                    gated_proto_accum = [_gated_proto_accum[i] for i in gated_indices if _gated_proto_accum[i]]
                    if gated_proto_accum:
                        global_csi_protos = aggregate_prototypes(gated_proto_accum)
                        for i in range(len(clients)):
                            _gated_prototypes[i] = global_csi_protos
                    print(f"  {'':6}      [gated] Aggregated adapters from {len(gated_indices)} gated client(s)")
                else:
                    wc = 0.0
                _gated_proto_accum = [{} for _ in clients]

            else:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Weight-based FL algorithms
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                agg_fn = AGGREGATION_FNS[algorithm]
                state_dicts = _state_dicts if _state_dicts is not None else [m.state_dict() for m in models]

                # Capture local-only layers before aggregation
                if algorithm == "fedbn":
                    local_bn = [_extract_local_keys(sd, is_bn_key) for sd in state_dicts]
                if algorithm == "fedper":
                    local_head = [_extract_local_keys(sd, is_head_key) for sd in state_dicts]

                new_global, wc = agg_fn(state_dicts[0], state_dicts)
                last_global_state_dict = copy.deepcopy(new_global)

                # Distribute back — restore local layers per client
                for i, model in enumerate(models):
                    merged = new_global
                    if algorithm == "fedbn":
                        merged = _restore_local_keys(merged, local_bn[i])
                    elif algorithm == "fedper":
                        merged = _restore_local_keys(merged, local_head[i])
                    elif algorithm == "fedema":
                        # Soft interpolation: w ← α·w_local + (1-α)·w_global
                        merged = _ema_interpolate(state_dicts[i], new_global, fedema_alpha)
                    model.load_state_dict(merged)

            # Ditto: train personalised models with proximal toward the
            # *freshly aggregated* global consensus.  Only the global
            # model participates in FedAvg; the personalised model stays
            # local and is never shared — it just trains on the current
            # day's data anchored to the new global.
            if algorithm == "ditto":

                def _ditto_train_client(
                    i: int,
                    day_key: str = day_key,
                    day_num: int = day_num,
                    last_global_state_dict: dict[str, Any] | None = last_global_state_dict,
                ) -> None:
                    _d_name = client_names[i]
                    _d_indices = day_maps[i].get(day_key, [])
                    _d_seed = seed + day_num * len(clients) + i
                    # Assemble same replay mix the global model used
                    _d_reservoir: list[tuple[Any, int]] = []
                    if replay_buffer_size > 0 and replay_ratio > 0.0 and _folsom_replay_buffer:
                        _d_reservoir = sample_replay_for_day(
                            _folsom_replay_buffer,
                            max(0, int(round(len(_d_indices) * replay_ratio))),
                            seed=_d_seed,
                        )
                    _d_recent: list[tuple[Any, int]] = []
                    if recent_buffer_size > 0 and recent_replay_ratio > 0.0:
                        _d_recent = recent_buffers[i].sample(
                            max(0, int(round(len(_d_indices) * recent_replay_ratio))),
                            seed=_d_seed + 1,
                        )
                    _d_replay = (_d_reservoir + _d_recent) or None
                    train_day(
                        ditto_models[i],
                        datasets[i],
                        _d_indices,
                        lr=adapt_lr,
                        weight_decay=adapt_weight_decay,
                        max_grad_norm=adapt_max_grad_norm,
                        batch_size=adapt_batch_size,
                        epochs=adapt_epochs,
                        device=device,
                        replay_entries=_d_replay,
                        global_state_dict=last_global_state_dict,
                        mu=ditto_lambda,
                        rng_seed=_d_seed,
                        client_name=f"{_d_name} ditto",
                    )

                if _pool is not None:
                    ditto_futures = [_pool.submit(_ditto_train_client, i) for i in range(len(clients))]
                    for fut in ditto_futures:
                        fut.result()
                else:
                    for i in range(len(clients)):
                        _ditto_train_client(i)

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

    # Shut down thread pool
    if _pool is not None:
        _pool.shutdown(wait=False)

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
    print(f"  {'Client':<{_nw}}  {'MAE':>9}  {'RMSE':>9}  {'WAPE':>8}  {'SMAPE':>8}  {'Count':>8}")
    print(f"  {'-'*_nw}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name in client_names:
        m = overall_metrics[name]
        print(
            f"  {name:<{_nw}}  "
            f"{m.get('mae', float('nan')):>9.5f}  "
            f"{m.get('rmse', float('nan')):>9.5f}  "
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
        "models": ditto_models if algorithm == "ditto" else models,
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
    type=click.Choice(
        ["static", "isolated", "fedavg", "fedbn", "fedper", "fedprox", "ditto", "fedema", "fedproto", "gated"],
        case_sensitive=False,
    ),
    default="fedbn",
    show_default=True,
    help=(
        "FL aggregation algorithm.\n"
        "  static   — inference only (no training, no aggregation).\n"
        "  isolated — no federation (lower bound).\n"
        "  fedavg   — standard weight averaging.\n"
        "  fedbn    — FedAvg excluding BatchNorm layers.\n"
        "  fedper   — aggregate backbone only, keep head local.\n"
        "  fedprox  — FedAvg + proximal term in local training.\n"
        "  ditto    — dual-model with personalised proximal anchor.\n"
        "  fedema   — FedAvg + EMA interpolation at distribution.\n"
        "  fedproto — share per-class feature prototypes, not weights.\n"
        "  gated — CSI-binned prototypes + adapter aggregation from gated clients."
    ),
)
@click.option(
    "--loso-holdout",
    type=str,
    default=None,
    help="Site name to hold out for LOSO (excluded from federation aggregation).",
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
    "--fedprox-mu",
    type=float,
    default=0.01,
    show_default=True,
    help=(
        "Proximal regularisation strength for FedProx.  "
        "Local loss is augmented with (μ/2)‖w-w_global‖².  Only used when --algorithm=fedprox."
    ),
)
@click.option(
    "--ditto-lambda",
    type=float,
    default=0.1,
    show_default=True,
    help=(
        "Proximal strength for Ditto's personalised model.  The global model "
        "trains with standard FedAvg; the personal model's loss is augmented "
        "with (λ/2)‖v-w_global‖².  Only used when --algorithm=ditto."
    ),
)
@click.option(
    "--fedema-alpha",
    type=float,
    default=0.5,
    show_default=True,
    help=(
        "EMA interpolation coefficient for FedEMA.  After aggregation, local "
        "weights are updated as w←α·w_local + (1-α)·w_global.  Higher α retains "
        "more local knowledge.  Only used when --algorithm=fedema."
    ),
)
@click.option(
    "--fedproto-lambda",
    type=float,
    default=0.1,
    show_default=True,
    help=(
        "Prototype regularisation relative weight for FedProto.  The prototype "
        "MSE term is adaptively scaled to match the task-loss magnitude, so "
        "this parameter acts as a relative contribution: 0.1 ≈ proto adds 10%% "
        "of the task loss.  Only used when --algorithm=fedproto."
    ),
)
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
@click.option(
    "--parallel-clients",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Number of clients to process in parallel per day.  "
        "Each client runs prediction + training on its own CUDA stream.  "
        "1 = sequential (default).  Set to the number of clients for full "
        "parallelism (overlaps CPU work and enables GPU stream concurrency)."
    ),
)
@click.option(
    "--freeze-early-backbone",
    is_flag=True,
    default=False,
    help=(
        "Freeze early backbone layers (conv1, bn1, layer1, layer2) in ResNet models. "
        "Prevents low-level feature extractors from adapting during continual learning, "
        "forcing adaptation to occur only in higher-level layers. Helps prevent catastrophic "
        "forgetting of low-level features while allowing task-specific adaptation in deeper layers."
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
    fedprox_mu: float,
    ditto_lambda: float,
    fedema_alpha: float,
    fedproto_lambda: float,
    save_results: Path | None,
    parallel_clients: int,
    freeze_early_backbone: bool,
    loso_holdout: str | None,
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
        fedprox_mu=fedprox_mu,
        ditto_lambda=ditto_lambda,
        fedema_alpha=fedema_alpha,
        fedproto_lambda=fedproto_lambda,
        parallel_clients=parallel_clients,
        freeze_early_backbone=freeze_early_backbone,
        loso_holdout=loso_holdout,
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
            "fedprox_mu": fedprox_mu,
            "ditto_lambda": ditto_lambda,
            "fedema_alpha": fedema_alpha,
            "fedproto_lambda": fedproto_lambda,
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
