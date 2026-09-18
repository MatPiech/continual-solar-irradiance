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

Note on fedbn / fedper communication cost: this script has every client
upload its *full* state dict regardless of algorithm — fedbn/fedper only
drop keys server-side during aggregation, so the reduced-bandwidth benefit
those algorithms are usually chosen for is not exercised here. For actual
upload/download payload measurements, use
`scripts/fl_edge_overhead_benchmark.py`.

Weighted aggregation: all weight-based algorithms average clients weighted
by each client's local sample count since the last round (canonical
FedAvg's n_k/n weighting), not a uniform per-client mean.

LOSO holdout (`--loso-holdout SITE`): excludes that client from every
federation round for *any* algorithm — it trains purely locally and never
contributes to or receives the aggregated result.

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

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
import yaml

from cirr.continual import build_gated_thresholds, train_gated_day
from cirr.data.loading import load_dataset
from cirr.evaluation.smart_persistence import predict_periods
from cirr.fl.kc_prototypes import (
    DEFAULT_KC_EDGES,
    PROTO_CLASS_NAMES,
    REGIME_LABELS,
    aggregate_global_prototypes,
    default_edges,
    extract_day_prototypes,
    DEFAULT_MIN_GHI_CLEAR,
    make_regime_labeller,
    regime_names,
)
from cirr.fl.prototypes import accumulate_prototypes, aggregate_prototypes, extract_prototypes
from cirr.fl.server import AGGREGATION_FNS, _compute_weight_change, is_bn_key, is_head_key, payload_nbytes
from cirr.losses import BerHuLoss
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.physics.clear_sky import load_clear_sky_lookup, resolve_clear_sky_value
from cirr.training.day_loop import predict_day, train_day
from cirr.training.replay import RecentBuffer, build_folsom_replay_buffer, sample_replay_for_day
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# Algorithms that exchange prototypes rather than weights.
PROTO_ALGORITHMS = ("fedproto",)
# Controls for the exchanged prototype, used to test whether the mechanism is
# active at all (H5.2) rather than only whether it is harmless (H5.1).
ALIGN_TARGETS = ("global", "own", "shuffled", "random")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def filter_lit_days(
    day_map: dict[str, list[int]],
    periods: list[dict[str, Any]],
    lookup: dict[str, float],
    min_ghi_clear: float,
) -> tuple[dict[str, list[int]], int]:
    """Drop samples whose clear-sky GHI is below *min_ghi_clear* (kW/m^2).

    Applied once, where the day index map is built, so local training, evaluation,
    prototype extraction and the Smart-Persistence denominator all score exactly the
    same population — there is no second place for the filter to disagree with itself.

    Night samples are trivially predictable (target and prediction are both ~0) and
    their share differs sharply by site: 0 % at Folsom, whose camera records 06-20 h,
    against 19 % at Karpicko and 31 % at Poznan, which record around the clock.
    Leaving them in depresses the Polish sites' error for *every* method and makes
    cross-site comparison meaningless.  0 disables the filter.

    Days left with no lit sample are dropped entirely, which is the same state the
    simulation already uses for a day a client has no data for.
    """
    if min_ghi_clear <= 0.0:
        return day_map, 0
    kept: dict[str, list[int]] = {}
    dropped = 0
    for day, idxs in day_map.items():
        lit = []
        for j in idxs:
            stamp = periods[j]["history"][-1]["image_name"].rsplit(".", 1)[0]
            ghi = resolve_clear_sky_value(lookup, stamp)
            if ghi is not None and ghi >= min_ghi_clear:
                lit.append(j)
            else:
                dropped += 1
        if lit:
            kept[day] = lit
    return kept, dropped


def set_determinism(seed: int) -> None:
    """Pin every RNG the day loop touches, and ask torch for deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # warn_only: some timm kernels have no deterministic implementation; a warning
    # is more useful than refusing to run the benchmark at all.
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # provenance is best-effort, never fatal
        return "unknown"


def parse_edges(spec: str | None, label: str) -> tuple[float, ...]:
    """Parse a comma-separated regime cut-point list; empty string = one regime."""
    if spec is None:
        return default_edges(label)
    spec = spec.strip()
    if not spec:
        return ()
    edges = tuple(float(x) for x in spec.split(","))
    if list(edges) != sorted(edges):
        raise ValueError(f"--regime-edges must be ascending, got {edges}")
    return edges


def make_participation(spec: str, client_names: list[str], seed: int) -> Callable[[int, int], bool]:
    """Build ``participates(client_idx, fl_round) -> bool`` from a schedule spec.

    ``full`` | ``dropout:<p>`` | ``outage:<client>:<start_round>:<n_rounds>``.
    Dropout draws are seeded from the run seed, so a rerun of the same seed drops
    the same clients in the same rounds.
    """
    if spec == "full":
        return lambda _i, _r: True

    kind, _, rest = spec.partition(":")
    if kind == "dropout":
        prob = float(rest)
        if not 0.0 <= prob < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {prob}")

        def _dropout(client_idx: int, fl_round: int) -> bool:
            rng = random.Random((seed, fl_round, client_idx).__hash__())
            return rng.random() >= prob

        return _dropout

    if kind == "outage":
        name, start_s, days_s = rest.split(":")
        if name not in client_names:
            raise ValueError(f"outage client {name!r} not among {client_names}")
        idx, start, span = client_names.index(name), int(start_s), int(days_s)
        return lambda client_idx, fl_round: not (client_idx == idx and start <= fl_round < start + span)

    raise ValueError(f"Unknown --participation spec {spec!r}")


def apply_align_target(
    per_client_protos: list[dict[int, torch.Tensor]],
    own_protos: list[dict[int, torch.Tensor]],
    align_target: str,
    fl_round: int,
    seed: int,
) -> list[dict[int, torch.Tensor]]:
    """Replace the aggregated prototype with a control variant.

    ``own`` removes the exchange entirely (nothing is sent or received, so the
    alignment term becomes a self-consistency penalty); ``shuffled`` keeps the
    received vectors but mis-assigns them across regimes; ``random`` keeps only
    the magnitude.  Together they separate "collaboration is harmless" from
    "collaboration does nothing".
    """
    if align_target == "global":
        return per_client_protos
    if align_target == "own":
        return [dict(p) for p in own_protos]

    rng = random.Random((seed, fl_round, align_target).__hash__())
    out: list[dict[int, torch.Tensor]] = []
    for protos in per_client_protos:
        classes = sorted(protos)
        if align_target == "shuffled":
            shuffled = list(classes)
            # A plain permutation is the identity with probability 1/C! — one round in
            # six at C = 3 — which silently turns the control into a no-op and dilutes
            # the H5.2 contrast.  Draw a derangement instead, so every regime really
            # does receive another regime's prototype.  (C < 2 has no derangement.)
            while len(classes) > 1:
                rng.shuffle(shuffled)
                if all(a != b for a, b in zip(classes, shuffled, strict=True)):
                    break
            out.append({cls: protos[other] for cls, other in zip(classes, shuffled, strict=True)})
        else:  # random, norm-matched
            out.append(
                {
                    cls: torch.randn(
                        protos[cls].shape,
                        generator=torch.Generator().manual_seed(rng.randrange(2**31)),
                    )
                    * (protos[cls].norm() / math.sqrt(protos[cls].numel()))
                    for cls in classes
                }
            )
    return out


def _seed_history(
    buffer: RecentBuffer,
    dataset: Any,
    day_map: dict[str, list[int]],
    all_days: list[str],
    join_day_num: int,
    history_days: int,
) -> None:
    """Give a joining client *history_days* of its own past data, nothing more."""
    if history_days <= 0:
        return
    for day in all_days[max(0, join_day_num - history_days) : join_day_num]:
        if day in day_map:
            buffer.add_day(dataset, day_map[day])


def _pooled_train(
    models: list[Forecaster],
    datasets: list[Any],
    day_maps: list[dict[str, list[int]]],
    day_key: str,
    joined: list[bool],
    *,
    adapt_lr: float,
    adapt_weight_decay: float,
    adapt_max_grad_norm: float,
    adapt_batch_size: int,
    adapt_epochs: int,
    device: str,
    rng_seed: int,
) -> float:
    """Oracle update: one model, one epoch over the union of every client's day.

    Other clients' samples ride in through the existing replay channel, which is
    already a list of ``(dataset, index)`` pairs, so no new data path is needed.
    Every client shares the same model object, so the update reaches all of them.
    """
    primary = next((i for i, ok in enumerate(joined) if ok and day_maps[i].get(day_key)), None)
    if primary is None:
        return 0.0
    union = [
        (datasets[j], idx) for j in range(len(models)) if joined[j] and j != primary for idx in day_maps[j].get(day_key, [])
    ]
    return train_day(
        models[primary],
        datasets[primary],
        day_maps[primary][day_key],
        lr=adapt_lr,
        weight_decay=adapt_weight_decay,
        max_grad_norm=adapt_max_grad_norm,
        batch_size=adapt_batch_size,
        epochs=adapt_epochs,
        device=device,
        replay_entries=union or None,
        rng_seed=rng_seed,
        client_name="pooled",
    )


def _sp_rmse(stats: tuple[float, int] | None) -> float:
    """RMSE from a (sum of squared errors, count) pair."""
    if not stats or stats[1] == 0:
        return float("nan")
    return float((stats[0] / stats[1]) ** 0.5)


def client_mean_prototypes(accum: dict[int, tuple[torch.Tensor, int]]) -> dict[int, torch.Tensor]:
    return {cls: feat_sum / count for cls, (feat_sum, count) in accum.items() if count > 0}


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
    classify_fn: Callable[[float, float], int] | None = None,
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
        train_steps = 0
        proto_bin_edges: tuple[float, float] = DEFAULT_KC_EDGES
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
                # Batch count is fully determined by the pool size, so it is
                # cheaper to derive than to thread a counter through train_day.
                _pool_size = len(day_indices) + len(_replay or [])
                train_steps = adapt_epochs * math.ceil(_pool_size / adapt_batch_size)

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
                    classify_fn=classify_fn,
                )

        # Step 4: Extract per-class prototypes (FedProto — after training
        # so features reflect the updated model)
        day_protos: dict[int, tuple[torch.Tensor, int]] = {}
        if train and extract_prototypes and day_samples:
            day_protos = extract_day_prototypes(
                model,
                day_samples,
                device,
                adapt_batch_size,
                edges=proto_bin_edges,
                labeller=classify_fn,
            )

        # Step 5: Update recent FIFO buffer
        if train and recent_buffer_size > 0:
            recent_buffer.add_day(dataset, day_indices)

    if stream is not None:
        stream.synchronize()

    # Per-sample regime ids, in the same order as preds/tgts, so error can be
    # reported per sky regime without a second pass over the data.
    regimes = (
        np.array([classify_fn(float(s[3].view(-1)[0]), float(s[2])) for s in day_samples], dtype=np.int16)
        if (classify_fn is not None and day_samples)
        else np.array([], dtype=np.int16)
    )

    return {
        "client_idx": client_idx,
        "name": name,
        "n_samples": len(day_indices),
        "preds": preds,
        "tgts": tgts,
        "regimes": regimes,
        "train_steps": train_steps,
        "mae": metrics.get("mae", float("nan")),
        "rmse": metrics.get("rmse", float("nan")),
        "loss": loss,
        "day_prototypes": day_protos,
        "proto_bin_edges": proto_bin_edges,
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
    regime_label: str = "clear_sky",
    regime_edges: tuple[float, ...] | None = None,
    proto_min_ghi_clear: float = DEFAULT_MIN_GHI_CLEAR,
    min_ghi_clear: float = DEFAULT_MIN_GHI_CLEAR,
    attention: bool = True,
    attention_temperature: float | None = None,
    align_target: str = "global",
    participation: str = "full",
    join_days: dict[str, int] | None = None,
    local_history_days: int = 0,
    ft_epochs: int = 1,
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
    set_determinism(seed)

    if len(clients) < 2 and algorithm != "pooled":
        raise ValueError("At least two clients are required.")

    algorithm = algorithm.lower()
    _extra_algorithms = ("pooled", "fedavg_ft")
    if algorithm not in ("static", "isolated", *AGGREGATION_FNS.keys(), *_extra_algorithms):
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: static, isolated, "
            f"{', '.join(AGGREGATION_FNS)}, {', '.join(_extra_algorithms)}"
        )
    if align_target not in ALIGN_TARGETS:
        raise ValueError(f"Unknown align target '{align_target}'. Choose from: {', '.join(ALIGN_TARGETS)}")

    is_static = algorithm == "static"
    is_pooled = algorithm == "pooled"
    join_days = dict(join_days or {})

    # Regime labelling: one labeller drives prototype extraction, the alignment
    # target inside train_day, and the per-regime error breakdown, so the three
    # can never disagree about which regime a sample belongs to.
    edges = default_edges(regime_label) if regime_edges is None else tuple(regime_edges)
    classify_fn = make_regime_labeller(regime_label, edges, proto_min_ghi_clear)
    n_regimes = len(edges) + 1
    regime_name_map = regime_names(n_regimes)

    client_names = [name for name, _ in clients]

    if loso_holdout is not None and loso_holdout not in client_names:
        raise ValueError(f"--loso-holdout '{loso_holdout}' does not match any client name: {client_names}")

    unknown_joins = set(join_days) - set(client_names)
    if unknown_joins:
        raise ValueError(f"--join-day names unknown clients: {sorted(unknown_joins)}")
    participates = make_participation(participation, client_names, seed)

    print(f"\n{'=' * 70}")
    print(f"  FL Benchmark — algorithm: {algorithm.upper()}  sync_interval: {sync_interval} days  warmup: {warmup_days} days")
    print(f"{'=' * 70}")
    for name, path in clients:
        print(f"  {name:20s}  lr={adapt_lr}  epochs={adapt_epochs}  path={path}")
    if loso_holdout is not None:
        print(f"  LOSO holdout: {loso_holdout}  (excluded from every federation round, all algorithms)")
    print()

    # --- Load datasets ---
    print("Loading datasets...")
    datasets = []
    day_maps = []
    # Smart-Persistence squared-error sums and counts per day.  Kept as (sse, n)
    # rather than an RMSE so the Forecast-Skill denominator can later be pooled
    # over exactly the days the model was scored on.
    sp_day_stats: list[dict[str, tuple[float, int]]] = []
    clear_sky_lookups: list[dict[str, float]] = []
    for name, path in clients:
        ds = load_dataset(path, split="test")
        dm = group_indices_by_day(ds.periods)
        lookup = load_clear_sky_lookup(path)
        dm, _dropped = filter_lit_days(dm, ds.periods, lookup, min_ghi_clear)

        clear_sky_lookups.append(lookup)
        sp_day_stats.append({})
        datasets.append(ds)
        day_maps.append(dm)
        _kept = sum(len(v) for v in dm.values())
        if _dropped:
            print(
                f"  {name}: {_kept} periods, {len(dm)} days "
                f"({_dropped} dropped below {min_ghi_clear * 1000:.0f} W/m^2 clear-sky)"
            )
        else:
            print(f"  {name}: {_kept} periods, {len(dm)} days")

    # --- Load models (all clients start from the same checkpoint) ---
    print(f"\nLoading checkpoint: {checkpoint_path}")
    models = [_load_checkpoint(checkpoint_path, device, freeze_early_backbone) for _ in clients]
    if is_pooled:
        # One shared model: the oracle has no per-site personalisation to lose.
        models = [models[0] for _ in clients]
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

    # Smart Persistence, restricted to the days actually simulated.  Computing it
    # over the whole calendar would put a year-wide RMSE in the denominator of a
    # Forecast Skill whose numerator covers only this run's days.
    print("Smart Persistence reference:")
    for i, (name, _path) in enumerate(clients):
        for day in all_days:
            idxs = day_maps[i].get(day, [])
            if not idxs:
                continue
            preds_sp, tgts_sp, _ = predict_periods(
                [datasets[i].periods[j] for j in idxs], clear_sky_lookups[i], period_scale=1 / 1000.0
            )
            if len(preds_sp):
                sp_day_stats[i][day] = (float(np.sum((preds_sp - tgts_sp) ** 2)), len(preds_sp))
        _sse = sum(v[0] for v in sp_day_stats[i].values())
        _n = sum(v[1] for v in sp_day_stats[i].values())
        print(f"  {name}: RMSE={np.sqrt(_sse / _n):.5f} over {_n} samples" if _n else f"  {name}: no samples")
    print()

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
    # Long-format logs (Section 5.4 of the benchmark plan)
    metrics_daily: list[dict[str, Any]] = []
    metrics_regime: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    attention_rounds: list[np.ndarray] = []
    started_at = time.time()
    # Days on which each client actually produced predictions.
    evaluated_days: list[set[str]] = [set() for _ in clients]
    days_since_sync = 0
    fl_round_idx = 0
    # Counts every sync *opportunity*, fired or not — see the participation draw below.
    sync_opportunity_idx = 0
    last_global_state_dict: dict[str, Any] | None = None  # updated after every FL round
    in_warmup = warmup_days > 0

    # Accumulators for final overall metrics (avoid a second inference pass)
    _all_preds: list[list[np.ndarray]] = [[] for _ in clients]
    _all_tgts: list[list[np.ndarray]] = [[] for _ in clients]

    # Per-client sample counts since the last FL round — used to weight each
    # client's contribution to weighted aggregation (canonical FedAvg weights
    # by local sample count rather than averaging clients uniformly).
    _round_sample_counts: list[int] = [0 for _ in clients]

    # --- FedProto state: per-class prototype accumulators and per-client global prototypes ---
    _proto_accum: list[dict[int, tuple[torch.Tensor, int]]] = [{} for _ in clients]
    # Aggregate KC bin edges over each sync interval (defaults unless overridden).
    _proto_edge_accum: list[dict[str, float]] = [{"low_sum": 0.0, "high_sum": 0.0, "count": 0.0} for _ in clients]
    # Per-client personalised global prototypes (attention-weighted).
    # After the first FedProto sync this becomes a list of dicts, one per client.
    _per_client_global_protos: list[dict[int, torch.Tensor]] | None = None
    fedproto_round_diagnostics: list[dict[str, Any]] = []

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
            # Long-format logs and the Forecast-Skill day set must come back too,
            # or a cached run silently loses its warmup days from every log while
            # an uncached run keeps them.
            metrics_daily = _cache.get("metrics_daily", [])
            metrics_regime = _cache.get("metrics_regime", [])
            events = _cache.get("events", [])
            for _rec in per_day:
                for _ci, _cname in enumerate(client_names):
                    if _rec.get(f"{_cname}_n", 0) > 0:
                        evaluated_days[_ci].add(_rec["day"])
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
        loss_by_idx: dict[int, float] = {}
        day_events: dict[int, dict[str, Any]] = {}

        # --- Process all clients (parallel or sequential) ---
        _day_key = day_key
        _day_num = day_num
        _infer_models = ditto_models if algorithm == "ditto" else models
        _global_sd = last_global_state_dict if algorithm == "fedprox" else None
        _prox_mu = fedprox_mu if algorithm == "fedprox" else 0.0
        _is_fedproto = algorithm == "fedproto"
        _is_gated = algorithm == "gated"
        # A client that has not joined yet is inert: it neither predicts, trains,
        # nor takes part in a round, and contributes nothing to its own metrics.
        joined = [day_num >= join_days.get(client_names[i], 0) for i in range(len(clients))]
        for i, is_joined in enumerate(joined):
            if is_joined and day_num == join_days.get(client_names[i], 0) and join_days.get(client_names[i], 0) > 0:
                _seed_history(recent_buffers[i], datasets[i], day_maps[i], all_days, day_num, local_history_days)

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
                "classify_fn": classify_fn,
            }
            for i in range(len(clients))
            if joined[i]
        ]
        # The pooled oracle is one model trained on the union of the day's data:
        # every client predicts with it first, then a single update is applied.
        if is_pooled:
            for kwargs in client_kwargs:
                kwargs["train"] = False

        if _pool is not None:
            futures = [_pool.submit(_process_client_day, **kwargs) for kwargs in client_kwargs]
            client_results = [f.result() for f in futures]
        else:
            client_results = [_process_client_day(**kwargs) for kwargs in client_kwargs]

        if is_pooled and not is_static:
            _pooled_train(
                models,
                datasets,
                day_maps,
                day_key,
                joined,
                adapt_lr=adapt_lr,
                adapt_weight_decay=adapt_weight_decay,
                adapt_max_grad_norm=adapt_max_grad_norm,
                adapt_batch_size=adapt_batch_size,
                adapt_epochs=adapt_epochs,
                device=device,
                rng_seed=seed + day_num,
            )

        for result in client_results:
            i = result["client_idx"]
            name = result["name"]
            day_record[f"{name}_n"] = result["n_samples"]
            day_record[f"{name}_mae"] = result["mae"]
            day_record[f"{name}_rmse"] = result["rmse"]
            _round_sample_counts[i] += result["n_samples"]
            if len(result["preds"]) > 0:
                _all_preds[i].append(result["preds"])
                _all_tgts[i].append(result["tgts"])
                evaluated_days[i].add(day_key)
            loss_by_idx[i] = result["loss"]

            # FedProto: accumulate per-class prototypes for this client
            if _is_fedproto and result.get("day_prototypes"):
                accumulate_prototypes(_proto_accum[i], result["day_prototypes"])
            if _is_fedproto and result.get("proto_bin_edges") is not None:
                low_edge, high_edge = result["proto_bin_edges"]
                _proto_edge_accum[i]["low_sum"] += float(low_edge)
                _proto_edge_accum[i]["high_sum"] += float(high_edge)
                _proto_edge_accum[i]["count"] += 1.0

            # Long-format daily metrics, including the Forecast-Skill denominator.
            _m = compute_metrics(result["preds"], result["tgts"]) if result["n_samples"] else {}
            metrics_daily.append(
                {
                    "date": day_key,
                    "client": name,
                    "n_samples": result["n_samples"],
                    "mae": _m.get("mae", float("nan")),
                    "rmse": _m.get("rmse", float("nan")),
                    "wape": _m.get("wape", float("nan")),
                    "smape": _m.get("smape", float("nan")),
                    "sp_rmse": _sp_rmse(sp_day_stats[i].get(day_key)),
                }
            )
            _reg = result["regimes"]
            for cls in sorted(set(_reg.tolist())):
                _mask = _reg == cls
                _rm = compute_metrics(result["preds"][_mask], result["tgts"][_mask])
                metrics_regime.append(
                    {
                        "date": day_key,
                        "client": name,
                        "regime": regime_name_map.get(int(cls), str(cls)),
                        "n_samples": int(_mask.sum()),
                        "mae": _rm["mae"],
                        "rmse": _rm["rmse"],
                    }
                )
            # Bytes and participation are only known once the round fires, so the
            # row is kept by reference and completed below.
            _event = {
                "date": day_key,
                "client": name,
                "samples_after_filter": result["n_samples"],
                "train_steps": result["train_steps"],
                "extraction_calls": 1 if result.get("day_prototypes") else 0,
                "participated": 0,
                "bytes_up": 0,
                "bytes_down": 0,
                "payload_dtype": "float32",
            }
            events.append(_event)
            day_events[i] = _event

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
            return f"{_rec.get(f'{n}_mae', float('nan')):>8.6f}  {_rec.get(f'{n}_rmse', float('nan')):>8.6f}  {n_samples:>5}"

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
                    "metrics_daily": list(metrics_daily),
                    "metrics_regime": list(metrics_regime),
                    "events": list(events),
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
        # Who takes part in *this* round: joined, not held out, not dropped or
        # in an outage.  Absent clients neither send nor receive.
        _federates = not is_static and algorithm not in ("isolated", "pooled")
        _sync_due = warmup_ended_today or days_since_sync >= sync_interval
        roster: list[int] = []
        if _federates and _sync_due:
            # Key the participation draw on the sync OPPORTUNITY, not on fl_round_idx.
            # fl_round_idx advances only when a round actually fires, so keying on it
            # made a round in which everybody dropped redraw the identical outcome at
            # every later opportunity: the all-dropped state was absorbing and
            # federation stopped for good (dropout:0.5 fired ~7 of 48 rounds).
            roster = [
                i
                for i in range(len(clients))
                if joined[i]
                and (loso_holdout is None or client_names[i] != loso_holdout)
                and participates(i, sync_opportunity_idx)
            ]
            sync_opportunity_idx += 1
        _do_aggregate = _federates and _sync_due and bool(roster)
        if _sync_due and not warmup_ended_today:
            # Rounds keep their cadence even when nobody turns up for one, so a missed
            # round costs one round rather than retrying every day until someone shows.
            days_since_sync = 0
        if warmup_ended_today and not in_warmup:
            # Reset counter so the next periodic sync starts fresh
            days_since_sync = 0

        if _do_aggregate:
            if algorithm == "fedproto":
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FedProto: aggregate prototypes, NOT model weights.
                # Each client keeps its own model entirely local.
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # LOSO holdout contributes an empty prototype dict: it never
                # shows up in `present` inside aggregate_global_prototypes, so
                # it neither influences the consensus prototypes nor receives
                # any (it gets back an empty dict, i.e. the proto-regularisation
                # term is a no-op for it) — fully excluded from federation.
                _proto_accum_for_agg = [(_proto_accum[i] if i in roster else {}) for i in range(len(clients))]
                _per_client_global_protos, _attn_weights = aggregate_global_prototypes(
                    _proto_accum_for_agg, attention=attention, temperature=attention_temperature
                )
                _own_protos = [client_mean_prototypes(acc) for acc in _proto_accum]
                _per_client_global_protos = apply_align_target(
                    _per_client_global_protos, _own_protos, align_target, fl_round_idx, seed
                )
                # Payload actually crossing the wire: accumulated sums + counts up,
                # the personalised consensus down.  The `own` control exchanges
                # nothing, so it is charged nothing.
                for i in roster:
                    _up_b = 0 if align_target == "own" else payload_nbytes(_proto_accum[i])
                    _down_b = 0 if align_target == "own" else payload_nbytes(_per_client_global_protos[i])
                    _ev = day_events.get(i)
                    if _ev is not None:
                        _ev.update(participated=1, bytes_up=_up_b, bytes_down=_down_b)
                for cls in sorted({c for pcp in _per_client_global_protos for c in pcp}):
                    for i in roster:
                        if cls in _per_client_global_protos[i]:
                            round_rows.append(
                                {
                                    "round": fl_round_idx,
                                    "date": day_key,
                                    "client": client_names[i],
                                    "regime": regime_name_map.get(int(cls), str(cls)),
                                    "align_loss_mean": loss_by_idx.get(i, float("nan")),
                                    "update_l2_norm": float("nan"),
                                    "prototype_norms": float(_per_client_global_protos[i][cls].norm()),
                                }
                            )
                _attn_arr = np.full((len(clients), len(clients), n_regimes), np.nan, dtype=np.float32)
                for cls, mat in _attn_weights.items():
                    if 0 <= int(cls) < n_regimes:
                        _attn_arr[:, :, int(cls)] = np.asarray(mat, dtype=np.float32)
                attention_rounds.append(_attn_arr)
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

                # Export-ready diagnostics rows for reproducibility / paper appendix.
                for cls in _all_classes:
                    cls_name = PROTO_CLASS_NAMES.get(cls, str(cls))
                    for i in range(len(client_names)):
                        support_count = 0
                        if cls in _proto_accum[i]:
                            support_count = int(_proto_accum[i][cls][1])
                        if _proto_edge_accum[i]["count"] > 0:
                            low_mean = _proto_edge_accum[i]["low_sum"] / _proto_edge_accum[i]["count"]
                            high_mean = _proto_edge_accum[i]["high_sum"] / _proto_edge_accum[i]["count"]
                        else:
                            low_mean, high_mean = float("nan"), float("nan")

                        for j in range(len(client_names)):
                            attn_ij = float(
                                _attn_weights.get(cls, [[0.0] * len(client_names) for _ in range(len(client_names))])[i][j]
                            )
                            fedproto_round_diagnostics.append(
                                {
                                    "fl_round": fl_round_idx,
                                    "day": day_key,
                                    "class_id": int(cls),
                                    "class_name": cls_name,
                                    "source_client_idx": int(i),
                                    "source_client": client_names[i],
                                    "target_client_idx": int(j),
                                    "target_client": client_names[j],
                                    "attention_weight": attn_ij,
                                    "source_support_count": support_count,
                                    "source_kc_edge_low_mean": low_mean,
                                    "source_kc_edge_high_mean": high_mean,
                                }
                            )

                # Reset accumulators for next sync interval
                _proto_accum = [{} for _ in clients]
                _proto_edge_accum = [{"low_sum": 0.0, "high_sum": 0.0, "count": 0.0} for _ in clients]

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
                        weights=[_round_sample_counts[i] for i in gated_indices],
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
                            # Keep LOSO holdout's local gating prototypes untouched:
                            # it is intentionally excluded from federation.
                            if loso_holdout and client_names[i] == loso_holdout:
                                continue
                            _gated_prototypes[i] = global_csi_protos
                    print(f"  {'':6}      [gated] Aggregated adapters from {len(gated_indices)} gated client(s)")
                else:
                    wc = 0.0
                _gated_proto_accum = [{} for _ in clients]

            else:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Weight-based FL algorithms (this also covers ditto's
                # global copy — its personalised model is handled separately,
                # every day, outside this aggregation block).
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                agg_fn = AGGREGATION_FNS["fedavg" if algorithm == "fedavg_ft" else algorithm]
                state_dicts_all = _state_dicts if _state_dicts is not None else [m.state_dict() for m in models]

                # `roster` already excludes the LOSO holdout, clients that have
                # not joined, and clients absent under the participation
                # schedule.  Absent clients neither contribute their weights nor
                # receive the aggregate back (distribution loop below).
                _participating = roster
                state_dicts = [state_dicts_all[i] for i in _participating]
                _weights = [_round_sample_counts[i] for i in _participating]

                # Capture local-only layers before aggregation
                if algorithm == "fedbn":
                    local_bn = [_extract_local_keys(sd, is_bn_key) for sd in state_dicts_all]
                if algorithm == "fedper":
                    local_head = [_extract_local_keys(sd, is_head_key) for sd in state_dicts_all]

                new_global, wc = agg_fn(state_dicts[0], state_dicts, weights=_weights)
                last_global_state_dict = copy.deepcopy(new_global)

                # Distribute back — restore local layers per client
                for i, model in enumerate(models):
                    if i not in roster:
                        continue  # absent this round: stays fully local, untouched
                    merged = new_global
                    if algorithm == "fedbn":
                        merged = _restore_local_keys(merged, local_bn[i])
                    elif algorithm == "fedper":
                        merged = _restore_local_keys(merged, local_head[i])
                    elif algorithm == "fedema":
                        # Soft interpolation: w ← α·w_local + (1-α)·w_global
                        merged = _ema_interpolate(state_dicts_all[i], new_global, fedema_alpha)
                    model.load_state_dict(merged)
                    _ev = day_events.get(i)
                    if _ev is not None:
                        _ev.update(
                            participated=1,
                            bytes_up=payload_nbytes(state_dicts_all[i]),
                            bytes_down=payload_nbytes(merged),
                        )
                    round_rows.append(
                        {
                            "round": fl_round_idx,
                            "date": day_key,
                            "client": client_names[i],
                            "regime": "all",
                            "align_loss_mean": float("nan"),
                            "update_l2_norm": _compute_weight_change(state_dicts_all[i], merged),
                            "prototype_norms": float("nan"),
                        }
                    )

                # FedAvg + local fine-tuning: recover the personalisation that
                # averaging just erased, before the next day's evaluation.
                if algorithm == "fedavg_ft":
                    for i in roster:
                        train_day(
                            models[i],
                            datasets[i],
                            day_maps[i].get(day_key, []),
                            lr=adapt_lr,
                            weight_decay=adapt_weight_decay,
                            max_grad_norm=adapt_max_grad_norm,
                            batch_size=adapt_batch_size,
                            epochs=ft_epochs,
                            device=device,
                            rng_seed=seed + day_num * len(clients) + i,
                            client_name=f"{client_names[i]} ft",
                        )

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
            # This round's counts have now been used for weighted aggregation
            # (or, for fedproto, ignored in favour of prototype support
            # counts) — start accumulating fresh for the next round.
            _round_sample_counts = [0 for _ in clients]

            _loss_str = "  ".join(
                f"{client_names[_i]}={_loss:.5f}"
                for _i, _loss in sorted(loss_by_idx.items())
                if day_record.get(f"{client_names[_i]}_n", 0) > 0
            )
            print(
                f"  {'':6}      └─ {round_tag}FL Round {fl_round_idx:3d}  ({per_round[-1]['days_in_round']}d)  ΔW={wc:.5f}  losses: {_loss_str}"
            )

        # -----------------------------------------------------------------
        # Ditto personalised-model training — runs every day (not just on
        # aggregation rounds), otherwise the personalised model is stale for
        # every day between sync rounds while still being used for that
        # day's evaluation metrics (`_infer_models` above). Anchored to the
        # latest known global consensus; the LOSO holdout trains with no
        # anchor (mu=0) since it never participates in federation either way.
        # -----------------------------------------------------------------
        if algorithm == "ditto" and not is_static:

            def _ditto_train_client(
                i: int,
                day_key: str = day_key,
                day_num: int = day_num,
                last_global_state_dict: dict[str, Any] | None = last_global_state_dict,
            ) -> None:
                _d_name = client_names[i]
                _d_indices = day_maps[i].get(day_key, [])
                _d_seed = seed + day_num * len(clients) + i
                _is_holdout = loso_holdout is not None and client_names[i] == loso_holdout
                _d_mu = 0.0 if _is_holdout else ditto_lambda
                _d_anchor = None if _is_holdout else last_global_state_dict
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
                    global_state_dict=_d_anchor,
                    mu=_d_mu,
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
            # Forecast skill against Smart Persistence pooled over exactly the
            # days this client contributed predictions on — same days, same
            # samples, so numerator and denominator are comparable.
            _sse = sum(sp_day_stats[i][d][0] for d in evaluated_days[i] if d in sp_day_stats[i])
            _n = sum(sp_day_stats[i][d][1] for d in evaluated_days[i] if d in sp_day_stats[i])
            _sp = float(np.sqrt(_sse / _n)) if _n else float("nan")
            overall_metrics[name]["sp_rmse"] = _sp
            overall_metrics[name]["sp_count"] = _n
            overall_metrics[name]["fs"] = (
                100.0 * (1.0 - overall_metrics[name]["rmse"] / _sp) if _n and np.isfinite(_sp) and _sp > 0 else float("nan")
            )
        else:
            overall_metrics[name] = {}

    _nw = max(10, max(len(n) for n in client_names))
    print(f"\n{'=' * 80}")
    print(f"  Final Results — {algorithm.upper()}  (FL rounds fired: {fl_round_idx})")
    print(f"{'=' * 80}")
    print(f"  {'Client':<{_nw}}  {'MAE':>9}  {'RMSE':>9}  {'WAPE':>8}  {'SMAPE':>8}  {'FS':>7}  {'Count':>8}")
    print(f"  {'-' * _nw}  {'-' * 9}  {'-' * 9}  {'-' * 8}  {'-' * 8}  {'-' * 7}  {'-' * 8}")
    for name in client_names:
        m = overall_metrics[name]
        print(
            f"  {name:<{_nw}}  "
            f"{m.get('mae', float('nan')):>9.5f}  "
            f"{m.get('rmse', float('nan')):>9.5f}  "
            f"{m.get('wape', float('nan')):>7.2f}%  "
            f"{m.get('smape', float('nan')):>7.2f}%  "
            f"{m.get('fs', float('nan')):>6.2f}%  "
            f"{m.get('count', 0):>8}"
        )
    print(f"{'=' * 80}\n")

    return {
        "algorithm": algorithm,
        "regime_label": regime_label,
        "regime_edges": list(edges),
        "proto_min_ghi_clear": proto_min_ghi_clear,
        "min_ghi_clear": min_ghi_clear,
        "attention": ("fixed" if attention_temperature is not None else "on") if attention else "uniform",
        "attention_temperature": attention_temperature,
        "align_target": align_target,
        "participation": participation,
        "join_days": join_days,
        "local_history_days": local_history_days,
        "wall_time_s": round(time.time() - started_at, 3),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "metrics_daily": metrics_daily,
        "metrics_regime": metrics_regime,
        "events": events,
        "round_rows": round_rows,
        "attention_rounds": attention_rounds,
        "regime_names": regime_name_map,
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
        "fedproto_round_diagnostics": fedproto_round_diagnostics,
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
        [
            "static",
            "isolated",
            "pooled",
            "fedavg",
            "fedavg_ft",
            "fedbn",
            "fedper",
            "fedprox",
            "ditto",
            "fedema",
            "fedproto",
            "gated",
        ],
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
        "  gated — CSI-binned prototypes + adapter aggregation from gated clients.\n"
        "  pooled — oracle: one model trained daily on the union of all clients.\n"
        "  fedavg_ft — FedAvg followed by local fine-tuning after each round."
    ),
)
@click.option(
    "--regime-label",
    type=click.Choice(REGIME_LABELS, case_sensitive=False),
    default="clear_sky",
    show_default=True,
    help=(
        "Source of the sky-regime label used for prototype stratification.\n"
        "  clear_sky — GHI / Ineichen-Perez clear-sky GHI (site- and time-aware).\n"
        "  proxy     — GHI / 1000 W/m^2 (the pre-2026 behaviour; latitude-dependent)."
    ),
)
@click.option(
    "--regime-edges",
    type=str,
    default=None,
    help=(
        "Comma-separated ascending regime cut points, e.g. '0.3,0.85'.  "
        "Default: the label's own defaults.  Empty string = one regime (no stratification)."
    ),
)
@click.option(
    "--min-ghi-clear",
    type=float,
    default=DEFAULT_MIN_GHI_CLEAR,
    show_default=True,
    help=(
        "Minimum clear-sky GHI in kW/m^2 for a sample to enter the simulation at all. "
        "Applied once where the day index map is built, so training, evaluation, "
        "prototypes and the Forecast-Skill denominator all score the same population. "
        "0 keeps every sample, including night."
    ),
)
@click.option(
    "--proto-min-ghi-clear",
    type=float,
    default=DEFAULT_MIN_GHI_CLEAR,
    show_default=True,
    help=(
        "Minimum clear-sky GHI in kW/m^2 for a sample to carry a sky regime.  Samples "
        "below it are still predicted and still counted in the metrics, but contribute "
        "to no prototype and are pulled toward none.  0 disables the floor and restores "
        "the pre-2026-09 behaviour, in which night frames landed in the clear regime."
    ),
)
@click.option(
    "--attention",
    type=click.Choice(["on", "fixed", "uniform"], case_sensitive=False),
    default="on",
    show_default=True,
    help=(
        "Prototype aggregation.  on = similarity+support attention at an adaptive "
        "temperature (the implemented mechanism); fixed = the same attention at the "
        "static tau of --attention-temperature (the mechanism the thesis text "
        "describes); uniform = a plain unweighted mean."
    ),
)
@click.option(
    "--attention-temperature",
    type=float,
    default=0.5,
    show_default=True,
    help="Softmax temperature tau, used only when --attention=fixed.",
)
@click.option(
    "--stratify",
    type=click.Choice(["on", "single"], case_sensitive=False),
    default="on",
    show_default=True,
    help="'single' collapses every sample into one regime (overrides --regime-edges).",
)
@click.option(
    "--align-target",
    type=click.Choice(ALIGN_TARGETS, case_sensitive=False),
    default="global",
    show_default=True,
    help=(
        "Prototype control.  global = the aggregated consensus; own = no exchange "
        "(zero bytes); shuffled = peers' prototypes assigned to the wrong regimes; "
        "random = norm-matched Gaussian noise."
    ),
)
@click.option(
    "--participation",
    type=str,
    default="full",
    show_default=True,
    help="Round participation: 'full', 'dropout:<p>', or 'outage:<client>:<start_round>:<n_rounds>'.",
)
@click.option(
    "--join-day",
    "join_day",
    type=str,
    multiple=True,
    help="Late/cold-start join as '<client>:<day_index>'.  Repeat per client.",
)
@click.option(
    "--local-history-days",
    type=int,
    default=0,
    show_default=True,
    help="Days of its own past data a joining client may replay (needs --recent-buffer-size > 0).",
)
@click.option(
    "--ft-epochs",
    type=int,
    default=1,
    show_default=True,
    help="Local fine-tuning epochs after each round.  Only used when --algorithm=fedavg_ft.",
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
        "  <dir>/fedproto_round_diagnostics.csv — per-round FedProto attention/support/edge logs (fedproto only).\n"
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
    regime_label: str,
    regime_edges: str | None,
    proto_min_ghi_clear: float,
    min_ghi_clear: float,
    attention: str,
    attention_temperature: float,
    stratify: str,
    align_target: str,
    participation: str,
    join_day: tuple[str, ...],
    local_history_days: int,
    ft_epochs: int,
) -> None:
    """Run the FL benchmark comparing federation algorithms on solar irradiance forecasting."""
    # Build ordered client list from all --dataset flags
    # Strip common suffixes (_prepared) to get a clean site name
    clients: list[tuple[str, Path]] = []
    for ds in dataset:
        raw = ds.parent.name if ds.name == "." else ds.name
        name = raw.removesuffix("_prepared")
        clients.append((name, ds))

    edges = () if stratify == "single" else parse_edges(regime_edges, regime_label)
    join_days = {}
    for spec in join_day:
        name, _, day = spec.partition(":")
        join_days[name] = int(day)

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
        regime_label=regime_label,
        regime_edges=edges,
        proto_min_ghi_clear=proto_min_ghi_clear,
        min_ghi_clear=min_ghi_clear,
        attention=attention != "uniform",
        attention_temperature=attention_temperature if attention == "fixed" else None,
        align_target=align_target,
        participation=participation,
        join_days=join_days,
        local_history_days=local_history_days,
        ft_epochs=ft_epochs,
    )

    if save_results is not None:
        save_results.mkdir(parents=True, exist_ok=True)

        # Per-day CSV (wide, kept for backward compatibility with older analyses)
        df_day = pd.DataFrame(results["per_day"])
        df_day.to_csv(save_results / "per_day.csv", index=False)

        # Long-format logs consumed by the Chapter 5 analysis
        pd.DataFrame(results["metrics_daily"]).to_csv(save_results / "metrics_daily.csv", index=False)
        pd.DataFrame(results["metrics_regime"]).to_csv(save_results / "metrics_regime.csv", index=False)
        pd.DataFrame(results["events"]).to_csv(save_results / "events.csv", index=False)
        pd.DataFrame(results["round_rows"]).to_csv(save_results / "rounds.csv", index=False)
        _attn = results["attention_rounds"]
        np.save(
            save_results / "attention.npy",
            np.stack(_attn) if _attn else np.empty((0, len(results["client_names"]), len(results["client_names"]), 0)),
        )

        # Fully resolved configuration + provenance, so a run can be replayed
        # without reconstructing the command line.
        with (save_results / "config.yaml").open("w") as fh:
            yaml.safe_dump(
                {
                    "git_commit": git_commit(),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "command": "fl_benchmark",
                    "algorithm": algorithm,
                    "clients": [(name, str(path)) for name, path in clients],
                    "sync_interval": sync_interval,
                    "num_days": num_days,
                    "start_day": start_day,
                    "warmup_days": warmup_days,
                    "seed": seed,
                    "device": device,
                    "adapt": {
                        "lr": adapt_lr,
                        "weight_decay": adapt_weight_decay,
                        "max_grad_norm": adapt_max_grad_norm,
                        "batch_size": adapt_batch_size,
                        "epochs": adapt_epochs,
                    },
                    "replay": {
                        "buffer_size": replay_buffer_size,
                        "ratio": replay_ratio,
                        "recent_buffer_size": recent_buffer_size,
                        "recent_ratio": recent_replay_ratio,
                    },
                    "regime_label": regime_label,
                    "regime_edges": list(edges),
                    "proto_min_ghi_clear": proto_min_ghi_clear,
                    "min_ghi_clear": min_ghi_clear,
                    "attention": attention,
                    "attention_temperature": attention_temperature,
                    "stratify": stratify,
                    "align_target": align_target,
                    "participation": participation,
                    "join_days": join_days,
                    "local_history_days": local_history_days,
                    "ft_epochs": ft_epochs,
                    "loso_holdout": loso_holdout,
                    "method_params": {
                        "fedprox_mu": fedprox_mu,
                        "ditto_lambda": ditto_lambda,
                        "fedema_alpha": fedema_alpha,
                        "fedproto_lambda": fedproto_lambda,
                    },
                },
                fh,
                sort_keys=False,
            )

        # Per-round CSV
        if results["per_round"]:
            df_round = pd.DataFrame(results["per_round"])
            df_round.to_csv(save_results / "per_round.csv", index=False)

        # FedProto diagnostics (only populated for --algorithm fedproto)
        if results.get("fedproto_round_diagnostics"):
            df_proto_diag = pd.DataFrame(results["fedproto_round_diagnostics"])
            df_proto_diag.to_csv(save_results / "fedproto_round_diagnostics.csv", index=False)

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
        summary = {
            k: v
            for k, v in results.items()
            if k
            not in (
                "per_day",
                "per_round",
                "fedproto_round_diagnostics",
                "models",
                "last_global_state_dict",
                "metrics_daily",
                "metrics_regime",
                "events",
                "round_rows",
                "attention_rounds",
            )
        }
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
        if results.get("fedproto_round_diagnostics"):
            print(f"  fedproto_round_diagnostics.csv ({len(results['fedproto_round_diagnostics'])} rows)")
        print("  metrics_daily.csv / metrics_regime.csv / events.csv / rounds.csv / attention.npy")
        print("  config.yaml")
        print("  summary.json")
        _n_global = 1 if results["last_global_state_dict"] is not None else 0
        print(f"  checkpoints/  ({len(_final_models)} local models{', 1 global' if _n_global else ''})")


if __name__ == "__main__":
    main()
