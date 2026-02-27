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
      --algorithm fedbn \\
      --sync-interval 7 \\
      --num-days 90 \\
      --save-results results/fedbn_7day
"""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cirr.datamodules.forecasting import ForecastingDataModule
from cirr.fl.server import AGGREGATION_FNS, _is_bn_key, _is_head_key
from cirr.models.forecaster import Forecaster
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day
from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FedProto: kc-stratified prototype classes
# ---------------------------------------------------------------------------
# Weather conditions are classified from normalised target irradiance
# (target_irradiance / 1000).  These thresholds approximate the clearness
# index (kc) bins commonly used in solar forecasting literature.
PROTO_OVERCAST = 0  # heavy cloud cover
PROTO_TRANSIENT = 1  # variable / partly cloudy
PROTO_CLEAR = 2  # clear or mostly clear sky
PROTO_CLASS_NAMES = {PROTO_OVERCAST: "overcast", PROTO_TRANSIENT: "transient", PROTO_CLEAR: "clear"}

_KC_OVERCAST_THRESHOLD = 0.15  # norm irradiance < 150 W/m²
_KC_CLEAR_THRESHOLD = 0.50  # norm irradiance >= 500 W/m²


def _classify_irradiance(norm_irradiance: float) -> int:
    """Classify a normalised irradiance value into a prototype weather class.

    Parameters
    ----------
    norm_irradiance : float
        Target irradiance normalised to [0, 1] (i.e. target / 1000 W/m²).

    Returns
    -------
    class_idx : int
        One of ``PROTO_OVERCAST``, ``PROTO_TRANSIENT``, ``PROTO_CLEAR``.
    """
    if norm_irradiance < _KC_OVERCAST_THRESHOLD:
        return PROTO_OVERCAST
    elif norm_irradiance < _KC_CLEAR_THRESHOLD:
        return PROTO_TRANSIENT
    else:
        return PROTO_CLEAR


def _extract_day_prototypes(
    model: Forecaster,
    day_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str,
    batch_size: int,
) -> dict[int, tuple[torch.Tensor, int]]:
    """Extract per-class prototype feature vectors from a day's samples.

    Passes samples through the image encoder and accumulates feature sums
    per weather class (Clear, Overcast, Transient).

    Parameters
    ----------
    model : Forecaster
        Model whose ``image_encoder`` produces the feature vectors.
    day_samples : list
        Pre-loaded ``(image, irr_hist, target)`` tensor tuples.
    device : str
        Compute device.
    batch_size : int
        Batch size for feature extraction.

    Returns
    -------
    prototypes : dict[int, tuple[Tensor, int]]
        Mapping from class index to ``(sum_of_features, sample_count)``.
        Feature sums are on CPU.
    """
    if not day_samples:
        return {}

    model.eval()
    proto_sums: dict[int, torch.Tensor] = {}
    proto_counts: dict[int, int] = {}

    with torch.no_grad():
        for start in range(0, len(day_samples), batch_size):
            batch = day_samples[start : start + batch_size]
            imgs = torch.stack([s[0] for s in batch]).to(device)
            tgts = [s[2] for s in batch]

            features = model.image_encoder(imgs)  # [B, num_features]

            for b in range(len(batch)):
                tgt_val = tgts[b].item() if isinstance(tgts[b], torch.Tensor) else float(tgts[b])
                cls = _classify_irradiance(tgt_val)
                feat = features[b].cpu()

                if cls not in proto_sums:
                    proto_sums[cls] = torch.zeros_like(feat)
                    proto_counts[cls] = 0
                proto_sums[cls] += feat
                proto_counts[cls] += 1

    return {cls: (proto_sums[cls], proto_counts[cls]) for cls in proto_sums}


def _accumulate_prototypes(
    accum: dict[int, tuple[torch.Tensor, int]],
    day_protos: dict[int, tuple[torch.Tensor, int]],
) -> dict[int, tuple[torch.Tensor, int]]:
    """Merge *day_protos* into running accumulator *accum* (in-place).

    Returns the modified *accum* for convenience.
    """
    for cls, (feat_sum, count) in day_protos.items():
        if cls in accum:
            old_sum, old_count = accum[cls]
            accum[cls] = (old_sum + feat_sum, old_count + count)
        else:
            accum[cls] = (feat_sum.clone(), count)
    return accum


def _aggregate_global_prototypes(
    client_protos: list[dict[int, tuple[torch.Tensor, int]]],
) -> dict[int, torch.Tensor]:
    """Average per-class prototypes across all clients (weighted by count).

    Each client's prototype for a class is the feature-vector sum and
    sample count accumulated since the last sync.  The global prototype
    is the *count-weighted* average across clients — clients with more
    samples in a given class contribute proportionally more.

    Parameters
    ----------
    client_protos : list[dict[int, tuple[Tensor, int]]]
        Per-client accumulated prototypes: class → ``(sum_features, count)``.

    Returns
    -------
    global_prototypes : dict[int, Tensor]
        Global mean prototype for each weather class (on CPU).
    """
    global_protos: dict[int, torch.Tensor] = {}

    for cls in (PROTO_OVERCAST, PROTO_TRANSIENT, PROTO_CLEAR):
        total_sum: torch.Tensor | None = None
        total_count = 0

        for client_proto in client_protos:
            if cls in client_proto:
                feat_sum, count = client_proto[cls]
                if total_sum is None:
                    total_sum = feat_sum.clone()
                else:
                    total_sum += feat_sum
                total_count += count

        if total_sum is not None and total_count > 0:
            global_protos[cls] = total_sum / total_count

    return global_protos


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


def _load_checkpoint(checkpoint_path: Path, device: str, freeze_early_backbone: bool = False) -> Forecaster:
    """Instantiate a Forecaster and load weights from *checkpoint_path*.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file.
    device : str
        Device to load the model on.
    freeze_early_backbone : bool, optional
        If True, freezes conv1, bn1, layer1, layer2 in the ResNet backbone.
    """
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

    if freeze_early_backbone:
        model.freeze_early_backbone()

    return model


# ---------------------------------------------------------------------------
# Per-day inference / training
# ---------------------------------------------------------------------------


def _predict_day(
    model: Forecaster,
    dataset: Any,
    day_indices: list[int],
    device: str,
    batch_size: int,
    client_name: str = "",
) -> tuple[np.ndarray, np.ndarray, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Run batched inference for every sample in *day_indices*.

    Returns
    -------
    predictions  : np.ndarray, shape (N,)
    targets      : np.ndarray, shape (N,)
    day_samples  : list of (image, irr_hist, target) tensor tuples on CPU,
                   one per sample — reusable by ``_train_day`` to avoid
                   redundant dataset loads.
    """
    if not day_indices:
        return np.array([]), np.array([]), []

    model.eval()
    preds, tgts = [], []
    day_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        _pbar = tqdm(
            total=len(day_indices),
            desc=f"  [{client_name}] load+infer",
            leave=False,
            unit="smp",
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for start in range(0, len(day_indices), batch_size):
            batch_idx = day_indices[start : start + batch_size]
            imgs, hists, targets = [], [], []
            for idx in batch_idx:
                images, irr_hist, target, _ = dataset[idx]
                imgs.append(images)
                hists.append(irr_hist)
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)
                targets.append(target)
                # Cache each sample as-is (already on CPU)
                day_samples.append((images, irr_hist, target))
                _pbar.update(1)
            imgs_t = torch.stack(imgs).to(device)
            hists_t = torch.stack(hists).to(device)
            mean, _ = model(imgs_t, hists_t)
            preds.append(mean.cpu().numpy().ravel())
            tgts.extend(t.item() for t in targets)
        _pbar.close()
    return (
        np.concatenate(preds) if preds else np.array([]),
        np.array(tgts),
        day_samples,
    )


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
    global_state_dict: dict[str, Any] | None = None,
    mu: float = 0.0,
    rng_seed: int | None = None,
    day_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    client_name: str = "",
    global_prototypes: dict[int, torch.Tensor] | None = None,
    proto_lambda: float = 0.0,
) -> float:
    """Train *model* for *epochs* epochs on *day_indices*.

    *replay_entries* is a pre-sampled list of ``(dataset, index)`` pairs
    (assembled by the caller via ``_sample_replay_for_day()``) that is
    concatenated with the current-day samples before batching — matching
    the design of ``train_on_day`` in ``local_training.py``.
    The pool is reshuffled at the start of every epoch.

    Parameters
    ----------
    global_state_dict : dict, optional
        Reference weights for the proximal term (FedProx / Ditto).  When
        provided together with ``mu > 0``, the loss is augmented with
        ``(μ/2) ‖w - w_global‖²``.
    mu : float
        Proximal regularisation strength.  0 disables the proximal term.
    day_samples : list, optional
        Pre-loaded (image, irr_hist, target) tensor tuples from
        ``_predict_day()``.  Avoids redundant dataset reads.
    global_prototypes : dict, optional
        FedProto global prototypes: mapping from weather class index
        (``PROTO_OVERCAST``, ``PROTO_TRANSIENT``, ``PROTO_CLEAR``) to a
        mean feature-vector tensor.  When provided together with
        ``proto_lambda > 0``, the loss is augmented with a prototype
        regularisation term that pulls local feature representations
        toward the global consensus.
    proto_lambda : float
        Prototype regularisation strength for FedProto.
        ``loss += λ · mean(‖feature(x) - global_proto[class(x)]‖²)``.
        0 disables the prototype term.

    Returns
    -------
    avg_loss : float
    """
    if not day_indices:
        return 0.0

    # Build the training pool.  When day_samples is available we already have
    # the tensors; replay entries are also materialised to the same format
    # so the entire pool is homogeneous (img, hist, tgt) tensor tuples.
    if day_samples is not None:
        preloaded_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(day_samples)
        if replay_entries:
            for ds, idx in tqdm(
                replay_entries,
                desc=f"  [{client_name}] load replay",
                leave=False,
                unit="smp",
                file=sys.stdout,
                dynamic_ncols=True,
            ):
                img, hist, tgt, _ = ds[idx]
                if not isinstance(tgt, torch.Tensor):
                    tgt = torch.tensor(tgt)
                preloaded_pool.append((img, hist, tgt))
        pool_base: list = preloaded_pool
        _preloaded = True
    else:
        current_ds: list[tuple[Any, int]] = [(dataset, idx) for idx in day_indices]
        pool_base = current_ds + (replay_entries or [])
        _preloaded = False

    model.train()

    # Pre-compute detached reference tensors for proximal regularisation
    # (FedProx: μ/2 · ‖w - w_global‖²,  Ditto: λ/2 · ‖v - w_global‖²)
    _prox_ref: dict[str, torch.Tensor] | None = None
    if mu > 0.0 and global_state_dict is not None:
        _prox_ref = {k: v.detach().to(device) for k, v in global_state_dict.items() if v.is_floating_point()}

    # Pre-move global prototypes to device for FedProto regularisation
    _use_proto = global_prototypes is not None and proto_lambda > 0.0
    _proto_on_device: dict[int, torch.Tensor] = {}
    if _use_proto:
        _proto_on_device = {cls: proto.detach().to(device) for cls, proto in global_prototypes.items()}

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_loss = 0.0
    count = 0
    _rng = random.Random(rng_seed)

    for ep in range(epochs):
        # Reshuffle the fixed pool each epoch (replay+current already sized by caller)
        pool = list(pool_base)
        _rng.shuffle(pool)
        batches = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]

        _pbar = tqdm(
            batches,
            desc=f"  [{client_name}] train e{ep+1}/{epochs}",
            leave=False,
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for batch in _pbar:
            if _preloaded:
                # Pool entries are (img, hist, target) tensor tuples
                imgs = torch.stack([item[0] for item in batch]).to(device)
                hist = torch.stack([item[1] for item in batch]).to(device)
                tgts = torch.stack([item[2] for item in batch]).to(device)
            else:
                # Pool entries are (dataset, idx) tuples; load from dataset
                imgs = torch.stack([ds[idx][0] for ds, idx in batch]).to(device)  # type: ignore
                hist = torch.stack([ds[idx][1] for ds, idx in batch]).to(device)  # type: ignore
                tgts = torch.stack([ds[idx][2] for ds, idx in batch]).to(device)  # type: ignore

            optimizer.zero_grad()

            # Forward pass — use forward_with_features when prototype
            # regularisation is active to get intermediate encoder features
            # in a single pass (no redundant encoder computation).
            if _use_proto:
                mean, var, img_feats = model.forward_with_features(imgs, hist)
            else:
                mean, var = model(imgs, hist)

            if var is not None:
                loss = model.loss_fn(mean, tgts, var)
            elif model.loss_name == "Berhu":
                loss = model.loss_fn(mean * 1000, tgts * 1000)
            else:
                loss = model.loss_fn(mean, tgts)

            # Proximal term: (μ/2) · ‖w - w_global‖²  (FedProx / Ditto)
            if _prox_ref is not None:
                prox = sum(
                    ((p - _prox_ref[n]) ** 2).sum() for n, p in model.named_parameters() if p.requires_grad and n in _prox_ref
                )
                loss = loss + (mu / 2.0) * prox

            # Prototype regularisation: λ · mean(‖f(x) - proto[class(x)]‖²)
            # Pulls local feature representations toward the global consensus
            # prototype for each weather condition (Clear / Overcast / Transient).
            if _use_proto:
                proto_loss = torch.tensor(0.0, device=device)
                n_proto = 0
                for b in range(imgs.size(0)):
                    tgt_val = tgts[b].item()
                    cls = _classify_irradiance(tgt_val)
                    if cls in _proto_on_device:
                        proto_loss = proto_loss + F.mse_loss(img_feats[b], _proto_on_device[cls])
                        n_proto += 1
                if n_proto > 0:
                    loss = loss + proto_lambda * (proto_loss / n_proto)

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
        preds, tgts, day_samples = _predict_day(
            infer_model, dataset, day_indices, device, batch_size=adapt_batch_size, client_name=name
        )
        metrics = compute_metrics(preds, tgts) if len(preds) > 0 else {}

        # Step 2: Prepare replay samples
        _reservoir_entries: list[tuple[Any, int]] = []
        if replay_buffer_size > 0 and replay_ratio > 0.0 and folsom_replay_buffer:
            _n = max(0, int(round(len(day_indices) * replay_ratio)))
            _reservoir_entries = _sample_replay_for_day(folsom_replay_buffer, _n, seed=rng_seed)
        _recent_entries: list[tuple[Any, int]] = []
        if recent_buffer_size > 0 and recent_replay_ratio > 0.0:
            _n = max(0, int(round(len(day_indices) * recent_replay_ratio)))
            _recent_entries = recent_buffer.sample(_n, seed=rng_seed + 1)
        _replay = (_reservoir_entries + _recent_entries) or None

        # Step 3: Train (reuses cached day_samples to skip redundant dataset loads)
        loss = _train_day(
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
        if extract_prototypes and day_samples:
            day_protos = _extract_day_prototypes(model, day_samples, device, adapt_batch_size)

        # Step 5: Update recent FIFO buffer
        if recent_buffer_size > 0:
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
    fedproto_lambda     : Prototype regularisation strength for FedProto.
                          During local training the loss is augmented with
                          λ · mean(‖feature(x) - global_proto[class(x)]‖²)
                          where class(x) is the kc-stratified weather class
                          (Clear / Overcast / Transient).  Higher values pull
                          the local feature space closer to the global
                          consensus.

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
        _folsom_replay_buffer, _folsom_dist = _build_initial_replay_buffer(_folsom_path, replay_buffer_size)
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

    # --- FedProto state: per-class prototype accumulators and global prototypes ---
    _proto_accum: list[dict[int, tuple[torch.Tensor, int]]] = [{} for _ in clients]
    _global_prototypes: dict[int, torch.Tensor] | None = None  # set after first FedProto sync

    # -----------------------------------------------------------------------
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
                "global_prototypes": _global_prototypes if _is_fedproto else None,
                "proto_lambda": fedproto_lambda if _is_fedproto else 0.0,
                "extract_prototypes": _is_fedproto and not in_warmup,
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
                _accumulate_prototypes(_proto_accum[i], result["day_prototypes"])

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
        _do_aggregate = algorithm != "isolated" and (warmup_ended_today or days_since_sync >= sync_interval)
        if warmup_ended_today and not in_warmup:
            # Reset counter so the next periodic sync starts fresh
            days_since_sync = 0

        if _do_aggregate:
            if algorithm == "fedproto":
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FedProto: aggregate prototypes, NOT model weights.
                # Each client keeps its own model entirely local.
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                _global_prototypes = _aggregate_global_prototypes(_proto_accum)
                wc = 0.0  # no weight-level change

                # Log prototype summary
                _proto_cls_str = "  ".join(
                    f"{PROTO_CLASS_NAMES[c]}={_global_prototypes[c].norm():.3f}" for c in sorted(_global_prototypes)
                )
                print(f"  {'':6}      [proto] Aggregated {len(_global_prototypes)} class prototypes: " f"{_proto_cls_str}")
                # Print per-client sample counts that contributed
                for ci, cname in enumerate(client_names):
                    _counts = {PROTO_CLASS_NAMES[c]: cnt for c, (_, cnt) in _proto_accum[ci].items()}
                    if _counts:
                        _cnt_str = "  ".join(f"{k}={v}" for k, v in _counts.items())
                        print(f"  {'':6}              {cname}: {_cnt_str}")

                # Reset accumulators for next sync interval
                _proto_accum = [{} for _ in clients]

            else:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Weight-based FL algorithms
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                agg_fn = AGGREGATION_FNS[algorithm]
                state_dicts = _state_dicts if _state_dicts is not None else [m.state_dict() for m in models]

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
                        _d_reservoir = _sample_replay_for_day(
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
                    _train_day(
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
        ["isolated", "fedavg", "fedbn", "fedper", "fedprox", "ditto", "fedema", "fedproto"], case_sensitive=False
    ),
    default="fedbn",
    show_default=True,
    help=(
        "FL aggregation algorithm.\n"
        "  isolated — no federation (lower bound).\n"
        "  fedavg   — standard weight averaging.\n"
        "  fedbn    — FedAvg excluding BatchNorm layers.\n"
        "  fedper   — aggregate backbone only, keep head local.\n"
        "  fedprox  — FedAvg + proximal term in local training.\n"
        "  ditto    — dual-model with personalised proximal anchor.\n"
        "  fedema   — FedAvg + EMA interpolation at distribution.\n"
        "  fedproto — share per-class feature prototypes, not weights."
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
        "Prototype regularisation strength for FedProto.  Local loss is "
        "augmented with λ·mean(‖feature(x)-global_proto[class(x)]‖²) where "
        "class(x) is the kc-stratified weather class (Clear / Overcast / "
        "Transient).  Only used when --algorithm=fedproto."
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
