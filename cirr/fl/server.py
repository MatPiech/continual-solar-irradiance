import copy
import io
from typing import Any

import torch

# Keys whose names indicate BatchNorm statistics / parameters.
# FedBN keeps these local (not aggregated) to preserve per-client data statistics.
_BN_KEY_PATTERNS = ("running_mean", "running_var", "num_batches_tracked", ".bn", "norm.")

# Key prefix for the network head used by FedPer / Base-Head split.
# Only the backbone (image_encoder) is aggregated; head stays local.
_HEAD_PREFIX = "network_head"


def payload_nbytes(payload: Any) -> int:
    """Serialized size of *payload* in bytes — what actually crosses the wire.

    Uses the same ``torch.save`` encoding as ``fl_edge_overhead_benchmark`` so
    the two payload figures are directly comparable, rather than counting
    ``numel * element_size`` and ignoring framing.
    """
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getbuffer().nbytes


def _weighted_stack_mean(tensors: list[torch.Tensor], weights: list[float] | None) -> torch.Tensor:
    """Weighted mean of *tensors* along a new leading dimension.

    ``weights`` are each client's relative contribution for this round (e.g.
    local sample count). ``None``, an all-zero, or an empty list falls back
    to a plain unweighted mean — this keeps every caller that doesn't pass
    weights (older scripts, ad-hoc tests) behaving exactly as before.
    """
    stacked = torch.stack(tensors)
    if not weights:
        return stacked.mean(dim=0)
    w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device)
    total = w.sum()
    if total <= 0:
        return stacked.mean(dim=0)
    w = w / total
    view_shape = (len(tensors),) + (1,) * (stacked.dim() - 1)
    return (stacked * w.view(view_shape)).sum(dim=0)


def federated_averaging(global_weights, client_updates, weights: list[float] | None = None):
    """Perform federated averaging on client model updates.

    Parameters
    ----------
    global_weights : dict
        The old model state dictionary.
    client_updates : list
        List of state_dicts from clients.
    weights : list[float], optional
        Per-client relative contribution (e.g. local sample count for this
        round), same order as *client_updates*. ``None`` reproduces the
        original unweighted mean; canonical FedAvg weights by ``n_k / n``,
        so pass sample counts here to match that definition.

    Returns
    -------
    new_weights : dict
        Averaged model weights.
    weight_change_mb : float
        L2 norm of global weight change.
    """
    if not client_updates:
        return global_weights, 0.0

    # Simple averaging
    new_weights = copy.deepcopy(global_weights)

    for key in new_weights:
        # Stack the tensors from all updates and take mean
        tensors = [update[key] for update in client_updates]

        # Only average floating-point tensors; keep first value for others (e.g., counters)
        if tensors[0].is_floating_point():
            new_weights[key] = _weighted_stack_mean(tensors, weights)
        else:
            new_weights[key] = tensors[0]

    # Compute weight change (global model change)
    total_change = 0.0
    for key in global_weights:
        if global_weights[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            diff = global_weights[key].float() - new_weights[key].float()
            total_change += torch.sum(diff**2).item()
    weight_change_mb = total_change**0.5

    return new_weights, weight_change_mb


def is_bn_key(key: str) -> bool:
    """Return True if *key* belongs to a BatchNorm layer (FedBN exclusion list)."""
    return any(pattern in key for pattern in _BN_KEY_PATTERNS)


def is_head_key(key: str) -> bool:
    """Return True if *key* belongs to the network head (FedPer exclusion list)."""
    return key.startswith(_HEAD_PREFIX)


def _compute_weight_change(old_weights: dict, new_weights: dict) -> float:
    """L2 norm of the difference between two weight dicts."""
    total = 0.0
    for key in old_weights:
        if old_weights[key].dtype in (torch.float32, torch.float16, torch.bfloat16):
            diff = old_weights[key].float() - new_weights[key].float()
            total += torch.sum(diff**2).item()
    return total**0.5


def fedbn_aggregate(
    global_weights: dict[str, Any],
    client_updates: list[dict[str, Any]],
    weights: list[float] | None = None,
) -> tuple[dict[str, Any], float]:
    """FedBN: aggregate all layers **except** BatchNorm statistics.

    BatchNorm parameters (running_mean, running_var, num_batches_tracked,
    and any key matching ``bn`` / ``norm``) are kept from the global model
    so that each client's local BN statistics are never overwritten by the
    server.  This preserves per-client covariate statistics and greatly
    reduces negative transfer across heterogeneous sites.

    Note: this benchmark still has each client upload its *full* state
    dict — only the server-side aggregation drops BN keys. The bandwidth
    savings FedBN is usually motivated by are not realised here; see
    ``scripts/fl_edge_overhead_benchmark.py`` for actual transport-cost
    measurements.

    Parameters
    ----------
    global_weights : dict
        Current global model state dictionary (used for BN keys).
    client_updates : list[dict]
        List of client state dictionaries submitted this round.
    weights : list[float], optional
        Per-client relative contribution (e.g. local sample count), same
        order as *client_updates*. ``None`` averages clients uniformly.

    Returns
    -------
    new_weights : dict
        Updated global state dict (BN keys are unchanged).
    weight_change : float
        L2 norm of the update to the global model.
    """
    if not client_updates:
        return global_weights, 0.0

    new_weights = copy.deepcopy(global_weights)
    for key in new_weights:
        if is_bn_key(key):
            # Keep global (= previous round) BN statistics — each client
            # restores its own local BN stats before fine-tuning.
            continue
        tensors = [update[key] for update in client_updates]
        if tensors[0].is_floating_point():
            new_weights[key] = _weighted_stack_mean(tensors, weights)
        else:
            new_weights[key] = tensors[0]

    return new_weights, _compute_weight_change(global_weights, new_weights)


def fedper_aggregate(
    global_weights: dict[str, Any],
    client_updates: list[dict[str, Any]],
    weights: list[float] | None = None,
) -> tuple[dict[str, Any], float]:
    """FedPer / Base-Head Split: aggregate **only** the backbone encoder.

    The ``network_head`` (final FC layers) is kept strictly local — every
    client retains its personalised regression head that maps global features
    to its own irradiance distribution.  Only the shared image encoder is
    averaged, giving clients globally generalised visual representations
    while allowing site-specific output mapping.

    Note: this benchmark still has each client upload its *full* state
    dict — only the server-side aggregation drops head keys. The bandwidth
    savings FedPer is usually motivated by are not realised here; see
    ``scripts/fl_edge_overhead_benchmark.py`` for actual transport-cost
    measurements.

    Parameters
    ----------
    global_weights : dict
        Current global model state dictionary (used for head keys).
    client_updates : list[dict]
        List of client state dictionaries submitted this round.
    weights : list[float], optional
        Per-client relative contribution (e.g. local sample count), same
        order as *client_updates*. ``None`` averages clients uniformly.

    Returns
    -------
    new_weights : dict
        Updated global state dict (network_head keys are unchanged).
    weight_change : float
        L2 norm of the update to the global model.
    """
    if not client_updates:
        return global_weights, 0.0

    new_weights = copy.deepcopy(global_weights)
    for key in new_weights:
        if is_head_key(key):
            # This client's head is personal — do not overwrite.
            continue
        tensors = [update[key] for update in client_updates]
        if tensors[0].is_floating_point():
            new_weights[key] = _weighted_stack_mean(tensors, weights)
        else:
            new_weights[key] = tensors[0]

    return new_weights, _compute_weight_change(global_weights, new_weights)


def gated_aggregate(
    global_weights: dict[str, Any],
    client_updates: list[dict[str, Any]],
    weights: list[float] | None = None,
) -> tuple[dict[str, Any], float]:
    """Gated FL: aggregate adapter (network_head) weights only.

    Backbone stays frozen/global; only clients that opened their gates should
    be included in *client_updates* by the benchmark runner.

    Parameters
    ----------
    weights : list[float], optional
        Per-client relative contribution (e.g. local sample count), same
        order as *client_updates*. ``None`` averages clients uniformly.
    """
    if not client_updates:
        return global_weights, 0.0

    new_weights = copy.deepcopy(global_weights)
    for key in new_weights:
        if not is_head_key(key):
            continue
        tensors = [update[key] for update in client_updates]
        if tensors[0].is_floating_point():
            new_weights[key] = _weighted_stack_mean(tensors, weights)
        else:
            new_weights[key] = tensors[0]

    return new_weights, _compute_weight_change(global_weights, new_weights)


# ---------------------------------------------------------------------------
# Soft-personalisation strategies
#   Server-side aggregation is identical to FedAvg for all three.  The
#   personalisation happens *locally*: either via a proximal training term
#   (FedProx / Ditto) or via EMA interpolation at weight distribution time
#   (FedEMA).  These thin wrappers carry self-documenting docstrings so
#   that every algorithm in the registry has its own entry point.
# ---------------------------------------------------------------------------


def fedprox_aggregate(
    global_weights: dict[str, Any],
    client_updates: list[dict[str, Any]],
    weights: list[float] | None = None,
) -> tuple[dict[str, Any], float]:
    """FedProx: FedAvg aggregation with proximal-term regularised local training.

    Server-side aggregation is identical to FedAvg — simple averaging of all
    client weight submissions.  The proximal regularisation term
    ``(μ/2) ‖w - w_global‖²`` is added **locally** during each client's
    training step, keeping local updates close to the last global model.

    Parameters
    ----------
    global_weights : dict
        Current global model state dictionary.
    client_updates : list[dict]
        List of client state dictionaries submitted this round.

    Returns
    -------
    new_weights : dict
        Averaged global state dict (identical to FedAvg).
    weight_change : float
        L2 norm of the update to the global model.
    """
    return federated_averaging(global_weights, client_updates, weights=weights)


def ditto_aggregate(
    global_weights: dict[str, Any],
    client_updates: list[dict[str, Any]],
    weights: list[float] | None = None,
) -> tuple[dict[str, Any], float]:
    """Ditto: FedAvg aggregation with per-client personalised models.

    Server-side aggregation is identical to FedAvg.  Each client maintains a
    *second* personalised model that is trained locally with a proximal term
    ``(λ/2) ‖v - w_global‖²`` anchoring it to the latest global model.  The
    personalised model is used for inference while the global copy participates
    in federation — implemented in ``run_fl_benchmark``.

    Parameters
    ----------
    global_weights : dict
        Current global model state dictionary.
    client_updates : list[dict]
        List of client state dictionaries submitted this round.

    Returns
    -------
    new_weights : dict
        Averaged global state dict (identical to FedAvg).
    weight_change : float
        L2 norm of the update to the global model.
    """
    return federated_averaging(global_weights, client_updates, weights=weights)


def fedema_aggregate(
    global_weights: dict[str, Any],
    client_updates: list[dict[str, Any]],
    weights: list[float] | None = None,
) -> tuple[dict[str, Any], float]:
    """FedEMA: FedAvg aggregation with EMA-interpolated weight distribution.

    Server-side aggregation is identical to FedAvg.  The soft-personalisation
    step happens at **distribution** time: instead of forcibly replacing local
    weights with the global average, each client interpolates::

        w_local ← α · w_local + (1 - α) · w_global  # noqa: RUF001

    This single-line change turns the global server into a gentle regulariser
    that whispers generalised physics to the local model without destroying
    its hard-earned site-specific feature extractors.  Implemented in
    ``run_fl_benchmark`` during the weight distribution step.

    Parameters
    ----------
    global_weights : dict
        Current global model state dictionary.
    client_updates : list[dict]
        List of client state dictionaries submitted this round.

    Returns
    -------
    new_weights : dict
        Averaged global state dict (identical to FedAvg).
    weight_change : float
        L2 norm of the update to the global model.
    """
    return federated_averaging(global_weights, client_updates, weights=weights)


def fedproto_aggregate(
    global_weights: dict[str, Any],
    _client_updates: list[dict[str, Any]],
    weights: list[float] | None = None,  # noqa: ARG001 — accepted for registry-signature parity, unused (no weight aggregation)
) -> tuple[dict[str, Any], float]:
    """FedProto: no weight aggregation — each client keeps fully local weights.

    In Federated Prototype Learning, clients exchange per-class prototype
    feature vectors (average image-encoder outputs for Clear / Overcast /
    Transient weather conditions) instead of model weights.  The server
    aggregates prototypes, not weights, so this function is a deliberate
    no-op on the weight dictionaries.

    Prototype extraction, aggregation, and the contrastive regularisation
    loss are implemented in the benchmark runner (``run_fl_benchmark``).

    Parameters
    ----------
    global_weights : dict
        Current global model state dictionary (returned **unchanged**).
    client_updates : list[dict]
        Client state dictionaries (ignored — prototypes are handled
        separately).

    Returns
    -------
    global_weights : dict
        Unchanged model weights.
    weight_change : float
        Always 0.0 (no weight-level aggregation).
    """
    return global_weights, 0.0


# ---------------------------------------------------------------------------
# Public registry: maps algorithm name → aggregation callable
# ---------------------------------------------------------------------------

AGGREGATION_FNS: dict[str, Any] = {
    "fedavg": federated_averaging,
    "fedbn": fedbn_aggregate,
    "fedper": fedper_aggregate,
    "fedprox": fedprox_aggregate,
    "ditto": ditto_aggregate,
    "fedema": fedema_aggregate,
    "fedproto": fedproto_aggregate,
    "gated": gated_aggregate,
}
