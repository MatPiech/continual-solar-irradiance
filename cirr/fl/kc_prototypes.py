"""Data-driven KC-stratified prototype extraction and attentive aggregation."""

from __future__ import annotations

from collections.abc import Callable

import torch

from cirr.fl.prototypes import extract_binned_prototypes
from cirr.models.forecaster import Forecaster
from cirr.physics.csi import CSI_CLEAR, CSI_OVERCAST, compute_csi

PROTO_OVERCAST = 0
PROTO_TRANSIENT = 1
PROTO_CLEAR = 2
PROTO_CLASS_NAMES = {
    PROTO_OVERCAST: "overcast",
    PROTO_TRANSIENT: "transient",
    PROTO_CLEAR: "clear",
}

# Physics-informed default KC/Kt regime boundaries:
#   overcast  : < 0.30
#   transient : [0.30, 0.65)
#   clear     : >= 0.65
DEFAULT_KC_EDGES = (0.30, 0.65)
DEFAULT_EDGE_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)

# Clear-sky-index regime boundaries, reusing the physics module's cut points so
# gating and prototype stratification label the sky the same way.
DEFAULT_CSI_EDGES = (CSI_OVERCAST, CSI_CLEAR)

# Regime label sources.  "proxy" normalises by a fixed 1000 W/m^2 (the target is
# already stored in kW/m^2), so the same label means different sky states at
# different latitudes; "clear_sky" divides by the site-and-time Ineichen-Perez
# clear-sky irradiance and does not.
REGIME_LABELS = ("proxy", "clear_sky")


def classify_value(value: float, edges: tuple[float, ...]) -> int:
    """Bin *value* against ascending *edges*; no edges means a single regime."""
    for idx, edge in enumerate(edges):
        if value < edge:
            return idx
    return len(edges)


def default_edges(label: str) -> tuple[float, float]:
    return DEFAULT_KC_EDGES if label == "proxy" else DEFAULT_CSI_EDGES


def make_regime_labeller(label: str, edges: tuple[float, ...]) -> Callable[[float, float], int]:
    """Return ``fn(target, ghi_clear) -> regime id`` for the chosen label source."""
    if label not in REGIME_LABELS:
        raise ValueError(f"Unknown regime label {label!r}. Choose from {REGIME_LABELS}.")
    if label == "proxy":
        return lambda target, _ghi: classify_value(target, edges)
    return lambda target, ghi: classify_value(compute_csi(target, ghi), edges)


def regime_names(n_classes: int) -> dict[int, str]:
    """Human-readable regime names; the physical triple keeps its usual names."""
    if n_classes == 3:
        return dict(PROTO_CLASS_NAMES)
    if n_classes == 1:
        return {0: "all"}
    return {i: f"regime{i}" for i in range(n_classes)}


def classify_irradiance(norm_irradiance: float, edges: tuple[float, float] = DEFAULT_KC_EDGES) -> int:
    """Map normalised irradiance to 3 prototype bins using provided edges."""
    overcast_edge, clear_edge = edges
    if norm_irradiance < overcast_edge:
        return PROTO_OVERCAST
    if norm_irradiance < clear_edge:
        return PROTO_TRANSIENT
    return PROTO_CLEAR


def _sample_target(sample: tuple[torch.Tensor, ...]) -> float:
    target = sample[3]
    return float(target.view(-1)[0].item() if isinstance(target, torch.Tensor) else target)


def fit_kc_bin_edges(
    values: list[float],
    quantile_levels: tuple[float, float] = DEFAULT_EDGE_QUANTILES,
    fallback_edges: tuple[float, float] = DEFAULT_KC_EDGES,
) -> tuple[float, float]:
    """Fit class edges from data quantiles (deterministic, no random thresholds)."""
    if len(values) < 3:
        return fallback_edges

    x = torch.tensor(values, dtype=torch.float32)
    q = torch.quantile(x, torch.tensor([quantile_levels[0], quantile_levels[1]], dtype=torch.float32))
    low = float(q[0].item())
    high = float(q[1].item())

    # Guard against collapsed quantiles on low-variance segments.
    if high <= low:
        x_min, x_max = float(torch.min(x).item()), float(torch.max(x).item())
        if x_max <= x_min:
            return fallback_edges
        span = x_max - x_min
        low = x_min + span / 3.0
        high = x_min + 2.0 * span / 3.0
    return low, high


def _sample_ghi_clear(sample: tuple[torch.Tensor, ...]) -> float:
    ghi = sample[2]
    return float(ghi.item() if isinstance(ghi, torch.Tensor) else ghi)


def extract_day_prototypes(
    model: Forecaster,
    day_samples: list[tuple[torch.Tensor, ...]],
    device: str,
    batch_size: int,
    edges: tuple[float, float] | None = None,
    labeller: Callable[[float, float], int] | None = None,
) -> dict[int, tuple[torch.Tensor, int]]:
    """Extract per-regime prototypes; *labeller* must match the one used in training."""
    if labeller is None:
        bin_edges = edges if edges is not None else DEFAULT_KC_EDGES
        labeller = make_regime_labeller("proxy", bin_edges)
    return extract_binned_prototypes(
        model,
        day_samples,
        device,
        batch_size,
        lambda sample: labeller(_sample_target(sample), _sample_ghi_clear(sample)),
    )


def _zscore(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if values.numel() == 0:
        return values
    std = values.std(unbiased=False)
    if float(std.item()) < eps:
        return torch.zeros_like(values)
    return (values - values.mean()) / std.clamp(min=eps)


def aggregate_global_prototypes(
    client_protos: list[dict[int, tuple[torch.Tensor, int]]],
    attention: bool = True,
    temperature: float | None = None,
) -> tuple[list[dict[int, torch.Tensor]], dict[int, list[list[float]]]]:
    """Aggregate class prototypes with adaptive attention.

    Attention logits combine:
    - cosine similarity between client prototypes (content relevance)
    - support-count prior (prototype reliability)
    Similarity scaling is data-driven (pairwise std) by default.  Passing
    *temperature* replaces that adaptive scale with a fixed tau, which is what the
    thesis text describes (tau = 0.5); everything else, including the support-count
    prior, is held identical so the two differ only in the scale.
    """
    n_clients = len(client_protos)

    client_means: list[dict[int, torch.Tensor]] = []
    for cp in client_protos:
        means: dict[int, torch.Tensor] = {}
        for cls, (feat_sum, count) in cp.items():
            if count > 0:
                means[cls] = feat_sum / count
        client_means.append(means)

    per_client_protos: list[dict[int, torch.Tensor]] = [{} for _ in range(n_clients)]
    attention_weights: dict[int, list[list[float]]] = {}

    all_classes = sorted({cls for means in client_means for cls in means})
    for cls in all_classes:
        present = [i for i in range(n_clients) if cls in client_means[i]]
        if not present:
            continue

        if len(present) == 1:
            i = present[0]
            per_client_protos[i][cls] = client_means[i][cls]
            attention_weights[cls] = [[0.0] * n_clients for _ in range(n_clients)]
            attention_weights[cls][i][i] = 1.0
            continue

        proto_stack = torch.stack([client_means[i][cls] for i in present])
        count_vec = torch.tensor([client_protos[i][cls][1] for i in present], dtype=torch.float32)
        norms = proto_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = proto_stack / norms
        sim_matrix = normed @ normed.T

        # Similarity scale (temperature): adaptive by default (matrix std over the present clients), or a fixed tau when one is given.
        if temperature is None:
            sim_scale = sim_matrix.std(unbiased=False).clamp(min=1e-4)
        else:
            sim_scale = torch.tensor(float(temperature)).clamp(min=1e-4)
        sim_logits = sim_matrix / sim_scale

        if attention:
            # Reliability prior: larger support count -> stronger contribution.
            count_logits = _zscore(torch.log1p(count_vec))
            attn_logits = sim_logits + count_logits.unsqueeze(0)
            attn = torch.softmax(attn_logits, dim=1)
        else:
            # Ablation: plain unweighted consensus, every peer counted equally.
            attn = torch.full((len(present), len(present)), 1.0 / len(present))
        weighted = attn @ proto_stack

        cls_attn: list[list[float]] = [[0.0] * n_clients for _ in range(n_clients)]
        for ki, i in enumerate(present):
            # .clone(): `weighted[ki]` is a row VIEW of the (n_present, d) stack, and
            # torch.save serialises a view's whole backing storage — which made the
            # logged prototype payload scale with the number of participating peers
            # (3x at three clients) instead of being a single d-vector per regime.
            per_client_protos[i][cls] = weighted[ki].clone()
            for kj, j in enumerate(present):
                cls_attn[i][j] = attn[ki, kj].item()
        attention_weights[cls] = cls_attn

    return per_client_protos, attention_weights
