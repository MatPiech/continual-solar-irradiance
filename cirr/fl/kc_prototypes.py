"""Data-driven KC-stratified prototype extraction and attentive aggregation."""

from __future__ import annotations

import torch

from cirr.fl.prototypes import extract_binned_prototypes
from cirr.models.forecaster import Forecaster

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


def extract_day_prototypes(
    model: Forecaster,
    day_samples: list[tuple[torch.Tensor, ...]],
    device: str,
    batch_size: int,
    edges: tuple[float, float] | None = None,
) -> dict[int, tuple[torch.Tensor, int]]:
    # Runtime adaptation uses fixed physics defaults unless caller overrides.
    bin_edges = edges if edges is not None else DEFAULT_KC_EDGES
    return extract_binned_prototypes(
        model,
        day_samples,
        device,
        batch_size,
        lambda sample: classify_irradiance(_sample_target(sample), bin_edges),
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
) -> tuple[list[dict[int, torch.Tensor]], dict[int, list[list[float]]]]:
    """Aggregate class prototypes with adaptive attention.

    Attention logits combine:
    - cosine similarity between client prototypes (content relevance)
    - support-count prior (prototype reliability)
    Similarity scaling is data-driven (pairwise std), avoiding static temperature.
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

    for cls in (PROTO_OVERCAST, PROTO_TRANSIENT, PROTO_CLEAR):
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

        # Adaptive similarity scale (temperature): use matrix std over present clients.
        sim_scale = sim_matrix.std(unbiased=False).clamp(min=1e-4)
        sim_logits = sim_matrix / sim_scale

        # Reliability prior: larger support count -> stronger contribution.
        count_logits = _zscore(torch.log1p(count_vec))
        attn_logits = sim_logits + count_logits.unsqueeze(0)
        attn = torch.softmax(attn_logits, dim=1)
        weighted = attn @ proto_stack

        cls_attn: list[list[float]] = [[0.0] * n_clients for _ in range(n_clients)]
        for ki, i in enumerate(present):
            per_client_protos[i][cls] = weighted[ki]
            for kj, j in enumerate(present):
                cls_attn[i][j] = attn[ki, kj].item()
        attention_weights[cls] = cls_attn

    return per_client_protos, attention_weights
