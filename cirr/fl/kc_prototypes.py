"""KC-threshold stratified prototype extraction and FedProto aggregation."""

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

KC_OVERCAST_THRESHOLD = 0.15
KC_CLEAR_THRESHOLD = 0.50


def classify_irradiance(norm_irradiance: float) -> int:
    if norm_irradiance < KC_OVERCAST_THRESHOLD:
        return PROTO_OVERCAST
    if norm_irradiance < KC_CLEAR_THRESHOLD:
        return PROTO_TRANSIENT
    return PROTO_CLEAR


def _sample_target(sample: tuple[torch.Tensor, ...]) -> float:
    target = sample[3]
    return float(target.view(-1)[0].item() if isinstance(target, torch.Tensor) else target)


def extract_day_prototypes(
    model: Forecaster,
    day_samples: list[tuple[torch.Tensor, ...]],
    device: str,
    batch_size: int,
) -> dict[int, tuple[torch.Tensor, int]]:
    return extract_binned_prototypes(
        model,
        day_samples,
        device,
        batch_size,
        lambda sample: classify_irradiance(_sample_target(sample)),
    )


def aggregate_global_prototypes(
    client_protos: list[dict[int, tuple[torch.Tensor, int]]],
    *,
    tau: float = 0.5,
) -> tuple[list[dict[int, torch.Tensor]], dict[int, list[list[float]]]]:
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
        norms = proto_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = proto_stack / norms
        sim_matrix = normed @ normed.T
        attn = torch.softmax(sim_matrix / tau, dim=1)
        weighted = attn @ proto_stack

        cls_attn: list[list[float]] = [[0.0] * n_clients for _ in range(n_clients)]
        for ki, i in enumerate(present):
            per_client_protos[i][cls] = weighted[ki]
            for kj, j in enumerate(present):
                cls_attn[i][j] = attn[ki, kj].item()
        attention_weights[cls] = cls_attn

    return per_client_protos, attention_weights
