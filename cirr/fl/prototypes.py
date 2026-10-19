"""CSI-binned prototype extraction and aggregation."""

from __future__ import annotations

from collections.abc import Callable

import torch

from cirr.models.forecaster import Forecaster
from cirr.physics.csi import compute_csi, discretize_csi


def classify_csi_bin(target_irradiance: float, ghi_clear: float, n_bins: int = 3) -> int:
    return discretize_csi(compute_csi(target_irradiance, ghi_clear), n_bins=n_bins)


def extract_binned_prototypes(
    model: Forecaster,
    samples: list[tuple[torch.Tensor, ...]],
    device: str,
    batch_size: int,
    classify_fn: Callable[[tuple[torch.Tensor, ...]], int],
) -> dict[int, tuple[torch.Tensor, int]]:
    if not samples:
        return {}

    model.eval()
    proto_sums: dict[int, torch.Tensor] = {}
    proto_counts: dict[int, int] = {}
    # Collapse diagnostics, accumulated in the pass we already make.  A representation
    # collapse shows up as the feature norm falling while mean pairwise cosine rises
    # toward 1 (every embedding pointing the same way).  Cheap streaming form:
    #   E[cos(zi,zj)], i != j  ==  (||sum of unit vectors||^2 - n) / (n(n-1))
    norm_sum = 0.0
    unit_sum: torch.Tensor | None = None
    n_seen = 0

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            imgs = torch.stack([s[0] for s in batch]).to(device)
            features = model.extract_proto_features(imgs)
            for b, sample in enumerate(batch):
                cls = classify_fn(sample)
                if cls < 0:
                    continue  # no usable sky regime: never contributes to a prototype
                feat = features[b].cpu()
                norm_sum += float(feat.norm())
                unit = feat / feat.norm().clamp(min=1e-12)
                unit_sum = unit.clone() if unit_sum is None else unit_sum + unit
                n_seen += 1
                if cls not in proto_sums:
                    proto_sums[cls] = torch.zeros_like(feat)
                    proto_counts[cls] = 0
                proto_sums[cls] += feat
                proto_counts[cls] += 1

    stats: dict[str, float] = {}
    if n_seen:
        stats["feature_norm_mean"] = norm_sum / n_seen
        stats["pairwise_cos_mean"] = (
            float((unit_sum @ unit_sum).item() - n_seen) / (n_seen * (n_seen - 1)) if n_seen > 1 else float("nan")
        )
    return {cls: (proto_sums[cls], proto_counts[cls]) for cls in proto_sums}, stats


def extract_prototypes(
    model: Forecaster,
    samples: list[tuple[torch.Tensor, ...]],
    device: str,
    batch_size: int,
    n_bins: int = 3,
) -> dict[int, tuple[torch.Tensor, int]]:
    def classify(sample: tuple[torch.Tensor, ...]) -> int:
        ghi_clear = float(sample[2].item() if isinstance(sample[2], torch.Tensor) else sample[2])
        target = float(sample[3].view(-1)[0].item() if isinstance(sample[3], torch.Tensor) else sample[3])
        return classify_csi_bin(target, ghi_clear, n_bins=n_bins)

    protos, _stats = extract_binned_prototypes(model, samples, device, batch_size, classify)
    return protos


def accumulate_prototypes(
    accum: dict[int, tuple[torch.Tensor, int]],
    day_protos: dict[int, tuple[torch.Tensor, int]],
) -> dict[int, tuple[torch.Tensor, int]]:
    for cls, (feat_sum, count) in day_protos.items():
        if cls in accum:
            old_sum, old_count = accum[cls]
            accum[cls] = (old_sum + feat_sum, old_count + count)
        else:
            accum[cls] = (feat_sum.clone(), count)
    return accum


def aggregate_prototypes(client_protos: list[dict[int, tuple[torch.Tensor, int]]]) -> dict[int, torch.Tensor]:
    global_sums: dict[int, torch.Tensor] = {}
    global_counts: dict[int, int] = {}
    for client in client_protos:
        for cls, (feat_sum, count) in client.items():
            if cls not in global_sums:
                global_sums[cls] = torch.zeros_like(feat_sum)
                global_counts[cls] = 0
            global_sums[cls] += feat_sum
            global_counts[cls] += count
    return {cls: global_sums[cls] / max(global_counts[cls], 1) for cls in global_sums}


def prototype_distance(embedding: torch.Tensor, prototype: torch.Tensor, metric: str = "euclidean") -> float:
    if metric == "euclidean":
        return torch.norm(embedding - prototype, p=2).item()
    if metric == "cosine":
        e = torch.nn.functional.normalize(embedding, dim=0)
        p = torch.nn.functional.normalize(prototype, dim=0)
        return (1.0 - torch.dot(e, p)).item()
    raise ValueError(f"Unknown metric: {metric}")
