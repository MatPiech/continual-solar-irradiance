"""Dual-threshold gating and gated day training."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum

import torch
import torch.nn as nn

from cirr.continual.buffer import DelayedFeedbackBuffer, DelayedState
from cirr.fl.prototypes import (
    aggregate_prototypes,
    classify_csi_bin,
    extract_prototypes,
    prototype_distance,
)
from cirr.physics.csi import compute_csi, discretize_csi
from cirr.physics.thresholds import compute_intra_bin_distances, fit_thresholds
from cirr.training.ogd import head_ogd_step


class GateZone(str, Enum):
    NORMAL = "normal"
    DRIFT = "drift"
    OOD = "ood"


def relative_clear_sky_error(y_hat: float, y_t: float, ghi_clear: float, eps: float = 1e-3) -> float:
    return abs(y_hat - y_t) / max(ghi_clear, eps)


def evaluate_gate(
    embedding: torch.Tensor,
    prototype: torch.Tensor,
    y_hat: float,
    y_t: float,
    ghi_clear: float,
    tau_norm: float,
    tau_ood: float,
    epsilon: float,
    random_gate: bool = False,
    distance_unaware: bool = False,
) -> tuple[GateZone, float, float]:
    dist = prototype_distance(embedding.cpu(), prototype.cpu())
    e_rel = relative_clear_sky_error(y_hat, y_t, ghi_clear)

    if random_gate:
        import random

        return (GateZone.DRIFT if random.random() > 0.5 else GateZone.NORMAL, dist, e_rel)

    if not distance_unaware and dist > tau_ood:
        return GateZone.OOD, dist, e_rel

    if distance_unaware or dist <= tau_norm:
        if e_rel <= epsilon:
            return GateZone.NORMAL, dist, e_rel
        return GateZone.DRIFT, dist, e_rel

    return GateZone.OOD, dist, e_rel


def should_backprop(zone: GateZone) -> bool:
    return zone == GateZone.DRIFT


def train_gated_day(
    model,
    dataset,
    day_indices: list[int],
    device: str,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    prototypes: dict[int, torch.Tensor],
    thresholds: dict[int, dict[str, float]],
    horizon_steps: int = 3,
    epsilon: float = 0.05,
) -> dict[str, int | float]:
    buffer = DelayedFeedbackBuffer(horizon_steps=horizon_steps)
    stats: dict[str, int | float] = defaultdict(int)
    model.eval()

    for idx in day_indices:
        x_t, hist, ghi_clear, y_t, ts = dataset[idx]
        ghi_val = float(ghi_clear.item())

        with torch.no_grad():
            z_t = model.extract_proto_features(x_t.unsqueeze(0).to(device)).squeeze(0).cpu()
            y_hat = model(x_t.unsqueeze(0).to(device), hist.unsqueeze(0).to(device))[0].item()

        resolved = buffer.enqueue(
            DelayedState(ts, z_t, x_t, y_hat, ghi_val, classify_csi_bin(y_t.item(), ghi_val), y_t.item(), hist)
        )

        if resolved is None:
            continue

        stats["considered"] += 1
        proto = prototypes.get(resolved.csi_bin)
        if proto is None:
            continue

        tau = thresholds.get(resolved.csi_bin, {"tau_norm": 1.0, "tau_ood": 2.0})
        zone, _, _ = evaluate_gate(
            resolved.z_t,
            proto,
            resolved.y_hat,
            resolved.y_target,
            resolved.ghi_clear,
            tau["tau_norm"],
            tau["tau_ood"],
            epsilon,
        )
        stats[zone.value] += 1
        if not should_backprop(zone):
            continue

        stats["backprop"] += 1
        head_ogd_step(
            model,
            optimizer,
            loss_fn,
            resolved.x_t,
            resolved.history,
            torch.tensor([[resolved.y_target]], device=device),
            device,
        )

    return dict(stats)


def eval_gated_online(
    model,
    dataset,
    day_map: dict[str, list[int]],
    days: list[str],
    device: str,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    prototypes: dict[int, torch.Tensor],
    thresholds: dict[int, dict[str, float]],
    *,
    horizon_steps: int = 3,
    epsilon: float = 0.05,
    random_gate: bool = False,
    distance_unaware: bool = False,
    show_progress: bool = True,
) -> tuple[list[float], list[float], list[dict], dict[str, int]]:
    """Run gated continual eval over multiple days; return preds, targets, rows, gate stats."""
    from tqdm import tqdm

    buffer = DelayedFeedbackBuffer(horizon_steps=horizon_steps)
    all_preds: list[float] = []
    all_tgts: list[float] = []
    predictions: list[dict] = []
    gate_stats: dict[str, int] = defaultdict(int)
    backprop_count = 0
    model.eval()

    for day in days:
        buffer.clear()
        indices = day_map.get(day, [])
        iterator = tqdm(indices, desc=day, leave=False) if show_progress else indices
        for idx in iterator:
            x_t, hist, ghi_clear, y_t, ts = dataset[idx]
            ghi_val = float(ghi_clear.item())

            with torch.no_grad():
                z_t = model.extract_proto_features(x_t.unsqueeze(0).to(device)).squeeze(0).cpu()
                y_hat = model(x_t.unsqueeze(0).to(device), hist.unsqueeze(0).to(device))[0].item()

            state = DelayedState(ts, z_t, x_t, y_hat, ghi_val, classify_csi_bin(y_t.item(), ghi_val), y_t.item(), hist)
            resolved = buffer.enqueue(state)
            pred_row = {"timestamp": ts, "prediction": y_hat, "target": y_t.item(), "gate_zone": "pending"}

            if resolved is not None:
                proto = prototypes.get(resolved.csi_bin)
                if proto is None:
                    zone = GateZone.NORMAL
                    dist, e_rel = 0.0, 0.0
                else:
                    tau = thresholds.get(resolved.csi_bin, {"tau_norm": 1.0, "tau_ood": 2.0})
                    zone, dist, e_rel = evaluate_gate(
                        resolved.z_t,
                        proto,
                        resolved.y_hat,
                        resolved.y_target,
                        resolved.ghi_clear,
                        tau["tau_norm"],
                        tau["tau_ood"],
                        epsilon,
                        random_gate=random_gate,
                        distance_unaware=distance_unaware,
                    )
                gate_stats[zone.value] += 1
                pred_row["gate_zone"] = zone.value
                pred_row["latent_dist"] = dist
                pred_row["e_rel"] = e_rel
                if should_backprop(zone):
                    backprop_count += 1
                    head_ogd_step(
                        model,
                        optimizer,
                        loss_fn,
                        resolved.x_t,
                        resolved.history,
                        torch.tensor([[resolved.y_target]], device=device),
                        device,
                    )

            all_preds.append(y_hat)
            all_tgts.append(y_t.item())
            predictions.append(pred_row)

    gate_stats["backprop"] = backprop_count
    return all_preds, all_tgts, predictions, dict(gate_stats)


def build_gated_thresholds(
    model, dataset, device: str, n_samples: int = 300
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    samples = [dataset[i] for i in range(min(n_samples, len(dataset)))]
    prototypes = aggregate_prototypes([extract_prototypes(model, samples, device, batch_size=32)])

    embeddings, bins = [], []
    with torch.no_grad():
        for sample in samples:
            z = model.extract_proto_features(sample[0].unsqueeze(0).to(device)).squeeze(0).cpu()
            embeddings.append(z)
            bins.append(discretize_csi(compute_csi(sample[3].item(), sample[2].item())))

    if not embeddings:
        return prototypes, {}
    return prototypes, fit_thresholds(compute_intra_bin_distances(torch.stack(embeddings), bins))
