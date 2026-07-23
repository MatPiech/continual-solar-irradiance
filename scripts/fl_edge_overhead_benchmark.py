"""Edge-overhead benchmark for FL algorithms on constrained platforms.

This script measures the *platform cost* of a single federated round for the
algorithms implemented in ``scripts/fl_benchmark.py``. It is designed for
deployment-oriented comparisons on edge devices such as Raspberry Pi and
NVIDIA Orin Nano, where communication volume, synchronization latency,
memory pressure, and energy matter as much as predictive quality.

Measured dimensions
-------------------
* Local training time per round.
* Additional FedProto prototype-extraction time.
* Upload / download payload size using real PyTorch serialization.
* Server aggregation latency.
* Client-side update-application latency.
* Peak host RSS and accelerator memory.
* Estimated per-round energy using device-specific power/network profiles.

Methodology notes
-----------------
* FedProto follows the same weather-class thresholds and attention-weighted
  personalised prototype aggregation as ``fl_benchmark.py``.
* FedBN, FedPer, FedProx, Ditto, and FedEMA are benchmarked according to the
  *current implementation* in this repository. In particular, FedBN and
  FedPer still upload full client state dicts and receive full merged models;
  the personalization happens during aggregation/distribution, not via a
  smaller transport payload.
* The benchmark uses synthetic local batches so it isolates platform overheads
  from dataset I/O variability. Run it directly on the target edge device for
  the most representative measurements.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import click
import pandas as pd
import torch
import torch.nn.functional as F

from cirr.fl.kc_prototypes import (
    aggregate_global_prototypes,
    classify_irradiance,
    extract_day_prototypes,
)
from cirr.fl.server import AGGREGATION_FNS, is_bn_key, is_head_key
from cirr.models.loading import load_forecaster_from_checkpoint

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
_FEDERATED_ALGORITHMS = ("fedavg", "fedbn", "fedper", "fedprox", "ditto", "fedema", "fedproto")


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    uplink_mbps: float
    downlink_mbps: float
    active_power_w: float
    idle_power_w: float
    network_power_w: float


DEVICE_PROFILES: dict[str, DeviceProfile] = {
    "raspberry-pi": DeviceProfile(
        name="raspberry-pi",
        uplink_mbps=30.0,
        downlink_mbps=30.0,
        active_power_w=7.0,
        idle_power_w=3.0,
        network_power_w=1.5,
    ),
    "orin-nano": DeviceProfile(
        name="orin-nano",
        uplink_mbps=150.0,
        downlink_mbps=150.0,
        active_power_w=15.0,
        idle_power_w=6.0,
        network_power_w=2.5,
    ),
}


def _build_synthetic_samples(
    n_samples: int,
    image_size: int,
    history_dim: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    class_targets = [0.05, 0.30, 0.80]
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for idx in range(n_samples):
        image = torch.randn((3, image_size, image_size), generator=generator)
        history = torch.randn((history_dim,), generator=generator)
        base_target = class_targets[idx % len(class_targets)]
        jitter = torch.rand((1,), generator=generator).item() * 0.02
        target = torch.tensor([min(0.99, base_target + jitter)], dtype=torch.float32)
        ghi_clear = torch.tensor(1.0, dtype=torch.float32)
        samples.append((image, history, ghi_clear, target))
    return samples


def _current_rss_mb() -> float:
    status = Path("/proc/self/status")
    if not status.exists():
        return float("nan")
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / 1024.0
    return float("nan")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _state_dict_tensor_bytes(state_dict: dict[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += _tensor_nbytes(value)
    return total


def _serialize_payload(payload: Any) -> tuple[bytes, float]:
    buffer = io.BytesIO()
    start = perf_counter()
    torch.save(payload, buffer)
    elapsed = perf_counter() - start
    return buffer.getvalue(), elapsed


def _deserialize_payload(payload_bytes: bytes, map_location: str = "cpu") -> tuple[Any, float]:
    buffer = io.BytesIO(payload_bytes)
    start = perf_counter()
    payload = torch.load(buffer, map_location=map_location, weights_only=False)
    elapsed = perf_counter() - start
    return payload, elapsed


def _network_time_seconds(payload_bytes: int, bandwidth_mbps: float) -> float:
    if bandwidth_mbps <= 0:
        return float("inf")
    return (payload_bytes * 8.0) / (bandwidth_mbps * 1_000_000.0)


def _extract_local_keys(state_dict: dict[str, Any], key_filter) -> dict[str, Any]:
    return {key: value.clone() for key, value in state_dict.items() if key_filter(key)}


def _restore_local_keys(aggregated: dict[str, Any], local_state: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(aggregated)
    for key, value in local_state.items():
        merged[key] = value
    return merged


def _ema_interpolate(local_sd: dict[str, Any], global_sd: dict[str, Any], alpha: float) -> dict[str, Any]:
    merged = {}
    for key in global_sd:
        if isinstance(global_sd[key], torch.Tensor) and global_sd[key].is_floating_point():
            merged[key] = alpha * local_sd[key] + (1.0 - alpha) * global_sd[key]
        else:
            merged[key] = local_sd[key]
    return merged


def _peak_accelerator_memory_mb(device: str) -> float:
    if device != "cpu" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0**2)
    return 0.0


def _reset_accelerator_peak(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _sync_device(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _train_local_model(
    model,
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    global_state_dict: dict[str, Any] | None = None,
    mu: float = 0.0,
    global_prototypes: dict[int, torch.Tensor] | None = None,
    proto_lambda: float = 0.0,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    prox_ref: dict[str, torch.Tensor] | None = None
    if mu > 0.0 and global_state_dict is not None:
        prox_ref = {
            key: value.detach().to(device)
            for key, value in global_state_dict.items()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        }

    proto_on_device: dict[int, torch.Tensor] = {}
    use_proto = global_prototypes is not None and proto_lambda > 0.0
    if use_proto:
        proto_on_device = {key: value.detach().to(device) for key, value in global_prototypes.items()}

    for _ in range(epochs):
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = torch.stack([sample[0] for sample in batch]).to(device)
            history = torch.stack([sample[1] for sample in batch]).to(device)
            targets = torch.stack([sample[3] for sample in batch]).to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_proto:
                mean, var, image_features = model.forward_with_features(images, history)
            else:
                mean, var = model(images, history)

            if var is not None:
                loss = model.loss_fn(mean, targets, var)
            elif model.loss_name == "Berhu":
                loss = model.loss_fn(mean * 1000, targets * 1000)
            else:
                loss = model.loss_fn(mean, targets)

            if prox_ref is not None:
                prox = sum(
                    ((parameter - prox_ref[name]) ** 2).sum()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and name in prox_ref
                )
                loss = loss + (mu / 2.0) * prox

            if use_proto:
                proto_loss = torch.tensor(0.0, device=device)
                proto_count = 0
                for batch_idx in range(images.size(0)):
                    cls = classify_irradiance(float(targets[batch_idx].item()))
                    if cls in proto_on_device:
                        proto_loss = proto_loss + F.mse_loss(image_features[batch_idx], proto_on_device[cls])
                        proto_count += 1
                if proto_count > 0:
                    avg_proto = proto_loss / proto_count
                    scale = loss.detach() / (avg_proto.detach() + 1e-8)
                    loss = loss + proto_lambda * scale * avg_proto

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def _build_prototype_payload(client_proto: dict[int, tuple[torch.Tensor, int]]) -> dict[int, dict[str, Any]]:
    payload: dict[int, dict[str, Any]] = {}
    for cls, (feat_sum, count) in client_proto.items():
        payload[cls] = {"feature_sum": feat_sum.clone(), "count": count}
    return payload


def _weight_payload_kind(algorithm: str) -> str:
    if algorithm == "fedproto":
        return "prototypes"
    if algorithm in {"fedbn", "fedper", "fedprox", "fedema", "ditto", "fedavg"}:
        return "weights"
    return "unknown"


def _count_model_parameters(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _benchmark_algorithm(
    algorithm: str,
    base_model,
    device: str,
    profile: DeviceProfile,
    num_clients: int,
    rounds: int,
    batch_size: int,
    local_batches: int,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    fedprox_mu: float,
    ditto_lambda: float,
    fedema_alpha: float,
    fedproto_lambda: float,
    image_size: int,
    history_dim: int,
    seed: int,
) -> dict[str, Any]:
    clients = [copy.deepcopy(base_model) for _ in range(num_clients)]
    personalised_models = [copy.deepcopy(model) for model in clients] if algorithm == "ditto" else []
    client_samples = [
        _build_synthetic_samples(batch_size * local_batches, image_size, history_dim, seed + 101 * client_idx)
        for client_idx in range(num_clients)
    ]

    initial_proto_accum = [
        extract_day_prototypes(model, samples, device, batch_size)
        for model, samples in zip(clients, client_samples, strict=False)
    ]
    proto_cache, _ = aggregate_global_prototypes(initial_proto_accum)

    local_train_s: list[float] = []
    proto_extract_s: list[float] = []
    upload_serialize_s: list[float] = []
    download_deserialize_s: list[float] = []
    aggregation_s: list[float] = []
    apply_update_s: list[float] = []
    peak_rss_mb: list[float] = []
    peak_accel_mb: list[float] = []
    upload_bytes_per_client: list[float] = []
    download_bytes_per_client: list[float] = []
    prototype_dims: list[int] = []
    prototype_classes: list[int] = []
    last_global_state_dict = copy.deepcopy(clients[0].state_dict())

    for _round_idx in range(rounds):
        round_proto_accum: list[dict[int, tuple[torch.Tensor, int]]] = []
        state_dicts: list[dict[str, Any]] = []
        local_bn: list[dict[str, Any]] = []
        local_head: list[dict[str, Any]] = []
        upload_bytes_current: list[int] = []
        download_bytes_current: list[int] = []

        for client_idx, model in enumerate(clients):
            _reset_accelerator_peak(device)
            rss_before = _current_rss_mb()
            train_start = perf_counter()
            _train_local_model(
                model=model,
                samples=client_samples[client_idx],
                device=device,
                batch_size=batch_size,
                epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                global_state_dict=last_global_state_dict if algorithm in {"fedprox", "ditto"} else None,
                mu=fedprox_mu if algorithm == "fedprox" else 0.0,
                global_prototypes=proto_cache[client_idx] if algorithm == "fedproto" else None,
                proto_lambda=fedproto_lambda if algorithm == "fedproto" else 0.0,
            )
            _sync_device(device)
            local_train_s.append(perf_counter() - train_start)
            peak_accel_mb.append(_peak_accelerator_memory_mb(device))
            peak_rss_mb.append(max(0.0, _current_rss_mb() - rss_before))

            if algorithm == "fedproto":
                proto_start = perf_counter()
                client_proto = extract_day_prototypes(model, client_samples[client_idx], device, batch_size)
                _sync_device(device)
                proto_extract_s.append(perf_counter() - proto_start)
                round_proto_accum.append(client_proto)
                prototype_classes.append(len(client_proto))
                if client_proto:
                    first_tensor = next(iter(client_proto.values()))[0]
                    prototype_dims.append(first_tensor.numel())

                payload = _build_prototype_payload(client_proto)
                payload_bytes, ser_elapsed = _serialize_payload(payload)
                upload_serialize_s.append(ser_elapsed)
                upload_bytes_current.append(len(payload_bytes))
            else:
                proto_extract_s.append(0.0)
                state_dict = copy.deepcopy(model.state_dict())
                state_dicts.append(state_dict)
                if algorithm == "fedbn":
                    local_bn.append(_extract_local_keys(state_dict, is_bn_key))
                elif algorithm == "fedper":
                    local_head.append(_extract_local_keys(state_dict, is_head_key))

                payload_bytes, ser_elapsed = _serialize_payload(state_dict)
                upload_serialize_s.append(ser_elapsed)
                upload_bytes_current.append(len(payload_bytes))

        agg_start = perf_counter()
        if algorithm == "fedproto":
            proto_cache, _ = aggregate_global_prototypes(round_proto_accum)
            _sync_device(device)
            aggregation_s.append(perf_counter() - agg_start)

            for client_idx in range(num_clients):
                download_payload = {cls: tensor.clone() for cls, tensor in proto_cache[client_idx].items()}
                payload_bytes, _ = _serialize_payload(download_payload)
                download_bytes_current.append(len(payload_bytes))
                _, deser_elapsed = _deserialize_payload(payload_bytes, map_location="cpu")
                download_deserialize_s.append(deser_elapsed)
                apply_update_s.append(0.0)
        else:
            global_state, _ = AGGREGATION_FNS[algorithm](state_dicts[0], state_dicts)
            _sync_device(device)
            aggregation_s.append(perf_counter() - agg_start)
            last_global_state_dict = copy.deepcopy(global_state)

            payload_bytes, _ = _serialize_payload(global_state)
            for client_idx, model in enumerate(clients):
                download_bytes_current.append(len(payload_bytes))
                downloaded_payload, deser_elapsed = _deserialize_payload(payload_bytes, map_location="cpu")
                download_deserialize_s.append(deser_elapsed)

                apply_start = perf_counter()
                merged = downloaded_payload
                if algorithm == "fedbn":
                    merged = _restore_local_keys(merged, local_bn[client_idx])
                elif algorithm == "fedper":
                    merged = _restore_local_keys(merged, local_head[client_idx])
                elif algorithm == "fedema":
                    merged = _ema_interpolate(state_dicts[client_idx], downloaded_payload, fedema_alpha)
                model.load_state_dict(merged)
                _sync_device(device)
                apply_update_s.append(perf_counter() - apply_start)

            if algorithm == "ditto":
                for client_idx, personal_model in enumerate(personalised_models):
                    _train_local_model(
                        model=personal_model,
                        samples=client_samples[client_idx],
                        device=device,
                        batch_size=batch_size,
                        epochs=local_epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm,
                        global_state_dict=last_global_state_dict,
                        mu=ditto_lambda,
                    )
                    _sync_device(device)

        upload_bytes_per_client.append(_mean([float(value) for value in upload_bytes_current]))
        download_bytes_per_client.append(_mean([float(value) for value in download_bytes_current]))

    mean_upload_bytes = _mean(upload_bytes_per_client)
    mean_download_bytes = _mean(download_bytes_per_client)
    uplink_s = _network_time_seconds(int(mean_upload_bytes), profile.uplink_mbps)
    downlink_s = _network_time_seconds(int(mean_download_bytes), profile.downlink_mbps)
    active_s = _mean(local_train_s) + _mean(proto_extract_s) + _mean(upload_serialize_s) + _mean(download_deserialize_s)
    active_s += _mean(apply_update_s)
    waiting_s = uplink_s + downlink_s + _mean(aggregation_s)
    total_round_s = active_s + waiting_s
    total_energy_j = profile.active_power_w * active_s
    total_energy_j += profile.network_power_w * (uplink_s + downlink_s)
    total_energy_j += profile.idle_power_w * _mean(aggregation_s)

    total_params, trainable_params = _count_model_parameters(base_model)
    prototype_dim = int(round(_mean([float(value) for value in prototype_dims]))) if prototype_dims else 0
    prototype_class_count = int(round(_mean([float(value) for value in prototype_classes]))) if prototype_classes else 0
    state_tensor_bytes = _state_dict_tensor_bytes(base_model.state_dict())

    return {
        "algorithm": algorithm,
        "payload_kind": _weight_payload_kind(algorithm),
        "num_clients": num_clients,
        "rounds": rounds,
        "batch_size": batch_size,
        "local_batches": local_batches,
        "local_epochs": local_epochs,
        "model_params": total_params,
        "trainable_params": trainable_params,
        "state_tensor_mb": state_tensor_bytes / (1024.0**2),
        "prototype_dim": prototype_dim,
        "prototype_classes_per_client": prototype_class_count,
        "upload_mb_per_client": mean_upload_bytes / (1024.0**2),
        "download_mb_per_client": mean_download_bytes / (1024.0**2),
        "round_trip_mb_per_client": (mean_upload_bytes + mean_download_bytes) / (1024.0**2),
        "fleet_upload_mb": (mean_upload_bytes * num_clients) / (1024.0**2),
        "fleet_download_mb": (mean_download_bytes * num_clients) / (1024.0**2),
        "local_train_ms": _mean(local_train_s) * 1000.0,
        "proto_extract_ms": _mean(proto_extract_s) * 1000.0,
        "upload_serialize_ms": _mean(upload_serialize_s) * 1000.0,
        "download_deserialize_ms": _mean(download_deserialize_s) * 1000.0,
        "server_aggregate_ms": _mean(aggregation_s) * 1000.0,
        "apply_update_ms": _mean(apply_update_s) * 1000.0,
        "uplink_ms_est": uplink_s * 1000.0,
        "downlink_ms_est": downlink_s * 1000.0,
        "edge_round_ms_est": total_round_s * 1000.0,
        "edge_active_ms": active_s * 1000.0,
        "estimated_energy_j": total_energy_j,
        "peak_rss_delta_mb": _mean(peak_rss_mb),
        "peak_accelerator_mb": _mean(peak_accel_mb),
    }


def _resolve_profile(
    device_profile: str,
    uplink_mbps: float | None,
    downlink_mbps: float | None,
    active_power_w: float | None,
    idle_power_w: float | None,
    network_power_w: float | None,
) -> DeviceProfile:
    if device_profile == "custom":
        missing = [
            name
            for name, value in (
                ("uplink_mbps", uplink_mbps),
                ("downlink_mbps", downlink_mbps),
                ("active_power_w", active_power_w),
                ("idle_power_w", idle_power_w),
                ("network_power_w", network_power_w),
            )
            if value is None
        ]
        if missing:
            raise click.UsageError(f"Custom profile requires: {', '.join(missing)}")
        return DeviceProfile(
            name="custom",
            uplink_mbps=float(uplink_mbps),
            downlink_mbps=float(downlink_mbps),
            active_power_w=float(active_power_w),
            idle_power_w=float(idle_power_w),
            network_power_w=float(network_power_w),
        )

    base = DEVICE_PROFILES[device_profile]
    return DeviceProfile(
        name=base.name,
        uplink_mbps=float(uplink_mbps if uplink_mbps is not None else base.uplink_mbps),
        downlink_mbps=float(downlink_mbps if downlink_mbps is not None else base.downlink_mbps),
        active_power_w=float(active_power_w if active_power_w is not None else base.active_power_w),
        idle_power_w=float(idle_power_w if idle_power_w is not None else base.idle_power_w),
        network_power_w=float(network_power_w if network_power_w is not None else base.network_power_w),
    )


def _normalise_algorithms(algorithms: tuple[str, ...]) -> list[str]:
    if not algorithms or algorithms == ("all",):
        return list(_FEDERATED_ALGORITHMS)
    if "all" in algorithms and len(algorithms) > 1:
        raise click.UsageError("Use either --algorithm all or explicit repeated --algorithm values.")
    selected = [algorithm.lower() for algorithm in algorithms]
    invalid = [algorithm for algorithm in selected if algorithm not in _FEDERATED_ALGORITHMS]
    if invalid:
        raise click.UsageError(f"Unsupported algorithms: {', '.join(invalid)}")
    return selected


def _add_relative_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    if "fedavg" not in set(results_df["algorithm"]):
        return results_df

    fedavg = results_df.loc[results_df["algorithm"] == "fedavg"].iloc[0]
    enriched = results_df.copy()
    for column in ("round_trip_mb_per_client", "edge_round_ms_est", "estimated_energy_j"):
        baseline = float(fedavg[column])
        relative_name = f"{column}_reduction_vs_fedavg_pct"
        if baseline == 0.0:
            enriched[relative_name] = 0.0
        else:
            enriched[relative_name] = 100.0 * (1.0 - (enriched[column] / baseline))
    return enriched


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the pretrained checkpoint used to instantiate the edge model.",
)
@click.option(
    "--algorithm",
    multiple=True,
    default=("all",),
    show_default=True,
    help=(
        "Algorithm(s) to compare. Repeat for an explicit subset, or use 'all'. "
        "Supported: fedavg, fedbn, fedper, fedprox, ditto, fedema, fedproto."
    ),
)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", show_default=True)
@click.option(
    "--device-profile",
    type=click.Choice(["raspberry-pi", "orin-nano", "custom"]),
    default="raspberry-pi",
    show_default=True,
    help="Power/network profile used for round-latency and energy estimates.",
)
@click.option("--uplink-mbps", type=float, default=None, help="Override profile uplink throughput.")
@click.option("--downlink-mbps", type=float, default=None, help="Override profile downlink throughput.")
@click.option("--active-power-w", type=float, default=None, help="Override profile active compute power draw.")
@click.option("--idle-power-w", type=float, default=None, help="Override profile idle wait power draw.")
@click.option("--network-power-w", type=float, default=None, help="Override profile network-interface power draw.")
@click.option("--num-clients", type=int, default=3, show_default=True, help="Number of FL clients in the round.")
@click.option("--rounds", type=int, default=3, show_default=True, help="Measured rounds per algorithm.")
@click.option("--batch-size", type=int, default=16, show_default=True, help="Mini-batch size per local step.")
@click.option("--local-batches", type=int, default=4, show_default=True, help="Number of batches processed per local epoch.")
@click.option("--local-epochs", type=int, default=1, show_default=True, help="Number of local epochs per round.")
@click.option("--history-dim", type=int, default=4, show_default=True, help="Synthetic irradiance-history feature dimension.")
@click.option("--image-size", type=int, default=128, show_default=True, help="Synthetic square image size.")
@click.option("--lr", type=float, default=1e-4, show_default=True, help="Local optimizer learning rate.")
@click.option("--weight-decay", type=float, default=1e-3, show_default=True, help="Local optimizer weight decay.")
@click.option("--max-grad-norm", type=float, default=1.0, show_default=True, help="Gradient clipping threshold.")
@click.option("--fedprox-mu", type=float, default=0.01, show_default=True, help="FedProx proximal regularization.")
@click.option("--ditto-lambda", type=float, default=0.1, show_default=True, help="Ditto proximal regularization.")
@click.option("--fedema-alpha", type=float, default=0.5, show_default=True, help="FedEMA interpolation coefficient.")
@click.option("--fedproto-lambda", type=float, default=0.1, show_default=True, help="FedProto prototype regularization.")
@click.option(
    "--freeze-early-backbone",
    is_flag=True,
    default=False,
    help="Freeze early backbone layers before benchmarking local adaptation cost.",
)
@click.option(
    "--save-results",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output directory for comparison CSV/JSON artifacts.",
)
def main(
    checkpoint: Path,
    algorithm: tuple[str, ...],
    device: str,
    device_profile: str,
    uplink_mbps: float | None,
    downlink_mbps: float | None,
    active_power_w: float | None,
    idle_power_w: float | None,
    network_power_w: float | None,
    num_clients: int,
    rounds: int,
    batch_size: int,
    local_batches: int,
    local_epochs: int,
    history_dim: int,
    image_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    fedprox_mu: float,
    ditto_lambda: float,
    fedema_alpha: float,
    fedproto_lambda: float,
    freeze_early_backbone: bool,
    save_results: Path | None,
) -> None:
    """Benchmark FL platform overheads on an edge device or device profile."""
    algorithms = _normalise_algorithms(algorithm)
    profile = _resolve_profile(
        device_profile=device_profile,
        uplink_mbps=uplink_mbps,
        downlink_mbps=downlink_mbps,
        active_power_w=active_power_w,
        idle_power_w=idle_power_w,
        network_power_w=network_power_w,
    )

    base_model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    if freeze_early_backbone:
        base_model.freeze_early_backbone()

    print(f"\n{'=' * 88}")
    print(f"  Edge FL Overhead Benchmark  device={device}  profile={profile.name}")
    print(f"{'=' * 88}")
    print(
        "  Relevant comparison axes: local compute, prototype extraction, payload size, "
        "server aggregation, synchronization latency, memory, energy"
    )
    print(
        f"  Profile: uplink={profile.uplink_mbps:.1f} Mbps  downlink={profile.downlink_mbps:.1f} Mbps  "
        f"active={profile.active_power_w:.1f} W  idle={profile.idle_power_w:.1f} W  network={profile.network_power_w:.1f} W"
    )
    print()

    results = []
    for algorithm_name in algorithms:
        print(f"  Benchmarking {algorithm_name}...")
        result = _benchmark_algorithm(
            algorithm=algorithm_name,
            base_model=base_model,
            device=device,
            profile=profile,
            num_clients=num_clients,
            rounds=rounds,
            batch_size=batch_size,
            local_batches=local_batches,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            fedprox_mu=fedprox_mu,
            ditto_lambda=ditto_lambda,
            fedema_alpha=fedema_alpha,
            fedproto_lambda=fedproto_lambda,
            image_size=image_size,
            history_dim=history_dim,
            seed=42,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = _add_relative_columns(results_df)
    results_df = results_df.sort_values("edge_round_ms_est").reset_index(drop=True)

    display_columns = [
        "algorithm",
        "payload_kind",
        "upload_mb_per_client",
        "download_mb_per_client",
        "local_train_ms",
        "proto_extract_ms",
        "server_aggregate_ms",
        "edge_round_ms_est",
        "estimated_energy_j",
        "peak_rss_delta_mb",
        "peak_accelerator_mb",
    ]
    print(results_df[display_columns].to_string(index=False, float_format=lambda value: f"{value:0.3f}"))

    if "fedavg" in set(results_df["algorithm"]) and "fedproto" in set(results_df["algorithm"]):
        fedavg_row = results_df.loc[results_df["algorithm"] == "fedavg"].iloc[0]
        fedproto_row = results_df.loc[results_df["algorithm"] == "fedproto"].iloc[0]
        print("\nFedProto vs FedAvg")
        print(
            f"  Round-trip payload/client: {fedproto_row['round_trip_mb_per_client']:.3f} MB vs "
            f"{fedavg_row['round_trip_mb_per_client']:.3f} MB"
        )
        print(
            f"  Estimated round time: {fedproto_row['edge_round_ms_est']:.2f} ms vs "
            f"{fedavg_row['edge_round_ms_est']:.2f} ms"
        )
        print(
            f"  Estimated round energy: {fedproto_row['estimated_energy_j']:.3f} J vs "
            f"{fedavg_row['estimated_energy_j']:.3f} J"
        )
        print(
            f"  Extra local prototype cost: {fedproto_row['proto_extract_ms']:.2f} ms  "
            f"(prototype_dim={int(fedproto_row['prototype_dim'])}, classes/client={int(fedproto_row['prototype_classes_per_client'])})"
        )

    if save_results is not None:
        save_results.mkdir(parents=True, exist_ok=True)
        comparison_csv = save_results / "comparison.csv"
        summary_json = save_results / "summary.json"
        results_df.to_csv(comparison_csv, index=False)
        with summary_json.open("w") as handle:
            json.dump(
                {
                    "checkpoint": str(checkpoint),
                    "device": device,
                    "device_profile": asdict(profile),
                    "algorithms": algorithms,
                    "num_clients": num_clients,
                    "rounds": rounds,
                    "batch_size": batch_size,
                    "local_batches": local_batches,
                    "local_epochs": local_epochs,
                    "history_dim": history_dim,
                    "image_size": image_size,
                    "results": results,
                },
                handle,
                indent=2,
            )
        print(f"\nSaved results to {save_results}")
        print(f"  {comparison_csv}")
        print(f"  {summary_json}")


if __name__ == "__main__":
    main()
