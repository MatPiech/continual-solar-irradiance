"""Quick edge-oriented benchmark for online adaptation and TTA methods.

Measures (per method):
- inference latency distribution (ms/sample)
- adaptation latency distribution (ms/update event)
- end-to-end latency distribution (ms/sample)
- update duty cycle (fraction of samples triggering adaptation)
- throughput (samples/sec)
- memory stats (GPU peak allocated/reserved, CPU RSS/HWM deltas)

The benchmark is intentionally lightweight: it evaluates only ``num_samples`` from
``split='test'`` and focuses on system performance rather than forecast quality.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import resource
import time
from typing import Any, Literal

import click
import numpy as np
import torch
import torch.nn as nn

from cirr.continual.buffer import reveal_after_horizon
from cirr.data.loading import load_dataset
from cirr.evaluation.online import infer_one
from cirr.evaluation.runner import load_eval_model
from cirr.losses import BerHuLoss
from cirr.training.ogd import algorithm1_scope, ogd_step, pred_mean, sliding_minibatch_step
from cirr.training.sm_rls import ShermanMorrisonHeadUpdater
from cirr.tta import MEMOReg, NOTEReg

MethodName = Literal["baseline", "memo", "note", "paper_ogd", "sliding", "sm_rls"]


def _sync_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _now() -> float:
    return time.perf_counter()


def _clamp_nonnegative(pred: torch.Tensor) -> torch.Tensor:
    return torch.clamp(pred, min=0.0)


def _summarize_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "count": 0.0,
        }
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "count": float(arr.size),
    }


def _format_ms(x: float) -> str:
    return "nan" if np.isnan(x) else f"{x:.3f}"


def _format_float(x: float, digits: int = 4) -> str:
    return "nan" if np.isnan(x) else f"{x:.{digits}f}"


def _read_proc_kb(field: str) -> int:
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(field):
                    # Example: VmHWM:\t 12345 kB
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except OSError:
        return -1
    return -1


def _cpu_mem_snapshot() -> dict[str, float]:
    rss_kb = _read_proc_kb("VmRSS:")
    hwm_kb = _read_proc_kb("VmHWM:")
    ru_maxrss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "rss_mb": float(rss_kb) / 1024.0 if rss_kb >= 0 else float("nan"),
        "hwm_mb": float(hwm_kb) / 1024.0 if hwm_kb >= 0 else float("nan"),
        "ru_maxrss_mb": ru_maxrss_kb / 1024.0,
    }


def _gpu_mem_snapshot(device: str) -> dict[str, float]:
    if device != "cuda":
        return {
            "peak_alloc_mb": float("nan"),
            "peak_reserved_mb": float("nan"),
            "current_alloc_mb": float("nan"),
            "current_reserved_mb": float("nan"),
        }
    return {
        "peak_alloc_mb": float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0),
        "current_alloc_mb": float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0),
        "current_reserved_mb": float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0),
    }


@dataclass
class MethodResult:
    method: str
    samples: int
    updates: int
    infer_ms: dict[str, float]
    adapt_ms: dict[str, float]
    e2e_ms: dict[str, float]
    duty_cycle: float
    throughput_sps: float
    cpu_mem_before: dict[str, float]
    cpu_mem_after: dict[str, float]
    gpu_mem: dict[str, float]


def _benchmark_method(
    method: MethodName,
    checkpoint: Path,
    dataset: Any | None,
    sample_indices: list[int],
    device: str,
    input_source: Literal["dataset", "random"],
    random_seed: int,
    horizon_steps: int,
    # TTA args
    tta_adapt_mode: Literal["all", "bn_only"],
    memo_n_augmentations: int,
    memo_niter: int,
    note_update_every_x: int,
    note_niter: int,
    note_use_learned_stats: bool,
    note_bn_momentum: float,
    # OGD/sliding args
    ogd_t1: float,
    ogd_t2: float,
    ogd_lr: float,
    ogd_loss_name: Literal["mse", "huber", "berhu"],
    ogd_beta1: float,
    ogd_beta2: float,
    ogd_weight_decay: float,
    # Sliding args
    sliding_buffer_size: int,
    sliding_scope: Literal["head_only", "head_and_encoder"],
    # SM-RLS args
    sm_update_scope: Literal["last_layer", "full_head"],
    sm_lambda_init: float,
    sm_forgetting_factor: float,
    sm_full_head_regularization_mult: float,
    sm_error_clip: float,
    sm_max_last_layer_update_norm: float,
    sm_max_first_layer_update_norm: float,
    sm_anchor_decay_last: float,
    sm_anchor_decay_first: float,
) -> MethodResult:
    model = load_eval_model(checkpoint, device)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(random_seed)

    infer_times_ms: list[float] = []
    adapt_times_ms: list[float] = []
    e2e_times_ms: list[float] = []

    updates = 0
    total_samples = 0

    cpu_before = _cpu_mem_snapshot()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Method-specific state
    memo: MEMOReg | None = None
    note: NOTEReg | None = None
    causal_queue: deque[dict[str, Any]] = deque()
    adapt_buffer: deque[dict[str, Any]] = deque(maxlen=sliding_buffer_size)
    optimizer: torch.optim.Optimizer | None = None
    loss_fn: nn.Module | None = None
    sm_updater: ShermanMorrisonHeadUpdater | None = None

    if method == "memo":
        memo = MEMOReg(
            model=model,
            lr=ogd_lr,
            weight_decay=ogd_weight_decay,
            n_augmentations=memo_n_augmentations,
            niter=memo_niter,
            episodic=True,
            adapt_mode=tta_adapt_mode,
        ).to(device)
    elif method == "note":
        note = NOTEReg(
            model=model,
            lr=ogd_lr,
            weight_decay=ogd_weight_decay,
            update_every_x=note_update_every_x,
            niter=note_niter,
            adapt_mode=tta_adapt_mode,
            use_learned_stats=note_use_learned_stats,
            bn_momentum=note_bn_momentum,
        ).to(device)
    elif method in {"paper_ogd", "sliding"}:
        if ogd_loss_name == "berhu":
            loss_fn = BerHuLoss(threshold=30, scale=1000).to(device)
        elif ogd_loss_name == "huber":
            loss_fn = nn.HuberLoss().to(device)
        else:
            loss_fn = nn.MSELoss().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=ogd_lr,
            betas=(ogd_beta1, ogd_beta2),
            weight_decay=ogd_weight_decay,
        )
    elif method == "sm_rls":
        sm_updater = ShermanMorrisonHeadUpdater(
            model=model,
            lambda_init=sm_lambda_init,
            forgetting_factor=sm_forgetting_factor,
            full_head_regularization_mult=sm_full_head_regularization_mult,
            error_clip=sm_error_clip,
            max_last_layer_update_norm=sm_max_last_layer_update_norm,
            max_first_layer_update_norm=sm_max_first_layer_update_norm,
            anchor_decay_last=sm_anchor_decay_last,
            anchor_decay_first=sm_anchor_decay_first,
            device=device,
        )

    _sync_if_cuda(device)
    bench_start = _now()

    for idx in sample_indices:
        if input_source == "random":
            image_input = torch.randn((3, 128, 128), generator=rng)
            irradiance_history = torch.randn((4,), generator=rng)
            target_irradiance = torch.randn((1,), generator=rng)
        else:
            if dataset is None:
                raise ValueError("Dataset must be provided when input_source='dataset'.")
            image_input, irradiance_history, ghi_clear, target_irradiance, _image_name = dataset[idx]

        total_samples += 1
        sample_start = _now()

        if method == "baseline":
            t0 = _now()
            images = image_input.unsqueeze(0).to(device)
            hist = irradiance_history.unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                out = model(images, hist)
                pred = pred_mean(out)
                _ = _clamp_nonnegative(pred)
            _sync_if_cuda(device)
            infer_times_ms.append((_now() - t0) * 1000.0)

        elif method == "memo":
            assert memo is not None

            # Adaptation timing (per-sample episodic reset + K adaptation steps)
            t_adapt0 = _now()
            memo.model.load_state_dict(memo._initial_model_state, strict=True)
            memo.configure_model()
            memo._adapt_single(image_input, irradiance_history.unsqueeze(0), torch.device(device))
            _sync_if_cuda(device)
            adapt_times_ms.append((_now() - t_adapt0) * 1000.0)
            updates += 1

            # Inference timing on original sample
            t_inf0 = _now()
            mean, _var = memo._predict_single(image_input, irradiance_history.unsqueeze(0), torch.device(device))
            _ = _clamp_nonnegative(mean)
            _sync_if_cuda(device)
            infer_times_ms.append((_now() - t_inf0) * 1000.0)

        elif method == "note":
            assert note is not None

            note._buffer_images.append(image_input.detach().cpu())
            note._buffer_histories.append(irradiance_history.detach().cpu())
            note._sample_count += 1

            if note._sample_count % note.update_every_x == 0:
                t_adapt0 = _now()
                note._adapt_from_buffer(torch.device(device))
                _sync_if_cuda(device)
                adapt_times_ms.append((_now() - t_adapt0) * 1000.0)
                updates += 1

            t_inf0 = _now()
            mean, _var = note._predict_single(image_input, irradiance_history, torch.device(device))
            _ = _clamp_nonnegative(mean)
            _sync_if_cuda(device)
            infer_times_ms.append((_now() - t_inf0) * 1000.0)

        elif method == "paper_ogd":
            assert optimizer is not None and loss_fn is not None

            t_inf0 = _now()
            mean_raw, _ = infer_one(model, image_input, irradiance_history, device)
            _sync_if_cuda(device)
            infer_times_ms.append((_now() - t_inf0) * 1000.0)

            oldest = reveal_after_horizon(
                causal_queue,
                horizon_steps,
                {
                    "image": image_input,
                    "history": irradiance_history,
                    "gt": target_irradiance.unsqueeze(0),
                    "raw_pred": float(mean_raw.detach().cpu().view(-1)[0].item()),
                },
            )
            if oldest is not None:
                delayed_gt = float(oldest["gt"].view(-1)[0].item())
                delayed_abs_error = abs(float(oldest["raw_pred"]) - delayed_gt)
                scope = algorithm1_scope(delayed_abs_error, ogd_t1, ogd_t2)

                if scope != "none":
                    t_adapt0 = _now()
                    ogd_step(
                        model=model,
                        optimizer=optimizer,
                        scope=scope,
                        loss_fn=loss_fn,
                        image=oldest["image"],
                        history=oldest["history"],
                        gt=oldest["gt"],
                        device=device,
                    )
                    _sync_if_cuda(device)
                    adapt_times_ms.append((_now() - t_adapt0) * 1000.0)
                    updates += 1

        elif method == "sliding":
            assert optimizer is not None and loss_fn is not None

            t_inf0 = _now()
            _, _ = infer_one(model, image_input, irradiance_history, device)
            _sync_if_cuda(device)
            infer_times_ms.append((_now() - t_inf0) * 1000.0)

            oldest = reveal_after_horizon(
                causal_queue,
                horizon_steps,
                {
                    "image": image_input,
                    "history": irradiance_history,
                    "gt": target_irradiance.unsqueeze(0),
                },
            )
            if oldest is not None:
                adapt_buffer.append(oldest)
                if len(adapt_buffer) >= sliding_buffer_size:
                    t_adapt0 = _now()
                    sliding_minibatch_step(
                        model=model,
                        optimizer=optimizer,
                        scope=sliding_scope,
                        loss_fn=loss_fn,
                        buffer=list(adapt_buffer),
                        device=device,
                    )
                    _sync_if_cuda(device)
                    adapt_times_ms.append((_now() - t_adapt0) * 1000.0)
                    updates += 1

        elif method == "sm_rls":
            assert sm_updater is not None

            t_inf0 = _now()
            _, _ = infer_one(model, image_input, irradiance_history, device)
            _sync_if_cuda(device)
            infer_times_ms.append((_now() - t_inf0) * 1000.0)

            oldest = reveal_after_horizon(
                causal_queue,
                horizon_steps,
                {
                    "image": image_input,
                    "history": irradiance_history,
                    "gt": target_irradiance.unsqueeze(0),
                },
            )
            if oldest is not None:
                t_adapt0 = _now()
                sm_updater.update(
                    model=model,
                    image=oldest["image"],
                    history=oldest["history"],
                    gt=oldest["gt"],
                    device=device,
                    scope=sm_update_scope,
                )
                _sync_if_cuda(device)
                adapt_times_ms.append((_now() - t_adapt0) * 1000.0)
                updates += 1

        else:
            raise ValueError(f"Unsupported method: {method!r}")

        e2e_times_ms.append((_now() - sample_start) * 1000.0)

    _sync_if_cuda(device)
    wall_s = _now() - bench_start

    cpu_after = _cpu_mem_snapshot()
    gpu_mem = _gpu_mem_snapshot(device)

    duty_cycle = float(updates) / float(total_samples) if total_samples > 0 else float("nan")
    throughput_sps = float(total_samples) / wall_s if wall_s > 0.0 else float("nan")

    return MethodResult(
        method=method,
        samples=total_samples,
        updates=updates,
        infer_ms=_summarize_ms(infer_times_ms),
        adapt_ms=_summarize_ms(adapt_times_ms),
        e2e_ms=_summarize_ms(e2e_times_ms),
        duty_cycle=duty_cycle,
        throughput_sps=throughput_sps,
        cpu_mem_before=cpu_before,
        cpu_mem_after=cpu_after,
        gpu_mem=gpu_mem,
    )


def _print_result_table(results: list[MethodResult], device: str) -> None:
    print("\n=== Edge Benchmark Summary ===")
    print(
        "method       samples updates duty   infer_mean  infer_p95  adapt_mean  adapt_p95  " "e2e_mean   e2e_p95   throughput"
    )
    for r in results:
        print(
            f"{r.method:<12} {r.samples:<7d} {r.updates:<7d} {r.duty_cycle:<6.3f} "
            f"{_format_ms(r.infer_ms['mean']):>10} {_format_ms(r.infer_ms['p95']):>10} "
            f"{_format_ms(r.adapt_ms['mean']):>11} {_format_ms(r.adapt_ms['p95']):>10} "
            f"{_format_ms(r.e2e_ms['mean']):>9} {_format_ms(r.e2e_ms['p95']):>9} "
            f"{_format_float(r.throughput_sps, 2):>11}"
        )

    print("\n=== Memory (per method process-level snapshots) ===")
    print("method       cpu_rss_mb  cpu_hwm_mb  cpu_hwm_delta_mb", end="")
    if device == "cuda":
        print("  gpu_peak_alloc_mb  gpu_peak_reserved_mb")
    else:
        print("")

    for r in results:
        hwm_delta = r.cpu_mem_after["hwm_mb"] - r.cpu_mem_before["hwm_mb"]
        line = (
            f"{r.method:<12} "
            f"{_format_float(r.cpu_mem_after['rss_mb'], 2):>10} "
            f"{_format_float(r.cpu_mem_after['hwm_mb'], 2):>11} "
            f"{_format_float(hwm_delta, 2):>16}"
        )
        if device == "cuda":
            line += (
                f" { _format_float(r.gpu_mem['peak_alloc_mb'], 2):>18}"
                f" { _format_float(r.gpu_mem['peak_reserved_mb'], 2):>21}"
            )
        print(line)


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True, help="Path to model checkpoint.")
@click.option("--dataset-path", type=click.Path(exists=True, path_type=Path), required=False, help="Prepared dataset path.")
@click.option(
    "--input-source",
    type=click.Choice(["dataset", "random"], case_sensitive=False),
    default="dataset",
    show_default=True,
    help="Input source for benchmarking.",
)
@click.option("--random-seed", type=int, default=42, show_default=True, help="Random seed used in random input mode.")
@click.option("--num-samples", type=click.IntRange(min=1), default=512, show_default=True, help="Number of test samples.")
@click.option("--offset", type=click.IntRange(min=0), default=0, show_default=True, help="Start index in the test split.")
@click.option(
    "--methods",
    type=str,
    default="baseline,memo,note,paper_ogd,sliding,sm_rls",
    show_default=True,
    help="Comma-separated methods: baseline,memo,note,paper_ogd,sliding,sm_rls",
)
@click.option("--device", type=click.Choice(["cuda", "cpu"], case_sensitive=False), default="cuda", show_default=True)
@click.option("--horizon-steps", type=click.IntRange(min=1), default=15, show_default=True)
# TTA parameters
@click.option(
    "--tta-adapt-mode", type=click.Choice(["all", "bn_only"], case_sensitive=False), default="all", show_default=True
)
@click.option("--memo-n-augmentations", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--memo-niter", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--note-update-every-x", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--note-niter", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--note-use-learned-stats/--note-use-batch-stats", default=True, show_default=True)
@click.option("--note-bn-momentum", type=float, default=0.1, show_default=True)
# OGD/sliding parameters
@click.option("--ogd-t1", type=float, default=0.01, show_default=True)
@click.option("--ogd-t2", type=float, default=0.1, show_default=True)
@click.option("--ogd-lr", type=float, default=1e-5, show_default=True)
@click.option(
    "--ogd-loss", type=click.Choice(["mse", "huber", "berhu"], case_sensitive=False), default="mse", show_default=True
)
@click.option("--ogd-beta1", type=float, default=0.5, show_default=True)
@click.option("--ogd-beta2", type=float, default=0.975, show_default=True)
@click.option("--ogd-weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--sliding-buffer-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option(
    "--sliding-scope",
    type=click.Choice(["head_only", "head_and_encoder"], case_sensitive=False),
    default="head_only",
    show_default=True,
)
# SM-RLS parameters
@click.option("--sm-update-scope", type=click.Choice(["last_layer", "full_head"], case_sensitive=False), default="last_layer")
@click.option("--sm-lambda-init", type=float, default=1.0, show_default=True)
@click.option("--sm-forgetting-factor", type=click.FloatRange(min=1e-6, max=1.0), default=1.0, show_default=True)
@click.option("--sm-full-head-regularization-mult", type=click.FloatRange(min=1.0), default=100.0, show_default=True)
@click.option("--sm-error-clip", type=click.FloatRange(min=1e-6), default=0.2, show_default=True)
@click.option("--sm-max-last-layer-update-norm", type=click.FloatRange(min=1e-8), default=0.01, show_default=True)
@click.option("--sm-max-first-layer-update-norm", type=click.FloatRange(min=1e-8), default=0.002, show_default=True)
@click.option("--sm-anchor-decay-last", type=click.FloatRange(min=0.0, max=1.0), default=1e-4, show_default=True)
@click.option("--sm-anchor-decay-first", type=click.FloatRange(min=0.0, max=1.0), default=5e-4, show_default=True)
def main(
    checkpoint: Path,
    dataset_path: Path | None,
    input_source: str,
    random_seed: int,
    num_samples: int,
    offset: int,
    methods: str,
    device: str,
    horizon_steps: int,
    tta_adapt_mode: str,
    memo_n_augmentations: int,
    memo_niter: int,
    note_update_every_x: int,
    note_niter: int,
    note_use_learned_stats: bool,
    note_bn_momentum: float,
    ogd_t1: float,
    ogd_t2: float,
    ogd_lr: float,
    ogd_loss: str,
    ogd_beta1: float,
    ogd_beta2: float,
    ogd_weight_decay: float,
    sliding_buffer_size: int,
    sliding_scope: str,
    sm_update_scope: str,
    sm_lambda_init: float,
    sm_forgetting_factor: float,
    sm_full_head_regularization_mult: float,
    sm_error_clip: float,
    sm_max_last_layer_update_norm: float,
    sm_max_first_layer_update_norm: float,
    sm_anchor_decay_last: float,
    sm_anchor_decay_first: float,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    if ogd_t1 >= ogd_t2:
        raise click.UsageError(f"--ogd-t1 ({ogd_t1}) must be < --ogd-t2 ({ogd_t2}).")

    requested_methods = [m.strip().lower() for m in methods.split(",") if m.strip()]
    valid_methods: set[str] = {"baseline", "memo", "note", "paper_ogd", "sliding", "sm_rls"}
    invalid = [m for m in requested_methods if m not in valid_methods]
    if invalid:
        raise click.UsageError(f"Unknown methods: {invalid}. Valid: {sorted(valid_methods)}")

    dataset: Any | None = None
    if input_source == "dataset":
        if dataset_path is None:
            raise click.UsageError("--dataset-path is required when --input-source=dataset.")
        print(f"Loading dataset from {dataset_path} ...")
        dataset = load_dataset(dataset_path, split="test")
        if len(dataset) == 0:
            raise click.UsageError("Test dataset is empty.")

        max_start = len(dataset) - 1
        if offset > max_start:
            raise click.UsageError(f"--offset {offset} exceeds last test index {max_start}.")

        end = min(len(dataset), offset + num_samples)
        sample_indices = list(range(offset, end))
    else:
        sample_indices = list(range(num_samples))

    print(
        f"Benchmark config: input_source={input_source} device={device} samples={len(sample_indices)} "
        f"offset={offset} methods={requested_methods} horizon_steps={horizon_steps}"
    )

    results: list[MethodResult] = []
    for method in requested_methods:
        print(f"\n--- Benchmarking method={method} ---")
        result = _benchmark_method(
            method=method,  # type: ignore[arg-type]
            checkpoint=checkpoint,
            dataset=dataset,
            sample_indices=sample_indices,
            device=device,
            input_source=input_source,  # type: ignore[arg-type]
            random_seed=random_seed,
            horizon_steps=horizon_steps,
            tta_adapt_mode=tta_adapt_mode,  # type: ignore[arg-type]
            memo_n_augmentations=memo_n_augmentations,
            memo_niter=memo_niter,
            note_update_every_x=note_update_every_x,
            note_niter=note_niter,
            note_use_learned_stats=note_use_learned_stats,
            note_bn_momentum=note_bn_momentum,
            ogd_t1=ogd_t1,
            ogd_t2=ogd_t2,
            ogd_lr=ogd_lr,
            ogd_loss_name=ogd_loss,  # type: ignore[arg-type]
            ogd_beta1=ogd_beta1,
            ogd_beta2=ogd_beta2,
            ogd_weight_decay=ogd_weight_decay,
            sliding_buffer_size=sliding_buffer_size,
            sliding_scope=sliding_scope,  # type: ignore[arg-type]
            sm_update_scope=sm_update_scope,  # type: ignore[arg-type]
            sm_lambda_init=sm_lambda_init,
            sm_forgetting_factor=sm_forgetting_factor,
            sm_full_head_regularization_mult=sm_full_head_regularization_mult,
            sm_error_clip=sm_error_clip,
            sm_max_last_layer_update_norm=sm_max_last_layer_update_norm,
            sm_max_first_layer_update_norm=sm_max_first_layer_update_norm,
            sm_anchor_decay_last=sm_anchor_decay_last,
            sm_anchor_decay_first=sm_anchor_decay_first,
        )
        results.append(result)

    _print_result_table(results, device=device)
    print("\nAdditional recommendation: use p95/p99 end-to-end latency and update duty cycle as edge-SLA metrics.")


if __name__ == "__main__":
    main()
