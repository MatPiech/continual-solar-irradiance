#!/usr/bin/env python3
"""On-device cost of the Chapter 4 exclusion flag (per-position detector), and its net daily cost.

Run ON the Raspberry Pi 5 / Jetson Orin Nano.  Per frame (batch 1, deployed input shape):

  inference         forecaster forward pass alone
  inference + flag  the same pass with the layer2 hook, 3x3 pooling, fixed random projection to 128 dims,
                    nearest-neighbour distance of each of the 172 cells to its own 2000-entry bank,
                    normalisation, global median and core maximum -- the exclusion score s_t
  flag (marginal)   the difference: what screening adds to a forecast that is issued anyway

plus the resident bank memory (process RSS before/after loading it) and energy per frame net of idle.

Net cost per day: detector cost x daylight frames per day, against the update cost the flag avoids, from the
withheld-update counts of tab:ch4-stream-actions and the per-action costs of scripts/edge_adaptation_benchmark.py
(--overhead-json).  The composition of the avoided updates is not recorded, so the saving is bracketed by the
head-update and damped-update costs.

    python scripts/edge_admissibility_cost.py --out results/edge/pi5/admissibility.json \
        --overhead-json results/edge/pi5/adaptation.json 2>&1 | tee logs/edge_flag_pi5.log
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
from time import perf_counter

import click
import numpy as np
import torch

from cirr.evaluation.admissibility import LocalGate, local_projection
from cirr.evaluation.runner import load_eval_model
from edge_power import PowerSampler, cooldown, thermal_state

HW, HISTORY = 128, 4
FRAMES_PER_DAY = 245538 / 362  # Folsom 2016 scored daylight frames per day (tab:ch4-solar-benchmark)
# Updates withheld by the flag, per day, from tab:ch4-stream-actions (policy minus policy + flag, 362 days).
AVOIDED_PER_DAY = {"clean": (103435 - 100791) / 362, "mixed episodes": (105116 - 100901) / 362}


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def hwm_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def timed(fn, n: int, warmup: int, cuda: bool) -> tuple[list[float], float, float]:
    times, t0 = [], 0.0
    for i in range(warmup + n):
        if cuda:
            torch.cuda.synchronize()
        if i == warmup:
            t0 = PowerSampler.now()
        a = perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((perf_counter() - a) * 1e3)
    return times, t0, PowerSampler.now()


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=Path("checkpoints/resnet50_epoch_015.ckpt"))
@click.option("--gate", type=click.Path(exists=True, path_type=Path), default=Path("results/adaptation/gate/local/local_gate.npz"))
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--bank-dtype", type=click.Choice(["fp32", "fp16"]), default="fp32", show_default=True)
@click.option("--samples", default=200, show_default=True)
@click.option("--warmup", default=20, show_default=True)
@click.option("--threads", type=int, default=None)
@click.option("--idle-seconds", default=20.0, show_default=True)
@click.option("--cool-to", default=50.0, show_default=True, help="Chip temperature (C) to wait for before each timed block.")
@click.option("--overhead-json", type=click.Path(exists=True, path_type=Path), default=None,
              help="Output of edge_adaptation_benchmark.py on the same device, for the avoided-update cost.")
@click.option("--out", type=click.Path(path_type=Path), default=None)
@torch.no_grad()
def main(checkpoint, gate, device, bank_dtype, samples, warmup, threads, idle_seconds, cool_to, overhead_json, out) -> None:
    if threads:
        torch.set_num_threads(threads)
    cuda = device == "cuda"
    thermal = {"before_idle": cooldown(cool_to, label="before idle baseline")}
    with PowerSampler() as ps:
        idle_w = ps.idle(idle_seconds)
        print(f"power backend: {ps.backend}; idle {idle_w:.3f} W", flush=True)

        model = load_eval_model(checkpoint, device)
        x, h = torch.rand(1, 3, HW, HW, device=device), torch.rand(1, HISTORY, device=device)
        rss_model = rss_mb()
        g = LocalGate.load(gate)
        dt = torch.float32 if bank_dtype == "fp32" else torch.float16
        bank = g.bank.to(device=device, dtype=dt).contiguous()
        del g.bank  # keep only the working copy resident
        scale = g.scale.to(device).float()
        core = torch.nonzero(g.core).flatten().to(device)
        flat = g.cells.flatten().to(device)
        proj = local_projection().to(device)

        grabbed: dict[str, torch.Tensor] = {}

        def infer() -> None:
            model(x, h)

        def infer_and_flag() -> None:
            model(x, h)
            f = torch.nn.functional.avg_pool2d(grabbed["f"].float(), 3, 1, 1)
            f = (f.flatten(2).transpose(1, 2)[:, flat] @ proj).transpose(0, 1).to(dt)  # [cells, 1, 128]
            d = torch.cdist(f, bank).amin(-1).squeeze(-1).float() / scale             # [cells]
            _s = max(float(d.median()), float(d[core].max()))                          # G and C statistics

        res = {}
        thermal["before_inference"] = cooldown(cool_to, label="before inference")
        ms, t0, t1 = timed(infer, samples, warmup, cuda)
        e = ps.energy(t0, t1)
        res["inference"] = dict(mean=float(np.mean(ms)), std=float(np.std(ms)), p95=float(np.percentile(ms, 95)),
                                energy_mj=1e3 * (e - idle_w * (t1 - t0)) / len(ms))
        thermal["before_flag"] = cooldown(cool_to, label="before inference + flag")
        hook = model.image_encoder.layer2.register_forward_hook(lambda _m, _i, o: grabbed.__setitem__("f", o))
        ms, t0, t1 = timed(infer_and_flag, samples, warmup, cuda)
        hook.remove()
        rss_bank = rss_mb()  # after scoring, so the bank pages are resident
        e = ps.energy(t0, t1)
        res["inference_flag"] = dict(mean=float(np.mean(ms)), std=float(np.std(ms)), p95=float(np.percentile(ms, 95)),
                                     energy_mj=1e3 * (e - idle_w * (t1 - t0)) / len(ms))
        backend = ps.backend

    flag_ms = res["inference_flag"]["mean"] - res["inference"]["mean"]
    flag_mj = res["inference_flag"]["energy_mj"] - res["inference"]["energy_mj"]
    memory = dict(rss_model_mb=rss_model, rss_after_scoring_mb=rss_bank, peak_rss_mb=hwm_mb(),
                  bank_bytes_mb=bank.numel() * bank.element_size() / 1e6,
                  peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20 if cuda else None)
    per_day = dict(frames=FRAMES_PER_DAY, flag_s=flag_ms * FRAMES_PER_DAY / 1e3, flag_j=flag_mj * FRAMES_PER_DAY / 1e3)

    avoided = {}
    if overhead_json:
        oh = json.loads(Path(overhead_json).read_text())["results"]
        costs = {k: (oh[k]["update"]["mean"], oh[k]["update"].get("energy_mj")) for k in ("head", "damped") if k in oh}
        for stream, n in AVOIDED_PER_DAY.items():
            avoided[stream] = {k: dict(updates_per_day=n, saved_s=n * c[0] / 1e3,
                                       saved_j=(n * c[1] / 1e3) if c[1] is not None else None) for k, c in costs.items()}

    print(f"\n{platform.node()} ({platform.machine()}, {device}, bank {bank_dtype})")
    print(f"  inference          {res['inference']['mean']:8.2f} ms   {res['inference']['energy_mj']:8.1f} mJ")
    print(f"  inference + flag   {res['inference_flag']['mean']:8.2f} ms   {res['inference_flag']['energy_mj']:8.1f} mJ")
    print(f"  flag (marginal)    {flag_ms:8.2f} ms   {flag_mj:8.1f} mJ   bank {memory['bank_bytes_mb']:.0f} MB, peak RSS {memory['peak_rss_mb']:.0f} MB")
    print(f"  per day ({FRAMES_PER_DAY:.0f} frames): flag {per_day['flag_s']:.1f} s, {per_day['flag_j']:.1f} J")
    for stream, d in avoided.items():
        parts = ", ".join(f"{k}: {v['saved_s']:.1f} s" + (f" / {v['saved_j']:.1f} J" if v["saved_j"] is not None else "") for k, v in d.items())
        print(f"  avoided updates, {stream} ({next(iter(d.values()))['updates_per_day']:.1f}/day): {parts}")

    if out:
        meta = dict(host=platform.node(), machine=platform.machine(), device=device, torch=torch.__version__,
                    threads=torch.get_num_threads(), bank_dtype=bank_dtype, samples=samples, warmup=warmup,
                    power_backend=backend, idle_w=idle_w, checkpoint=str(checkpoint), gate=str(gate))
        thermal["end"] = thermal_state()
        rec = dict(meta=meta, thermal=thermal, results=res, flag_marginal=dict(ms=flag_ms, mj=flag_mj), memory=memory,
                   per_day=per_day, avoided_updates=avoided)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(json.loads(json.dumps(rec), parse_constant=lambda _c: None), indent=2))
        print("wrote", out)


if __name__ == "__main__":
    main()
