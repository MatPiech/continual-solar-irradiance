#!/usr/bin/env python3
"""Per-sample cost of Chapter 4 adaptation, measured on the target device.

Run this ON the Raspberry Pi 5 / Jetson Orin Nano.  Every method runs the exact update code
and hyper-parameters of its campaign run (results/adaptation/*/<site>/<method>/seed0/config.yaml),
each in a fresh process so peak memory is attributable to that method alone:

  static     no_grad forward only
  head       conditional policy, head-only action (frozen encoder)
  damped     conditional policy, proposed action: layer3+layer4 + head, encoder LR 0.1 x head LR
  full       conditional policy, full-encoder action
  window     Lara-Benitez et al.: one Adam step on the 4 latest labelled samples, every sample
  rls        Zhang et al.: Sherman-Morrison RLS on the last layer, every sample
  memo       MEMO: episodic reset + 8-view entropy step, every sample
  note       NOTE: prediction-variance step on a 64-sample buffer, every 64 samples

Two costs are reported.  ``update`` is one update event.  ``per-sample`` is the stream cost,
inference plus the update amortised by how often the method updates; for the conditional policy
that rate is the recorded action mix of tab:ch4-actions, so its per-sample cost is site dependent.

Energy: board power is sampled during every timed block (``scripts/edge_power.py``: Pi 5 PMIC or Jetson
tegrastats); an idle window before the runs gives the idle power, and each event's energy is reported net of
idle (mJ).  PMIC and tegrastats omit converter/carrier losses; check one run against an inline meter.

Inputs are synthetic tensors of the deployed shape (C x 128 x 128 image, 4-step history), so no
dataset is needed on the device; latency does not depend on pixel values.  The channel count is
read from the checkpoint, so the RGB and IEP checkpoints both work.

    python scripts/edge_adaptation_benchmark.py --checkpoint checkpoints/resnet50_epoch_015.ckpt \
        --out results/edge/pi5/adaptation.json 2>&1 | tee logs/edge_adaptation_pi5.log
"""

from __future__ import annotations

from collections import deque
import json
import multiprocessing as mp
from pathlib import Path
import platform
from time import perf_counter

import click
import numpy as np
import torch

from edge_power import PowerSampler, cooldown, thermal_state

HISTORY, HW = 4, 128
METHODS = ("static", "head", "damped", "full", "window", "rls", "memo", "note")
LABEL = {"static": "Static model", "head": "Conditional, head only", "damped": "Conditional, damped (proposed)",
         "full": "Conditional, full encoder", "window": "Lara-Benítez et al.", "rls": "Zhang et al.",
         "memo": "MEMO", "note": "NOTE"}
# Update events per sample for the comparators (window/rls/memo update on every sample, NOTE every 64th).
RATE = {"static": 0.0, "window": 1.0, "rls": 1.0, "memo": 1.0, "note": 1 / 64}
# Recorded action shares of the damped policy (head, encoder), tab:ch4-actions.
DAMPED_MIX = {"Folsom": (0.344, 0.086), "Poznań": (0.485, 0.170), "Karpicko": (0.505, 0.192)}


def _hwm_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def _timed(fn, n: int, warmup: int, cuda: bool, clock=perf_counter) -> tuple[list[float], float, float]:
    """Per-call latencies (ms) of the timed calls, and the monotonic window they span."""
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


def _energy_mj(ps: PowerSampler, t0: float, t1: float, n: int, idle_w: float) -> dict[str, float]:
    """Mean board power over the window and energy per call above idle (mJ)."""
    e = ps.energy(t0, t1)
    return {"power_w": e / (t1 - t0) if t1 > t0 else float("nan"),
            "energy_mj": 1e3 * (e - idle_w * (t1 - t0)) / n}


def _summary(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95))}


def measure(method: str, checkpoint: str, device: str, samples: int, warmup: int, threads: int | None, idle_w: float) -> dict:
    """One method in a fresh process: inference and update-event latency, peak memory."""
    from cirr.evaluation.runner import load_eval_model
    from cirr.training.ogd import freeze_all, make_regression_loss, ogd_step, sliding_minibatch_step
    from cirr.training.sm_rls import ShermanMorrisonHeadUpdater
    from cirr.tta import MEMOReg, NOTEReg

    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(0)
    cuda = device == "cuda"
    model = load_eval_model(Path(checkpoint), device)
    c = model.image_encoder.conv1.in_channels
    image, history, gt = torch.rand(c, HW, HW), torch.rand(HISTORY), torch.rand(1, 1)
    x, h = image.unsqueeze(0).to(device), history.unsqueeze(0).to(device)

    def infer() -> None:
        with torch.no_grad():
            model(x, h)

    update = None
    if method in ("head", "damped", "full"):
        # campaign optimiser: evaluate_lagged_ogd.py, config of conditional_last2_lowlr
        opt = torch.optim.AdamW([{"params": model.network_head.parameters(), "lr": 1e-5},
                                 {"params": model.image_encoder.parameters(), "lr": 1e-6}],
                                betas=(0.5, 0.975), weight_decay=1e-4)
        loss = make_regression_loss("mse").to(device)
        scope, blocks = {"head": ("head_only", None), "damped": ("head_and_encoder", 2), "full": ("head_and_encoder", None)}[method]
        update = lambda: ogd_step(model, opt, scope, loss, image, history, gt, device, encoder_blocks=blocks)  # noqa: E731
    elif method == "window":
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss = make_regression_loss("mse").to(device)
        buffer = [{"image": torch.rand(c, HW, HW), "history": torch.rand(HISTORY), "gt": torch.rand(1, 1)} for _ in range(4)]
        update = lambda: sliding_minibatch_step(model, opt, "head_only", loss, buffer, device)  # noqa: E731
    elif method == "rls":
        rls = ShermanMorrisonHeadUpdater(model=model, lambda_init=1.0, forgetting_factor=1.0, error_clip=0.2, device=device)
        update = lambda: rls.update(model=model, image=image, history=history, gt=gt, device=device, scope="last_layer")  # noqa: E731
    elif method == "memo":
        memo = MEMOReg(model=model, lr=1e-5, weight_decay=0.01, n_augmentations=8, niter=1, episodic=True, adapt_mode="all").to(device)

        def update() -> None:
            memo.model.load_state_dict(memo._initial_model_state, strict=True)
            memo.configure_model()
            memo._adapt_single(x[0], h, torch.device(device))  # inputs on the device, as in MEMOReg.forward

        infer = lambda: memo._predict_single(x[0], h, torch.device(device))  # noqa: E731
    elif method == "note":
        note = NOTEReg(model=model, lr=1e-5, weight_decay=0.01, update_every_x=64, niter=1, adapt_mode="all",
                       use_learned_stats=False, bn_momentum=0.1).to(device)
        note._buffer_images = deque((torch.rand(c, HW, HW) for _ in range(64)), maxlen=64)
        note._buffer_histories = deque((torch.rand(HISTORY) for _ in range(64)), maxlen=64)
        update = lambda: note._adapt_from_buffer(torch.device(device))  # noqa: E731
        infer = lambda: note._predict_single(image, history, torch.device(device))  # noqa: E731

    if method != "memo" and method != "note":
        freeze_all(model)
    with PowerSampler() as ps:
        ms, t0, t1 = _timed(infer, samples, warmup, cuda)
        out = {"inference": _summary(ms) | _energy_mj(ps, t0, t1, len(ms), idle_w)}
        if update is not None:
            n = max(10, samples // 10) if method == "note" else samples  # a NOTE event is a batch-64 step
            ms, t0, t1 = _timed(update, n, min(warmup, 5) if method == "note" else warmup, cuda)
            out["update"] = _summary(ms) | _energy_mj(ps, t0, t1, len(ms), idle_w)
        out["power_backend"] = ps.backend
    out["peak_rss_mb"] = _hwm_mb()
    if cuda:
        out["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 2**20
    out["input_channels"] = c
    return out


def _run_isolated(*args) -> dict:
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(measure, args)


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--methods", default=",".join(METHODS), show_default=True, help="Comma-separated subset.")
@click.option("--samples", default=200, show_default=True, help="Timed repetitions per measurement.")
@click.option("--warmup", default=20, show_default=True)
@click.option("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice).")
@click.option("--idle-seconds", default=20.0, show_default=True, help="Idle window for the idle-power baseline.")
@click.option("--cool-to", default=50.0, show_default=True, help="Chip temperature (C) to wait for before each method.")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Write the results as JSON.")
def main(checkpoint: Path, device: str, methods: str, samples: int, warmup: int, threads: int | None, idle_seconds: float, cool_to: float, out: Path | None) -> None:
    chosen = [m.strip() for m in methods.split(",") if m.strip()]
    if unknown := set(chosen) - set(METHODS):
        raise click.BadParameter(f"unknown methods {sorted(unknown)}; choose from {METHODS}")
    cooldown(cool_to, label="before idle baseline")
    with PowerSampler() as ps:
        idle_w = ps.idle(idle_seconds)
        backend = ps.backend
    print(f"power backend: {backend}; idle {idle_w:.3f} W over {idle_seconds:.0f} s", flush=True)
    res: dict[str, dict] = {}
    for m in chosen:
        start = cooldown(cool_to, label=f"before {m}")
        print(f"measuring {m} ...", flush=True)
        res[m] = _run_isolated(m, str(checkpoint), device, samples, warmup, threads, idle_w)
        res[m]["thermal"] = {"start": start, "end": thermal_state()}

    for m, r in res.items():  # per-sample stream cost
        upd = r.get("update", {}).get("mean", 0.0)
        if m in RATE:
            r["per_sample_ms"] = {"all sites": r["inference"]["mean"] + RATE[m] * upd}
        elif m == "damped" and "head" in res:
            head = res["head"]["update"]["mean"]
            r["per_sample_ms"] = {s: r["inference"]["mean"] + ph * head + pe * upd for s, (ph, pe) in DAMPED_MIX.items()}

    meta = {"checkpoint": str(checkpoint), "input_channels": next(iter(res.values()))["input_channels"], "device": device,
            "host": platform.node(), "machine": platform.machine(), "torch": torch.__version__,
            "threads": threads or torch.get_num_threads(), "samples": samples, "warmup": warmup,
            "power_backend": backend, "idle_w": idle_w, "cool_to_c": cool_to}
    print(f"\n{meta['host']} ({meta['machine']}, {device}, {meta['input_channels']}-channel input, {meta['threads']} threads)")
    print(f"{'method':<32} {'infer ms':>9} {'update ms':>17} {'per-sample ms':>28} {'Tput/s':>7} {'peak RSS MB':>12} {'infer mJ':>9} {'update mJ':>10}")
    for m, r in res.items():
        u = r.get("update")
        ups = f"{u['mean']:8.2f} ± {u['std']:6.2f}" if u else f"{'–':>17}"
        per = r.get("per_sample_ms", {})
        pers = " / ".join(f"{v:.1f}" for v in per.values()) if per else "–"
        tput = f"{1e3 / next(iter(per.values())):7.2f}" if len(per) == 1 else f"{'mix':>7}" if per else f"{'–':>7}"
        ue = f"{u['energy_mj']:10.1f}" if u else f"{'–':>10}"
        print(f"{LABEL[m]:<32} {r['inference']['mean']:9.2f} {ups:>17} {pers:>28} {tput} {r['peak_rss_mb']:12.1f} {r['inference']['energy_mj']:9.1f} {ue}")
    if "damped" in res and "per_sample_ms" in res["damped"]:
        print("  damped per-sample cost is per site: " + ", ".join(DAMPED_MIX) + " (action mix of tab:ch4-actions)")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        clean = json.loads(json.dumps({"meta": meta, "results": res}), parse_constant=lambda _c: None)  # NaN -> null
        out.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
        print("wrote", out)


if __name__ == "__main__":
    main()
