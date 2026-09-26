#!/usr/bin/env python3
"""Per-phase device cost of the Chapter 5 client, three matched arms on the same real days.

Run ON the Raspberry Pi 5 / Jetson Orin Nano.  Each arm follows the configuration of record of Chapter 5
(one local epoch per day, AdamW 1e-5, batch 32, replay of the 1000 most recent samples at ratio 0.5) and runs
in its own process, so its peak memory is its own:

  isolated     predict -> train                                     (no exchange; the chapter's reference)
  fedavg       predict -> train, then on sync: serialize the full state, deserialize + load the aggregate
  prototypes   predict -> train with the alignment term (C = 12, lambda 1.0, tau 0.5) -> extract prototypes
               after the update (every day, accumulated over the round), then on sync: serialize the payload,
               deserialize the returned prototypes

The per-phase differences between arms are the incremental client costs the chapter cannot state from the
legacy profile: alignment and post-update extraction for prototypes, state (de)serialization for FedAvg.  The
returned prototypes come from the server's attention aggregation over this client's payload and two
perturbed copies (distinct tensors); on-device compute does not depend on their content.  Server aggregation
and network transfer are measured separately (priority 5, host + clients).

Energy per phase comes from scripts/edge_power.py (Pi 5 PMIC or Jetson tegrastats), net of an idle window.

    python scripts/edge_collaboration_benchmark.py --dataset data/folsom_prepared --days 7 \
        --out results/edge/pi5/collaboration.json 2>&1 | tee logs/edge_collaboration_benchmark_pi5.log
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = ("isolated", "fedavg", "prototypes")
PHASES = ("predict", "train", "extract", "sync_client")


def _hwm_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def select_days(dataset: Path, days: int) -> list[str]:
    """Days spread across the calendar (day length is what varies most), plus one warm-up day first."""
    from cirr.data.loading import load_dataset
    from cirr.physics.clear_sky import DEFAULT_MIN_GHI_CLEAR, filter_lit_days, load_clear_sky_lookup
    from cirr.utils.dataset_utils import group_indices_by_day

    ds = load_dataset(dataset, split="test")
    day_map, _ = filter_lit_days(group_indices_by_day(ds.periods), ds.periods, load_clear_sky_lookup(dataset), DEFAULT_MIN_GHI_CLEAR)
    avail = sorted(day_map)
    stride = max(1, len(avail) // (days + 1))
    return avail[::stride][: days + 1]


def run_arm(arm: str, dataset: str, checkpoint: str, days: list[str], device: str, threads: int | None,
            idle_w: float, quiet: bool) -> dict:
    """One arm in a fresh process: per-day, per-phase time and energy, and peak memory."""
    from cirr.data.loading import load_dataset
    from cirr.fl.kc_prototypes import aggregate_global_prototypes, default_edges, extract_day_prototypes, make_regime_labeller
    from cirr.models.loading import load_forecaster_from_checkpoint
    from cirr.physics.clear_sky import DEFAULT_MIN_GHI_CLEAR, filter_lit_days, load_clear_sky_lookup
    from cirr.training.day_loop import predict_day, train_day
    from cirr.training.replay import RecentBuffer
    from cirr.utils.dataset_utils import group_indices_by_day
    from edge_power import PowerSampler

    if threads:
        torch.set_num_threads(threads)
    cuda = device == "cuda"
    ds = load_dataset(Path(dataset), split="test")
    day_map, _ = filter_lit_days(group_indices_by_day(ds.periods), ds.periods, load_clear_sky_lookup(Path(dataset)), DEFAULT_MIN_GHI_CLEAR)
    model = load_forecaster_from_checkpoint(Path(checkpoint), device=device, use_hparams=True)
    labeller = make_regime_labeller("clear_sky", default_edges("clear_sky"), DEFAULT_MIN_GHI_CLEAR, 4)
    recent = RecentBuffer(1000)
    returned: dict | None = None     # prototypes returned by the server, used as alignment targets
    accum: dict[int, tuple[torch.Tensor, int]] = {}
    sink = open(os.devnull, "w")
    rows = []

    with PowerSampler() as ps:
        def phase(row: dict, name: str, fn):
            if cuda:
                torch.cuda.synchronize()
            t0, a = ps.now(), perf_counter()
            if quiet:
                with redirect_stdout(sink):
                    out = fn()
            else:
                out = fn()
            if cuda:
                torch.cuda.synchronize()
            t1, dt = ps.now(), perf_counter() - a
            row[f"{name}_s"] = dt
            row[f"{name}_j"] = ps.energy(t0, t1) - idle_w * (t1 - t0)
            return out

        for pos, day in enumerate(days):
            idx = day_map[day]
            row = {"day": day, "n_samples": len(idx)}
            preds = phase(row, "predict", lambda: predict_day(model, ds, idx, device, 32, day))
            day_samples = preds[2]
            n_rep = int(round(len(idx) * 0.5))
            replay = recent.sample(n_rep, seed=pos + 1) or None
            align = arm == "prototypes" and returned is not None
            phase(row, "train", lambda: train_day(
                model, ds, idx, lr=1e-5, weight_decay=1e-4, max_grad_norm=1.0, batch_size=32, epochs=1, device=device,
                replay_entries=replay, rng_seed=pos, day_samples=day_samples, client_name=day,
                global_prototypes=returned if align else None, proto_lambda=1.0 if align else 0.0,
                align_loss="prototypical", align_temperature=0.5, classify_fn=labeller))
            row["n_replay"] = len(replay or [])
            row["aligned"] = align
            if arm == "prototypes":
                protos, _ = phase(row, "extract", lambda: extract_day_prototypes(model, day_samples, device, 32, labeller=labeller))
                for c, (s, n) in protos.items():
                    s0, n0 = accum.get(c, (torch.zeros_like(s), 0))
                    accum[c] = (s0 + s, n0 + n)
            recent.add_day(ds, idx)

            # client side of a synchronisation, measured every day so the median is robust
            if arm == "fedavg":
                def sync_fedavg():
                    buf = io.BytesIO(); torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
                    up = buf.tell(); buf.seek(0)
                    state = torch.load(buf, map_location=device); model.load_state_dict(state)
                    return up
                row["payload_up_bytes"] = phase(row, "sync_client", sync_fedavg)
            elif arm == "prototypes":
                def sync_proto():
                    payload = {c: (s.detach().cpu(), n) for c, (s, n) in accum.items()}
                    buf = io.BytesIO(); torch.save(payload, buf); up = buf.tell()
                    peers = [payload] + [{c: (s * (1 + 0.01 * k), n) for c, (s, n) in payload.items()} for k in (1, 2)]
                    per_client, _ = aggregate_global_prototypes(peers, attention=True, temperature=0.5)  # server, here for content
                    back = io.BytesIO(); torch.save(per_client[0], back); back.seek(0)
                    return up, torch.load(back, map_location=device)
                row["payload_up_bytes"], returned = phase(row, "sync_client", sync_proto)
            row["warmup"] = pos == 0
            rows.append(row)
            print(f"  [{arm}] {day} n={len(idx)} " + " ".join(f"{p}={row[p + '_s']:.2f}s" for p in PHASES if p + "_s" in row), flush=True)
        backend = ps.backend

    return {"rows": rows, "peak_rss_mb": _hwm_mb(), "power_backend": backend,
            "peak_cuda_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None}


def _isolated(*args) -> dict:
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(run_arm, args)


def _median(v: list[float]) -> float:
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


@click.command()
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=Path("checkpoints/resnet50_epoch_015.ckpt"))
@click.option("--days", default=7, show_default=True, help="Timed days (one warm-up day is added and discarded).")
@click.option("--arms", default=",".join(ARMS), show_default=True)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--threads", type=int, default=None)
@click.option("--idle-seconds", default=20.0, show_default=True)
@click.option("--cool-to", default=50.0, show_default=True, help="Chip temperature (C) to wait for before each arm.")
@click.option("--progress/--no-progress", default=False, help="Show the per-batch progress bars.")
@click.option("--out", type=click.Path(path_type=Path), default=None)
def main(dataset, checkpoint, days, arms, device, threads, idle_seconds, cool_to, progress, out) -> None:
    from edge_power import PowerSampler, cooldown, thermal_state

    chosen = [a.strip() for a in arms.split(",") if a.strip()]
    assert set(chosen) <= set(ARMS), f"arms must be among {ARMS}"
    sel = select_days(dataset, days)
    cooldown(cool_to, label="before idle baseline")
    with PowerSampler() as ps:
        idle_w = ps.idle(idle_seconds)
        backend = ps.backend
    print(f"power backend: {backend}; idle {idle_w:.3f} W | days (first is warm-up): {sel}", flush=True)
    res = {}
    for a in chosen:
        start = cooldown(cool_to, label=f"before {a}")
        res[a] = _isolated(a, str(dataset), str(checkpoint), sel, device, threads, idle_w, not progress)
        res[a]["thermal"] = {"start": start, "end": thermal_state()}

    summary = {}
    for a, r in res.items():
        timed = [x for x in r["rows"] if not x["warmup"]]
        summary[a] = {f"{p}_{u}": _median([x.get(f"{p}_{u}", float("nan")) for x in timed]) for p in PHASES for u in ("s", "j")}
        summary[a]["n_samples"] = _median([x["n_samples"] for x in timed])
        summary[a]["peak_rss_mb"] = r["peak_rss_mb"]
        summary[a]["payload_up_bytes"] = _median([x.get("payload_up_bytes", float("nan")) for x in timed])

    print(f"\n{platform.node()} ({platform.machine()}, {device}) — medians over {days} days")
    print(f"{'arm':<11}" + "".join(f"{p + ' s':>16}{p + ' J':>16}" for p in PHASES) + f"{'peak RSS MB':>13}")
    for a, s in summary.items():
        print(f"{a:<11}" + "".join(f"{s[p + '_s']:16.2f}{s[p + '_j']:16.1f}" for p in PHASES) + f"{s['peak_rss_mb']:13.0f}")
    if "isolated" in summary:
        base = summary["isolated"]
        for a in ("fedavg", "prototypes"):
            if a in summary:
                s = summary[a]
                inc_s = sum((s[p + "_s"] if s[p + "_s"] == s[p + "_s"] else 0) - (base[p + "_s"] if base[p + "_s"] == base[p + "_s"] else 0) for p in PHASES)
                inc_j = sum((s[p + "_j"] if s[p + "_j"] == s[p + "_j"] else 0) - (base[p + "_j"] if base[p + "_j"] == base[p + "_j"] else 0) for p in PHASES)
                print(f"  incremental over isolated, {a}: {inc_s:+.2f} s/day, {inc_j:+.1f} J/day (excl. transport and server)")

    if out:
        meta = dict(host=platform.node(), machine=platform.machine(), device=device, torch=torch.__version__,
                    threads=threads or torch.get_num_threads(), dataset=str(dataset), checkpoint=str(checkpoint),
                    days=sel, power_backend=backend, idle_w=idle_w, cool_to_c=cool_to)
        rec = json.loads(json.dumps(dict(meta=meta, summary=summary, arms=res)), parse_constant=lambda _c: None)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
        print("wrote", out)


if __name__ == "__main__":
    main()
