#!/usr/bin/env python3
"""Per-round edge cost of federated exchange, measured on the target device.

Run this ON the Raspberry Pi / Jetson Orin.  It replays a representative subset of
real days from one site's prepared dataset through the exact code path the campaign
uses (``predict_day`` -> ``extract_day_prototypes`` -> ``train_day``) and reports
everything ``tab:ch5-edge-overhead`` needs:

  * inference over a day
  * prototype extraction  (T_extract -- the term that decides the break-even)
  * one local training epoch with the alignment term active
  * round-trip payload bytes, full-model exchange vs prototype exchange
  * aggregation latency for both payload kinds
  * peak host RSS and accelerator memory
  * mean board power and per-round energy, where a rail sensor exists

Break-even link speed:  B* = 8 (S_weights - S_proto) / T_extract.  Below B* prototype
exchange completes a round sooner; above it, shipping the weights does.

Typical use (log to a file -- a run on a Pi is slow enough that losing it hurts):

    python scripts/edge_hw_benchmark.py --dataset data/karpicko_prepared \
        --out results/edge/pi5.json 2>&1 | tee logs/edge_pi5.log
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, suppress
import json
import os
from pathlib import Path
import platform
import sys
import threading
from time import perf_counter
from typing import Any

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fl_benchmark import filter_lit_days  # single-sourced: the night-sample filter

from cirr.data.loading import load_dataset
from cirr.fl.kc_prototypes import (
    DEFAULT_MIN_GHI_CLEAR,
    aggregate_global_prototypes,
    default_edges,
    extract_day_prototypes,
    make_regime_labeller,
)
from cirr.fl.server import federated_averaging, payload_nbytes
from cirr.models.loading import load_forecaster_from_checkpoint
from cirr.physics.clear_sky import load_clear_sky_lookup
from cirr.training.day_loop import predict_day, train_day
from cirr.utils.dataset_utils import group_indices_by_day

MIB = 1024.0**2


# --------------------------------------------------------------------------- #
# Platform probes
# --------------------------------------------------------------------------- #
def board_name() -> str:
    """Board string from the device tree (Pi and Jetson both publish one)."""
    model = Path("/proc/device-tree/model")
    if model.exists():
        return model.read_bytes().decode("utf-8", "ignore").strip("\x00").strip()
    return f"{platform.system()} {platform.machine()}"


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):  # peak RSS, not current
            return int(line.split()[1]) / 1024.0
    return float("nan")


def power_sensors() -> list[Path]:
    """INA3221 rails exposed through hwmon.  Jetson has them; a Pi does not."""
    found: list[Path] = []
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name_file = hwmon / "name"
        if not name_file.exists() or "ina" not in name_file.read_text().strip().lower():
            continue
        found.extend(sorted(hwmon.glob("power*_input")))
    return found


class PowerMeter:
    """Sample the board rails while a phase runs; NaN when there is no sensor.

    ponytail: 10 Hz polling of a sysfs file, which is plenty against phases that
    last seconds.  The scale is a knob because the INA3221 unit is not consistent
    across JetPack releases -- hwmon says microwatts, the older ``in_power0_input``
    nodes report milliwatts.  Check one reading against a wall meter before
    trusting the energy column, and pass --power-scale 1e-3 if it reads 1000x low.
    """

    def __init__(self, sensors: list[Path], scale: float, fallback_w: float | None):
        self.sensors, self.scale, self.fallback_w = sensors, scale, fallback_w
        self.samples: list[float] = []

    def __enter__(self) -> PowerMeter:
        self._stop = threading.Event()
        if self.sensors:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(0.1):
            with suppress(OSError, ValueError):
                self.samples.append(sum(int(p.read_text()) for p in self.sensors) * self.scale)

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self.sensors:
            self._thread.join(timeout=1.0)

    @property
    def mean_w(self) -> float:
        if self.samples:
            return sum(self.samples) / len(self.samples)
        return self.fallback_w if self.fallback_w is not None else float("nan")


@contextmanager
def quiet(enabled: bool):
    """Swallow the per-batch tqdm bars raised inside predict_day / train_day.

    They bury the timings in a teed log.  This tqdm build has no TQDM_DISABLE
    support, and the bars are handed ``file=sys.stdout`` at call time, so
    redirecting stdout around the phase is what actually silences them.
    """
    if not enabled:
        yield
        return
    with Path(os.devnull).open("w") as sink, redirect_stdout(sink):
        yield


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@contextmanager
def timed(store: dict[str, Any], key: str, device: str):
    sync(device)
    start = perf_counter()
    yield
    sync(device)
    store[key] = perf_counter() - start


def median(values: list[float]) -> float:
    """Median, not mean: one thermally throttled day should not move the estimate."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def break_even_mbps(saved_bytes: int, t_extract: float) -> float:
    """Link speed at which the transfer saved equals the extraction pass added."""
    if t_extract <= 0:
        return float("inf")
    return saved_bytes * 8.0 / t_extract / 1e6


# --------------------------------------------------------------------------- #
@click.command()
@click.option("--self-test", is_flag=True, help="Check the arithmetic and the platform probes, then exit.")
@click.option("--dataset", type=Path, help="Prepared site directory.")
@click.option("--checkpoint", default=Path("checkpoints/resnet50_epoch_015.ckpt"), type=Path)
@click.option("--days", default=5, help="Representative days to time (plus one warm-up).")
@click.option("--clients", default=3, help="Federation size used for the aggregation timing.")
@click.option("--batch-size", default=32)
@click.option("--adapt-lr", default=1e-5, help="Campaign value; the 1e-4 default degrades the model.")
@click.option("--adapt-epochs", default=1)
@click.option("--regime-label", default="clear_sky")
@click.option("--elevation-bins", default=4, help="3 sky bins x 4 elevation bins = the 12 regimes.")
@click.option("--proto-lambda", default=1.0)
@click.option("--align-temperature", default=0.5)
@click.option("--device", default=None, help="Default: cuda when present, else cpu.")
@click.option("--threads", default=0, help="torch CPU threads; 0 leaves the default.")
@click.option("--power-scale", default=1e-6, help="Sensor unit -> watts. hwmon is uW.")
@click.option("--power-w", default=None, type=float, help="Assumed board power when no sensor exists.")
@click.option("--progress/--no-progress", default=False, help="Show per-batch tqdm bars.")
@click.option("--out", type=Path, default=None, help="Write the full record as JSON.")
def main(
    self_test: bool,
    dataset: Path | None,
    checkpoint: Path,
    days: int,
    clients: int,
    batch_size: int,
    adapt_lr: float,
    adapt_epochs: int,
    regime_label: str,
    elevation_bins: int,
    proto_lambda: float,
    align_temperature: float,
    device: str | None,
    threads: int,
    power_scale: float,
    power_w: float | None,
    progress: bool,
    out: Path | None,
) -> None:
    if self_test:
        # The break-even is the one derived quantity the chapter quotes, so it is the
        # one thing worth pinning: these are the ch5 payloads against a 36.33 s and a
        # 10 s extraction pass.
        assert abs(break_even_mbps(192_800_000, 36.33) - 42.46) < 0.05
        assert abs(break_even_mbps(192_800_000, 10.0) - 154.24) < 0.05
        assert break_even_mbps(1, 0) == float("inf")
        assert median([3.0, 1.0, 2.0]) == 2.0 and median([4.0, 1.0, 2.0, 3.0]) == 2.5
        assert median([]) != median([])  # empty -> NaN, which is never equal to itself
        print(
            f"self-test OK | board: {board_name()} | rails: {len(power_sensors())} | "
            f"cuda: {torch.cuda.is_available()} | peak RSS {rss_mib():.0f} MiB"
        )
        return
    if dataset is None:
        raise click.UsageError("--dataset is required (or pass --self-test).")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if threads:
        torch.set_num_threads(threads)
    sensors = power_sensors()

    print(f"board       : {board_name()}")
    print(f"device      : {device}   torch {torch.__version__}   threads {torch.get_num_threads()}")
    print(f"power       : {len(sensors)} rail(s)" if sensors else f"power       : no sensor (--power-w {power_w})")

    ds = load_dataset(dataset, split="test")
    lookup = load_clear_sky_lookup(dataset)
    assert lookup, f"empty clear-sky lookup for {dataset}; pass the dataset DIRECTORY, not a file"
    day_map, dropped = filter_lit_days(group_indices_by_day(ds.periods), ds.periods, lookup, DEFAULT_MIN_GHI_CLEAR)

    # Spread the subset across the calendar: day length, and so samples per round,
    # is the thing that varies most, and a run of consecutive days would miss it.
    available = sorted(day_map)
    stride = max(1, len(available) // (days + 1))
    selected = available[::stride][: days + 1]
    print(f"dataset     : {len(available)} lit days ({dropped} night samples dropped), timing {selected[1:]}\n")

    model = load_forecaster_from_checkpoint(checkpoint, device=device, use_hparams=True)
    labeller = make_regime_labeller(regime_label, default_edges(regime_label), DEFAULT_MIN_GHI_CLEAR, elevation_bins)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    rows: list[dict[str, Any]] = []
    global_protos: dict[int, torch.Tensor] | None = None
    client_proto: dict[int, tuple[torch.Tensor, int]] = {}
    # The prototype payload scales with how many of the regimes a day actually
    # populated -- a short winter day fills far fewer than a long summer one.  Size
    # the payload from the fullest day seen, so the reported saving is the smallest
    # one the mechanism delivers rather than a flattering mid-winter figure.
    payload_proto: dict[int, tuple[torch.Tensor, int]] = {}
    payload_day = ""

    for position, day in enumerate(selected):
        indices = day_map[day]
        row: dict[str, Any] = {"day": day, "n_samples": len(indices)}
        with PowerMeter(sensors, power_scale, power_w) as meter, quiet(not progress):
            with timed(row, "t_predict_s", device):
                _preds, _tgts, day_samples = predict_day(model, ds, indices, device, batch_size, day)
            with timed(row, "t_extract_s", device):
                client_proto, _stats = extract_day_prototypes(model, day_samples, device, batch_size, labeller=labeller)
            with timed(row, "t_train_s", device):
                train_day(
                    model,
                    ds,
                    indices,
                    lr=adapt_lr,
                    weight_decay=1e-4,
                    max_grad_norm=1.0,
                    batch_size=batch_size,
                    epochs=adapt_epochs,
                    device=device,
                    day_samples=day_samples,
                    client_name=day,
                    rng_seed=0,
                    global_prototypes=global_protos,
                    proto_lambda=proto_lambda,
                    align_loss="prototypical",
                    align_temperature=align_temperature,
                    classify_fn=labeller,
                )
        row["power_w"] = meter.mean_w
        row["n_regimes"] = len(client_proto)
        row["t_round_s"] = row["t_predict_s"] + row["t_extract_s"] + row["t_train_s"]
        global_protos = {cls: total / n for cls, (total, n) in client_proto.items() if n > 0}

        # The first day pays for lazy CUDA/cuDNN init and a cold page cache, so it is
        # timed like the rest and then dropped rather than silently inflating day one.
        tag = "warm-up (discarded)" if position == 0 else ""
        print(
            f"  {day}  n={row['n_samples']:4d}  regimes={row['n_regimes']:2d}  "
            f"predict {row['t_predict_s']:7.2f}s  extract {row['t_extract_s']:7.2f}s  "
            f"train {row['t_train_s']:8.2f}s  {row['power_w']:6.2f}W  {tag}"
        )
        if position:
            rows.append(row)
            if len(client_proto) > len(payload_proto):
                payload_proto, payload_day = client_proto, day

    # ---- payloads and aggregation, timed once on the same device ---------------
    agg: dict[str, Any] = {}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    up_weights = payload_nbytes(state)
    up_proto = payload_nbytes(payload_proto)

    with timed(agg, "t_agg_weights_s", "cpu"):
        merged = federated_averaging(state, [state] * clients, [1.0] * clients)
    with timed(agg, "t_agg_proto_s", "cpu"):
        per_client, _attn = aggregate_global_prototypes(
            [payload_proto] * clients, attention=True, temperature=align_temperature
        )
    down_weights = payload_nbytes(merged)
    down_proto = payload_nbytes(per_client[0])

    trip_weights, trip_proto = up_weights + down_weights, up_proto + down_proto
    saved = trip_weights - trip_proto
    t_extract = median([r["t_extract_s"] for r in rows])
    t_train = median([r["t_train_s"] for r in rows])
    b_star = break_even_mbps(saved, t_extract)
    mean_power = median([r["power_w"] for r in rows])

    record = {
        "board": board_name(),
        "device": device,
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "clients": clients,
        "payload_sized_from_day": payload_day,
        "payload_regimes": len(payload_proto),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "per_day": rows,
        "payload_mib": {
            "weights_uplink": up_weights / MIB,
            "weights_round_trip": trip_weights / MIB,
            "proto_uplink": up_proto / MIB,
            "proto_round_trip": trip_proto / MIB,
        },
        "aggregation_ms": {k: v * 1e3 for k, v in agg.items()},
        "median_t_extract_s": t_extract,
        "median_t_train_s": t_train,
        "median_power_w": mean_power,
        "bits_saved_per_round_mbit": saved * 8 / 1e6,
        "break_even_mbps": b_star,
        "peak_rss_mib": rss_mib(),
        "peak_accelerator_mib": (torch.cuda.max_memory_allocated() / MIB if device.startswith("cuda") else 0.0),
    }

    print(f"\n  {'':38s}{'full model':>14s}{'prototypes':>14s}")
    print(f"  {'round-trip payload (MiB)':38s}{trip_weights / MIB:14.3f}{trip_proto / MIB:14.3f}")
    print(f"  {'uplink per client (MiB)':38s}{up_weights / MIB:14.3f}{up_proto / MIB:14.3f}")
    print(
        f"  {f'aggregation, {clients} clients (ms)':38s}"
        f"{agg['t_agg_weights_s'] * 1e3:14.3f}{agg['t_agg_proto_s'] * 1e3:14.3f}"
    )
    print(f"  {'prototype extraction (s)':38s}{0.0:14.3f}{t_extract:14.3f}")
    print(f"\n  local training epoch      : {t_train:.2f} s")
    print(f"  saved per round           : {saved * 8 / 1e6:.1f} Mbit")
    print(f"  BREAK-EVEN LINK SPEED     : {b_star:.1f} Mbit/s")
    print(f"  peak RSS / accelerator    : {record['peak_rss_mib']:.0f} / {record['peak_accelerator_mib']:.0f} MiB")
    if mean_power == mean_power:  # not NaN
        print(
            f"  round energy              : {mean_power * median([r['t_round_s'] for r in rows]):.1f} J "
            f"at {mean_power:.2f} W"
        )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        # NaN is not valid JSON, and a board with no power rail (every Pi) produces
        # one -- write null so the file survives jq and any non-Python reader.
        clean = json.loads(json.dumps(record), parse_constant=lambda _c: None)
        out.write_text(json.dumps(clean, indent=2))
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
