"""Board power sampling for the edge benchmarks (Raspberry Pi 5 PMIC, Jetson tegrastats).

    from edge_power import PowerSampler
    with PowerSampler() as ps:            # backend chosen automatically
        idle_w = ps.idle(10.0)            # seconds of idle sampling
        t0 = ps.now(); work(); t1 = ps.now()
        joules = ps.energy(t0, t1)        # integrated over the samples in [t0, t1]
        net = joules - idle_w * (t1 - t0) # energy above idle

Backends
  pmic        Raspberry Pi 5: ``vcgencmd pmic_read_adc``; power = sum over rails of current x voltage for
              every rail reporting both.  This is the PMIC's view of the board rails: it omits the 5 V input
              converter losses and USB peripherals, so it reads below an inline meter on the supply.  Verify
              one run against the meter and report which instrument each number comes from.
  tegrastats  Jetson: ``tegrastats --interval <ms>``, the VDD_IN rail (module input power).  Module telemetry
              excludes carrier-board losses; the same caveat applies.
  none        no sensor: every energy value is NaN.

Thermal control: ``cooldown()`` waits until the hottest thermal zone is below a target temperature (with a minimum
and maximum wait) so every measurement starts from the same thermal state, and ``thermal_state()`` records the
temperature and, on a Pi, the firmware throttling flags; both are stored with each result.

Samples are (monotonic time, watts); energy is the trapezoidal integral between two timestamps.  A phase shorter
than about two sampling intervals cannot be resolved; time many repetitions inside one window instead.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import threading
import time

_PMIC_LINE = re.compile(r"^\s*(\S+)_([AV])\s+(?:current|volt)\(\d+\)=([\d.]+)[AV]\s*$")
_VDD_IN = re.compile(r"VDD_IN (\d+)mW")


def detect_backend() -> str:
    if shutil.which("vcgencmd"):
        try:
            out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2).stdout
            if "current(" in out:
                return "pmic"
        except (OSError, subprocess.SubprocessError):
            pass
    if shutil.which("tegrastats"):
        return "tegrastats"
    return "none"


def pmic_watts() -> float:
    """One PMIC reading: sum of I x V over rails that report both."""
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2).stdout
    amps, volts = {}, {}
    for line in out.splitlines():
        m = _PMIC_LINE.match(line)
        if m:
            (amps if m[2] == "A" else volts)[m[1]] = float(m[3])
    return sum(a * volts[r] for r, a in amps.items() if r in volts)


def chip_temp_c() -> float:
    """Hottest thermal zone in deg C (Pi: cpu-thermal; Jetson: max over CPU/GPU/SoC zones); NaN if none."""
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:  # offline Jetson zones fail on read (EAGAIN); os.read raises instead of returning None
            fd = os.open(z, os.O_RDONLY)
            try:
                v = int(os.read(fd, 32).strip())
            finally:
                os.close(fd)
            if 0 < v < 150_000:  # millidegrees; skip disabled or bogus zones
                temps.append(v / 1000.0)
        except Exception:
            pass
    return max(temps) if temps else float("nan")


def throttled() -> str | None:
    """Pi firmware throttling flags (``vcgencmd get_throttled``): 0x0 means none now and none since boot."""
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return out.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


def thermal_state() -> dict:
    return {"temp_c": chip_temp_c(), "throttled": throttled()}


def cooldown(target_c: float = 50.0, min_s: float = 30.0, max_s: float = 300.0, label: str = "") -> dict:
    """Idle at least *min_s*, then until the chip is at or below *target_c* (at most *max_s*); report the state."""
    t0 = time.monotonic()
    time.sleep(min_s)
    while time.monotonic() - t0 < max_s:
        t = chip_temp_c()
        if t != t or t <= target_c:  # NaN (no sensor) or cool enough
            break
        time.sleep(5)
    state = thermal_state() | {"waited_s": round(time.monotonic() - t0, 1)}
    print(f"  cooldown{' ' + label if label else ''}: {state['temp_c']:.1f} C after {state['waited_s']:.0f} s"
          + (f", throttled={state['throttled']}" if state["throttled"] else ""), flush=True)
    return state


class PowerSampler:
    def __init__(self, backend: str = "auto", interval_s: float = 0.1):
        self.backend = detect_backend() if backend == "auto" else backend
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def now() -> float:
        return time.monotonic()

    def __enter__(self) -> PowerSampler:
        if self.backend == "pmic":
            self._thread = threading.Thread(target=self._pmic_loop, daemon=True)
            self._thread.start()
        elif self.backend == "tegrastats":
            self._proc = subprocess.Popen(["tegrastats", "--interval", str(int(self.interval_s * 1000))],
                                          stdout=subprocess.PIPE, text=True)
            self._thread = threading.Thread(target=self._tegra_loop, daemon=True)
            self._thread.start()
        time.sleep(3 * self.interval_s)  # first samples before any timed work
        return self

    def _pmic_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.monotonic(), pmic_watts()))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_s)

    def _tegra_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = _VDD_IN.search(line)
            if m:
                self.samples.append((time.monotonic(), int(m[1]) / 1000.0))

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        if self.backend != "none":
            self._thread.join(timeout=2.0)

    def energy(self, t0: float, t1: float) -> float:
        """Joules between t0 and t1 (trapezoidal over the samples, clipped to the window)."""
        pts = [(t, w) for t, w in list(self.samples) if t0 <= t <= t1]
        if self.backend == "none" or len(pts) < 2:
            return float("nan")
        e = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))
        span = pts[-1][0] - pts[0][0]
        return e * (t1 - t0) / span if span > 0 else float("nan")  # extend to the window edges

    def mean_w(self, t0: float, t1: float) -> float:
        e = self.energy(t0, t1)
        return e / (t1 - t0) if t1 > t0 else float("nan")

    def idle(self, seconds: float = 10.0) -> float:
        """Mean power over an idle window (nothing else should run)."""
        t0 = self.now()
        time.sleep(seconds)
        return self.mean_w(t0, self.now())


if __name__ == "__main__":
    if sys.argv[1:2] == ["cooldown"]:  # used by run_edge_device.sh between steps
        cooldown(float(sys.argv[2]) if len(sys.argv) > 2 else 50.0, label="between steps")
        sys.exit(0)
    print(f"temperature: {chip_temp_c():.1f} C, throttled: {throttled()}")
    with PowerSampler() as ps:
        print(f"backend: {ps.backend}")
        w = ps.idle(5.0)
        print(f"idle over 5 s: {w:.3f} W from {len(ps.samples)} samples")
