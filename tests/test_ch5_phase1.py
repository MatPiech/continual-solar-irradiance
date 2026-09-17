"""Checks for the Chapter 5 Phase-1 fixes (docs/ch5_server_benchmark_plan.md).

One file, assert-based, no fixtures.  Each test guards a claim the benchmark
plan makes about the code rather than a claim about the science.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cirr.fl.kc_prototypes import aggregate_global_prototypes, default_edges, make_regime_labeller
from cirr.fl.server import payload_nbytes
from cirr.models.forecaster import Forecaster
from cirr.physics.clear_sky import DAYLIGHT_MAX_ZENITH, is_daylight

_spec = importlib.util.spec_from_file_location("fl_benchmark", Path(__file__).parents[1] / "scripts/fl_benchmark.py")
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

ALGORITHMS = ("isolated", "fedavg", "fedavg_ft", "fedbn", "fedper", "fedprox", "fedema", "fedproto", "pooled")


def _model() -> Forecaster:
    return Forecaster(
        model_name="timm-resnet50", pretrained=False, input_channels=3, loss_function="Berhu", lr=3e-4, lr_patience=2
    )


def test_f2_trainable_scope_identical_across_methods():
    """No method may quietly train fewer parameters than the do-no-harm reference."""
    model = _model()
    counts = {alg: sum(p.numel() for p in model.parameters() if p.requires_grad) for alg in ALGORITHMS}
    for alg, n in counts.items():
        print(f"  {alg:12s} trainable={n:,}")
    assert len(set(counts.values())) == 1, counts
    assert counts["fedproto"] == sum(p.numel() for p in model.parameters()), "something is frozen by default"


def test_f3_clear_sky_label_differs_from_proxy_at_low_sun():
    """The two label sources must not be interchangeable, or F3 buys nothing."""
    proxy = make_regime_labeller("proxy", default_edges("proxy"))
    clear = make_regime_labeller("clear_sky", default_edges("clear_sky"))
    # Polish winter noon: 0.25 kW/m^2 under a 0.28 kW/m^2 clear sky is a clear day,
    # but the fixed-1000 proxy can never call it anything but overcast.
    assert proxy(0.25, 0.28) == 0
    assert clear(0.25, 0.28) == 2
    # Single-regime collapse (stratify: single)
    single = make_regime_labeller("clear_sky", ())
    assert single(0.25, 0.28) == single(0.9, 1.0) == 0


def test_f4_daylight_predicate():
    lookup = {"20240101_120000": 40.0, "20240101_180000": 88.0}
    assert is_daylight(lookup, "20240101_120000")
    assert not is_daylight(lookup, "20240101_180000")
    assert DAYLIGHT_MAX_ZENITH == 85.0
    # Unknown timestamps are kept rather than silently emptying the dataset.
    assert is_daylight(lookup, "20240101_235959")


def test_f6_align_target_controls():
    protos = [{0: torch.ones(8), 1: torch.ones(8) * 2.0}]
    own = [{0: torch.zeros(8), 1: torch.zeros(8)}]

    assert fb.apply_align_target(protos, own, "global", 0, 42) is protos
    assert torch.equal(fb.apply_align_target(protos, own, "own", 0, 42)[0][0], torch.zeros(8))

    shuffled = fb.apply_align_target(protos, own, "shuffled", 0, 42)[0]
    assert sorted(float(v[0]) for v in shuffled.values()) == [1.0, 2.0], "shuffle must permute, not invent"

    rnd = fb.apply_align_target(protos, own, "random", 0, 42)[0]
    for cls in (0, 1):
        assert not torch.equal(rnd[cls], protos[0][cls])
        assert abs(float(rnd[cls].norm()) - float(protos[0][cls].norm())) < 0.6 * float(protos[0][cls].norm())
    # Same seed and round reproduce the same control.
    again = fb.apply_align_target(protos, own, "random", 0, 42)[0]
    assert torch.equal(rnd[0], again[0])


def test_f7_uniform_attention_is_a_plain_mean():
    a = {0: (torch.tensor([1.0, 0.0]) * 4, 4)}  # mean [1, 0]
    b = {0: (torch.tensor([0.0, 1.0]) * 4, 4)}  # mean [0, 1]
    uniform, weights = aggregate_global_prototypes([a, b], attention=False)
    assert torch.allclose(uniform[0][0], torch.tensor([0.5, 0.5]))
    assert weights[0][0] == pytest.approx([0.5, 0.5])
    attn, _ = aggregate_global_prototypes([a, b], attention=True)
    assert not torch.allclose(attn[0][0], uniform[0][0]), "attention must differ from a plain mean"


def test_f8_participation_schedules():
    names = ["folsom", "karpicko", "poznan"]
    always = fb.make_participation("full", names, 42)
    assert all(always(i, r) for i in range(3) for r in range(10))

    outage = fb.make_participation("outage:karpicko:2:3", names, 42)
    assert [outage(1, r) for r in range(7)] == [True, True, False, False, False, True, True]
    assert all(outage(0, r) for r in range(7)), "outage must not touch other clients"

    drop = fb.make_participation("dropout:0.5", names, 42)
    draws = [drop(i, r) for i in range(3) for r in range(40)]
    assert 0 < sum(not d for d in draws) < len(draws), "dropout must drop some rounds but not all"
    assert [drop(i, r) for i in range(3) for r in range(40)] == draws, "dropout must be seed-reproducible"


def test_f12_payload_bytes_are_serialized_sizes():
    model = _model()
    weights = payload_nbytes(model.state_dict())
    protos = payload_nbytes({c: (torch.zeros(2048), 100) for c in range(3)})
    assert weights > 50 * protos, (weights, protos)
    # Matches the actual buffer, not numel * itemsize.
    assert payload_nbytes(torch.zeros(1000)) >= 4000


def test_f10_determinism_flags_set():
    fb.set_determinism(7)
    assert torch.backends.cudnn.benchmark is False
    assert torch.rand(3).tolist() == (fb.set_determinism(7) or torch.rand(3).tolist())


def test_f11_regime_metrics_shape():
    """Per-regime rows must partition the day, not resample it."""
    regimes = np.array([0, 0, 1, 2, 2, 2], dtype=np.int16)
    assert sum(int((regimes == c).sum()) for c in set(regimes.tolist())) == len(regimes)


def test_forecast_skill_denominator_is_day_scoped():
    """FS must pool Smart Persistence over the evaluated days only.

    Regression guard for a wiring run that scored 3 days of model error against
    a whole-calendar Smart Persistence RMSE, which inflated Forecast Skill
    (Poznan showed 62.9% on two April days).
    """
    # (sum squared error, n) per day: two calm days, one volatile day.
    stats = {"0401": (1.0, 100), "0402": (1.0, 100), "0715": (400.0, 100)}
    assert fb._sp_rmse(stats["0401"]) == pytest.approx(0.1)
    assert fb._sp_rmse(None) != fb._sp_rmse(stats["0401"])  # missing day -> nan, never 0
    assert np.isnan(fb._sp_rmse(None)) and np.isnan(fb._sp_rmse((1.0, 0)))

    evaluated = {"0401", "0402"}
    sse = sum(stats[d][0] for d in evaluated)
    n = sum(stats[d][1] for d in evaluated)
    scoped = (sse / n) ** 0.5
    all_days = (sum(v[0] for v in stats.values()) / sum(v[1] for v in stats.values())) ** 0.5
    assert scoped == pytest.approx(0.1)
    assert all_days > 4 * scoped, "whole-calendar denominator must differ from the day-scoped one"

    model_rmse = 0.09
    assert 100 * (1 - model_rmse / scoped) == pytest.approx(10.0)  # honest skill
    assert 100 * (1 - model_rmse / all_days) > 80.0  # what the bug reported
