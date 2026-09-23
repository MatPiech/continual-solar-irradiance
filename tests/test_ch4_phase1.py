"""Checks for the Chapter 4 Phase-1 fixes (docs/ch4_conditional_adaptation_plan.md).

One file, assert-based, no fixtures.  Each test guards a claim the plan makes
about the code, not a claim about the science.  Tests that need a finished run
read the verification folder produced by Phase 1 and skip when it is absent, so
the suite stays runnable without a GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cirr.evaluation.campaign import ACTIONS, SOURCE_CHECKPOINT, buffer_bytes
from cirr.models.forecaster import Forecaster
from cirr.physics.clear_sky import DEFAULT_MIN_GHI_CLEAR, filter_lit_days
from cirr.training.ogd import algorithm1_scope, set_trainable
from cirr.utils.provenance import set_determinism, sha256_file

ROOT = Path(__file__).parents[1]
VERIFY = ROOT / "results" / "ch4" / "_verify"
RUN_FILES = ("config.yaml", "metrics_daily.csv", "events.csv", "predictions.csv", "summary.json")


def _load_script(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _model() -> Forecaster:
    return Forecaster(
        model_name="timm-resnet50", pretrained=False, input_channels=3, loss_function="Berhu", lr=3e-4, lr_patience=2
    )


def _run(name: str) -> Path:
    d = VERIFY / name
    if not (d / "summary.json").exists():
        pytest.skip(f"no verification run at {d}; see docs/ch4_conditional_adaptation_plan.md Phase 1")
    return d


# --- F1: one daylight filter, shared with Chapter 5 ------------------------------


def test_f1_chapter4_uses_the_federated_filter():
    """The filter must be the same object both chapters call, not a copy."""
    fb = _load_script("scripts/fl_benchmark.py")
    assert fb.filter_lit_days is filter_lit_days
    assert fb.DEFAULT_MIN_GHI_CLEAR == DEFAULT_MIN_GHI_CLEAR == 0.001

    cam = importlib.import_module("cirr.evaluation.campaign")
    assert cam.filter_lit_days is filter_lit_days


def test_f1_filter_drops_exactly_the_dark_samples():
    lookup = {"20250101_120000": 0.5, "20250101_120100": 0.0, "20250101_120200": 0.0005}
    periods = [{"history": [{"image_name": f"{k}.jpg", "irradiance": 0.1}], "target_irradiance": 0.1} for k in lookup]
    kept, dropped = filter_lit_days({"0101": [0, 1, 2]}, periods, lookup, DEFAULT_MIN_GHI_CLEAR)
    assert kept == {"0101": [0]}
    assert dropped == 2
    assert filter_lit_days({"0101": [0, 1, 2]}, periods, lookup, 0.0) == ({"0101": [0, 1, 2]}, 0)


def test_f1_no_scored_sample_is_dark():
    run = _run("cond_seed0")
    preds = pd.read_csv(run / "predictions.csv")
    assert len(preds) > 0
    assert (preds["ghi_clear"] >= DEFAULT_MIN_GHI_CLEAR).all(), "a dark sample survived the filter"
    assert preds["lit"].all()


# --- F2: one source checkpoint, hashed into every run ----------------------------


def test_f2_source_checkpoint_is_the_chapter5_model():
    assert SOURCE_CHECKPOINT.name == "resnet50_epoch_015.ckpt"


def test_f2_every_run_records_checkpoint_hash_and_commit():
    run = _run("cond_seed0")
    cfg = (run / "config.yaml").read_text()
    assert "checkpoint_sha256:" in cfg and "git_commit:" in cfg
    sha = next(ln.split(": ", 1)[1].strip() for ln in cfg.splitlines() if ln.startswith("checkpoint_sha256:"))
    ckpt = ROOT / SOURCE_CHECKPOINT
    if ckpt.exists():
        assert sha == sha256_file(ckpt)


# --- F3 / F4: the run folder schema and both poolings ----------------------------


def test_f3_run_folder_carries_every_file_and_column():
    run = _run("cond_seed0")
    for name in RUN_FILES:
        assert (run / name).exists(), f"{name} missing from {run}"

    daily = pd.read_csv(run / "metrics_daily.csv")
    assert set(daily.columns) >= {"date", "n_samples", "n_dropped_dark", "mae", "rmse", "wape", "smape", "sp_rmse"}
    assert daily["sp_rmse"].notna().all(), "Smart Persistence must be scored on the model population"

    events = pd.read_csv(run / "events.csv")
    assert set(events.columns) >= {"date", "considered", "queue_flushed", "scope_changed_if_raw", *ACTIONS}
    assert set(events.columns) >= {f"{p}_{a}" for p in ("lit", "dark") for a in ACTIONS}


def test_f4_summary_reports_pooled_and_day_weighted():
    summary = json.loads((_run("cond_seed0") / "summary.json").read_text())
    assert {"pooled", "day_weighted", "pooling_delta"} <= summary.keys()
    for key in ("mae", "rmse", "wape", "smape"):
        assert key in summary["pooled"] and key in summary["day_weighted"]
    # the delta is the size of a choice the published tables made silently
    assert np.isclose(summary["pooling_delta"]["mae"], summary["pooled"]["mae"] - summary["day_weighted"]["mae"])


# --- F5: the trigger and the metric use the same prediction ----------------------


def test_f5_trigger_uses_the_reported_forecast():
    """L is the error of the clamped forecast, as eq:ch4-abs-error defines it."""
    run = _run("cond_seed0")
    preds = pd.read_csv(run / "predictions.csv")
    considered = preds[preds["update_considered"]]
    assert len(considered) > 0

    cfg = (run / "config.yaml").read_text()
    t1 = float(next(ln for ln in cfg.splitlines() if ln.startswith("t1:")).split(": ")[1])
    t2 = float(next(ln for ln in cfg.splitlines() if ln.startswith("t2:")).split(": ")[1])
    mapping = {"none": "none", "head_only": "head", "head_and_encoder": "whole"}

    # the trigger input is the same quantity the metric scores
    assert np.allclose(considered["delayed_abs_error"], considered["absolute_error"], atol=1e-6)
    expected = [mapping[algorithm1_scope(e, t1, t2)] for e in considered["delayed_abs_error"]]
    assert list(considered["adaptation_scope"]) == expected


def test_f5_raw_trigger_sentinel_is_recorded_and_quiet():
    """Disagreement here means the model started predicting negative irradiance."""
    run = _run("cond_seed0")
    preds = pd.read_csv(run / "predictions.csv")
    considered = preds[preds["update_considered"]]
    assert (considered["scope_if_raw"] != "not_available").all()
    assert (preds["prediction"] >= 0).all()

    events = pd.read_csv(run / "events.csv")
    assert "scope_changed_if_raw" in events.columns
    assert events["scope_changed_if_raw"].sum() == 0, (
        "the clamp bound on some sample: the trigger saw a smaller error than the model made"
    )


# --- F6 / F12: what each action and each method actually trains -------------------


def _trainable(model: Forecaster, scope: str) -> int:
    set_trainable(model, scope)  # type: ignore[arg-type]
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_f6_the_policy_has_exactly_two_update_subspaces():
    """Section 4.3 defines head-only and full-model; the code must offer those."""
    model = _model()
    total = sum(p.numel() for p in model.parameters())
    head = _trainable(model, "head_only")
    full = _trainable(model, "head_and_encoder")

    print(f"  head_only={head:,}  head_and_encoder={full:,}  total={total:,}")
    assert head < full, (head, full)
    assert full == total, "the high-error action must be S = R^P"
    assert head == sum(p.numel() for p in model.network_head.parameters())


def test_f6_the_published_high_error_action_trained_only_the_head():
    """The defect F6 corrects, guarded so it cannot quietly come back.

    The old code unfroze ``encoder_children[-2:]``.  On a timm backbone built
    with ``num_classes=0`` those are ``SelectAdaptivePool2d`` and ``Identity``,
    which hold no parameters, so the full-model action trained only the head.
    """
    model = _model()
    children = list(model.image_encoder.children())
    assert sum(p.numel() for c in children[-2:] for p in c.parameters()) == 0
    assert _trainable(model, "head_and_encoder") > _trainable(model, "head_only")


def test_f12_head_scope_is_the_same_parameter_set_for_every_gradient_method():
    """Sliding-window and the conditional policy must not compare across scopes."""
    model = _model()
    set_trainable(model, "head_only")
    policy_head = {n for n, p in model.named_parameters() if p.requires_grad}
    assert policy_head == {n for n, _ in model.network_head.named_parameters(prefix="network_head")}
    print(f"  head-scope parameters: {len(policy_head)} tensors")


# --- F8: the action mix, split by daylight --------------------------------------


def test_f8_events_reconcile_with_the_per_sample_rows():
    run = _run("cond_seed0")
    events = pd.read_csv(run / "events.csv")
    preds = pd.read_csv(run / "predictions.csv")
    considered = preds[preds["update_considered"]]

    assert events["considered"].sum() == len(considered)
    for action in ACTIONS:
        assert events[action].sum() == int((considered["adaptation_scope"] == action).sum())
        assert (events[action] == events[f"lit_{action}"] + events[f"dark_{action}"]).all()


# --- F10: what the delay queue costs --------------------------------------------


def test_f10_buffer_bytes_scale_with_the_horizon():
    class _Stub:
        def __getitem__(self, _i):
            return (torch.zeros(3, 128, 128), torch.zeros(4), torch.zeros(1), torch.zeros(1), "x.jpg")

    one, fifteen = buffer_bytes(_Stub(), 1), buffer_bytes(_Stub(), 15)
    assert fifteen == 15 * one
    # the queue holds images, which is why a head update recomputes features
    assert one >= 3 * 128 * 128 * 4


def test_f10_summary_reports_buffer_and_memory():
    summary = json.loads((_run("cond_seed0") / "summary.json").read_text())
    assert summary["buffer_bytes"] > 0
    assert summary["peak_rss_mb"] > 0


# --- F11: determinism ------------------------------------------------------------


def test_f11_set_determinism_pins_every_generator():
    set_determinism(3)
    a = (torch.rand(4).tolist(), np.random.rand(4).tolist())
    set_determinism(3)
    b = (torch.rand(4).tolist(), np.random.rand(4).tolist())
    assert a == b
    set_determinism(4)
    assert (torch.rand(4).tolist(), np.random.rand(4).tolist()) != a


# --- population regression against the published run -----------------------------


def test_published_karpicko_run_scored_the_unfiltered_population():
    """Guards the finding that motivated F1; fails if the legacy artifact changes."""
    legacy = ROOT / "results/karpicko/karpicko_paper_alg1_ogd_results.txt"
    if not legacy.exists():
        pytest.skip("published artifact not present")
    text = legacy.read_text()
    assert "Samples: 226114" in text, "the published run scored every sample, night included"
    assert "checkpoints/epoch_010.ckpt" in text, "the published run used the pre-Phase-1 checkpoint"


def test_f6_partial_encoder_ablation_is_layer3_and_layer4():
    """--encoder-blocks 2 is what the original code meant: the last two parameterised groups."""
    model = _model()
    set_trainable(model, "head_and_encoder", encoder_blocks=2)
    trained = {n.split(".")[1] for n, p in model.image_encoder.named_parameters(prefix="e") if p.requires_grad}
    assert trained == {"layer3", "layer4"}, trained
    partial = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert _trainable(model, "head_only") < partial < _trainable(model, "head_and_encoder")
