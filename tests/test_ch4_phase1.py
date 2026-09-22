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

import click
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
    assert set(events.columns) >= {"date", "considered", "queue_flushed", *ACTIONS}
    assert set(events.columns) >= {f"{p}_{a}" for p in ("lit", "dark") for a in ACTIONS}


def test_f4_summary_reports_pooled_and_day_weighted():
    summary = json.loads((_run("cond_seed0") / "summary.json").read_text())
    assert {"pooled", "day_weighted", "pooling_delta"} <= summary.keys()
    for key in ("mae", "rmse", "wape", "smape"):
        assert key in summary["pooled"] and key in summary["day_weighted"]
    # the delta is the size of a choice the published tables made silently
    assert np.isclose(summary["pooling_delta"]["mae"], summary["pooled"]["mae"] - summary["day_weighted"]["mae"])


# --- F5: the trigger sees a different prediction than the metric -----------------


def test_f5_clamped_scope_is_recorded_for_every_considered_sample():
    preds = pd.read_csv(_run("cond_seed0") / "predictions.csv")
    considered = preds[preds["update_considered"]]
    assert len(considered) > 0
    assert (considered["scope_if_clamped"] != "not_available").all()

    cfg = (_run("cond_seed0") / "config.yaml").read_text()
    t1 = float(next(ln for ln in cfg.splitlines() if ln.startswith("t1:")).split(": ")[1])
    t2 = float(next(ln for ln in cfg.splitlines() if ln.startswith("t2:")).split(": ")[1])
    mapping = {"none": "none", "head_only": "head", "head_and_encoder": "whole"}
    expected = [mapping[algorithm1_scope(e, t1, t2)] for e in considered["absolute_error"]]
    assert list(considered["scope_if_clamped"]) == expected


# --- F6 / F12: what each action and each method actually trains -------------------


def _trainable(model: Forecaster, scope: str, encoder_unfreeze: str = "none") -> int:
    set_trainable(model, scope, encoder_unfreeze=encoder_unfreeze)  # type: ignore[arg-type]
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_f6_the_three_encoder_scopes_are_actually_distinct():
    """Each setting must reach further into the encoder than the last."""
    model = _model()
    total = sum(p.numel() for p in model.parameters())
    head = _trainable(model, "head_only")
    none = _trainable(model, "head_and_encoder", "none")
    last_blocks = _trainable(model, "head_and_encoder", "last_blocks")
    full = _trainable(model, "head_and_encoder", "full")

    print(f"  head_only={head:,}  none={none:,}  last_blocks={last_blocks:,}  full={full:,}  total={total:,}")
    assert head == none, "encoder_unfreeze='none' must train exactly the head"
    assert none < last_blocks < full == total, (none, last_blocks, full, total)


def test_f6_the_published_high_error_action_trained_only_the_head():
    """The defect F6 corrects: the 'full-model' action never touched the encoder.

    ``encoder_children[-2:]`` on a timm backbone built with ``num_classes=0`` is
    ``[SelectAdaptivePool2d, Identity]``, which hold no parameters.  Guarding it
    here so the finding cannot quietly regress once the chapter is rewritten.
    """
    model = _model()
    children = list(model.image_encoder.children())
    assert sum(p.numel() for c in children[-2:] for p in c.parameters()) == 0
    assert _trainable(model, "head_and_encoder", "none") == _trainable(model, "head_only")


def test_f12_head_scope_is_the_same_parameter_set_for_every_gradient_method():
    """Sliding-window and the conditional policy must not compare across scopes."""
    model = _model()
    set_trainable(model, "head_only")
    policy_head = {n for n, p in model.named_parameters() if p.requires_grad}
    assert policy_head == {n for n, _ in model.network_head.named_parameters(prefix="network_head")}
    print(f"  head-scope parameters: {len(policy_head)} tensors")


# --- F7: the trigger controls --------------------------------------------------


def test_f7_controls_select_the_intended_actions():
    mod = _load_script("scripts/related_works/evaluate_lagged_ogd.py")
    errs = [0.0, 0.005, 0.05, 0.5]

    assert [mod.make_scope_fn("never", 0.01, 0.1, 0, None, None)(e) for e in errs] == ["none"] * 4
    assert "none" not in [mod.make_scope_fn("always", 0.01, 0.1, 0, None, None)(e) for e in errs]
    assert [mod.make_scope_fn("conditional", 0.01, 0.1, 0, None, None)(e) for e in errs] == [
        "none",
        "none",
        "head_only",
        "head_and_encoder",
    ]
    with pytest.raises(click.UsageError):
        mod.make_scope_fn("random", 0.01, 0.1, 0, None, None)


def test_f7_random_trigger_matches_its_target_rate_and_is_seeded():
    mod = _load_script("scripts/related_works/evaluate_lagged_ogd.py")
    # one closure, many calls: the generator is per-run, not per-sample
    choose = mod.make_scope_fn("random", 0.01, 0.1, 7, 0.4, 0.25)
    draws = [choose(0.02) for _ in range(20000)]
    fired = [d for d in draws if d != "none"]
    assert abs(len(fired) / len(draws) - 0.4) < 0.02
    assert abs(sum(d == "head_and_encoder" for d in fired) / len(fired) - 0.25) < 0.02

    repeat = mod.make_scope_fn("random", 0.01, 0.1, 7, 0.4, 0.25)
    assert [repeat(0.02) for _ in range(1000)] == draws[:1000], "the random trigger is not seed-reproducible"
    other = mod.make_scope_fn("random", 0.01, 0.1, 8, 0.4, 0.25)
    assert [other(0.02) for _ in range(1000)] != draws[:1000], "a different seed gave the same stream"


def test_f7_never_performs_no_update():
    run = VERIFY / "never_resnet50_epoch_015"
    if not (run / "summary.json").exists():
        pytest.skip("no frozen control run")
    summary = json.loads((run / "summary.json").read_text())
    assert summary["actions"]["head"] == summary["actions"]["whole"] == 0
    assert summary["update_stats"]["head"] == summary["update_stats"]["whole"] == 0


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
