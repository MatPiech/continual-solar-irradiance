"""Checks for the Chapter 4 admissibility code (cirr/evaluation/admissibility.py).

Synthetic inputs only, so the suite runs without data or a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cirr.evaluation.admissibility import KINDS, CorruptedStream, build_schedule, corrupt, nominal_split


def _frame(seed: int = 0) -> torch.Tensor:
    """A normalised 3x64x64 frame with a circular field of view, as the dataset returns it."""
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(3, 64, 64, generator=g) * 0.5
    y, x = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    return img * (((y - 31.5) ** 2 + (x - 31.5) ** 2) <= 32**2).float()


@pytest.mark.parametrize("kind", KINDS)
def test_faults_are_reproducible_and_stay_inside_the_field_of_view(kind):
    img = _frame()
    a, b = corrupt(img, kind, 3, "20160415_114700"), corrupt(img, kind, 3, "20160415_114700")
    assert torch.equal(a, b), "a fault must be a pure function of (kind, episode, frame)"
    assert not torch.equal(a, img), "the fault changed nothing"
    outside = img.abs().sum(0) == 0
    assert (a[:, outside] == 0).all(), "the fault leaked outside the circular crop"


def test_episode_layout_is_fixed_but_droplets_move_between_frames():
    img = _frame()
    assert torch.equal(corrupt(img, "soiling", 1, "20160101_100000"), corrupt(img, "soiling", 1, "20160101_100100"))
    assert not torch.equal(corrupt(img, "droplets", 1, "20160101_100000"), corrupt(img, "droplets", 1, "20160101_100100"))


def test_schedule_is_deterministic_and_assigns_one_fault_per_frame():
    stamps = [
        f"2016{m:02d}{d:02d}_{h:02d}{mi:02d}00"
        for m in (1, 2)
        for d in range(1, 29)
        for h in (10, 11, 12)
        for mi in range(0, 60, 5)
    ]
    a, b = build_schedule(stamps, 0), build_schedule(stamps, 0)
    pd.testing.assert_frame_equal(a, b)
    assert a.timestamp.is_unique
    assert set(a.kind) <= set(KINDS)
    assert not build_schedule(stamps, 1).equals(a)


def test_corrupted_stream_changes_images_but_never_labels():
    class _Base:
        periods = ["p"]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return _frame(i), torch.ones(4), torch.tensor(0.5), torch.tensor(0.3 + i), f"2016010{i + 1}_100000"

    sched = pd.DataFrame({"timestamp": ["20160101_100000"], "kind": ["occlusion"], "episode": [0]})
    stream = CorruptedStream(_Base(), sched)
    faulted, clean = stream[0], stream[1]
    assert not torch.equal(faulted[0], _Base()[0][0])
    assert torch.equal(clean[0], _Base()[1][0])
    assert float(faulted[3]) == pytest.approx(0.3), "labels must stay true: that is what makes a faulty frame harmful"
    assert stream.periods == ["p"], "everything but __getitem__ is delegated"


def test_nominal_split_keeps_to_the_fit_years_and_splits_whole_days():
    periods = [
        {"history": [{"image_name": f"{y}01{d:02d}_{h:02d}0000.jpg"}]}
        for y in (2014, 2015, 2016)
        for d in range(1, 21)
        for h in range(8, 16)
    ]
    fit, cal = nominal_split(periods, {"2014", "2015"}, 1, 0.2, 0)
    years = {periods[i]["history"][0]["image_name"][:4] for i in fit + cal}
    assert years == {"2014", "2015"}, "an evaluation-year frame reached the detector"
    day = lambda i: periods[i]["history"][0]["image_name"][:8]  # noqa: E731
    assert not {day(i) for i in fit} & {day(i) for i in cal}, "a day appears in both halves"
    assert np.isclose(len({day(i) for i in cal}) / 40, 0.2)


def test_alert_statistic_is_causal_per_day_and_keeps_input_order():
    from cirr.evaluation.admissibility import alert_statistic

    ts = pd.Series([f"20160101_10{m:02d}00" for m in range(0, 60, 5)] + ["20160102_100000"])
    scores = np.arange(len(ts), dtype=float)
    base = alert_statistic(ts, scores)
    # a late spike must not change any earlier frame's statistic
    spiked = scores.copy()
    spiked[-2] = 1e6
    assert np.array_equal(alert_statistic(ts, spiked)[:-2], base[:-2])
    # window (t - 30 min, t]: the 10:35 frame sees 10:10..10:35 -> median of 2..7 = 4.5
    assert base[7] == 4.5
    # a new day starts a new window
    assert base[-1] == scores[-1]
    # shuffling the input only shuffles the output
    order = np.random.default_rng(0).permutation(len(ts))
    assert np.array_equal(alert_statistic(ts[order].reset_index(drop=True), scores[order]), base[order])


def test_local_gate_blames_the_single_abnormal_cell():
    from cirr.evaluation.admissibility import LocalGate

    torch.manual_seed(0)
    bank = torch.randn(5, 50, 128) * 0.1                      # 5 cells, 50 nominal frames each
    gate = LocalGate(bank, torch.ones(5, dtype=torch.bool), scale=torch.ones(5))
    frames = torch.randn(3, 5, 128) * 0.1
    frames[1, 3] += 5.0                                       # one soiled patch in frame 1
    score, where = gate.score(frames, "cpu")
    assert score.argmax() == 1 and where[1] == 3
