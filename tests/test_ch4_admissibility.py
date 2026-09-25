"""Checks for the Chapter 4 admissibility code (cirr/evaluation/admissibility.py).

Synthetic inputs only, so the suite runs without data or a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cirr.evaluation.admissibility import KINDS, RETIRED_KINDS, CorruptedStream, build_schedule, corrupt, nominal_split


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
    assert set(a.kind) <= set(KINDS) | set(RETIRED_KINDS)  # the §5 stream keeps its droplets episodes
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
    bank = torch.randn(5, 50, 128) * 0.1  # 5 cells, 50 nominal frames each
    gate = LocalGate(bank, torch.ones(5, dtype=torch.bool), scale=torch.ones(5))
    frames = torch.randn(3, 5, 128) * 0.1
    frames[1, 3] += 5.0  # one soiled patch in frame 1
    score, where = gate.score(frames, "cpu")
    assert score.argmax() == 1 and where[1] == 3


def _disc_cells(n: int = 16) -> torch.Tensor:
    """Valid cells of a synthetic circular field of view, as valid_cells() would give at 128 x 128."""
    from cirr.evaluation.admissibility import valid_cells

    y, x = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
    inside = ((y - 63.5) ** 2 + (x - 63.5) ** 2 <= 63.5**2).float()
    return valid_cells(inside.expand(3, 128, 128))


def test_edge_cells_are_the_ring_touching_the_mask_border():
    from cirr.evaluation.admissibility import edge_cells

    cells = _disc_cells()
    edge = edge_cells(cells)
    rc = torch.nonzero(cells)
    assert edge.shape == (len(rc),)
    # the centre cell is interior, the topmost valid cell is edge
    assert not edge[(rc == torch.tensor([7, 7])).all(1)].item()
    assert edge[rc[:, 0].argmin()].item()
    # a full square has edge cells only on its border
    square = torch.ones(4, 4, dtype=torch.bool)
    assert edge_cells(square).sum() == 12
    # a flat cell list has no geometry
    assert not edge_cells(torch.ones(5, dtype=torch.bool)).any()


def test_local_gate_split_scores_each_region_separately():
    from cirr.evaluation.admissibility import LocalGate

    cells = _disc_cells()
    n = int(cells.sum())
    torch.manual_seed(0)
    gate = LocalGate(torch.randn(n, 20, 128) * 0.1, cells, scale=torch.ones(n))
    z = torch.rand(4, n) * 0.1
    edge_idx, interior_idx = torch.nonzero(gate.edge).flatten(), torch.nonzero(~gate.edge).flatten()
    z[0, edge_idx[0]] = 5.0  # an edge event in frame 0
    z[1, interior_idx[0]] = 3.0  # an interior event in frame 1
    s = gate.split(z)
    assert s["all"][0] == 5.0 and s["all_cell"][0] == edge_idx[0]
    assert s["edge"][0] == 5.0 and s["interior"][0] < 1.0  # the interior score ignores the rim
    assert s["interior"][1] == 3.0 and s["interior_cell"][1] == interior_idx[0]
    assert s["edge"][1] < 1.0
    # the §5 score is what LocalGate.score gives
    frames = torch.randn(2, n, 128) * 0.1
    sc, where = gate.score(frames, "cpu")
    ss = gate.split(gate.normalised(frames, "cpu"))
    assert np.allclose(sc, ss["all"]) and np.array_equal(where, ss["all_cell"])


def test_local_gate_thresholds_survive_save_and_load(tmp_path):
    from cirr.evaluation.admissibility import LocalGate

    cells = _disc_cells()
    n = int(cells.sum())
    gate = LocalGate(torch.zeros(n, 3, 128), cells, scale=torch.ones(n), threshold=1.5, alert_threshold=1.2)
    gate.thresholds = {"all": 1.5, "interior": 1.1, "edge": 1.7, "interior_alert": 1.0}
    gate.save(tmp_path / "g.npz", note="test")
    back = LocalGate.load(tmp_path / "g.npz")
    assert back.thresholds == gate.thresholds and back.threshold == 1.5
    assert torch.equal(back.edge, gate.edge)


def test_edge_faults_sit_at_the_rim_and_interior_faults_do_not():
    from cirr.evaluation.admissibility import _MEAN, _STD, corrupt

    y, x = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
    r = ((y - 63.5) ** 2 + (x - 63.5) ** 2).sqrt() / 63.5
    clean = ((torch.full((3, 128, 128), 0.6) - _MEAN) / _STD) * (r <= 1).float()
    for kind, rim in (("occlusion_edge", True), ("occlusion", False), ("soiling_edge", True), ("soiling", False)):
        changed = ((corrupt(clean, kind, 0, "f") - clean).abs().sum(0) > 1e-3) & (r <= 1)
        centre = float(r[changed].mean())
        assert (centre > 0.6) if rim else (centre < 0.6), (kind, centre)
        assert changed[r > 1].sum() == 0  # nothing outside the field of view


def test_monitor_faults_render_inside_the_field_of_view_and_differ_as_designed():
    from cirr.evaluation.admissibility import _MEAN, _STD, corrupt

    y, x = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
    r = ((y - 63.5) ** 2 + (x - 63.5) ** 2).sqrt() / 63.5
    clean = ((torch.full((3, 128, 128), 0.6) - _MEAN) / _STD) * (r <= 1).float()
    changed = {}
    for kind in ("banding", "insect", "wet_dome"):  # wet_dome retired but still renderable
        a, b = corrupt(clean, kind, 3, "20160101_120000"), corrupt(clean, kind, 3, "20160101_120100")
        diff = (a - clean).abs().sum(0) > 1e-3
        assert diff[r > 1].sum() == 0, kind  # nothing outside the field of view
        changed[kind] = float(diff[r <= 1].float().mean())
        assert torch.equal(a, corrupt(clean, kind, 3, "20160101_120000"))  # deterministic per frame
        assert not torch.equal(a, b)  # every kind changes between frames
    assert changed["insect"] < 0.01 < changed["wet_dome"]  # an insect is a few pixels, a wet dome is everywhere
    assert changed["banding"] > 0.3  # banding spans the frame


def test_monitor_schedule_orders_kinds_and_never_overlaps():
    from cirr.evaluation.admissibility import build_monitor_schedule

    days = pd.date_range("2016-01-01", periods=40, freq="D")
    ts = [(d + pd.Timedelta(minutes=m)).strftime("%Y%m%d_%H%M%S") for d in days for m in range(420, 1080)]
    s = build_monitor_schedule(ts, seed=1)
    assert s.timestamp.is_unique
    assert set(s.kind) == {"soiling", "occlusion", "banding", "insect"}
    assert s[s.kind == "soiling"].episode.nunique() == 4


def test_monitor_statistics_are_median_and_core_max():
    from cirr.evaluation.admissibility import LocalGate, cell_depth

    cells = _disc_cells()
    n = int(cells.sum())
    gate = LocalGate(torch.zeros(n, 2, 128), cells, scale=torch.ones(n))
    depth = cell_depth(cells)
    assert int((depth >= 3).sum()) == int(gate.core.sum()) and depth.min() == 1
    z = torch.full((2, n), 0.5)
    z[0, torch.nonzero(~gate.core).flatten()[0]] = 9.0  # a rim spike: not core, barely moves the median
    z[1] = 2.0  # frame-wide departure
    m = gate.monitor(z)
    assert m["core"][0] == 0.5 and m["global"][0] == 0.5
    assert m["global"][1] == 2.0 and m["core"][1] == 2.0


def test_occluders_vary_in_colour_and_shape_across_episodes():
    from cirr.evaluation.admissibility import _MEAN, _STD, corrupt

    y, x = torch.meshgrid(torch.arange(128.0), torch.arange(128.0), indexing="ij")
    r = ((y - 63.5) ** 2 + (x - 63.5) ** 2).sqrt() / 63.5
    clean = ((torch.full((3, 128, 128), 0.6) - _MEAN) / _STD) * (r <= 1).float()
    colours, areas = set(), set()
    for ep in range(12):
        out = corrupt(clean, "occlusion", ep, "f")
        changed = ((out - clean).abs().sum(0) > 0.05) & (r <= 1)
        pix = (out * _STD + _MEAN).clamp(0, 1)
        colours.add(tuple((pix[:, changed].mean(1) * 4).round().tolist()))  # coarse colour
        areas.add(int(changed.sum() // 200))  # coarse area
    assert len(colours) >= 3 and len(areas) >= 3


def _synthetic_year():
    days = pd.date_range("2016-01-01", periods=120, freq="D")
    return [(d + pd.Timedelta(minutes=m)).strftime("%Y%m%d_%H%M%S") for d in days for m in range(420, 1080)]


def test_dose_schedule_hits_its_budget_without_overlap():
    from cirr.evaluation.admissibility import DOSE_MIX, build_dose_schedule

    ts = _synthetic_year()
    for rate in (1.0, 5.0, 15.0):
        s = build_dose_schedule(ts, rate, seed=int(rate))
        share = 100 * len(s) / len(ts)
        # whole-day soiling blocks cannot be split: allow one 3-day block of overshoot
        assert rate * 0.8 <= share <= max(rate * 1.3, rate + 100 * 3 * 660 / len(ts)), (rate, share)
        assert s.timestamp.is_unique
        assert set(s.kind) <= {"soiling", *[k for k, *_ in DOSE_MIX]}
        soil = (s.kind == "soiling").sum()
        assert 0.3 <= soil / len(s) <= 0.7  # about half persistent, half short-term


def test_continuous_schedule_covers_exactly_the_window():
    from cirr.evaluation.admissibility import build_continuous_schedule

    ts = _synthetic_year()
    s = build_continuous_schedule(ts, "soiling", "20160201", "20160210")
    assert set(s.timestamp.str[:8]) == {f"201602{d:02d}" for d in range(1, 11)}
    assert (s.kind == "soiling").all() and (s.episode == 0).all()
    assert len(s) == 10 * 660
