"""Clear-sky GHI lookup utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_clear_sky_lookup(dataset_path: Path) -> dict[str, float]:
    """Load clear-sky irradiance indexed by ``YYYYMMDD_HHMMSS`` timestamp.

    Returns values scaled to ``kW/m^2`` to match dataset target scaling.
    """
    clear_sky_files = sorted(dataset_path.glob("*clear_sky.csv"))
    if not clear_sky_files:
        return {}

    clear_sky_path = clear_sky_files[0]
    frame = pd.read_csv(clear_sky_path)
    if "datetime" not in frame.columns or "ghi_clear" not in frame.columns:
        raise ValueError(f"Invalid clear-sky file format: {clear_sky_path}")

    dt = pd.to_datetime(frame["datetime"], errors="coerce")
    ghi_kw = frame["ghi_clear"].astype(float) / 1000.0
    valid = ~dt.isna() & ghi_kw.notna()

    keys = dt.loc[valid].dt.strftime("%Y%m%d_%H%M%S")
    values = ghi_kw.loc[valid].astype(float)
    return dict(zip(keys.tolist(), values.tolist(), strict=False))


def resolve_clear_sky_value(clear_sky_lookup: dict[str, float], timestamp: str) -> float | None:
    """Resolve clear-sky value for a sample timestamp.

    Image timestamps are second-level; clear-sky tables are often 1-minute.
    Tries exact key first, then falls back to ``...SS=00`` for the same minute.
    """
    value = clear_sky_lookup.get(timestamp)
    if value is not None:
        return value

    if len(timestamp) >= 15:
        minute_key = f"{timestamp[:-2]}00"
        return clear_sky_lookup.get(minute_key)

    return None


# Minimum clear-sky GHI (kW/m^2) for a sample to be scored at all.
#
# Below this the clear-sky index is meaningless: ``compute_csi`` clamps the
# denominator at 1e-3 kW/m^2, so a night frame with ``ghi_clear == 0`` yields
# CSI = target/0.001, and any sensor noise above 0.85 W/m^2 was being labelled
# "clear".  That put ~40 % night frames into the clear prototype at Karpicko and
# Poznan while leaving daylight-only Folsom untouched.
#
# The floor is set to the clamp itself (1 W/m^2 = 1e-3 kW/m^2): every sample that
# survives it has ghi_clear >= the clamp, so the clamp can never bind on an
# eligible sample and CSI is always a true ratio rather than a division by eps.
DEFAULT_MIN_GHI_CLEAR = 0.001


def filter_lit_days(
    day_map: dict[str, list[int]],
    periods: list[dict],
    lookup: dict[str, float],
    min_ghi_clear: float,
) -> tuple[dict[str, list[int]], int]:
    """Drop samples whose clear-sky GHI is below *min_ghi_clear* (kW/m^2).

    Applied once, where the day index map is built, so local training, evaluation,
    prototype extraction and the Smart-Persistence denominator all score exactly the
    same population — there is no second place for the filter to disagree with itself.
    Chapters 4 and 5 both call this function, so their populations cannot drift apart.

    Night samples are trivially predictable (target and prediction are both ~0) and
    their share differs sharply by site: 0 % at Folsom, whose camera records 06-20 h,
    against 19 % at Karpicko and 31 % at Poznan, which record around the clock.
    Leaving them in depresses the Polish sites' error for *every* method and makes
    cross-site comparison meaningless.  0 disables the filter.

    Days left with no lit sample are dropped entirely, which is the same state the
    simulation already uses for a day a client has no data for.
    """
    if min_ghi_clear <= 0.0:
        return day_map, 0
    kept: dict[str, list[int]] = {}
    dropped = 0
    for day, idxs in day_map.items():
        lit = []
        for j in idxs:
            stamp = periods[j]["history"][-1]["image_name"].rsplit(".", 1)[0]
            ghi = resolve_clear_sky_value(lookup, stamp)
            if ghi is not None and ghi >= min_ghi_clear:
                lit.append(j)
            else:
                dropped += 1
        if lit:
            kept[day] = lit
    return kept, dropped
