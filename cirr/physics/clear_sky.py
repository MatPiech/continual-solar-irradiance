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


# Solar zenith beyond which a sample is treated as night/twilight and dropped
# from training and evaluation.  Shared by every chapter so error populations
# stay comparable across sites and experiments.
DAYLIGHT_MAX_ZENITH = 85.0


def load_zenith_lookup(dataset_path: Path) -> dict[str, float]:
    """Load apparent solar zenith (degrees) indexed by ``YYYYMMDD_HHMMSS``.

    Reads the ``zenith`` column of the site's ``*clear_sky.csv``.  Returns an
    empty dict when the file predates that column, which callers must treat as
    "cannot filter" and refuse rather than as "no daylight".
    """
    clear_sky_files = sorted(dataset_path.glob("*clear_sky.csv"))
    if not clear_sky_files:
        return {}

    frame = pd.read_csv(clear_sky_files[0])
    if "datetime" not in frame.columns or "zenith" not in frame.columns:
        return {}

    dt = pd.to_datetime(frame["datetime"], errors="coerce")
    zenith = frame["zenith"].astype(float)
    valid = ~dt.isna() & zenith.notna()
    keys = dt.loc[valid].dt.strftime("%Y%m%d_%H%M%S")
    return dict(zip(keys.tolist(), zenith.loc[valid].tolist(), strict=False))


def is_daylight(zenith_lookup: dict[str, float], timestamp: str, max_zenith: float = DAYLIGHT_MAX_ZENITH) -> bool:
    """True when the sample's apparent zenith is below *max_zenith*.

    Unknown timestamps are kept, so a missing solar table degrades to the
    unfiltered behaviour instead of silently emptying the dataset.
    """
    zenith = resolve_clear_sky_value(zenith_lookup, timestamp)
    return True if zenith is None else zenith < max_zenith
