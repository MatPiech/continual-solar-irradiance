"""Utility functions for per-day dataset processing and organization."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any


def generate_year_days() -> list[str]:
    """Generate all days of a standard year as MMDD strings.

    Returns
    -------
    list[str]
        List of 365 day strings in MMDD format (e.g., '0101', '0102', ..., '1231').
    """
    days = []
    start_date = datetime(2016, 1, 1)
    for day_offset in range(366):  # 2016 is a leap year
        current_date = start_date + timedelta(days=day_offset)
        # Skip Feb 29 to make it a standard 365-day year
        if current_date.month == 2 and current_date.day == 29:
            continue
        days.append(current_date.strftime("%m%d"))  # MMDD format
    return days


def group_indices_by_day(periods: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Group period indices by day using year-invariant MMDD format.

    Extracts MMDD (month and day) from image names, treating all years the same.

    Parameters
    ----------
    periods : list[dict[str, Any]]
        List of period dictionaries containing history with image_name field.

    Returns
    -------
    dict[str, list[int]]
        Dictionary mapping MMDD strings to lists of period indices for that day.
    """
    day_to_indices = defaultdict(list)
    sorted_indices = sorted(range(len(periods)), key=lambda idx: periods[idx]["history"][-1]["image_name"])
    for idx in sorted_indices:
        image_name = periods[idx]["history"][-1]["image_name"]
        # Extract MMDD only (positions 4-8 of YYYYMMDD), year-invariant
        day_key = image_name[4:8]  # MMDD
        day_to_indices[day_key].append(idx)
    return day_to_indices


def select_day_indices(
    day_index_map: dict[str, list[int]],
    num_days: int,
    start_day: str,
) -> list[int]:
    """Return flat list of dataset indices for *num_days* days from *start_day*.

    Parameters
    ----------
    day_index_map : dict[str, list[int]]
        Mapping ``MMDD → list[period_index]`` as returned by
        :func:`group_indices_by_day`.
    num_days : int
        Number of days to include.  Pass ``-1`` (or any value ``<= 0``) to
        select all days from *start_day* to year-end.
    start_day : str
        First day in MMDD format (e.g. ``"0101"``).

    Returns
    -------
    list[int]
        Ordered period indices covering the requested day range.

    Raises
    ------
    ValueError
        If *start_day* is not a valid MMDD string.
    """
    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day {start_day!r}. Expected MMDD format (0101–1231).")
    start_idx = all_days.index(start_day)
    selected_days = all_days[start_idx : start_idx + num_days] if num_days > 0 else all_days[start_idx:]
    indices: list[int] = []
    for day_key in selected_days:
        if day_key in day_index_map:
            indices.extend(day_index_map[day_key])
    return indices
