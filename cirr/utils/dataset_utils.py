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
