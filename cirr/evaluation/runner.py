"""Shared eval script bootstrap helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cirr.data.loading import load_dataset
from cirr.models.forecaster import Forecaster
from cirr.models.loading import load_forecaster_for_eval
from cirr.utils.dataset_utils import generate_year_days, group_indices_by_day


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def load_eval_model(checkpoint: Path, device: str) -> Forecaster:
    return load_forecaster_for_eval(checkpoint, device=device)


def dataset_name_from_path(dataset_path: Path) -> str:
    return dataset_path.name.split("_")[0]


def load_eval_days(
    dataset_path: Path,
    start_day: str,
    num_days: int,
) -> tuple[Any, dict[str, list[int]], list[str], str]:
    dataset_name = dataset_name_from_path(dataset_path)
    dataset = load_dataset(dataset_path, split="test")
    day_index_map = group_indices_by_day(dataset.periods)
    all_days = generate_year_days()
    if start_day not in all_days:
        raise ValueError(f"Invalid start_day: {start_day!r}")
    start_idx = all_days.index(start_day)
    days_to_eval = all_days[start_idx : start_idx + num_days] if num_days > 0 else all_days[start_idx:]
    days_to_eval = [d for d in days_to_eval if d in day_index_map]
    return dataset, day_index_map, days_to_eval, dataset_name
