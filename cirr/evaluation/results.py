"""Helpers for persisting evaluation predictions and metric reports to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_predictions_csv(predictions_data: list[dict[str, Any]], path: Path | str) -> None:
    """Save *predictions_data* to a CSV file at *path*.

    Parameters
    ----------
    predictions_data : list[dict]
        List of per-sample dicts as produced by the evaluation loops
        (keys: ``timestamp``, ``day``, ``prediction``, ``target``,
        ``error``, ``absolute_error``).
    path : Path or str
        Destination file path.
    """
    pd.DataFrame(predictions_data).to_csv(path, index=False)


def save_metrics_report(
    tag: str,
    dataset_name: str,
    checkpoint: Path | str,
    dataset_path: Path | str,
    overall_metrics: dict[str, float],
    per_day_metrics: dict[str, dict[str, float]],
    path: Path | str,
    extra_lines: list[str] | None = None,
) -> None:
    """Write a human-readable metrics report to a text file.

    Parameters
    ----------
    tag : str
        Short identifier for the run (e.g. ``"baseline"`` or
        ``"memo_reg_all"``).  Shown in the file header.
    dataset_name : str
        Short dataset name (e.g. ``"poznan"``).
    checkpoint : Path or str
        Path to the model checkpoint (for provenance).
    dataset_path : Path or str
        Path to the dataset directory (for provenance).
    overall_metrics : dict
        As returned by :func:`~cirr.utils.metrics.compute_metrics`.
    per_day_metrics : dict
        Mapping ``day_key (MMDD) → metric_dict``.
    path : Path or str
        Destination file path.
    extra_lines : list[str], optional
        Additional lines to insert between provenance and metrics sections
        (e.g. TTA configuration details).
    """
    path = Path(path)
    with path.open("w") as f:
        f.write(f"Evaluation Results: {dataset_name}  [{tag}]\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Dataset: {dataset_path}\n")
        if extra_lines:
            for line in extra_lines:
                f.write(line + "\n")
        f.write("=" * 60 + "\n\n")

        f.write("Overall Metrics:\n")
        f.write(f"  MAE:   {overall_metrics['mae']:.8f}\n")
        f.write(f"  RMSE:  {overall_metrics['rmse']:.8f}\n")
        f.write(f"  WAPE:  {overall_metrics['wape']:.2f}%\n")
        f.write(f"  SMAPE: {overall_metrics['smape']:.2f}%\n")
        f.write(f"  Samples: {overall_metrics['count']}\n\n")

        f.write("Per-Day Metrics:\n")
        f.write(f"{'Day':<6} {'MAE':<10} {'RMSE':<10} {'WAPE':<10} {'SMAPE':<10} {'Samples':<24}\n")
        f.write("-" * 78 + "\n")
        for day_key, m in per_day_metrics.items():
            regime1_count = m.get("regime1_count")
            regime2_count = m.get("regime2_count")
            if regime1_count is not None and regime2_count is not None:
                samples_display = f"{m['count']} (R1:{regime1_count}; R2:{regime2_count})"
            else:
                samples_display = str(m["count"])

            f.write(
                f"{day_key:<6} {m['mae']:<10.4f} {m['rmse']:<10.4f} "
                f"{m['wape']:<10.2f}% {m['smape']:<10.2f}% {samples_display:<24}\n"
            )
