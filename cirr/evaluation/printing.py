"""Console helpers for printing per-day and summary evaluation metrics."""

import numpy as np

from cirr.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Table layout
# ---------------------------------------------------------------------------

_HEADER = "Day    MAE        RMSE       MAPE       WAPE       SMAPE      Samples "
_SEPARATOR = "-" * len(_HEADER)


def print_metrics_header() -> None:
    """Print the column header and separator line for the per-day table."""
    print(_HEADER)
    print(_SEPARATOR)


def print_day_metrics(day: str, preds: list[float], targets: list[float]) -> None:
    """Compute metrics for *preds* / *targets* and print a single table row.

    Parameters
    ----------
    day : str
        Day label in MMDD format (e.g. ``"0315"``).
    preds : list[float]
        Model predictions for this day.
    targets : list[float]
        Ground-truth values for this day.
    """
    m = compute_metrics(np.array(preds), np.array(targets))
    print(
        f"{day:<6} {m['mae']:<10.4f} {m['rmse']:<10.4f} "
        f"{m['mape']:<16.2%} {m['wape']:<10.2f}% {m['smape']:<10.2f}% {m['count']:<8}"
    )


def print_overall_metrics(overall_metrics: dict[str, float], tag: str = "") -> None:
    """Print an overall metrics summary block.

    Parameters
    ----------
    overall_metrics : dict
        Dictionary as returned by :func:`~cirr.utils.metrics.compute_metrics`.
    tag : str
        Optional label appended to the section header (e.g. dataset name or
        method tag).
    """
    width = 60
    header = f"Evaluation Results{': ' + tag if tag else ''}"
    print(f"\n{'=' * width}")
    print(header)
    print(f"{'=' * width}\n")
    print("Overall Metrics:")
    print(f"  MAE:   {overall_metrics['mae']:.8f}")
    print(f"  RMSE:  {overall_metrics['rmse']:.8f}")
    print(f"  MAPE:  {overall_metrics['mape']:.2%}")
    print(f"  WAPE:  {overall_metrics['wape']:.2f}%")
    print(f"  SMAPE: {overall_metrics['smape']:.2f}%")
    print(f"  Samples: {overall_metrics['count']}")
