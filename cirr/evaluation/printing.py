"""Console helpers for printing per-day and summary evaluation metrics."""

import numpy as np

# ---------------------------------------------------------------------------
# Table layout
# ---------------------------------------------------------------------------

_HEADER = "Day    MAE        RMSE       WAPE       SMAPE      Samples "
_SEPARATOR = "-" * len(_HEADER)


def print_metrics_header() -> None:
    """Print the column header and separator line for the per-day table."""
    print(_HEADER)
    print(_SEPARATOR)


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

    # Handle NaN values for percentage metrics
    wape_val = overall_metrics["wape"]
    smape_val = overall_metrics["smape"]
    wape_str = f"{wape_val:.2f}%" if not np.isnan(wape_val) else "N/A"
    smape_str = f"{smape_val:.2f}%" if not np.isnan(smape_val) else "N/A"

    print(f"  WAPE:  {wape_str}")
    print(f"  SMAPE: {smape_str}")
    print(f"  Samples: {overall_metrics['count']}")
