"""Metrics for evaluating models."""

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def robust_wape(targets: np.ndarray, predictions: np.ndarray, min_threshold: float = 0.0) -> float:
    """Compute Weighted Absolute Percentage Error.

    Parameters
    ----------
    targets : np.ndarray
        Ground truth values.
    predictions : np.ndarray
        Predicted values.
    min_threshold : float, optional
        Minimum target value to include in computation. Values below this
        threshold are excluded to avoid division by near-zero denominators.
        Default is 0.0 (no filtering). For solar irradiance, 50-100 W/m²
        is recommended.

    Returns
    -------
    float
        WAPE as a percentage, or np.nan if no valid samples remain after filtering.
    """
    # Filter out samples below threshold
    mask = targets >= min_threshold
    if not np.any(mask):
        return np.nan

    filtered_targets = targets[mask]
    filtered_predictions = predictions[mask]

    denominator = np.sum(np.abs(filtered_targets))
    if denominator == 0:
        return np.nan

    return float(np.sum(np.abs(filtered_targets - filtered_predictions)) / denominator * 100)


def robust_smape(targets: np.ndarray, predictions: np.ndarray, min_threshold: float = 0.0) -> float:
    """Compute Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    targets : np.ndarray
        Ground truth values.
    predictions : np.ndarray
        Predicted values.
    min_threshold : float, optional
        Minimum target value to include in computation. Values below this
        threshold are excluded to avoid division by near-zero denominators.
        Default is 0.0 (no filtering). For solar irradiance, 50-100 W/m²
        is recommended.

    Returns
    -------
    float
        SMAPE as a percentage, or np.nan if no valid samples remain after filtering.
    """
    # Filter out samples below threshold
    mask = targets >= min_threshold
    if not np.any(mask):
        return np.nan

    filtered_targets = targets[mask]
    filtered_predictions = predictions[mask]

    numerator = 2 * np.abs(filtered_targets - filtered_predictions)
    denominator = np.abs(filtered_targets) + np.abs(filtered_predictions)

    smape_array = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return float(100 * np.mean(smape_array))


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, min_threshold: float = 0.005) -> dict[str, float]:
    """Compute MAE, RMSE, WAPE, SMAPE, and sample count.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values.
    targets : np.ndarray
        Ground truth values.
    min_threshold : float, optional
        Minimum target value for computing percentage-based metrics (WAPE/SMAPE).
        Samples below this threshold are excluded from percentage calculations to
        avoid very high values when targets are near zero.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: 'mae', 'rmse', 'wape', 'smape', 'count'.
        WAPE and SMAPE may be np.nan if no samples exceed min_threshold.
    """
    mae = mean_absolute_error(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)
    wape = robust_wape(targets, predictions, min_threshold=min_threshold)
    smape = robust_smape(targets, predictions, min_threshold=min_threshold)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wape": float(wape),
        "smape": float(smape),
        "count": len(predictions),
    }
