"""Metrics for evaluating models."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


def robust_wape(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error."""
    denominator = np.sum(np.abs(targets))
    if denominator == 0:
        return 0.0

    return float(np.sum(np.abs(targets - predictions)) / denominator * 100)


def robust_smape(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute Symmetric Mean Absolute Percentage Error."""
    numerator = 2 * np.abs(targets - predictions)
    denominator = np.abs(targets) + np.abs(predictions)

    smape_array = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return float(100 * np.mean(smape_array))


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, WAPE, SMAPE, and sample count."""
    mae = mean_absolute_error(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    wape = robust_wape(targets, predictions)
    smape = robust_smape(targets, predictions)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "wape": float(wape),
        "smape": float(smape),
        "count": len(predictions),
    }
