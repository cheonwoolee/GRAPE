from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return RMSE, MAE, and R2 as plain floats."""
    y_true_arr = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true, dtype=float)
    y_pred_arr = np.asarray(list(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}


def compute_split_metrics(split_predictions: dict[str, tuple[Iterable[float], Iterable[float]]]) -> dict[str, dict[str, float]]:
    """Compute metrics for each split from a mapping of split -> (y_true, y_pred)."""
    return {
        split: compute_metrics(y_true, y_pred)
        for split, (y_true, y_pred) in split_predictions.items()
    }
