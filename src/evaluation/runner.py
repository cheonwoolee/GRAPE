"""Model building and evaluation."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import compute_metrics
from models.grape import GrapeModel
from utils.config import get_hyperparameter_pools, get_runtime_config, get_search_active_keys
from utils.logger import get_logger

logger = get_logger("runner")


def build_model(run_config: dict, use_cpu: bool = False) -> GrapeModel:
    """Build GrapeModel from run configuration."""
    runtime = get_runtime_config()
    return GrapeModel(
        alpha=run_config["alpha"],
        adapter=run_config["adapter"],
        na=run_config["na"],
        ema=run_config["ema"],
        nlms_lr=runtime.get("nlms_lr", 0.01),
        rls_forgetting=runtime.get("rls_forgetting", 0.99),
        context=runtime.get("context", "hierarchical"),
        use_gpu=not use_cpu,
    )


def stream_evaluate(model: GrapeModel, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Stream predictions and updates through model."""
    preds = np.zeros(len(X), dtype=float)
    for idx in range(len(X)):
        ctx_key = f"{meta.iloc[idx]['EQP_ID']}_{meta.iloc[idx]['CHM_ID']}"
        preds[idx] = model.predict(X[idx], eqp_ch_key=ctx_key)
        model.update(X[idx], y[idx], eqp_ch_key=ctx_key)
    return preds, compute_metrics(y, preds)


def evaluate_single(run_config: dict, data_splits: dict, top_indices: np.ndarray, output_path: Path | None = None, use_cpu: bool = False) -> dict:
    """Evaluate single configuration across all splits."""
    n_features = min(int(run_config["nf"]), data_splits["train"][0].shape[1])
    feature_indices = top_indices[:n_features]
    unseen_indices = np.arange(n_features)
    model = build_model(run_config, use_cpu=use_cpu)

    X_train = data_splits["train"][0][:, feature_indices]
    y_train = data_splits["train"][1]
    model.set_initial_data(X_train, y_train)

    train_preds = model.global_model.predict(X_train)
    metrics = {"train": compute_metrics(y_train, train_preds)}
    predictions = {"train": train_preds.tolist()}

    for split in ["validation", "test"]:
        split_preds, split_metrics = stream_evaluate(model, data_splits[split][0][:, feature_indices], data_splits[split][1], data_splits[split][2])
        metrics[split] = split_metrics
        predictions[split] = split_preds.tolist()

    unseen_preds, unseen_metrics = stream_evaluate(model, data_splits["unseen"][0][:, unseen_indices], data_splits["unseen"][1], data_splits["unseen"][2])
    metrics["unseen"] = unseen_metrics
    predictions["unseen"] = unseen_preds.tolist()

    summary = {
        "config": {
            key: run_config[key]
            for key in ["dataset", "alpha", "nf", "adapter", "na", "ema"]
        },
        "metrics": metrics,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    return summary


def run_grid_search(run_config: dict, data_splits: dict, top_indices: np.ndarray, output_path: Path | None, use_cpu: bool = False) -> list[dict]:
    """Run grid search across hyperparameter pools."""
    pools = get_hyperparameter_pools()
    keys = get_search_active_keys()
    results = []
    total = int(np.prod([len(pools[key]) for key in keys]))
    for idx, combo in enumerate(itertools.product(*(pools[key] for key in keys)), start=1):
        hp = dict(zip(keys, combo))
        config = dict(run_config)
        config.update(hp)
        summary = evaluate_single(config, data_splits, top_indices, output_path=None, use_cpu=use_cpu)
        row = {
            **hp,
            "train_rmse": summary["metrics"]["train"]["RMSE"],
            "train_mae": summary["metrics"]["train"]["MAE"],
            "train_r2": summary["metrics"]["train"]["R2"],
            "valid_rmse": summary["metrics"]["validation"]["RMSE"],
            "valid_mae": summary["metrics"]["validation"]["MAE"],
            "valid_r2": summary["metrics"]["validation"]["R2"],
            "test_rmse": summary["metrics"]["test"]["RMSE"],
            "test_mae": summary["metrics"]["test"]["MAE"],
            "test_r2": summary["metrics"]["test"]["R2"],
            "unseen_rmse": summary["metrics"]["unseen"]["RMSE"],
            "unseen_mae": summary["metrics"]["unseen"]["MAE"],
            "unseen_r2": summary["metrics"]["unseen"]["R2"],
        }
        results.append(row)
        logger.info("[%s/%s] %s valid_r2=%.6f", idx, total, " ".join(f"{k}={v}" for k, v in hp.items()), row["valid_r2"])

    results.sort(key=lambda row: row["valid_r2"], reverse=True)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    return results
