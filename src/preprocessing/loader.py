"""Data loading and preparation for GRAPE evaluation."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger

logger = get_logger("loader")
SPLITS = ["train", "validation", "test", "unseen"]


def _resolve_bundle_prefix(bundle_dir: Path) -> str:
    expected = bundle_dir / f"{bundle_dir.name}_train_features.pkl"
    if expected.exists():
        return bundle_dir.name
    candidates = sorted(bundle_dir.glob("*_train_features.pkl"))
    if len(candidates) == 1:
        return candidates[0].name.replace("_train_features.pkl", "")
    if not candidates:
        raise FileNotFoundError(f"No exported split bundle found in {bundle_dir}")
    raise RuntimeError(f"Multiple bundle prefixes detected in {bundle_dir}; pass a specific file path instead.")


def _build_ids(split_ctx: dict | None, n_rows: int) -> pd.DataFrame:
    if not split_ctx:
        return pd.DataFrame({"EQP": ["UNKNOWN"] * n_rows, "CH": ["UNKNOWN"] * n_rows})

    eqp = np.asarray(split_ctx.get("eqp", ["UNKNOWN"] * n_rows))
    ch = np.asarray(split_ctx.get("ch", ["UNKNOWN"] * n_rows))
    if len(eqp) != n_rows:
        eqp = np.asarray(["UNKNOWN"] * n_rows)
    if len(ch) != n_rows:
        ch = np.asarray(["UNKNOWN"] * n_rows)
    return pd.DataFrame({"EQP": eqp, "CH": ch})


def _load_exported_split_bundle(bundle_dir: Path) -> dict:
    prefix = _resolve_bundle_prefix(bundle_dir)
    context_path = bundle_dir / f"{prefix}_context_ids.pkl"
    context_data = {}
    if context_path.exists():
        with context_path.open("rb") as handle:
            context_data = pickle.load(handle)

    split_alias = {"train": "train", "validation": "valid", "test": "test", "unseen": "unseen"}
    payload = {}
    for split in SPLITS:
        raw_split = split_alias[split]
        x_path = bundle_dir / f"{prefix}_{raw_split}_features.pkl"
        y_path = bundle_dir / f"{prefix}_{raw_split}_label.pkl"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing exported split files for '{split}': {x_path.name}, {y_path.name}")

        with x_path.open("rb") as handle:
            x_values = np.asarray(pickle.load(handle), dtype=float)
        with y_path.open("rb") as handle:
            y_values = np.asarray(pickle.load(handle), dtype=float).reshape(-1)
        if x_values.shape[0] != y_values.shape[0]:
            raise ValueError(f"Row mismatch for split '{split}': X={x_values.shape[0]} y={y_values.shape[0]}")

        x_df = pd.DataFrame(x_values, columns=[f"f{idx}" for idx in range(x_values.shape[1])])
        ids = _build_ids(context_data.get(raw_split), n_rows=x_values.shape[0])
        payload[split] = {"X": x_df, "y": y_values, "ids": ids}

    return payload


def load_feature_payload(feature_cache) -> dict:
    """Load feature cache and validate splits."""
    feature_path = Path(feature_cache)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {feature_cache}")

    if feature_path.is_dir():
        data = _load_exported_split_bundle(feature_path)
    else:
        with feature_path.open("rb") as handle:
            data = pickle.load(handle)

        if not isinstance(data, dict) or "train" not in data:
            parent = feature_path.parent
            data = _load_exported_split_bundle(parent)

    if "unseen" not in data and "test" in data:
        data["unseen"] = data["test"]
    for split in SPLITS:
        if split not in data:
            raise KeyError(f"Missing split: {split}")
        if hasattr(data[split]["y"], "values"):
            data[split]["y"] = data[split]["y"].values
        data[split]["y"] = np.asarray(data[split]["y"], dtype=float).reshape(-1)
        if "CTX_lot_target_mean" in data[split]["X"].columns:
            data[split]["X"] = data[split]["X"].drop(columns=["CTX_lot_target_mean"])
    keep_mask = data["train"]["y"] > -10.0
    if (~keep_mask).any():
        filtered = {}
        for key, value in data["train"].items():
            if isinstance(value, pd.DataFrame):
                filtered[key] = value.loc[keep_mask].reset_index(drop=True)
            elif isinstance(value, pd.Series):
                filtered[key] = value.loc[keep_mask].reset_index(drop=True)
            else:
                filtered[key] = np.asarray(value)[keep_mask]
        data["train"] = filtered
    return data


def prepare_signal_splits(data: dict) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]], np.ndarray, list[str]]:
    """Prepare signal splits with StandardScaler normalization."""
    x_train_df = data["train"]["X"]
    signal_cols = [column for column in x_train_df.columns if column.startswith("f")]
    if not signal_cols:
        raise ValueError("No signal feature columns found. Expected columns starting with 'f'.")

    scaler = StandardScaler().fit(np.nan_to_num(x_train_df[signal_cols].values))
    data_splits = {}
    for split in SPLITS:
        ids = data[split]["ids"]
        meta = pd.DataFrame({
            "EQP_ID": ids.iloc[:, 0] if len(ids.columns) > 0 else "UNKNOWN",
            "CHM_ID": ids.iloc[:, 1] if len(ids.columns) > 1 else "UNKNOWN",
        })
        scaled = scaler.transform(np.nan_to_num(data[split]["X"][signal_cols].values))
        data_splits[split] = (scaled, data[split]["y"], meta)

    corrs = np.array([abs(np.corrcoef(data_splits["train"][0][:, idx], data_splits["train"][1])[0, 1]) for idx in range(data_splits["train"][0].shape[1])])
    corrs[np.isnan(corrs)] = 0.0
    top_indices = np.argsort(corrs)[::-1]
    return data_splits, top_indices, signal_cols
