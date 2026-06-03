from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import yaml


def set_threading_constraints(n_threads: int = 4) -> None:
    """Limit BLAS/OpenMP threads for reproducible runs."""
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yml"
DEFAULT_SEARCH_PATH = ROOT_DIR / "configs" / "search.yml"
DATA_DIR = ROOT_DIR / "data"
RANDOM_SEED = 42


@dataclass(frozen=True)
class DatasetProfile:
    dataset_name: str
    split_ratios: tuple[int, int, int, int]
    feature_cache: Path | None = None


@lru_cache(maxsize=1)
def load_project_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"config.yml not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def load_search_config(search_path: str | Path = DEFAULT_SEARCH_PATH) -> dict[str, Any]:
    search_file = Path(search_path)
    if not search_file.exists():
        raise FileNotFoundError(f"search.yml not found: {search_file}")
    with search_file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_runtime_config() -> dict[str, Any]:
    return load_project_config().get("runtime", {})


def get_hyperparameter_pools() -> dict[str, list[Any]]:
    pools = dict(load_search_config().get("pools", {}))
    adapters = [adapter for adapter in pools.get("adapter", []) if adapter != "online_head_sgd"]
    pools["adapter"] = adapters
    return pools


def get_search_active_keys() -> list[str]:
    return list(load_search_config().get("active_keys", ["alpha", "nf", "adapter", "na", "ema"]))


def validate_adapter(adapter: str) -> str:
    if adapter == "online_head_sgd":
        raise ValueError("adapter 'online_head_sgd' has been removed")
    if adapter not in {"nlms", "rls"}:
        raise ValueError(f"Unsupported adapter '{adapter}'. Expected one of: nlms, rls")
    return adapter


def get_dataset_profile(dataset_name: str) -> DatasetProfile:
    datasets = load_project_config().get("datasets", {})
    dataset_cfg = datasets.get(dataset_name, {})
    split_ratios = tuple(dataset_cfg.get("split_ratios", [25, 5, 10, 60]))
    return DatasetProfile(
        dataset_name=dataset_name,
        split_ratios=split_ratios,
        feature_cache=Path(dataset_cfg["feature_cache"]) if dataset_cfg.get("feature_cache") else None,
    )


def get_dataset_config(dataset_name: str) -> DatasetProfile:
    return get_dataset_profile(dataset_name)


def resolve_feature_cache(dataset_name: str | None = None, override_path: str | Path | None = None) -> Path:
    if override_path is not None:
        return Path(override_path)

    if dataset_name:
        profile = get_dataset_profile(dataset_name)
        if profile.feature_cache is not None:
            return profile.feature_cache
        processed_root = Path(get_runtime_config().get("processed_root", DATA_DIR / "processed"))
        return processed_root / dataset_name / "wafer_features.pkl"

    raise ValueError("Feature input path is required. Provide --feature-cache or configure datasets.<name>.feature_cache.")


def resolve_run_config(config_path: str | Path, dataset: str | None = None, **kwargs) -> dict[str, Any]:
    """Resolve run configuration from config file and command-line overrides."""
    config = load_project_config(config_path)
    runtime = get_runtime_config()
    defaults = config.get("model_defaults") or {}
    resolved = dict(defaults)

    if dataset:
        resolved["dataset"] = dataset
        dataset_cfg = config.get("datasets", {}).get(dataset, {})
        dataset_override_keys = [
            "alpha", "nf", "adapter", "na", "ema",
            "context", "nlms_lr", "rls_forgetting",
        ]
        for key in dataset_override_keys:
            if key in dataset_cfg:
                resolved[key] = dataset_cfg[key]

    for key, value in kwargs.items():
        if value is not None:
            resolved[key] = value

    resolved.setdefault("alpha", 100.0)
    resolved.setdefault("nf", 50)
    resolved.setdefault("adapter", "nlms")
    resolved.setdefault("na", 10)
    resolved.setdefault("ema", 0.1)
    resolved.setdefault("context", runtime.get("context", "hierarchical"))
    resolved.setdefault("nlms_lr", runtime.get("nlms_lr", 0.01))
    resolved.setdefault("rls_forgetting", runtime.get("rls_forgetting", 0.99))

    validate_adapter(resolved["adapter"])
    return resolved
