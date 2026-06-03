#!/usr/bin/env python
"""GRAPE main execution script."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluation.runner import evaluate_single, run_grid_search
from preprocessing.loader import load_feature_payload, prepare_signal_splits
from utils.config import (
    RANDOM_SEED,
    get_runtime_config,
    resolve_feature_cache,
    resolve_run_config,
    set_threading_constraints,
)
from utils.logger import get_logger

logger = get_logger("run")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GRAPE streaming ridge regression runner")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yml"), help="Path to config.yml")
    parser.add_argument("--dataset", help="Optional dataset name for labeling and config lookup")
    parser.add_argument("--feature-cache", help="Input feature file (.pkl) or exported split bundle directory")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search across all hyperparameter combinations")
    
    parser.add_argument("--alpha", type=float, help="Ridge regularization parameter")
    parser.add_argument("--nf", type=int, help="Number of top signal features")
    parser.add_argument("--adapter", choices=["nlms", "rls"], help="Streaming adapter type")
    parser.add_argument("--na", type=int, help="Number of adapter features")
    parser.add_argument("--ema", type=float, help="EMA alpha for context correction")
    
    parser.add_argument("--threads", type=int, help="Number of threading threads")
    parser.add_argument("--output", help="Output path for results (JSON or CSV)")
    parser.add_argument("--use-cpu", action="store_true", help="Use CPU instead of CUDA")
    
    return parser.parse_args()


def print_summary(summary: dict) -> None:
    """Print evaluation summary to console."""
    config = summary["config"]
    metrics = summary["metrics"]
    logger.info("%s", "=" * 64)
    logger.info("Config: alpha=%s nf=%s adapter=%s na=%s ema=%s",
                config["alpha"], config["nf"], config["adapter"], config["na"],
                config["ema"])
    logger.info("%-12s %10s %10s %10s", "split", "rmse", "mae", "r2")
    logger.info("%s", "-" * 48)
    for split in ["train", "validation", "test", "unseen"]:
        m = metrics[split]
        logger.info("%-12s %10.6f %10.6f %10.6f", split, m["RMSE"], m["MAE"], m["R2"])
    logger.info("%s", "=" * 64)

def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    runtime = get_runtime_config()
    set_seed(runtime.get("random_seed", RANDOM_SEED))
    set_threading_constraints(args.threads or runtime.get("default_threads", 4))
    
    run_config = resolve_run_config(
        args.config,
        dataset=args.dataset,
        alpha=args.alpha,
        nf=args.nf,
        adapter=args.adapter,
        na=args.na,
        ema=args.ema,
    )

    feature_cache = resolve_feature_cache(run_config.get("dataset"), args.feature_cache)
    if "dataset" not in run_config or not run_config["dataset"]:
        run_config["dataset"] = Path(feature_cache).stem
    logger.info("Loading feature cache: %s", feature_cache)
    data = load_feature_payload(feature_cache)
    data_splits, top_indices, _ = prepare_signal_splits(data)
    
    if args.grid_search:
        if args.output:
            output_path = Path(args.output)
        else:
            results_dir = runtime.get("results_dir", "results/grid-search")
            output_path = PROJECT_ROOT / results_dir / f"{run_config['dataset']}-grid.csv"
        logger.info("Running grid search, output: %s", output_path)
        results = run_grid_search(run_config, data_splits, top_indices, output_path=output_path, use_cpu=args.use_cpu)
        best = results[0]
        logger.info("Best config (rank 1): %s", {k: best[k] for k in ["alpha", "nf", "adapter", "na", "ema", "valid_r2"]})
        return
    
    output_path = Path(args.output) if args.output else None
    summary = evaluate_single(run_config, data_splits, top_indices, output_path=output_path, use_cpu=args.use_cpu)
    print_summary(summary)


if __name__ == "__main__":
    main()
