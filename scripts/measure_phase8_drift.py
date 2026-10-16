"""Phase 8 -- drift measurement (design Section 5 / 14.4).

  python -u scripts/measure_phase8_drift.py

Reads each arm/seed's already-saved per-window predictions (written by
train_phase8_{frozen,adaptive,replay_control}.py) and the TRAIN reference feature/score
distribution, and computes feature-distribution, prediction-distribution, and performance drift
for every window, per arm, per seed -- all RETROSPECTIVE (Section 14.4; see
src/adaptive/drift.py's module docstring for why none of this can influence any prediction).

Requires train_phase8_*.py to have already been run (they have not been, in this implementation
pass) -- this script reads their output files and will raise FileNotFoundError until they exist.
Written now as implementation; NOT executed by this implementation pass.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.adaptive.drift import feature_distribution_drift, performance_drift, prediction_distribution_drift
from src.adaptive.windows import build_windows
from src.config import load_config
from src.graph.temporal import load_raw_arrays

SEEDS = [42, 123, 2024]
ARMS = ["frozen", "adaptive", "replay_control"]


def main() -> None:
    cfg = load_config()
    base = cfg.experiments_dir / "phase8_adaptive_drift"
    out_dir = base / "drift_measurements"
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = build_windows(cfg.split.val_day_min, cfg.split.val_day_max)

    raw, graph, scaler = load_raw_arrays(cfg)
    train_mask = (raw["day"] >= cfg.split.train_day_min) & (raw["day"] <= cfg.split.train_day_max)
    train_features = raw["X"][train_mask]

    results = {"feature_drift": {}, "prediction_drift": {}, "performance_drift": {}}
    for arm in ARMS:
        results["feature_drift"][arm] = {}
        results["prediction_drift"][arm] = {}
        results["performance_drift"][arm] = {}
        for seed in SEEDS:
            seed_dir = base / arm / f"seed_{seed}"
            window_metrics = []
            prev_scores = None
            fd_by_window, pd_by_window = {}, {}
            for k in range(1, len(windows) + 1):
                preds = pd.read_parquet(seed_dir / f"predictions_val_W{k}.parquet")
                m = json.load(open(seed_dir / f"metrics_val_W{k}.json"))
                window_metrics.append(m["pooled"])

                day_mask = np.isin(raw["day"], preds["day"].unique())
                fd_by_window[k] = feature_distribution_drift(train_features, raw["X"][day_mask])

                scores = preds["y_score"].to_numpy()
                ref = prev_scores if prev_scores is not None else scores
                pd_by_window[k] = prediction_distribution_drift(ref, scores)
                prev_scores = scores

            results["feature_drift"][arm][seed] = fd_by_window
            results["prediction_drift"][arm][seed] = pd_by_window
            results["performance_drift"][arm][seed] = performance_drift(window_metrics, primary_key="pr_auc")

    with open(out_dir / "feature_drift.json", "w", encoding="utf-8") as fh:
        json.dump(results["feature_drift"], fh, indent=1, default=str)
    with open(out_dir / "prediction_drift.json", "w", encoding="utf-8") as fh:
        json.dump(results["prediction_drift"], fh, indent=1, default=str)
    with open(out_dir / "performance_drift.json", "w", encoding="utf-8") as fh:
        json.dump(results["performance_drift"], fh, indent=1, default=str)
    print(f"wrote drift measurements for {len(ARMS)} arms x {len(SEEDS)} seeds to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
