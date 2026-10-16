"""Phase 8 — Arm F (frozen-parameter baseline).

  python -u scripts/train_phase8_frozen.py --seed 42

Loads the shared initial TRAIN-only checkpoint produced by scripts/train_phase8_init.py (NOT
produced in this implementation pass -- that script currently raises NotImplementedError at its
final wiring step, so this script cannot run to completion yet either; see its docstring).
Weights are NEVER updated after loading. Causal temporal/entity state evolves during inference
exactly as Phase 7 specifies (persistent `TemporalState`, carried forward, never reset, across
all 5 windows) -- see src/adaptive/update_loop.py's module docstring for why this is not the same
thing as "frozen internal state" (Section 14.1 of the locked design).

THIS SCRIPT PERFORMS REAL INFERENCE (forward passes over VAL) WHEN RUN. It is written now, as
implementation, and is NOT executed by this implementation pass.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.adaptive.update_loop import run_walk_forward
from src.adaptive.windows import build_windows
from src.config import load_config
from src.graph.schema import compute_pseudo_account_segments
from src.graph.temporal import build_temporal_dataset, load_raw_arrays
from src.models.metrics import compute_metrics
from src.models.temporal_graphsage import TemporalGraphSAGE

EXPERIMENT = "phase8_adaptive_drift/frozen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()
    base = cfg.experiments_dir / "phase8_adaptive_drift"
    out_dir = cfg.experiments_dir / EXPERIMENT / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    init_ckpt = base / "shared_init" / f"model_checkpoint_initial_seed{args.seed}.pt"
    hp_path = base / "shared_init" / f"hparams_seed{args.seed}.json"
    if not init_ckpt.exists() or not hp_path.exists():
        raise FileNotFoundError(
            f"shared initial checkpoint/hyperparameters not found ({init_ckpt}, {hp_path}) -- run "
            f"scripts/train_phase8_init.py --seed {args.seed} first (currently incomplete, see its docstring)")
    hp = json.load(open(hp_path))

    raw, graph, scaler = load_raw_arrays(cfg)
    ds = build_temporal_dataset(raw, cfg.split.val_day_max, dev, cfg)  # VAL loaded for SCORING only; F never trains on it
    windows = build_windows(cfg.split.val_day_min, cfg.split.val_day_max)

    model = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)
    model.load_state_dict(torch.load(init_ckpt, map_location=dev))
    sd_before = {k: v.clone() for k, v in model.state_dict().items()}

    per_window = run_walk_forward("F", model, ds, windows)  # no opt/pos_weight/fine_tune_epochs: F never updates

    sd_after = {k: v.clone() for k, v in model.state_dict().items()}
    for k in sd_before:
        assert torch.equal(sd_before[k], sd_after[k]), f"arm F parameter {k} changed -- this must never happen"

    y_all = pd.Series(raw["y"], index=raw["txn_id"])
    seg_all = pd.Series(compute_pseudo_account_segments(graph), index=graph["transaction"].transaction_id.numpy())
    for k, (ids, scores, days) in enumerate(per_window, start=1):
        y_true = y_all.loc[ids].to_numpy()
        seg = seg_all.loc[ids].to_numpy()
        m_pooled = compute_metrics(y_true, scores, threshold=hp["threshold"])
        metrics = {"pooled": m_pooled}
        for name in ["has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"]:
            mask = seg == name
            yt, sc = y_true[mask], scores[mask]
            metrics[name] = (compute_metrics(yt, sc, threshold=hp["threshold"]) if 0 < yt.sum() < len(yt)
                             else {"n_rows": int(len(yt)), "n_fraud": int(yt.sum()), "note": "insufficient class balance"})
        with open(out_dir / f"metrics_val_W{k}.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=1)
        pd.DataFrame({"TransactionID": ids, "isFraud": y_true, "y_score": scores, "day": days, "window": k,
                      "segment": seg}).to_parquet(out_dir / f"predictions_val_W{k}.parquet", index=False)
    print(f"seed {args.seed}: arm F scored {len(windows)} windows; parameters verified unchanged throughout", flush=True)


if __name__ == "__main__":
    main()
