"""Phase 8 -- Arm E (replay-control ablation, design Section 10).

  python -u scripts/train_phase8_replay_control.py --seed 42

Identical to Arm Ad (same shared initial checkpoint, same update cadence, same K/LR, same
optimizer settings) EXCEPT: at each update round, the training data is `TRAIN(1..121)` plus a
reproducibly-resampled-with-replacement set of TRAIN days (never a VAL or TEST day), matched in
day-count to Ad's expanding window at that round (src/adaptive/replay.py's `ReplayPoolBuilder`).
Arm E is still SCORED on the real W1..W5 calendar, identically to F/Ad -- only its UPDATE's
training-data source differs. See src/adaptive/update_loop.py's module docstring for why this
cannot corrupt the live scoring state.

THIS SCRIPT PERFORMS REAL TRAINING (forward+backward passes, GPU/CPU) WHEN RUN. It is written
now, as implementation, and is NOT executed by this implementation pass.
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

EXPERIMENT = "phase8_adaptive_drift/replay_control"


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
    ds = build_temporal_dataset(raw, cfg.split.val_day_max, dev, cfg)
    windows = build_windows(cfg.split.val_day_min, cfg.split.val_day_max)

    model = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)
    model.load_state_dict(torch.load(init_ckpt, map_location=dev))

    tr_rows = ds.txn_day <= cfg.split.train_day_max
    y_tr = raw["y"][:len(tr_rows)][tr_rows]
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=hp["fine_tune_lr"], weight_decay=1e-5)

    per_window = run_walk_forward("E", model, ds, windows, opt=opt, pos_weight=pos_weight,
                                  fine_tune_epochs=hp["fine_tune_epochs"], base_seed=args.seed,
                                  train_day_min=cfg.split.train_day_min, train_day_max=cfg.split.train_day_max)
    torch.save(model.state_dict(), out_dir / "model_checkpoint_final.pt")

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
    print(f"seed {args.seed}: arm E scored+updated across {len(windows)} windows "
          f"(K={hp['fine_tune_epochs']} epochs/round, lr={hp['fine_tune_lr']}, TRAIN-only replay)", flush=True)


if __name__ == "__main__":
    main()
