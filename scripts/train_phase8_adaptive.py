"""Phase 8 -- Arm Ad (adaptive).

  python -u scripts/train_phase8_adaptive.py --seed 42

Loads the same shared initial checkpoint as Arm F (scripts/train_phase8_init.py; not yet
producible in this implementation pass -- see that script's docstring). At each of the 5
pre-registered window boundaries, AFTER that window is fully scored, fine-tunes for the
pre-registered `K` epochs (hp["fine_tune_epochs"]) at the pre-registered learning rate
(hp["fine_tune_lr"]) on the expanding window TRAIN(1..121) union every already-scored VAL window
-- via `train_epoch` (Phase 7, unmodified) called with an advancing `train_last` -- never on the
window just scored or any later one (Section 14.5).

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

EXPERIMENT = "phase8_adaptive_drift/adaptive"


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
    # A single, persistent optimizer, continued (not reset) across every update round -- this is
    # the "warm start" referred to in the locked design (Section C): momentum/variance state
    # carries forward, only the data each round trains over expands.
    opt = torch.optim.Adam(model.parameters(), lr=hp["fine_tune_lr"], weight_decay=1e-5)

    per_window = run_walk_forward("Ad", model, ds, windows, opt=opt, pos_weight=pos_weight,
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
    print(f"seed {args.seed}: arm Ad scored+updated across {len(windows)} windows "
          f"(K={hp['fine_tune_epochs']} epochs/round, lr={hp['fine_tune_lr']})", flush=True)


if __name__ == "__main__":
    main()
