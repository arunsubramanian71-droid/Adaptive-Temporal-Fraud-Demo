"""Phase 7A arm C: memory-ablated temporal control (order-agnostic cumulative-mean memory).

Identical to Phase 7 in everything except the memory mechanism: the SAME `train_epoch` (imported from
scripts/train_temporal_graphsage.py, not re-implemented), same causal rolling evaluation, optimizer, lr,
weight decay, clipping, TBPTT window, seeds, 150 epochs / patience 20. See
experiments/temporal_ablation/DESIGN_BEFORE_EXECUTION.md.

  python -u scripts/train_temporal_mean_memory.py            # full 3-seed experiment
  python -u scripts/train_temporal_mean_memory.py --smoke    # 2-epoch timing check, writes nothing
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from phase7a_common import aggregate_and_plot, final_metrics, save_seed
from src.config import load_config
from src.graph.schema import compute_pseudo_account_segments
from src.graph.temporal import build_temporal_dataset, load_raw_arrays, snapshot_stats
from src.models.temporal_graphsage import rolling_temporal_predict
from src.models.temporal_meanmem import TemporalGraphSAGEMeanMemory
from train_temporal_graphsage import (DROPOUT, GRAD_CLIP, HIDDEN, LR, MAX_EPOCHS, PATIENCE, SEEDS, TBPTT_DAYS,
                                      WEIGHT_DECAY, train_epoch)

EXPERIMENT = "temporal_ablation/memory_ablated_temporal"


def run_seed(seed, ds, raw, seg_by_id, cfg, dev, out_dir, max_epochs, patience, save):
    train_last, first_pred, last_pred = cfg.split.train_day_max, cfg.split.val_day_min, cfg.split.val_day_max
    print(f"\n{'='*70}\nSEED {seed} (memory-ablated temporal; train days 1-{train_last}, eval {first_pred}-{last_pred}, "
          f"max_epochs={max_epochs}, patience={patience})\n{'='*70}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TemporalGraphSAGEMeanMemory(raw["X"].shape[1], hidden=HIDDEN, dropout=DROPOUT).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    y_all = pd.Series(raw["y"], index=raw["txn_id"])
    tr_rows = (ds.txn_day <= train_last)
    y_tr = raw["y"][:len(tr_rows)][tr_rows]
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=dev)
    n_windows = len(range(0, len([d for d in ds.days if d <= train_last]), TBPTT_DAYS))
    print(f"  params={n_params:,} | optimizer steps/epoch={n_windows}", flush=True)
    assert n_windows == 18

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_pr, best_epoch, patience_left, best_state = -1.0, -1, patience, None
    history, stopped_early, total_steps = [], False, 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        te = time.time()
        loss = train_epoch(model, ds, opt, train_last, pos_weight)
        total_steps += n_windows
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - te
        tv = time.time()
        ids, scores, _ = rolling_temporal_predict(model, ds, first_pred, last_pred)
        y_true = y_all.loc[ids].to_numpy()
        pr, roc = average_precision_score(y_true, scores), roc_auc_score(y_true, scores)
        t_eval = time.time() - tv
        history.append({"epoch": epoch, "train_loss": loss, "val_pr_auc": float(pr), "val_roc_auc": float(roc),
                        "epoch_seconds": round(time.time() - te, 3), "train_seconds": round(t_train, 3), "eval_seconds": round(t_eval, 3)})
        improved = pr > best_pr + 1e-4
        if epoch % 5 == 0 or improved or epoch <= 2:
            print(f"  epoch {epoch:>3}: loss={loss:.4f}  VAL PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}  "
                  f"(train {t_train:.1f}s + eval {t_eval:.1f}s){' *' if improved else ''}", flush=True)
        if improved:
            best_pr, best_epoch, patience_left = pr, epoch, patience
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  EARLY STOPPING at epoch {epoch} (best epoch {best_epoch}, PR-AUC {best_pr:.4f})", flush=True)
                stopped_early = True
                break
    wall = time.time() - t0
    model.load_state_dict(best_state)
    print(f"  done in {wall:.1f}s: {len(history)} epochs ({'early-stopped' if stopped_early else 'hit max_epochs - not converged'}), "
          f"best epoch {best_epoch}, {total_steps} optimizer steps", flush=True)
    resource = {"parameter_count": n_params, "training_wall_seconds": round(wall, 2), "epochs_run": len(history),
                "best_epoch": best_epoch, "stopped_early": stopped_early, "optimizer_steps_per_epoch": n_windows,
                "total_optimizer_steps": total_steps, "optimizer_steps_to_best_epoch": best_epoch * n_windows,
                "mean_seconds_per_epoch": round(wall / len(history), 3)}
    if dev.type == "cuda":
        resource["peak_gpu_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        resource["peak_gpu_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    if not save:
        return {"seed": seed, "history": history, "resource": resource}

    ids, scores, days_arr = rolling_temporal_predict(model, ds, first_pred, last_pred)
    y_true = y_all.loc[ids].to_numpy()
    seg = seg_by_id.loc[ids].to_numpy()
    metrics, th = final_metrics(ids, y_true, scores, seg)
    print(f"  FINAL: pooled PR-AUC={metrics['pooled']['pr_auc']:.4f}  cold={metrics['has_edge_cold']['pr_auc']:.4f}  "
          f"warm={metrics['has_edge_warm']['pr_auc']:.4f}  no-edge={metrics['no_pseudo_account_edge']['pr_auc']:.4f}", flush=True)
    cfgd = {"seed": seed, "arm": "C_memory_ablated_temporal", "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "max_epochs": max_epochs, "patience": patience, "tbptt_days": TBPTT_DAYS,
            "grad_clip": GRAD_CLIP, "optimizer_steps_per_epoch": n_windows, "scale_pos_weight": float(pos_weight),
            "parameter_count": n_params, "memory": "cumulative-mean (order-agnostic), MLP 64-192-64 per entity type"}
    save_seed(out_dir, ids, y_true, scores, days_arr, seg, th, metrics, history, resource, best_state, cfgd)
    return {"seed": seed, "history": history, "resource": resource, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    cfg = load_config()
    base_dir = cfg.experiments_dir / EXPERIMENT
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    raw, graph, scaler = load_raw_arrays(cfg)
    ds = build_temporal_dataset(raw, cfg.split.val_day_max, dev, cfg)
    print(f"device={dev}; dataset {ds.X.shape[0]:,} txns (days 1-{ds.max_day}); TEST rows never loaded", flush=True)
    seg_by_id = pd.Series(compute_pseudo_account_segments(graph), index=graph["transaction"].transaction_id.numpy())
    if args.smoke:
        r = run_seed(42, ds, raw, seg_by_id, cfg, dev, None, 2, 20, save=False)
        print("SMOKE (2 epochs, not a result):", r["resource"], flush=True)
        return
    base_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, base_dir / "feature_scaler.joblib")
    results = []
    for seed in SEEDS:
        results.append(run_seed(seed, ds, raw, seg_by_id, cfg, dev, base_dir / f"seed_{seed}", MAX_EPOCHS, PATIENCE, save=True))
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    aggregate_and_plot(results, base_dir, "Temporal (cumulative-mean memory)", SEEDS,
                       {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "tbptt_days": TBPTT_DAYS, "grad_clip": GRAD_CLIP,
                        "optimizer_steps_per_epoch": 18}, t_start,
                       {"arm": "C_memory_ablated_temporal", "regime": "Phase 7 pipeline/regime (18 windowed steps/epoch, clip 1.0), order-agnostic cumulative-mean memory"})
    print(f"\nTotal runtime {time.time()-t_start:.1f}s. TEST split was not loaded or touched.", flush=True)


if __name__ == "__main__":
    main()
