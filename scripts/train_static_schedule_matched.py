"""Phase 7A arm B: schedule-matched STATIC GraphSAGE.

Phase 6A architecture (HeteroSAGE, 162,817 params), graph, features, split, loss, class weighting, optimizer,
lr, weight decay, rolling per-day evaluation, 150 epochs / patience 20 are all unchanged. ONLY the training
regime changes, to match Phase 7's: each epoch is 18 optimizer steps, one per 7-day window of TRAIN days
([1-7] ... [113-119], [120-121]) in chronological order; each step = full forward over the TRAIN-masked graph
(edge_time <= 121), BCE(pos_weight) over that window's transactions only, backward, grad-norm clip 1.0, Adam step.
See experiments/temporal_ablation/DESIGN_BEFORE_EXECUTION.md.

  python -u scripts/train_static_schedule_matched.py            # full 3-seed experiment
  python -u scripts/train_static_schedule_matched.py --smoke    # 2-epoch timing check, writes to scratchpad only
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from phase7a_common import aggregate_and_plot, final_metrics, save_seed
from src.config import load_config
from src.graph.build import apply_time_mask
from src.graph.rolling_eval import rolling_predict
from src.graph.schema import compute_pseudo_account_segments
from src.models.graphsage import HeteroSAGE
from train_graphsage_robustness import build_scaled_graph   # identical Phase 6A graph/scaling helper (import-safe)

SEEDS = [42, 123, 2024]
HIDDEN, DROPOUT, LR, WEIGHT_DECAY = 64, 0.3, 1e-3, 1e-5
MAX_EPOCHS, PATIENCE = 150, 20
WINDOW_DAYS, GRAD_CLIP = 7, 1.0
EXPERIMENT = "temporal_ablation/static_schedule_matched"


def make_windows(txn_day: torch.Tensor, train_last: int):
    days = sorted(int(d) for d in torch.unique(txn_day[txn_day <= train_last]).tolist())
    wins = []
    for w0 in range(0, len(days), WINDOW_DAYS):
        wd = days[w0:w0 + WINDOW_DAYS]
        wins.append((wd[0], wd[-1], ((txn_day >= wd[0]) & (txn_day <= wd[-1])).nonzero(as_tuple=True)[0]))
    return wins


def run_seed(seed, graph, seg_by_id, cfg, dev, out_dir, max_epochs, patience, save):
    print(f"\n{'='*70}\nSEED {seed} (schedule-matched static; max_epochs={max_epochs}, patience={patience})\n{'='*70}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    model = HeteroSAGE(graph.metadata(), feature_dims, hidden=HIDDEN, dropout=DROPOUT).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    txn_day, y_full = graph["transaction"].day, graph["transaction"].y
    train_idx = (txn_day <= cfg.split.train_day_max).nonzero(as_tuple=True)[0]
    pos_weight = torch.tensor(float((y_full[train_idx] == 0).sum() / max((y_full[train_idx] == 1).sum(), 1)), device=dev)
    train_graph = apply_time_mask(graph, cfg.split.train_day_max)
    wins = make_windows(txn_day, cfg.split.train_day_max)
    steps_per_epoch = len(wins)
    print(f"  params={n_params:,} | optimizer steps/epoch={steps_per_epoch} | windows: "
          f"{[(a, b) for a, b, _ in wins][:2]} ... {[(a, b) for a, b, _ in wins][-2:]}", flush=True)
    assert steps_per_epoch == 18
    val_days = list(range(cfg.split.val_day_min, cfg.split.val_day_max + 1))
    y_by_id = pd.Series(y_full.cpu().numpy(), index=graph["transaction"].transaction_id.cpu().numpy())

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_pr, best_epoch, patience_left, best_state = -1.0, -1, patience, None
    history, stopped_early, total_steps = [], False, 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        te = time.time()
        model.train()
        tot, n_tot = 0.0, 0
        for _, _, idx in wins:                                   # chronological windows, one optimizer step each
            logits = model(train_graph.x_dict, train_graph.edge_index_dict)
            loss = F.binary_cross_entropy_with_logits(logits[idx], y_full[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total_steps += 1
            tot += float(loss.item()) * idx.numel()
            n_tot += idx.numel()
        loss_ep = tot / n_tot
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - te
        tv = time.time()
        ids, scores, _ = rolling_predict(model, graph, val_days)
        y_true = y_by_id.loc[ids].to_numpy()
        pr, roc = average_precision_score(y_true, scores), roc_auc_score(y_true, scores)
        history.append({"epoch": epoch, "train_loss": loss_ep, "val_pr_auc": float(pr), "val_roc_auc": float(roc),
                        "epoch_seconds": round(time.time() - te, 3), "train_seconds": round(t_train, 3),
                        "eval_seconds": round(time.time() - tv, 3)})
        improved = pr > best_pr + 1e-4
        if epoch % 5 == 0 or improved or epoch <= 2:
            print(f"  epoch {epoch:>3}: loss={loss_ep:.4f}  VAL PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}  "
                  f"(train {t_train:.1f}s + eval {time.time()-tv:.1f}s){' *' if improved else ''}", flush=True)
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
                "best_epoch": best_epoch, "stopped_early": stopped_early, "optimizer_steps_per_epoch": steps_per_epoch,
                "total_optimizer_steps": total_steps, "optimizer_steps_to_best_epoch": best_epoch * steps_per_epoch,
                "mean_seconds_per_epoch": round(wall / len(history), 3)}
    if dev.type == "cuda":
        resource["peak_gpu_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        resource["peak_gpu_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    if not save:
        return {"seed": seed, "history": history, "resource": resource}

    ids, scores, days_arr = rolling_predict(model, graph, val_days)
    y_true = y_by_id.loc[ids].to_numpy()
    seg = seg_by_id.loc[ids].to_numpy()
    metrics, th = final_metrics(ids, y_true, scores, seg)
    print(f"  FINAL: pooled PR-AUC={metrics['pooled']['pr_auc']:.4f}  cold={metrics['has_edge_cold']['pr_auc']:.4f}  "
          f"warm={metrics['has_edge_warm']['pr_auc']:.4f}  no-edge={metrics['no_pseudo_account_edge']['pr_auc']:.4f}", flush=True)
    cfgd = {"seed": seed, "arm": "B_static_schedule_matched", "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "max_epochs": max_epochs, "patience": patience, "window_days": WINDOW_DAYS, "grad_clip": GRAD_CLIP,
            "optimizer_steps_per_epoch": steps_per_epoch, "scale_pos_weight": float(pos_weight), "parameter_count": n_params}
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
    graph, scaler = build_scaled_graph(cfg)
    graph = graph.to(dev)
    print(f"device={dev}", flush=True)
    seg_by_id = pd.Series(compute_pseudo_account_segments(graph), index=graph["transaction"].transaction_id.cpu().numpy())
    if args.smoke:
        r = run_seed(42, graph, seg_by_id, cfg, dev, None, 2, 20, save=False)
        print("SMOKE (2 epochs, not a result):", r["resource"], flush=True)
        return
    base_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        results.append(run_seed(seed, graph, seg_by_id, cfg, dev, base_dir / f"seed_{seed}", MAX_EPOCHS, PATIENCE, save=True))
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    aggregate_and_plot(results, base_dir, "Static GraphSAGE (schedule-matched)", SEEDS,
                       {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "window_days": WINDOW_DAYS, "grad_clip": GRAD_CLIP,
                        "optimizer_steps_per_epoch": 18}, t_start,
                       {"arm": "B_static_schedule_matched", "regime": "18 windowed optimizer steps/epoch, grad clip 1.0, full-graph forward on TRAIN-masked graph"})
    print(f"\nTotal runtime {time.time()-t_start:.1f}s. TEST split was not used for any decision.", flush=True)


if __name__ == "__main__":
    main()
