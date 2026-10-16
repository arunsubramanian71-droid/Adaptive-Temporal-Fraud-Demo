"""Phase 5: the official GraphSAGE fraud-detection experiment.

TRAIN (days 1-121) -> fit, using one static TRAIN-masked graph view.
VALIDATION (days 122-151) -> rolling per-day evaluation (src/graph/rolling_eval.py),
used for early stopping and threshold selection. TEST is never touched.

Run: .venv/Scripts/python.exe scripts/train_graphsage.py
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.graph.build import apply_time_mask
from src.graph.rolling_eval import rolling_predict
from src.graph.schema import compute_pseudo_account_segments, load_graph_config
from src.models.graphsage import HeteroSAGE
from src.models.metrics import compute_metrics, select_best_f1_threshold

RANDOM_STATE = 42
EXPERIMENT_NAME = "graphsage"
HIDDEN = 64
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 60
PATIENCE = 10


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    t_start = time.time()

    cfg = load_config()
    gcfg = load_graph_config()
    exp_dir = cfg.experiments_dir / EXPERIMENT_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading Phase 4 graph (read-only — not modified or overwritten) ===")
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    print(f"  node types: {graph.node_types}")
    print(f"  {sum(graph[nt].num_nodes for nt in graph.node_types):,} nodes, "
          f"{sum(graph[et].edge_index.shape[1] for et in graph.edge_types):,} edges")

    print("\n=== 2. Feature normalization (StandardScaler, fit on TRAIN transaction rows only) ===")
    day = graph["transaction"].day.numpy()
    train_mask_np = (day >= cfg.split.train_day_min) & (day <= cfg.split.train_day_max)
    x = graph["transaction"].x.numpy()
    scaler = StandardScaler().fit(x[train_mask_np])
    graph["transaction"].x = torch.tensor(scaler.transform(x), dtype=torch.float32)
    print(f"  fit on {train_mask_np.sum():,} TRAIN rows; applied to all {len(x):,} transaction nodes")
    print("  (Phase 4's saved hetero_graph.pt on disk is untouched — this is an in-memory copy for training)")

    print("\n=== 3. Resource discipline: estimates before the official run ===")
    feature_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    model = HeteroSAGE(graph.metadata(), feature_dims, hidden=HIDDEN, dropout=DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  feature dims: {feature_dims}")
    print(f"  parameter count: {n_params:,}")
    print(f"  estimated GPU memory: ~4.5 GiB per forward/backward step (Phase 4's measured feasibility "
          f"benchmark on this identical graph/hidden-size combination — reused, not re-estimated from scratch)")
    print(f"  expected runtime: up to {MAX_EPOCHS} epochs x (~1 train step + 30 rolling VAL passes); "
          f"Phase 4 measured ~1.6s/train-step and ~85ms/rolling-pass -> a full epoch with VAL eval is "
          f"~seconds, so the whole run is expected in low single-digit minutes. Proceeding automatically "
          f"per the Phase 5 execution policy.")

    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\n=== 4. Moving graph + model to {dev} (once — masking then stays on-device) ===")
    graph = graph.to(dev)
    model = model.to(dev)

    train_graph = apply_time_mask(graph, cfg.split.train_day_max)  # static view, reused every epoch
    train_idx = (graph["transaction"].day <= cfg.split.train_day_max).nonzero(as_tuple=True)[0]
    y_train_full = graph["transaction"].y

    scale_pos_weight = float((y_train_full[train_idx] == 0).sum() / max((y_train_full[train_idx] == 1).sum(), 1))
    pos_weight = torch.tensor(scale_pos_weight, device=dev)
    print(f"  scale_pos_weight={scale_pos_weight:.2f} on TRAIN (class-weighted BCE loss; NOT accuracy-optimized, NOT SMOTE)")

    val_days = list(range(cfg.split.val_day_min, cfg.split.val_day_max + 1))
    y_true_by_id = pd.Series(graph["transaction"].y.cpu().numpy(), index=graph["transaction"].transaction_id.cpu().numpy())

    print("\n=== 5. Training (early stopping on rolling VAL PR-AUC — VALIDATION only, TEST untouched) ===")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_pr_auc, best_epoch, patience_left = -1.0, -1, PATIENCE
    best_state = None
    history = []
    t_train0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(train_graph.x_dict, train_graph.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(logits[train_idx], y_train_full[train_idx], pos_weight=pos_weight)
        loss.backward()
        opt.step()

        ids, scores, _ = rolling_predict(model, graph, val_days)
        y_true = y_true_by_id.loc[ids].to_numpy()
        val_pr_auc = average_precision_score(y_true, scores)
        t_ep = time.time() - t_ep0

        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": float(val_pr_auc),
                        "epoch_seconds": round(t_ep, 2)})
        improved = val_pr_auc > best_pr_auc + 1e-4
        marker = " *" if improved else ""
        print(f"  epoch {epoch:>3}: loss={loss.item():.4f}  VAL PR-AUC (rolling, pooled)={val_pr_auc:.4f}  "
              f"({t_ep:.2f}s){marker}")
        if improved:
            best_pr_auc, best_epoch, patience_left = val_pr_auc, epoch, PATIENCE
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping: no improvement for {PATIENCE} epochs (best epoch {best_epoch}, PR-AUC {best_pr_auc:.4f})")
                break

    t_train = time.time() - t_train0
    model.load_state_dict(best_state)
    print(f"\n  training complete in {t_train:.1f}s, {len(history)} epochs run, best epoch={best_epoch} (VAL PR-AUC {best_pr_auc:.4f})")

    print("\n=== 6. Final rolling VAL evaluation with best checkpoint ===")
    ids, scores, days_arr = rolling_predict(model, graph, val_days)
    y_true = y_true_by_id.loc[ids].to_numpy()
    thresh_info = select_best_f1_threshold(y_true, scores)
    print(f"  best-F1 threshold: {thresh_info['threshold']:.4f} "
          f"(F1={thresh_info['f1']:.4f}, P={thresh_info['precision']:.4f}, R={thresh_info['recall']:.4f})")
    metrics_pooled = compute_metrics(y_true, scores, threshold=thresh_info["threshold"])
    metrics_pooled["threshold_selection"] = thresh_info
    print(f"  POOLED: PR-AUC={metrics_pooled['pr_auc']:.4f}  ROC-AUC={metrics_pooled['roc_auc']:.4f}  "
          f"Precision={metrics_pooled['precision']:.4f}  Recall={metrics_pooled['recall']:.4f}  F1={metrics_pooled['f1']:.4f}")
    for k, v in metrics_pooled["top_k"].items():
        print(f"    top {k:>6}: alerts={v['k_alerts']:>6,}  fraud_captured={v['fraud_captured']:>5,}  "
              f"P@K={v['precision_at_k']:.4f}  R@K={v['recall_at_k']:.4f}")

    print("\n=== 7. Stratification: has_edge_warm / has_edge_cold / no_pseudo_account_edge ===")
    print("  (3-way, not 2-way — see compute_pseudo_account_segments docstring for why 'no edge at all' "
          "must be reported separately from genuine cold-start rather than merged with it)")
    segment_all = compute_pseudo_account_segments(graph)
    txn_id_all = graph["transaction"].transaction_id.cpu().numpy()
    segment_by_id = pd.Series(segment_all, index=txn_id_all)
    seg = segment_by_id.loc[ids].to_numpy()
    is_warm = seg == "has_edge_warm"  # kept for predictions_val.parquet's convenience column

    metrics_by_segment = {"pooled": metrics_pooled}
    for seg_name in ["has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"]:
        seg_mask = seg == seg_name
        yt, sc = y_true[seg_mask], scores[seg_mask]
        if len(yt) == 0 or yt.sum() == 0 or yt.sum() == len(yt):
            metrics_by_segment[seg_name] = {"n_rows": int(len(yt)), "n_fraud": int(yt.sum()), "note": "insufficient class balance for PR/ROC-AUC"}
            continue
        m = compute_metrics(yt, sc, threshold=thresh_info["threshold"])
        metrics_by_segment[seg_name] = m
        print(f"  {seg_name:<24}: n={m['n_rows']:>6,} fraud={m['n_fraud']:>5,} rate={m['fraud_rate']*100:.3f}%  "
              f"PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

    print("\n=== 8. Saving artifacts (Phase 3/4 artifacts left untouched) ===")
    torch.save(best_state, exp_dir / "model_checkpoint.pt")
    joblib.dump(scaler, exp_dir / "feature_scaler.joblib")

    config_record = {
        "experiment": EXPERIMENT_NAME,
        "random_state": RANDOM_STATE,
        "architecture": {"type": "HeteroSAGE", "hidden": HIDDEN, "layers": 2, "dropout": DROPOUT,
                         "activation": "ReLU", "aggr": "sum", "parameter_count": n_params},
        "training": {"optimizer": "Adam", "lr": LR, "weight_decay": WEIGHT_DECAY, "loss": "BCEWithLogits",
                    "scale_pos_weight": scale_pos_weight, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                    "best_epoch": best_epoch, "epochs_run": len(history)},
        "split": {"train_days": [cfg.split.train_day_min, cfg.split.train_day_max],
                  "val_days": [cfg.split.val_day_min, cfg.split.val_day_max],
                  "test_days": [cfg.split.test_day_min, cfg.split.test_day_max]},
        "temporal_evaluation": "rolling per-day cutoff (src/graph/rolling_eval.py): each day t scored using "
                              "graph masked to edge_time<=t, keeping only that day's own transaction predictions",
        "feature_dims": feature_dims,
        "device": str(dev),
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config_record, fh, indent=1)

    with open(exp_dir / "metrics_val.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_by_segment, fh, indent=1)

    with open(exp_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)

    preds = pd.DataFrame({"TransactionID": ids, "isFraud": y_true, "y_score": scores, "day": days_arr,
                          "y_pred": (scores >= thresh_info["threshold"]).astype(int), "segment": seg})
    preds.to_parquet(exp_dir / "predictions_val.parquet", index=False)

    resource_measurements = {
        "parameter_count": n_params,
        "training_wall_seconds": round(t_train, 2),
        "epochs_run": len(history),
        "seconds_per_epoch_mean": round(t_train / max(len(history), 1), 3),
        "device": str(dev),
    }
    if dev.type == "cuda":
        resource_measurements["peak_gpu_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        resource_measurements["peak_gpu_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    with open(exp_dir / "resource_measurements.json", "w", encoding="utf-8") as fh:
        json.dump(resource_measurements, fh, indent=1)

    metadata = {
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": pyg.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "config_path": "configs/data.yaml",
        "graph_config_path": "configs/graph.yaml",
        "graph_source": "data/processed/graph/hetero_graph.pt (Phase 4, unmodified)",
        "total_script_seconds": round(time.time() - t_start, 2),
    }
    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=1)

    for f in ["model_checkpoint.pt", "feature_scaler.joblib", "config.json", "metrics_val.json",
              "training_history.json", "predictions_val.parquet", "resource_measurements.json", "metadata.json"]:
        print(f"  wrote {exp_dir / f}")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s. TEST split was not loaded or touched.")


if __name__ == "__main__":
    main()
