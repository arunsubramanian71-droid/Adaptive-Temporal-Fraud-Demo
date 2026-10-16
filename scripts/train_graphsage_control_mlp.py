"""Phase 5 ablation item B: non-relational control.

A plain 2-hidden-layer MLP on the IDENTICAL 470-dim transaction feature
vector GraphSAGE uses (same StandardScaler, loaded from the GraphSAGE run
for byte-identical preprocessing), same hidden width (64), same dropout,
same class-weighted loss, same optimizer/training recipe — the only
difference is zero graph structure. This isolates the graph's marginal
contribution while holding everything else fixed.

Why this design rather than "GraphSAGE with edges removed": zeroing out
edges inside the same HeteroConv architecture leaves every entity node with
no neighbors at all, which is a degenerate case for SAGEConv's neighbor-
aggregation term (not a clean "no relational information" condition, just
an edge case of the same conv). A separately-trained, equal-capacity MLP is
the methodologically clean choice.

No rolling per-day evaluation is needed here — an MLP has no notion of a
graph, so there is no structural-leakage channel to guard against. This is
noted explicitly rather than silently reusing the rolling machinery for
something that doesn't need it.
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

from src.config import load_config
from src.models.graphsage import MLPControl
from src.models.metrics import compute_metrics, select_best_f1_threshold

RANDOM_STATE = 42
EXPERIMENT_NAME = "graphsage_control_mlp"
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
    exp_dir = cfg.experiments_dir / EXPERIMENT_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading the same transaction features GraphSAGE used (same scaler, same graph file) ===")
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    scaler = joblib.load(cfg.experiments_dir / "graphsage" / "feature_scaler.joblib")
    x = scaler.transform(graph["transaction"].x.numpy())
    y = graph["transaction"].y.numpy()
    day = graph["transaction"].day.numpy()
    txn_id = graph["transaction"].transaction_id.numpy()
    print(f"  reused scaler fit on GraphSAGE's TRAIN rows — byte-identical preprocessing for a clean ablation")

    train_mask = (day >= cfg.split.train_day_min) & (day <= cfg.split.train_day_max)
    val_mask = (day >= cfg.split.val_day_min) & (day <= cfg.split.val_day_max)
    print(f"  train: {train_mask.sum():,} rows, val: {val_mask.sum():,} rows (TEST not touched)")

    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    X_train = torch.tensor(x[train_mask], dtype=torch.float32, device=dev)
    y_train = torch.tensor(y[train_mask], dtype=torch.float32, device=dev)
    X_val = torch.tensor(x[val_mask], dtype=torch.float32, device=dev)
    y_val_np = y[val_mask]

    model = MLPControl(in_dim=x.shape[1], hidden=HIDDEN, dropout=DROPOUT).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== 2. Resource discipline ===")
    print(f"  feature dim: {x.shape[1]}  |  parameter count: {n_params:,} (HeteroSAGE had 162,817 — comparable "
          f"capacity, no graph)")
    print(f"  estimated GPU memory: negligible (<0.5 GiB — no entity nodes, no message passing)")
    print(f"  expected runtime: seconds — dense matmuls on ~418K x 470, far cheaper than graph message passing")

    scale_pos_weight = float((y[train_mask] == 0).sum() / max((y[train_mask] == 1).sum(), 1))
    pos_weight = torch.tensor(scale_pos_weight, device=dev)
    print(f"  scale_pos_weight={scale_pos_weight:.2f} (same imbalance treatment as GraphSAGE)")

    print("\n=== 3. Training ===")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_pr_auc, best_epoch, patience_left, best_state = -1.0, -1, PATIENCE, None
    history = []
    t_train0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(X_val)).cpu().numpy()
        val_pr_auc = average_precision_score(y_val_np, val_scores)
        t_ep = time.time() - t_ep0
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": float(val_pr_auc), "epoch_seconds": round(t_ep, 3)})
        improved = val_pr_auc > best_pr_auc + 1e-4
        if epoch % 10 == 0 or improved:
            print(f"  epoch {epoch:>3}: loss={loss.item():.4f}  VAL PR-AUC={val_pr_auc:.4f}  ({t_ep:.3f}s){' *' if improved else ''}")
        if improved:
            best_pr_auc, best_epoch, patience_left = val_pr_auc, epoch, PATIENCE
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch} (best epoch {best_epoch}, PR-AUC {best_pr_auc:.4f})")
                break
    t_train = time.time() - t_train0
    model.load_state_dict(best_state)
    print(f"  training complete in {t_train:.1f}s, {len(history)} epochs, best epoch={best_epoch} (VAL PR-AUC {best_pr_auc:.4f})")

    print("\n=== 4. Final VAL evaluation (single pass — no rolling needed, no graph dependency) ===")
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_val)).cpu().numpy()
    thresh_info = select_best_f1_threshold(y_val_np, scores)
    metrics_pooled = compute_metrics(y_val_np, scores, threshold=thresh_info["threshold"])
    metrics_pooled["threshold_selection"] = thresh_info
    print(f"  POOLED: PR-AUC={metrics_pooled['pr_auc']:.4f}  ROC-AUC={metrics_pooled['roc_auc']:.4f}  F1={metrics_pooled['f1']:.4f}")

    print("\n=== 5. Same 3-way stratification as GraphSAGE, for direct comparison ===")
    from src.graph.schema import compute_pseudo_account_segments
    segment_all = compute_pseudo_account_segments(graph)
    seg = segment_all[val_mask]
    metrics_by_segment = {"pooled": metrics_pooled}
    for seg_name in ["has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"]:
        m = seg == seg_name
        yt, sc = y_val_np[m], scores[m]
        if len(yt) == 0 or yt.sum() == 0 or yt.sum() == len(yt):
            metrics_by_segment[seg_name] = {"n_rows": int(len(yt)), "n_fraud": int(yt.sum()), "note": "insufficient class balance"}
            continue
        mm = compute_metrics(yt, sc, threshold=thresh_info["threshold"])
        metrics_by_segment[seg_name] = mm
        print(f"  {seg_name:<24}: n={mm['n_rows']:>6,} fraud={mm['n_fraud']:>5,} PR-AUC={mm['pr_auc']:.4f}  ROC-AUC={mm['roc_auc']:.4f}")

    print("\n=== 6. Saving artifacts ===")
    torch.save(best_state, exp_dir / "model_checkpoint.pt")
    with open(exp_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump({"experiment": EXPERIMENT_NAME, "random_state": RANDOM_STATE,
                   "architecture": {"type": "MLPControl", "hidden": HIDDEN, "layers": 2, "dropout": DROPOUT,
                                    "parameter_count": n_params},
                   "training": {"optimizer": "Adam", "lr": LR, "weight_decay": WEIGHT_DECAY,
                               "scale_pos_weight": scale_pos_weight, "best_epoch": best_epoch, "epochs_run": len(history)},
                   "note": "reuses GraphSAGE's fitted feature scaler for a byte-identical preprocessing comparison"}, fh, indent=1)
    with open(exp_dir / "metrics_val.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_by_segment, fh, indent=1)
    with open(exp_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)
    preds = pd.DataFrame({"TransactionID": txn_id[val_mask], "isFraud": y_val_np, "y_score": scores,
                          "y_pred": (scores >= thresh_info["threshold"]).astype(int), "segment": seg})
    preds.to_parquet(exp_dir / "predictions_val.parquet", index=False)
    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"timestamp_utc": pd.Timestamp.now("UTC").isoformat(), "python_version": platform.python_version(),
                   "torch_version": torch.__version__, "torch_geometric_version": pyg.__version__,
                   "total_script_seconds": round(time.time() - t_start, 2)}, fh, indent=1)

    for f in ["model_checkpoint.pt", "config.json", "metrics_val.json", "training_history.json",
              "predictions_val.parquet", "metadata.json"]:
        print(f"  wrote {exp_dir / f}")
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
