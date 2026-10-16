"""Phase 6A fairness addition: the non-relational MLP control, retrained at
the SAME extended epoch budget/patience as the Phase 6A GraphSAGE sweep.

Phase 5's original experiments/graphsage_control_mlp/ (60 epochs, seed 42,
also never early-stopped) is left completely untouched — this writes to a
separate directory. Reason this exists: comparing a newly-converged
150-epoch GraphSAGE against a stale, admittedly-underfit 60-epoch MLP
control would unfairly bias the "does the graph add signal" conclusion.
Trivially cheap (~6s at ~0.04s/epoch), so there's no resource-discipline
reason not to do it.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.graph.schema import compute_pseudo_account_segments
from src.models.graphsage import MLPControl
from src.models.metrics import compute_metrics, select_best_f1_threshold

RANDOM_STATE = 42
EXPERIMENT_NAME = "graphsage_robust"
HIDDEN = 64
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 150
PATIENCE = 20


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    t_start = time.time()

    cfg = load_config()
    out_dir = cfg.experiments_dir / EXPERIMENT_NAME / "mlp_control_extended"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading identical features/scaler as Phase 5's MLP control (same graph file) ===")
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    scaler = joblib.load(cfg.experiments_dir / "graphsage" / "feature_scaler.joblib")
    x = scaler.transform(graph["transaction"].x.numpy())
    y = graph["transaction"].y.numpy()
    day = graph["transaction"].day.numpy()
    txn_id = graph["transaction"].transaction_id.numpy()
    segment_all = compute_pseudo_account_segments(graph)

    train_mask = (day >= cfg.split.train_day_min) & (day <= cfg.split.train_day_max)
    val_mask = (day >= cfg.split.val_day_min) & (day <= cfg.split.val_day_max)

    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    X_train = torch.tensor(x[train_mask], dtype=torch.float32, device=dev)
    y_train = torch.tensor(y[train_mask], dtype=torch.float32, device=dev)
    X_val = torch.tensor(x[val_mask], dtype=torch.float32, device=dev)
    y_val_np = y[val_mask]
    seg_val = segment_all[val_mask]

    model = MLPControl(in_dim=x.shape[1], hidden=HIDDEN, dropout=DROPOUT).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    scale_pos_weight = float((y[train_mask] == 0).sum() / max((y[train_mask] == 1).sum(), 1))
    pos_weight = torch.tensor(scale_pos_weight, device=dev)
    print(f"  parameter count: {n_params:,}  |  max_epochs={MAX_EPOCHS}, patience={PATIENCE} "
          f"(same budget as the Phase 6A GraphSAGE sweep, for a fair comparison)")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_pr_auc, best_epoch, patience_left, best_state = -1.0, -1, PATIENCE, None
    history = []
    t_train0 = time.time()
    stopped_early = False
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
        val_roc_auc = roc_auc_score(y_val_np, val_scores)
        t_ep = time.time() - t_ep0
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": float(val_pr_auc),
                        "val_roc_auc": float(val_roc_auc), "epoch_seconds": round(t_ep, 4)})
        improved = val_pr_auc > best_pr_auc + 1e-4
        if epoch % 20 == 0 or improved and epoch <= 5:
            print(f"  epoch {epoch:>3}: loss={loss.item():.4f}  VAL PR-AUC={val_pr_auc:.4f}  VAL ROC-AUC={val_roc_auc:.4f}")
        if improved:
            best_pr_auc, best_epoch, patience_left = val_pr_auc, epoch, PATIENCE
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  EARLY STOPPING at epoch {epoch} (best epoch {best_epoch})")
                stopped_early = True
                break
    t_train = time.time() - t_train0
    model.load_state_dict(best_state)
    print(f"  training complete in {t_train:.1f}s, {len(history)} epochs "
          f"({'early-stopped' if stopped_early else 'hit max_epochs'}), best epoch={best_epoch}, PR-AUC={best_pr_auc:.4f}")

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_val)).cpu().numpy()
    thresh_info = select_best_f1_threshold(y_val_np, scores)
    metrics_pooled = compute_metrics(y_val_np, scores, threshold=thresh_info["threshold"])
    metrics_pooled["threshold_selection"] = thresh_info

    metrics_by_segment = {"pooled": metrics_pooled}
    for seg_name in ["has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"]:
        m = seg_val == seg_name
        yt, sc = y_val_np[m], scores[m]
        if len(yt) == 0 or yt.sum() == 0 or yt.sum() == len(yt):
            metrics_by_segment[seg_name] = {"n_rows": int(len(yt)), "n_fraud": int(yt.sum()), "note": "insufficient class balance"}
            continue
        metrics_by_segment[seg_name] = compute_metrics(yt, sc, threshold=thresh_info["threshold"])

    print(f"  FINAL: pooled={metrics_pooled['pr_auc']:.4f}  cold={metrics_by_segment['has_edge_cold']['pr_auc']:.4f}  "
          f"warm={metrics_by_segment['has_edge_warm']['pr_auc']:.4f}  no-edge={metrics_by_segment['no_pseudo_account_edge']['pr_auc']:.4f}")

    torch.save(best_state, out_dir / "model_checkpoint.pt")
    with open(out_dir / "metrics_val.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_by_segment, fh, indent=1)
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump({"seed": RANDOM_STATE, "hidden": HIDDEN, "dropout": DROPOUT, "max_epochs": MAX_EPOCHS,
                   "patience": PATIENCE, "best_epoch": best_epoch, "epochs_run": len(history),
                   "stopped_early": stopped_early, "parameter_count": n_params,
                   "note": "Phase 6A fairness addition — same epoch budget as the GraphSAGE sweep; "
                           "original Phase 5 60-epoch result retained unchanged at experiments/graphsage_control_mlp/"}, fh, indent=1)
    preds = pd.DataFrame({"TransactionID": txn_id[val_mask], "isFraud": y_val_np, "y_score": scores,
                          "y_pred": (scores >= thresh_info["threshold"]).astype(int), "segment": seg_val})
    preds.to_parquet(out_dir / "predictions_val.parquet", index=False)

    print(f"\nWrote artifacts to {out_dir}")
    print(f"Total runtime: {time.time()-t_start:.1f}s. Phase 5's original MLP control artifact was not modified.")


if __name__ == "__main__":
    main()
