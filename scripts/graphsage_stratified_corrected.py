"""Corrected cold-start stratification for the GraphSAGE run.

train_graphsage.py's own stratification conflated two different things into
one "cold_start" bucket: transactions with a genuinely new pseudo_account
(has_edge & not warm) and transactions with NO pseudo_account mapping at all
(missing addr1/D1 — 11.1% of rows, strongly correlated with ProductCD='C',
Phase 1 Section 6). The second group has a 12.5% base fraud rate (vs. ~2%
for the other two buckets) and dominated the flawed "cold_start" number
(1,044 of its 1,522 frauds). This script re-derives the honest 3-way split
directly from the saved predictions, without retraining anything.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.models.metrics import compute_metrics


def main() -> None:
    cfg = load_config()
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    day = graph["transaction"].day.numpy()
    txn_id = graph["transaction"].transaction_id.numpy()
    rel = next(et for et in graph.edge_types if et[0] == "transaction" and et[2] == "pseudo_account")
    ei = graph[rel].edge_index.numpy()
    key_of_txn = np.full(len(day), -1, dtype=np.int64)
    key_of_txn[ei[0]] = ei[1]
    has_edge = key_of_txn != -1

    order = np.argsort(day, kind="stable")
    key_sorted = key_of_txn[order]
    present_sorted = key_sorted != -1
    s = pd.Series(np.where(present_sorted, key_sorted, np.nan))
    prior_count = s.groupby(s).cumcount()
    warm_sorted = present_sorted & (prior_count.fillna(0).to_numpy() > 0)
    warm = np.zeros(len(day), dtype=bool)
    warm[order] = warm_sorted

    flags = pd.DataFrame({"TransactionID": txn_id, "has_edge": has_edge, "warm": warm})

    preds = pd.read_parquet(cfg.experiments_dir / "graphsage" / "predictions_val.parquet")
    preds = preds.merge(flags, on="TransactionID", how="left")

    def bucket(row):
        if not row["has_edge"]:
            return "no_pseudo_account_edge"
        return "has_edge_warm" if row["warm"] else "has_edge_cold"

    preds["bucket"] = preds.apply(bucket, axis=1)

    with open(cfg.experiments_dir / "graphsage" / "metrics_val.json", "r", encoding="utf-8") as fh:
        existing = json.load(fh)
    threshold = existing["pooled"]["threshold_selection"]["threshold"]

    print("=== Corrected 3-way stratification (VAL, GraphSAGE predictions) ===")
    results = {}
    for name, sub in preds.groupby("bucket"):
        yt, sc = sub["isFraud"].to_numpy(), sub["y_score"].to_numpy()
        n, nf = len(yt), int(yt.sum())
        if nf == 0 or nf == n:
            results[name] = {"n_rows": n, "n_fraud": nf, "note": "insufficient class balance"}
            print(f"  {name:<24} n={n:>6,} fraud={nf:>5,} (insufficient class balance for PR/ROC-AUC)")
            continue
        m = compute_metrics(yt, sc, threshold=threshold)
        results[name] = m
        print(f"  {name:<24} n={n:>6,} fraud={nf:>5,} rate={nf/n*100:.3f}%  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

    # Also: a "true cold-start" combined view = has_edge_cold + no_pseudo_account_edge
    # is NOT reported as one number here, deliberately — see script docstring;
    # they are different populations for different reasons and conflating them
    # is exactly the mistake being corrected.

    combined_true_cold = preds[preds["bucket"] == "has_edge_cold"]
    combined_warm = preds[preds["bucket"] == "has_edge_warm"]
    print(f"\n  For direct comparison with Phase 3's cold-start number (XGBoost 0.5592, historical control 0.1018):")
    print(f"  GraphSAGE TRUE cold-start (has_edge_cold only): PR-AUC={results['has_edge_cold']['pr_auc']:.4f}")
    print(f"  GraphSAGE warm-start (has_edge_warm only)      : PR-AUC={results['has_edge_warm']['pr_auc']:.4f}")

    out_path = cfg.experiments_dir / "graphsage" / "metrics_val_stratified_corrected.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
