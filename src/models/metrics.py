"""Shared evaluation metrics and threshold selection.

This is the canonical comparison format: every baseline (Logistic Regression,
Random Forest, XGBoost, the historical-feature control) calls
`compute_metrics` and writes its result the same way, so
experiments/*/metrics_val.json files are directly comparable later without
each script inventing its own schema.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_K_FRACTIONS = (0.001, 0.005, 0.01, 0.05)


def select_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Threshold selection using ONLY the data passed in (must be VAL, never
    TEST — Phase 1 risk R8 / this project's explicit leakage rule)."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # prec/rec have one more element than thr (the last point is
    # threshold=+inf, recall=0); drop it so indices align with thr.
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    best = int(np.argmax(f1))
    return {"threshold": float(thr[best]), "f1": float(f1[best]), "precision": float(prec[best]), "recall": float(rec[best])}


def top_k_stats(y_true: np.ndarray, y_score: np.ndarray, k_fractions=DEFAULT_K_FRACTIONS) -> dict:
    n = len(y_true)
    n_fraud = int(y_true.sum())
    order = np.argsort(-y_score, kind="stable")
    y_sorted = np.asarray(y_true)[order]
    out = {}
    for frac in k_fractions:
        k = max(1, int(round(n * frac)))
        captured = int(y_sorted[:k].sum())
        out[f"{frac*100:g}%"] = {
            "k_alerts": k,
            "fraud_captured": captured,
            "precision_at_k": captured / k,
            "recall_at_k": (captured / n_fraud) if n_fraud else float("nan"),
        }
    return out


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float,
                     k_fractions=DEFAULT_K_FRACTIONS) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    n_fraud = int(y_true.sum())

    return {
        "n_rows": n,
        "n_fraud": n_fraud,
        "fraud_rate": n_fraud / n if n else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if n_fraud not in (0, n) else float("nan"),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "n_predicted_positive": int(y_pred.sum()),
        "n_alerts": int(y_pred.sum()),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "top_k": top_k_stats(y_true, y_score, k_fractions),
    }
