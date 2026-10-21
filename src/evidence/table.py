"""Phase 12: recomputation, population checks and assembly of the consolidated validation-evidence table.

Scope record: experiments/phase12_validation_evidence_table/SCOPE_AND_ACCEPTANCE.md (P1-P6, AC2-AC8). One metric implementation only:
``src.models.metrics.compute_metrics`` (imported, unmodified). No threshold is selected or tuned; operating-point metrics use only the
thresholds recorded in the source files. Nothing here trains, scores or predicts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evidence import sources as S
from src.models.metrics import compute_metrics   # the single registered metric implementation

SEGMENTS = ("has_edge_warm", "has_edge_cold", "no_pseudo_account_edge")
WINDOWS = ((1, 122, 127), (2, 128, 133), (3, 134, 139), (4, 140, 145), (5, 146, 151))
TOL = 1e-9
K_KEYS = ("0.1%", "0.5%", "1%", "5%")
REF_REL = "experiments/temporal_graphsage/seed_42/predictions_val.parquet"
STANDING = ("Descriptive table of recorded validation-period (VAL, days 122-151) predictions. Every model was tuned on VAL (early stopping, "
            "epoch selection or thresholds) except where stated for F. No held-out (TEST) comparison exists, TEST is sealed, and nothing here "
            "supports a claim of generalization, statistical significance, causal attribution, fraud or coordination.")
F_BANNER = ("F rows are on a distinct evaluation basis: the shared checkpoint was trained on days 1-111, the threshold came from TRAIN days "
            "112-121, and the adaptive arm updates on already-scored VAL labels. F is not comparable with A, C, D or E.")
B_ROW = {"cell": "B", "status": "MISSING", "note": "no artifact exists (no graph-feature XGBoost); the historical-feature baseline is a different "
         "construct and is not substituted; H2 cannot be assessed from this table"}


class BuildBlocked(RuntimeError):
    """A required source is missing, a population check fails, or a recorded metric disagrees beyond the registered tolerance."""


# ------------------------------------------------------------------------------------------------ comparison with recorded values
def compare_recorded(recorded, recomputed, tol=TOL):
    """Compare every key of ``recorded`` that ``recomputed`` also holds. Returns max abs diff, n compared, mismatches, skipped keys."""
    mism, skipped = [], []
    state = {"max": 0.0, "n": 0}

    def walk(a, b, p):
        if isinstance(a, dict):
            for k, v in a.items():
                if not isinstance(b, dict) or k not in b:
                    skipped.append(f"{p}/{k}")
                    continue
                walk(v, b[k], f"{p}/{k}")
            return
        state["n"] += 1
        if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
            if a != b:
                mism.append({"key": p, "recorded": a, "recomputed": b})
            return
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return
        d = abs(fa - fb) if not (math.isnan(fa) or math.isnan(fb)) else float("inf")
        state["max"] = max(state["max"], d)
        if d > tol:
            mism.append({"key": p, "recorded": a, "recomputed": b, "abs_diff": d})

    walk(recorded, recomputed, "")
    return {"max_abs_diff": state["max"], "n_compared": state["n"], "mismatches": mism, "skipped_keys": sorted(set(skipped))}


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate_frame(frame, threshold, windows=True):
    """Pooled, strict-segment and (optionally) window metrics in the frame's stored row order (stable-sort ties follow row order)."""
    y = frame["isFraud"].to_numpy()
    s = frame["y_score"].to_numpy()
    seg = frame["segment"].to_numpy()
    out = {"pooled": compute_metrics(y, s, threshold)}
    for name in SEGMENTS:
        m = seg == name
        out[name] = compute_metrics(y[m], s[m], threshold)
    if windows:
        day = frame["day"].to_numpy()
        out["windows"] = {}
        for w, lo, hi in WINDOWS:
            m = (day >= lo) & (day <= hi)
            r = compute_metrics(y[m], s[m], threshold)
            out["windows"][w] = {k: r[k] for k in ("pr_auc", "roc_auc", "n_rows", "n_fraud")}
    return out


def window_of_day(d):
    for w, lo, hi in WINDOWS:
        if lo <= d <= hi:
            return w
    return None


# ------------------------------------------------------------------------------------------------ population identity (AC4)
def make_reference(reader):
    ref = reader.read_parquet(REF_REL, columns=["TransactionID", "isFraud", "day", "segment"])
    ref = ref.sort_values("TransactionID").reset_index(drop=True)
    if ref["TransactionID"].duplicated().any():
        raise BuildBlocked("reference file has duplicate TransactionIDs")
    return ref


def population_problems(ref, frame, name):
    probs = []
    if len(frame) != len(ref):
        probs.append(f"{name}: {len(frame)} rows, reference has {len(ref)}")
    if frame["TransactionID"].duplicated().any():
        probs.append(f"{name}: duplicated TransactionID")
    cols = ["TransactionID", "isFraud"] + [c for c in ("day", "segment") if c in frame.columns]
    m = frame[cols].merge(ref, on="TransactionID", how="outer", suffixes=("", "_ref"), indicator=True)
    left, right = int((m["_merge"] == "left_only").sum()), int((m["_merge"] == "right_only").sum())
    if left or right:
        probs.append(f"{name}: id sets differ ({left} only in the file, {right} only in the reference)")
    b = m[m["_merge"] == "both"]
    n = int((b["isFraud"].astype(int) != b["isFraud_ref"].astype(int)).sum())
    if n:
        probs.append(f"{name}: isFraud differs from the reference for {n} ids")
    for c in ("day", "segment"):
        if c in frame.columns:
            n = int((b[c] != b[c + "_ref"]).sum())
            if n:
                probs.append(f"{name}: {c} differs from the reference for {n} ids")
    return probs


def attach_missing(frame, ref):
    """Add day / segment from the reference by id where a file lacks them; the join keeps the file's row order."""
    missing = [c for c in ("day", "segment") if c not in frame.columns]
    if missing:
        frame = frame.merge(ref[["TransactionID"] + missing], on="TransactionID", how="left", validate="many_to_one")
        if frame[missing].isna().any().any():
            raise BuildBlocked("segment/day join left missing values")
    return frame


# ------------------------------------------------------------------------------------------------ run catalogue
def non_f_specs():
    E = S.E
    specs = []
    for d in ("logistic_regression", "random_forest", "xgboost"):
        specs.append(dict(id=f"A:{d}", cell="A", role="A-F cell", family=f"A:{d}", model=d, seed=None, dir=f"{E}{d}"))
    specs.append(dict(id="Aref:xgboost_historical_control", cell="A-reference", role="reference line (not an A-F cell)",
                      family="Aref:xgboost_historical_control", model="xgboost_historical_control", seed=None, dir=f"{E}xgboost_historical_control"))
    for s in S.SEEDS:
        specs.append(dict(id=f"C:seed{s}", cell="C", role="A-F cell", family="C:graphsage_robust", model="GraphSAGE (Phase 6A)", seed=s,
                          dir=f"{E}graphsage_robust/seed_{s}"))
    specs.append(dict(id="Ccontrol:mlp_control_extended", cell="C-control", role="control for C (not an A-F cell)", family="Ccontrol:mlp_control_extended",
                      model="MLP control, no relations (Phase 6A extended)", seed=None, dir=f"{E}graphsage_robust/mlp_control_extended"))
    for s in S.SEEDS:
        specs.append(dict(id=f"D:seed{s}", cell="D", role="A-F cell", family="D:gat", model="GAT (Phase 6B)", seed=s, dir=f"{E}gat/seed_{s}"))
    for s in S.SEEDS:
        specs.append(dict(id=f"E:seed{s}", cell="E", role="A-F cell", family="E:temporal_graphsage", model="Temporal GraphSAGE (GRU memory, Phase 7)", seed=s,
                          dir=f"{E}temporal_graphsage/seed_{s}"))
    for arm, nm in (("static_schedule_matched", "Static GraphSAGE, schedule-matched (Phase 7A)"), ("memory_ablated_temporal", "Temporal, memory ablated (Phase 7A)")):
        for s in S.SEEDS:
            specs.append(dict(id=f"E7A:{arm}:seed{s}", cell="E-control", role="control for E (not an A-F cell)", family=f"E7A:{arm}", model=nm, seed=s,
                              dir=f"{E}temporal_ablation/{arm}/seed_{s}"))
    for sp in specs:
        sp["pred"], sp["metrics"], sp["config"] = (f"{sp['dir']}/{f}" for f in ("predictions_val.parquet", "metrics_val.json", "config.json"))
        sp["files"] = [sp["pred"], sp["metrics"], sp["config"]] + ([f"{sp['dir']}/metadata.json"] if sp["cell"] in ("A", "A-reference") else [])
    return specs


IDENTITY = {
    "A": dict(fit_window="TRAIN 1-121", operating_point_origin="best-F1 threshold chosen on VAL (recorded)", val_tuned="yes: thresholds on VAL (XGBoost also early stopping on VAL)",
              val_label_use="threshold and early stopping used VAL labels"),
    "A-reference": dict(fit_window="TRAIN 1-121", operating_point_origin="best-F1 threshold chosen on VAL (recorded)", val_tuned="yes: threshold on VAL",
                        val_label_use="threshold used VAL labels"),
    "C": dict(fit_window="TRAIN 1-121", operating_point_origin="best-F1 threshold chosen on VAL (recorded)", val_tuned="yes: epoch and threshold selected on VAL",
              val_label_use="epoch selection and threshold used VAL labels"),
    "C-control": dict(fit_window="TRAIN 1-121", operating_point_origin="best-F1 threshold chosen on VAL (recorded)", val_tuned="yes: epoch and threshold selected on VAL",
                      val_label_use="epoch selection and threshold used VAL labels; reached its 150-epoch cap"),
    "D": dict(fit_window="TRAIN 1-121", operating_point_origin="best-F1 threshold chosen on VAL (recorded)", val_tuned="yes: epoch and threshold selected on VAL",
              val_label_use="epoch selection and threshold used VAL labels; 150-epoch cap reached in all seeds"),
    "E": dict(fit_window="TRAIN 1-121", operating_point_origin="best-F1 threshold chosen on VAL (recorded)", val_tuned="yes: epoch and threshold selected on VAL",
              val_label_use="epoch selection and threshold used VAL labels"),
    "E-control": dict(fit_window="TRAIN 1-121", operating_point_origin="best-F1 threshold chosen on VAL (recorded)", val_tuned="yes: epoch and threshold selected on VAL",
                      val_label_use="epoch selection and threshold used VAL labels"),
}
F_IDENTITY = {
    "frozen": "no update after the shared TRAIN-only checkpoint; no VAL labels used",
    "adaptive": "weights updated after each window on already-scored VAL labels plus TRAIN",
    "replay_control": "same update schedule as adaptive but on TRAIN resamples only; no VAL labels used",
}


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def metric_columns(ev):
    """Flat metric columns from an ``evaluate_frame`` result."""
    p = ev["pooled"]
    row = {"n_rows": p["n_rows"], "n_fraud": p["n_fraud"], "pr_auc": p["pr_auc"], "roc_auc": p["roc_auc"]}
    for seg in SEGMENTS:
        row[f"pr_auc_{seg}"] = ev[seg]["pr_auc"]
    for k in K_KEYS:
        t = k.replace("%", "pct").replace(".", "p")
        row[f"precision_at_{t}"] = p["top_k"][k]["precision_at_k"]
        row[f"recall_at_{t}"] = p["top_k"][k]["recall_at_k"]
    row.update({"threshold": p["threshold"], "precision": p["precision"], "recall": p["recall"], "f1": p["f1"], "n_alerts": p["n_alerts"],
                "false_positives": p["false_positives"]})
    return {k: _clean(v) for k, v in row.items()}


METRIC_KEYS = ["pr_auc", "roc_auc"] + [f"pr_auc_{s}" for s in SEGMENTS] + \
              [f"{a}_at_{k.replace('%', 'pct').replace('.', 'p')}" for k in K_KEYS for a in ("precision", "recall")] + ["precision", "recall", "f1", "n_alerts", "false_positives"]


def process_non_f(reader, ref, spec, hashes):
    frame = reader.read_parquet(spec["pred"])
    if not {"TransactionID", "isFraud", "y_score"} <= set(frame.columns):
        raise BuildBlocked(f"{spec['id']}: required columns missing")
    probs = population_problems(ref, frame, spec["id"])
    if probs:
        raise BuildBlocked("population mismatch: " + "; ".join(probs))
    frame = attach_missing(frame, ref)
    rec = reader.read_json(spec["metrics"])
    view = rec if "pooled" in rec else {"pooled": rec}
    thr = float(view["pooled"]["threshold"])
    ev = evaluate_frame(frame, thr)
    chk = compare_recorded(view, ev)
    cfg = reader.read_json(spec["config"])
    seed = cfg.get("seed", cfg.get("random_state"))
    ident = IDENTITY[spec["cell"]]
    row = {"run_id": spec["id"], "cell": spec["cell"], "role": spec["role"], "model": spec["model"], "seed": seed, "metric_basis":
           "pooled over the 83,655 VAL rows (days 122-151), stored row order", "fit_window": ident["fit_window"], "data_window": "VAL 122-151",
           "operating_point_origin": ident["operating_point_origin"], "includes_val_tuned_selection": ident["val_tuned"], "val_label_use": ident["val_label_use"],
           "parameter_count": cfg.get("parameter_count"), "recomputation_max_abs_diff": chk["max_abs_diff"],
           "source_paths": ";".join(spec["files"]), "source_sha256": ";".join(f"{p}={hashes[p]}" for p in spec["files"])}
    row.update(metric_columns(ev))
    return {"spec": spec, "row": row, "eval": ev, "check": {"run_id": spec["id"], **chk, "n_rows": int(len(frame)), "threshold": thr}}


def process_f(reader, ref, arm, seed, hashes):
    base = f"{S.E}phase8_adaptive_drift/{arm}/seed_{seed}"
    hp = f"{S.E}phase8_adaptive_drift/shared_init/hparams_seed{seed}.json"
    thr = float(reader.read_json(hp)["threshold"])
    frames, per_window, checks, files = [], {}, [], [hp]
    for w in range(1, 6):
        pf, mf = f"{base}/predictions_val_W{w}.parquet", f"{base}/metrics_val_W{w}.json"
        frame = reader.read_parquet(pf)
        if not {"TransactionID", "isFraud", "y_score", "day", "window", "segment"} <= set(frame.columns):
            raise BuildBlocked(f"{pf}: required columns missing")
        if not (frame["window"] == w).all() or not (frame["day"].map(window_of_day) == w).all():
            raise BuildBlocked(f"{pf}: window column inconsistent with the registered window definition")
        rec = reader.read_json(mf)
        ev = evaluate_frame(frame, thr, windows=False)
        chk = compare_recorded(rec, ev)
        checks.append({"run_id": f"F:{arm}:seed{seed}:W{w}", **chk, "n_rows": int(len(frame)), "threshold": thr})
        per_window[w] = ev
        frames.append(frame)
        files += [pf, mf]
    concat = pd.concat(frames, ignore_index=True)
    probs = population_problems(ref, concat, f"F:{arm}:seed{seed}")
    if probs:
        raise BuildBlocked("population mismatch: " + "; ".join(probs))
    ev_c = evaluate_frame(concat, thr)
    ident = F_IDENTITY[arm]
    common = {"cell": "F", "role": "F (distinct basis)", "model": f"Temporal GraphSAGE, F-{arm}", "seed": seed, "fit_window": "TRAIN 1-111 (shared checkpoint)",
              "data_window": "VAL 122-151 in five 6-day windows", "operating_point_origin": "TRAIN days 112-121 holdout (hparams, recorded)",
              "includes_val_tuned_selection": "no VAL-based epoch or threshold selection", "val_label_use": ident, "parameter_count": 197889,
              "source_paths": ";".join(files), "source_sha256": ";".join(f"{p}={hashes[p]}" for p in files),
              "recomputation_max_abs_diff": max(c["max_abs_diff"] for c in checks)}
    wins = [per_window[w] for w in range(1, 6)]

    def mean_over_windows(f):
        vals = [f(e) for e in wins]
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.mean(vals)) if vals else None

    row_mean = {"run_id": f"F:{arm}:seed{seed}:basis_mean_of_windows", **common, "metric_basis": "mean of the five window PR-AUCs (Phase 8 basis); threshold-based metrics not defined on this basis",
                "pr_auc": mean_over_windows(lambda e: e["pooled"]["pr_auc"]), "roc_auc": mean_over_windows(lambda e: e["pooled"]["roc_auc"]),
                **{f"pr_auc_{s}": mean_over_windows(lambda e, s=s: e[s]["pr_auc"]) for s in SEGMENTS}}
    row_cat = {"run_id": f"F:{arm}:seed{seed}:basis_concatenated", **common, "metric_basis": "five stored windows concatenated (W1..W5 order), pooled over 83,655 rows with the "
               "same metric code as A-E; later windows of the adaptive and replay arms are scored by updated weights", **metric_columns(ev_c)}
    return {"arm": arm, "seed": seed, "rows": [row_mean, row_cat], "windows": {w: {"pr_auc": per_window[w]["pooled"]["pr_auc"], "roc_auc": per_window[w]["pooled"]["roc_auc"],
            "n_rows": per_window[w]["pooled"]["n_rows"], "n_fraud": per_window[w]["pooled"]["n_fraud"]} for w in range(1, 6)},
            "checks": checks, "eval_concat": ev_c, "per_window_eval": per_window}


# ------------------------------------------------------------------------------------------------ aggregation (P3)
def aggregate(values):
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    a = np.array(v, dtype=float)
    return {"n": int(len(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None, "min": float(a.min()), "max": float(a.max())}


def aggregate_family(rows):
    out = {}
    for k in METRIC_KEYS:
        out[k] = aggregate([r.get(k) for r in rows])
    return out


# ------------------------------------------------------------------------------------------------ cross-checks against recorded comparison files (AC8)
def _diff(a, b):
    if a is None or b is None:
        return float("inf") if a != b else 0.0
    return abs(float(a) - float(b))


def cross_checks(reader, results, f_results):
    """Compare recomputed values with the recorded comparison files. Never used as inputs. Returns a list of checks."""
    E = S.E
    out = []

    def add(source, item, rec, new):
        out.append({"source": source, "item": item, "recorded": _clean(rec), "recomputed": _clean(new), "abs_diff": _diff(rec, new)})

    by_family = {}
    for r in results:
        by_family.setdefault(r["spec"]["family"], []).append(r)
    seg_key = {"pooled": "pooled", "has_edge_warm": "has_edge_warm", "has_edge_cold": "has_edge_cold", "no_pseudo_account_edge": "no_pseudo_account_edge"}
    c7 = reader.read_json(f"{E}temporal_graphsage/comparison.json")
    name_map = {"Logistic Regression": "A:logistic_regression", "Random Forest": "A:random_forest", "XGBoost": "A:xgboost",
                "XGBoost historical control": "Aref:xgboost_historical_control", "MLP control (Phase 6A extended)": "Ccontrol:mlp_control_extended",
                "GraphSAGE Phase 6A (3 seeds)": "C:graphsage_robust", "GAT Phase 6B (3 seeds)": "D:gat", "Temporal GraphSAGE (3 seeds)": "E:temporal_graphsage"}
    for nm, fam in name_map.items():
        for seg, key in seg_key.items():
            vals = [r["eval"][seg]["pr_auc"] for r in by_family[fam]]
            add("temporal_graphsage/comparison.json strict_segment_pr_auc", f"{nm} / {seg} PR-AUC (mean over runs)", c7["strict_segment_pr_auc"][nm][key], float(np.mean(vals)))
    seg_name = {"pooled": "pooled", "cold": "has_edge_cold", "warm": "has_edge_warm", "no_edge": "no_pseudo_account_edge"}
    for item in c7["static_vs_temporal"]:
        for kind, fam, vals_key in (("static", "C:graphsage_robust", "static_values"), ("temporal", "E:temporal_graphsage", "temporal_values")):
            runs = sorted(by_family[fam], key=lambda r: S.SEEDS.index(r["spec"]["seed"]))
            for i, r in enumerate(runs):
                add("temporal_graphsage/comparison.json static_vs_temporal", f"{kind} {item['segment']} {item['metric']} seed {r['spec']['seed']}",
                    item[vals_key][i], r["eval"][seg_name[item["segment"]]][item["metric"]])
    c7a = reader.read_json(f"{E}temporal_ablation/comparison.json")["arms"]
    arm_map = {"A": "C:graphsage_robust", "B": "E7A:static_schedule_matched", "C": "E7A:memory_ablated_temporal", "D": "E:temporal_graphsage"}
    key_map = {"pooled_pr_auc": ("pooled", "pr_auc"), "pooled_roc_auc": ("pooled", "roc_auc"), "cold_pr_auc": ("has_edge_cold", "pr_auc"),
               "warm_pr_auc": ("has_edge_warm", "pr_auc"), "no_edge_pr_auc": ("no_pseudo_account_edge", "pr_auc"), "precision": ("pooled", "precision")}
    for arm, fam in arm_map.items():
        for k, (seg, met) in key_map.items():
            vals = [r["eval"][seg][met] for r in by_family[fam]]
            add("temporal_ablation/comparison.json arms mean", f"7A arm {arm} {k}", c7a[arm]["mean"][k], float(np.mean(vals)))
            add("temporal_ablation/comparison.json arms std (ddof=1)", f"7A arm {arm} {k}", c7a[arm]["std"][k], float(np.std(vals, ddof=1)))
    c8 = reader.read_json(f"{E}phase8_adaptive_drift/comparison.json")["per_seed_window_pr_auc"]
    for rec_arm, arm in (("F", "frozen"), ("Ad", "adaptive"), ("E", "replay_control")):
        for fr in f_results:
            if fr["arm"] != arm:
                continue
            for w in range(1, 6):
                add("phase8_adaptive_drift/comparison.json per_seed_window_pr_auc", f"{arm} seed {fr['seed']} W{w}", c8[rec_arm][str(fr["seed"])][w - 1], fr["windows"][w]["pr_auc"])
    return out
