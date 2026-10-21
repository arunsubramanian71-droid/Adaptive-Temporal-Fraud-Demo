"""Comparison B (tabular + handcrafted graph features): run the pre-registered arms and apply the pre-registered decision rules.

Protocol: experiments/comparison_B_graph_features/PREREGISTRATION_B.md (hash in PREREGISTRATION_B.sha256).
Reads ONLY the files listed in the pre-registration, through a guarded reader that refuses any TEST-named path, checkpoint or
excluded folder and logs every open. Uses days 1-151 only. Writes only into experiments/comparison_B_graph_features/outputs/.
Arms already completed (same configuration) are skipped, so an interrupted run can be resumed.

Usage:  .venv/Scripts/python.exe -B scripts/run_comparison_b.py [--arms A_refit,B_full,...]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost as xgb  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evidence import sources as S  # noqa: E402
from src.evidence import table as T  # noqa: E402
from src.features import b_decision as D  # noqa: E402
from src.features import graph_features as G  # noqa: E402
from src.features.preprocessing import assert_features_exclude_blocklist  # noqa: E402
from src.models.metrics import select_best_f1_threshold  # noqa: E402
from src.models.tree_features import TreeFeatureBuilder  # noqa: E402

EXP = "experiments/comparison_B_graph_features"
OUT = ROOT / EXP / "outputs"
PREREG = f"{EXP}/PREREGISTRATION_B.md"
PREREG_SIDECAR = f"{EXP}/PREREGISTRATION_B.sha256"
TRAIN_F, VAL_F = "data/features/train_features.parquet", "data/features/val_features.parquet"
CACHE = "experiments/phase9_coordinated_fraud/cache/real_le151.npz"
EVID = "experiments/phase12_validation_evidence_table/outputs/evidence_runs.csv"
ALLOWED = [PREREG, PREREG_SIDECAR, TRAIN_F, VAL_F, CACHE, T.REF_REL, EVID]
SEEDS = (42, 123, 2024)
A_RECORDED_PR_AUC = 0.5395
BASE = dict(n_estimators=1000, max_depth=6, learning_rate=0.05, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50, n_jobs=-1)


class Reader(S.GuardedReader):
    """GuardedReader that also hands out guarded paths for npz loading."""

    def path(self, rel, kind="npz"):
        return self._check(rel, kind)


def arm_specs():
    abl = G.ablation_groups()
    specs = [("A_refit", "A", 42, {}, None), ("B_full", "B", 42, {}, None)]
    specs += [(f"A_sub_s{s}", "A", s, {"subsample": 0.8, "colsample_bytree": 0.8}, None) for s in SEEDS]
    specs += [(f"B_sub_s{s}", "B", s, {"subsample": 0.8, "colsample_bytree": 0.8}, None) for s in SEEDS]
    specs += [(f"B_{g}", "B", 42, {}, g) for g in abl]
    return specs   # (name, feature set, seed, extra params, ablation group)


def load_inputs(reader):
    cfg = load_config()
    train_df = reader.read_parquet(TRAIN_F)
    val_df = reader.read_parquet(VAL_F)
    assert train_df["TransactionDT"].max() // cfg.day_seconds <= cfg.split.train_day_max
    assert val_df["TransactionDT"].min() // cfg.day_seconds >= cfg.split.val_day_min and val_df["TransactionDT"].max() // cfg.day_seconds <= cfg.split.val_day_max
    z = np.load(reader.path(CACHE), allow_pickle=False)
    both = pd.concat([train_df[["TransactionID", "TransactionDT"]], val_df[["TransactionID", "TransactionDT"]]], ignore_index=True)
    checks = {"txn_id_order_equals_cache": bool(np.array_equal(both["TransactionID"].to_numpy(), z["txn_id"])),
              "day_equals_cache": bool(np.array_equal((both["TransactionDT"] // cfg.day_seconds).to_numpy(), z["day"])),
              "max_day_le_151": int(z["day"].max()) <= 151}
    full = pd.concat([train_df, val_df], ignore_index=True)
    # entity-id cache agrees with the raw key columns (presence and partition)
    def partition_equal(ids, key_cols):
        has = ids >= 0
        keys = full[key_cols]
        present = keys.notna().all(axis=1).to_numpy()
        if not np.array_equal(has, present):
            return False, "presence differs"
        sub = keys[has]
        f1 = pd.factorize(pd.MultiIndex.from_frame(sub))[0] if len(key_cols) > 1 else pd.factorize(sub.iloc[:, 0])[0]
        pair = pd.DataFrame({"a": ids[has], "b": f1})
        return bool(pair.groupby("a")["b"].nunique().max() == 1 and pair.groupby("b")["a"].nunique().max() == 1), "ok"
    checks["card_partition_equals_card1"] = partition_equal(z["card"], ["card1"])[0]
    checks["netproxy_partition_equals_id19_id20"] = partition_equal(z["net"], ["id_19", "id_20"])[0]
    d1 = full["D1"].fillna(0)
    full["_d1anchor"] = (full["TransactionDT"] // cfg.day_seconds) - d1
    pa_present_expected = full[["card1", "addr1"]].notna().all(axis=1).to_numpy()
    checks["pseudo_account_presence_equals_card1_addr1"] = bool(np.array_equal(z["pa"] >= 0, pa_present_expected))
    ents = {"pa": z["pa"], "card": z["card"], "net": z["net"], "dev": z["dev"]}
    return cfg, train_df, val_df, z, ents, checks


def truncation_check(time_s, amount, ents, day, full_feats):
    """BA2 on real data: features computed on rows of days <= cutoff equal the full-timeline features of those rows, bit for bit."""
    res = {}
    for cutoff in (60, 100, 121):
        keep = day <= cutoff
        part = G.compute_graph_features(time_s[keep], amount[keep], {k: v[keep] for k, v in ents.items()})
        res[str(cutoff)] = {"rows": int(keep.sum()), "bit_identical": bool(np.array_equal(full_feats[keep].to_numpy(), part.to_numpy(), equal_nan=True))}
    return res


def fit_arm(name, fset, seed, extra, group, X, y, spw, cols, n_train, abl):
    drop = set(abl[group]) if group else set()
    use = [i for i, c in enumerate(cols) if c not in drop]
    Xa = X[:, use]
    clf = xgb.XGBClassifier(**BASE, **extra, scale_pos_weight=spw, random_state=seed)
    t0 = time.time()
    clf.fit(Xa[:n_train], y[:n_train], eval_set=[(Xa[n_train:], y[n_train:])], verbose=False)
    fit_s = time.time() - t0
    score = clf.predict_proba(Xa[n_train:])[:, 1]
    return clf, score, [cols[i] for i in use], fit_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arms").mkdir(exist_ok=True)
    reader = Reader(ROOT, allowed=ALLOWED, extra=())
    prereg_hash = reader.sha256(PREREG)
    assert prereg_hash == reader.read_text(PREREG_SIDECAR).strip(), "pre-registration hash does not match its sidecar"
    print(f"pre-registration verified: {prereg_hash}")

    cfg, train_df, val_df, z, ents, checks = load_inputs(reader)
    n_train, n_val = len(train_df), len(val_df)
    ref = T.make_reference(reader)
    print(f"train {n_train:,} val {n_val:,}; input checks: {checks}")
    if not all(checks.values()):
        raise SystemExit(f"BLOCKED: input consistency check failed: {checks}")

    # --- graph features over the TRAIN+VAL timeline
    full = pd.concat([train_df, val_df], ignore_index=True)
    time_s = full["TransactionDT"].to_numpy(dtype=np.int64)
    amount = full["TransactionAmt"].to_numpy(dtype=np.float64)
    day = (time_s // cfg.day_seconds).astype(np.int64)
    t0 = time.time()
    gf = G.compute_graph_features(time_s, amount, ents)
    print(f"graph features: {gf.shape} in {time.time() - t0:.1f}s; NaN share {gf.isna().mean().mean():.3f}")
    trunc = truncation_check(time_s, amount, ents, day, gf)
    print("truncation check:", trunc)
    if not all(v["bit_identical"] for v in trunc.values()):
        raise SystemExit("BLOCKED: truncation (future-invariance) check failed")

    # --- tabular matrix exactly as comparison A (builder fit on TRAIN only)
    builder = TreeFeatureBuilder(cfg=cfg).fit(train_df)
    assert_features_exclude_blocklist(builder.feature_names_, cfg)
    Xa = np.vstack([builder.to_matrix(train_df, for_model="xgb"), builder.to_matrix(val_df, for_model="xgb")]).astype(np.float32)
    a_cols = list(builder.feature_names_)
    assert Xa.shape[1] == 470 == len(a_cols), Xa.shape
    clash = set(a_cols) & set(G.feature_names())
    assert not clash, clash
    Xb = np.hstack([Xa, gf.to_numpy(dtype=np.float32)])
    b_cols = a_cols + G.feature_names()
    y = np.concatenate([train_df["isFraud"].to_numpy(), val_df["isFraud"].to_numpy()])
    spw = float((y[:n_train] == 0).sum() / max((y[:n_train] == 1).sum(), 1))
    abl = G.ablation_groups()
    val_ids = val_df["TransactionID"].to_numpy()
    wanted = set(a for a in args.arms.split(",") if a)

    results = {}
    for name, fset, seed, extra, group in arm_specs():
        if wanted and name not in wanted:
            continue
        adir = OUT / "arms" / name
        done = adir / "metrics_val.json"
        cfg_rec = {"arm": name, "features": fset, "seed": seed, "extra_params": extra, "ablation_group": group, "base_params": {**BASE, "scale_pos_weight": spw}}
        if done.exists():
            prev = json.loads(done.read_text(encoding="utf-8"))
            if prev.get("config") == cfg_rec:
                print(f"[skip] {name}: already complete")
                results[name] = prev
                continue
        adir.mkdir(exist_ok=True)
        X = Xa if fset == "A" else Xb
        cols = a_cols if fset == "A" else b_cols
        clf, score, used, fit_s = fit_arm(name, fset, seed, extra, group, X, y, spw, cols, n_train, abl)
        yv = y[n_train:]
        thr = select_best_f1_threshold(yv, score)
        frame = pd.DataFrame({"TransactionID": val_ids, "isFraud": yv, "y_score": score})
        frame = T.attach_missing(frame, ref)
        probs = T.population_problems(ref, frame, name)
        if probs:
            raise SystemExit(f"BLOCKED: population problems in arm {name}: {probs}")
        ev = T.evaluate_frame(frame, thr["threshold"], windows=True)
        rec = {"config": cfg_rec, "n_features": len(used), "best_iteration": int(clf.best_iteration), "fit_seconds": round(fit_s, 1),
               "threshold_selection": thr, "pooled": ev["pooled"], "segments": {k: ev[k] for k in T.SEGMENTS},
               "windows": {str(k): v for k, v in ev["windows"].items()}, "pr_auc": ev["pooled"]["pr_auc"],
               "pr_auc_cold": ev["has_edge_cold"]["pr_auc"]}
        pd.DataFrame({"TransactionID": val_ids, "isFraud": yv, "y_score": score, "y_pred": (score >= thr["threshold"]).astype(int)}).to_parquet(adir / "predictions_val.parquet", index=False)
        imp = sorted(zip(used, clf.feature_importances_.tolist()), key=lambda t: -t[1])[:30]
        rec["top30_feature_importance_gain_share"] = [{"feature": f, "importance": v} for f, v in imp]
        done.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        if name in ("A_refit", "B_full"):
            joblib.dump(clf, adir / "model.joblib")
            (adir / "feature_columns.json").write_text(json.dumps(used), encoding="utf-8")
        results[name] = rec
        print(f"[done] {name}: PR-AUC {rec['pr_auc']:.4f} cold {rec['pr_auc_cold']:.4f} ROC {ev['pooled']['roc_auc']:.4f} "
              f"best_iter {rec['best_iteration']} fit {fit_s:.0f}s", flush=True)

    summary = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "preregistration_sha256": prereg_hash,
               "input_checks": checks, "truncation_check": trunc,
               "inputs_sha256": {p: reader.sha256(p) for p in (TRAIN_F, VAL_F, CACHE, T.REF_REL, EVID)}, "io_audit": reader.audit(),
               "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
               "seconds": round(time.time() - t_start, 1), "test_data_read": False,
               "implementation_choice_disclosed": "window features use an inclusive lower bound: a prior transaction exactly 86,400 s (604,800 s) earlier counts in the 1-day (7-day) window"}
    (OUT / "run_record.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    # --- decisions (only when every arm they need is complete)
    need = {"A_refit", "B_full", *[f"A_sub_s{s}" for s in SEEDS], *[f"B_sub_s{s}" for s in SEEDS]}
    all_res = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in (OUT / "arms").glob("*/metrics_val.json")}
    if need <= set(all_res):
        ev_runs = pd.read_csv(ROOT / EVID, float_precision="round_trip")
        def per_seed(prefix, cell):
            d = ev_runs[(ev_runs["cell"] == cell) & ev_runs["model"].str.startswith(prefix)].sort_values("seed")
            assert list(d["seed"]) == list(SEEDS), (prefix, list(d["seed"]))
            return d["pr_auc"].to_numpy()
        e_vals = per_seed("Temporal GraphSAGE (GRU", "E")
        c_vals = per_seed("GraphSAGE (Phase 6A)", "C")
        a_sub = [all_res[f"A_sub_s{s}"]["pr_auc"] for s in SEEDS]
        b_sub = [all_res[f"B_sub_s{s}"]["pr_auc"] for s in SEEDS]
        dec = {"H1_direct_B_vs_A": D.decide_h1(all_res["B_full"]["pr_auc"], all_res["A_refit"]["pr_auc"], b_sub, a_sub),
               "H2_E_vs_B_sub": D.decide_h2(e_vals, b_sub), "H2_secondary_C_vs_B_sub": D.decide_h2(c_vals, b_sub),
               "BA6_pipeline_control": {"A_refit_pr_auc": all_res["A_refit"]["pr_auc"], "recorded_A_pr_auc": A_RECORDED_PR_AUC,
                                        "abs_diff": abs(all_res["A_refit"]["pr_auc"] - A_RECORDED_PR_AUC),
                                        "within_0.005": abs(all_res["A_refit"]["pr_auc"] - A_RECORDED_PR_AUC) <= 0.005},
               "per_seed": {"A_sub": a_sub, "B_sub": b_sub, "E_recorded": e_vals.tolist(), "C_recorded": c_vals.tolist()}}
        abl_have = {f"B_{g}": all_res[f"B_{g}"]["pr_auc"] for g in abl if f"B_{g}" in all_res}
        dec["ablations"] = D.ablation_table(all_res["B_full"]["pr_auc"], abl_have, b_sub)
        (OUT / "decision.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
        print(json.dumps({k: (v["verdict"] if isinstance(v, dict) and "verdict" in v else None) for k, v in dec.items() if k.startswith("H")}, indent=1))
    print(f"done in {time.time() - t_start:.0f}s; TEST data read: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
