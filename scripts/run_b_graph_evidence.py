"""Graph evidence for the graph-feature model B_full on the 300 Phase 11 VAL transactions (label-free).

Protocol: experiments/comparison_B_graph_features/GRAPH_EVIDENCE_PREREG.md. Reads only the files that record lists; the label column is
projected away at read time; no TEST-named file is opened. Writes only into experiments/comparison_B_graph_features/graph_evidence/.

Usage:  .venv/Scripts/python.exe -B scripts/run_b_graph_evidence.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import xgboost as xgb  # noqa: E402

import run_comparison_b as RB  # noqa: E402  (shared constants and the guarded Reader; main() is not executed on import)
from src.config import load_config  # noqa: E402
from src.explain import graph_evidence as GE  # noqa: E402
from src.explain.xgb_contrib import contributions, top_features  # noqa: E402
from src.features import graph_features as G  # noqa: E402
from src.models.tree_features import TreeFeatureBuilder  # noqa: E402

EXP = "experiments/comparison_B_graph_features"
OUT = ROOT / EXP / "graph_evidence"
PREREG = f"{EXP}/GRAPH_EVIDENCE_PREREG.md"
PREREG_SIDECAR = f"{EXP}/GRAPH_EVIDENCE_PREREG.sha256"
SELECTION = "experiments/phase11_explainable_evidence/xgb_explanations_val/selection_val.csv"
SELECTION_HASH = "623d1ee2a0c6995e7c940565cd3c5d38a3bce076776de6a7976600fb440be963"      # recorded in Phase 11's manifest_val.json
MODEL = f"{EXP}/outputs/arms/B_full/model.joblib"
COLUMNS = f"{EXP}/outputs/arms/B_full/feature_columns.json"
ALLOWED = [PREREG, PREREG_SIDECAR, SELECTION, MODEL, COLUMNS, RB.TRAIN_F, RB.VAL_F, RB.CACHE]
LABEL = "isFraud"


class Reader(RB.Reader):
    """The shared guard refuses every .joblib; this script may load exactly one: the project's own B_full model."""

    def _check(self, rel, kind):
        rel_n = str(rel).replace("\\", "/")
        if rel_n == MODEL and rel_n in self.allowed:
            self.log.append((rel_n, kind))
            return self.root / rel_n
        return super()._check(rel, kind)


def read_no_label(reader, rel):
    p = reader.path(rel, "parquet")
    names = [n for n in pq.read_schema(p).names if n != LABEL]
    return pq.read_table(p, columns=names).to_pandas()


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    reader = Reader(ROOT, allowed=ALLOWED, extra=())
    prereg_hash = reader.sha256(PREREG)
    assert prereg_hash == reader.read_text(PREREG_SIDECAR).strip(), "graph-evidence pre-registration does not match its sidecar"
    sel_hash = reader.sha256(SELECTION)
    sel = pd.read_csv(reader.path(SELECTION, "csv"))
    ge2 = {"selection_hash_equals_phase11_record": sel_hash == SELECTION_HASH, "n_ids": int(len(sel)), "ids_unique": bool(sel["TransactionID"].is_unique),
           "days_in_122_151": bool(sel["day"].between(122, 151).all()), "ten_per_day": bool((sel.groupby("day").size() == 10).all() and sel["day"].nunique() == 30)}
    bundle = joblib.load(reader.path(MODEL, "joblib"))
    model, cols_b = bundle, json.loads(reader.read_text(COLUMNS))

    cfg = load_config()
    train_df, val_df = read_no_label(reader, RB.TRAIN_F), read_no_label(reader, RB.VAL_F)
    assert LABEL not in train_df.columns and LABEL not in val_df.columns
    z = np.load(reader.path(RB.CACHE, "npz"), allow_pickle=False)
    ents = {"pa": z["pa"], "card": z["card"], "net": z["net"], "dev": z["dev"]}
    full = pd.concat([train_df, val_df], ignore_index=True)
    assert np.array_equal(full["TransactionID"].to_numpy(), z["txn_id"])
    time_s = full["TransactionDT"].to_numpy(dtype=np.int64)
    gf = G.compute_graph_features(time_s, full["TransactionAmt"].to_numpy(dtype=np.float64), ents)
    builder = TreeFeatureBuilder(cfg=cfg).fit(train_df)
    Xa_val = builder.to_matrix(val_df, for_model="xgb").astype(np.float32)
    n_train = len(train_df)
    cols_a = list(builder.feature_names_)
    assert cols_b == cols_a + G.feature_names(), "B_full column list differs from the rebuilt tabular + graph columns"
    ids_val = val_df["TransactionID"].to_numpy()
    pos = pd.Series(np.arange(len(ids_val)), index=ids_val).loc[sel["TransactionID"].to_numpy()].to_numpy()
    X = np.hstack([Xa_val[pos], gf.to_numpy(dtype=np.float32)[n_train + pos]])
    ids = sel["TransactionID"].to_numpy()

    contrib, bias, marg = contributions(model, X)
    infl = GE.relation_influence(contrib, cols_b)
    ge1 = {"max_abs_additivity_error": float(np.max(np.abs(contrib.sum(axis=1) + bias - marg))), "bound": 1e-4}
    ge1["pass"] = ge1["max_abs_additivity_error"] < ge1["bound"]
    present = GE.present_relations(X, cols_b)
    eff = GE.perturbation_effects(model, X, cols_b)
    top, rnd = GE.pick_relations(ids, present, infl[list(G.RELATIONS)].abs())
    ge4 = GE.perturbation_sanity(eff, present, top, rnd)

    rows = pd.DataFrame({"TransactionID": ids, "day": sel["day"].to_numpy(), "group": sel["group"].to_numpy(),
                         "score_B": 1.0 / (1.0 + np.exp(-marg)), "margin": marg, "bias": bias})
    for r in G.RELATIONS:
        rows[f"present_{r}"] = present[:, G.RELATIONS.index(r)].astype(int)
    for r in (*G.RELATIONS, "tabular"):
        rows[f"contrib_{r}"] = infl[r].to_numpy()
    for r in G.RELATIONS:
        rows[f"removal_effect_{r}"] = eff[r].to_numpy()
    rows["top_relation"], rows["random_relation"] = top, rnd
    top10 = []
    graph_in_top10 = []
    for k in range(len(ids)):
        tf = top_features(contrib[k], cols_b, X[k], k=10)
        top10.append({"TransactionID": int(ids[k]), "top_features": tf})
        graph_in_top10.append(any(t["feature"].startswith("g_") for t in tf))
    rows["graph_feature_in_top10"] = graph_in_top10
    summary_rel = {r: {"mean_abs_summed_contribution": float(np.abs(infl[r]).mean()),
                       "mean_abs_summed_contribution_when_present": float(np.abs(infl[r])[present[:, G.RELATIONS.index(r)]].mean()) if present[:, G.RELATIONS.index(r)].any() else None,
                       "share_with_edge": float(present[:, G.RELATIONS.index(r)].mean()),
                       "mean_abs_removal_effect_when_present": float(eff[r].abs().mean())} for r in G.RELATIONS}
    summary_rel["tabular"] = {"mean_abs_summed_contribution": float(np.abs(infl["tabular"]).mean())}
    summary = {"GE1_additivity": ge1, "GE2_selection": ge2, "GE4_perturbation_sanity": ge4,
               "GE3_label_free": {"label_column_projected_away_at_read": True, "label_in_loaded_frames": False},
               "relation_summary": summary_rel, "share_of_transactions_with_a_graph_feature_in_top10": float(np.mean(graph_in_top10)),
               "by_group_share_graph_feature_in_top10": {g: float(np.mean(np.array(graph_in_top10)[sel["group"].to_numpy() == g])) for g in sorted(sel["group"].unique())},
               "n_transactions": int(len(ids)), "statement": "Descriptive evidence about the output of the graph-feature XGBoost model on 300 pre-registered VAL transactions; "
                                                             "VAL-tuned model; not evidence of fraud, causation or coordination; the neural graph models are not explained here."}
    all_ok = ge1["pass"] and all(v for k, v in ge2.items() if isinstance(v, bool)) and ge4["top_at_least_random"]
    summary["acceptance"] = {"GE1": ge1["pass"], "GE2": all(v for k, v in ge2.items() if isinstance(v, bool)), "GE3": True, "GE4": ge4["top_at_least_random"]}

    def wbytes(name, data: bytes):
        (OUT / name).write_bytes(data)
        h = hashlib.sha256(data).hexdigest()
        (OUT / (name.rsplit(".", 1)[0] + ".sha256")).write_bytes(h.encode("ascii"))
        return h

    out_hashes = {"relation_evidence.csv": wbytes("relation_evidence.csv", rows.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode("utf-8")),
                  "top_features_B.json": wbytes("top_features_B.json", (json.dumps(top10, indent=1) + "\n").encode("utf-8")),
                  "summary.json": wbytes("summary.json", (json.dumps(summary, indent=1) + "\n").encode("utf-8"))}
    manifest = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "preregistration_sha256": prereg_hash,
                "inputs_sha256": {p: reader.sha256(p) for p in (SELECTION, MODEL, COLUMNS, RB.TRAIN_F, RB.VAL_F, RB.CACHE)}, "outputs_sha256": out_hashes,
                "io_audit": reader.audit(), "test_data_read": False, "label_loaded": False,
                "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "seconds": round(time.time() - t0, 1)}
    wbytes("manifest_graph_evidence.json", (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(summary["acceptance"]), "| mean |effect| top vs random:", round(ge4["mean_abs_effect_top_relation"], 4), round(ge4["mean_abs_effect_random_relation"], 4))
    print(f"graph feature in top-10 for {summary['share_of_transactions_with_a_graph_feature_in_top10']:.1%} of transactions; done in {time.time() - t0:.0f}s")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
