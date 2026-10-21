"""Held-out protocol Amendment 1 (experiments/final_heldout_protocol/AMENDMENT_1.md): scoring and authorization for the added modules H and U.
(Module F, the adaptive arms, is run by scripts/run_f_walkforward.py and uses the same authorization function.)

The base protocol (HELDOUT_EVALUATION_PROTOCOL.md) and its 19-model registry are unchanged. The amendment is additive: it is honoured only if AUTHORIZATION.txt ALSO quotes the amendment's
SHA-256 and a sentence naming the modules the user authorizes. With only the base authorization, the amendment modules are refused. ``split="val"`` runs the same code on the validation period
(readiness gate); it reads no held-out row.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd

from src.heldout import evaluator as E

ROOT = E.ROOT
AMENDMENT = "experiments/final_heldout_protocol/AMENDMENT_1.md"
AMEND_PHRASE = "I authorize the following Amendment 1 modules:"
MODULES = ("F", "H", "U")
HIST_DIR = "experiments/xgboost_historical_control"
HIST_FEATURES = "data/features/historical_features.parquet"
UNW_XGB = {"A_none": "experiments/final_heldout_protocol/frozen_amendment/A_none", "B_none": "experiments/final_heldout_protocol/frozen_amendment/B_none"}
UNW_STATIC = {"graphsage_unweighted": ("experiments/unweighted_graphsage/seed_{s}", "sage")}
UNW_TEMPORAL = {"temporal_graphsage_unweighted": ("experiments/unweighted_temporal_graphsage/seed_{s}", "temporal")}


def authorized_modules(text: str) -> set:
    """Modules named after the amendment sentence (letters F, H, U separated by commas/spaces)."""
    m = re.search(re.escape(AMEND_PHRASE) + r"([^\n]*)", text)
    return set(re.findall(r"\b([FHU])\b", m.group(1))) if m else set()


def require_amendment_authorization(split: str, modules, root: pathlib.Path = ROOT) -> dict:
    """Base authorization (protocol hash + sentence) AND the amendment hash AND a sentence naming every requested module. Validation needs none."""
    modules = tuple(modules)
    bad = [m for m in modules if m not in MODULES]
    if bad:
        raise ValueError(f"unknown amendment module(s) {bad}")
    base = E.require_authorization(split, root)
    if split == "val":
        return base
    amend_hash = E.sha256_file(root / AMENDMENT)
    text = (root / E.AUTH).read_text(encoding="utf-8")
    if amend_hash not in text:
        raise E.AuthorizationError(f"{E.AUTH} does not quote the current amendment hash {amend_hash}; the amendment modules {modules} are refused (the base protocol alone may still be authorized).")
    missing = sorted(set(modules) - authorized_modules(text))
    if missing:
        raise E.AuthorizationError(f"{E.AUTH} does not contain the sentence '{AMEND_PHRASE} <modules>' naming module(s) {missing}; refusing.")
    return {**base, "amendment_sha256": amend_hash, "amendment_modules_authorized": sorted(authorized_modules(text) & set(MODULES)), "modules_run": list(modules)}


def hist_columns() -> list:
    from src.features.historical import historical_feature_columns
    return list(historical_feature_columns())


def hist_scores(cfg, split: str, root: pathlib.Path = ROOT) -> dict:
    """Score the frozen historical-feature XGBoost on the rows of ``split``.

    The historical feature file is a single row group covering all periods, so it is streamed in small batches in file order and reading stops as soon as every TransactionID of the split
    has been found; rows of other periods beyond that point are never decoded, and rows of other periods inside the last batch are dropped on the spot (same load-then-truncate rule as the graph arrays)."""
    import joblib
    import pyarrow.parquet as pq
    ids = E.read_features(root / E.SPLITS[split]["features"])["TransactionID"].to_numpy()
    cols = hist_columns()
    want = set(int(i) for i in ids)
    parts, found = [], 0
    for batch in pq.ParquetFile(root / HIST_FEATURES).iter_batches(batch_size=4096, columns=["TransactionID"] + cols):
        b = batch.to_pandas()
        b = b[b["TransactionID"].isin(want)]
        if len(b):
            parts.append(b)
            found += len(b)
        if found >= len(want):
            break
    t = pd.concat(parts, ignore_index=True)
    df = pd.DataFrame({"TransactionID": ids}).merge(t, on="TransactionID", how="left", validate="one_to_one")
    assert len(df) == len(ids) and len(t) == len(want), "historical feature rows do not match the split's TransactionIDs"
    X = df[cols].to_numpy(dtype=np.float32)
    model = joblib.load(root / HIST_DIR / "model.joblib")
    return {"xgboost_historical": pd.DataFrame({"TransactionID": ids, "y_score": model.predict_proba(X)[:, 1]})}


def u_registry() -> dict:
    reg = {}
    for name, d in UNW_XGB.items():
        reg[name] = {"family": "xgb_unweighted", "dir": d, "files": ["model.joblib", "feature_columns.json"]}
    for table, fam in ((UNW_STATIC, "static"), (UNW_TEMPORAL, "temporal")):
        for name, (pat, _kind) in table.items():
            for s in E.SEEDS:
                reg[f"{name}_s{s}"] = {"family": fam, "dir": pat.format(s=s), "files": ["model_checkpoint.pt", "metrics_val.json"]}
    return reg


def a_none_scores(cfg, split: str, root: pathlib.Path) -> dict:
    """A_none: the 500 comparison-A features, same builder as the XGBoost baseline."""
    import joblib
    from src.models.tree_features import TreeFeatureBuilder
    df = E.read_features(root / E.SPLITS[split]["features"])
    builder = TreeFeatureBuilder.load(cfg, root / E.TABULAR["xgboost"] / "feature_builder")
    cols = json.loads((root / UNW_XGB["A_none"] / "feature_columns.json").read_text(encoding="utf-8"))
    assert cols == list(builder.feature_names_)
    model = joblib.load(root / UNW_XGB["A_none"] / "model.joblib")
    return {"A_none": pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy(), "y_score": model.predict_proba(builder.to_matrix(df, for_model="xgb"))[:, 1]})}


def recorded_threshold_hist(root: pathlib.Path = ROOT) -> float:
    return float(json.loads((root / HIST_DIR / "metrics_val.json").read_text(encoding="utf-8"))["threshold"])


def recorded_threshold_u(name: str, root: pathlib.Path = ROOT) -> float:
    """Validation best-F1 threshold recorded for each unweighted model (comparison-A / follow-up metrics files)."""
    if name == "A_none":
        return E.recorded_threshold(root / "experiments/imbalance_check_xgboost/outputs/arms/none")
    if name == "B_none":
        return E.recorded_threshold(root / "experiments/imbalance_check_xgboost/outputs/arms_unweighted/B_none")
    base = name.rsplit("_s", 1)
    pat = (UNW_STATIC.get(base[0]) or UNW_TEMPORAL.get(base[0]))[0]
    return E.recorded_threshold(root / pat.format(s=int(base[1])))


def run_modules(split: str, modules, root: pathlib.Path = ROOT) -> dict:
    """Score the frozen models of modules H and/or U. Returns {"scores", "labels", "auth"} like ``evaluator.run``."""
    modules = tuple(m for m in modules if m in ("H", "U"))
    auth = require_amendment_authorization(split, modules, root)
    import torch
    from src.config import load_config
    from src.graph.schema import compute_pseudo_account_segments
    from src.graph.temporal import load_raw_arrays
    cfg = load_config()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw, graph_cpu, _ = load_raw_arrays(cfg)
    seg = compute_pseudo_account_segments(graph_cpu)
    lo, hi = E.SPLITS[split]["days"]
    raw = E.truncate_raw(raw, hi)
    seg = seg[:len(raw["day"])]
    m = (raw["day"] >= lo) & (raw["day"] <= hi)
    labels = pd.DataFrame({"TransactionID": raw["txn_id"][m], "isFraud": raw["y"][m], "day": raw["day"][m], "segment": seg[m]})
    scores = {}
    if "H" in modules:
        scores.update(hist_scores(cfg, split, root))
    if "U" in modules:
        scores.update(a_none_scores(cfg, split, root))
        scores.update(E._b_scores(cfg, split, root, raw, model_dir=UNW_XGB["B_none"], name="B_none"))
        scores.update(E._static_scores(cfg, split, root, dev, list(UNW_STATIC), table=UNW_STATIC))
        scores.update(E._temporal_scores(cfg, split, root, dev, raw, list(UNW_TEMPORAL), table=UNW_TEMPORAL))
    return {"scores": scores, "labels": labels, "auth": auth}
