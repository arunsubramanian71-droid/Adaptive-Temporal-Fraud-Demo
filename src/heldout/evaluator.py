"""Frozen-model evaluator for the held-out protocol (experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md).

``split="val"`` runs exactly the same code on the validation period so the evaluator can be checked against the recorded VAL predictions
(the code-readiness gate, protocol section 6). ``split="heldout"`` (days 152-182) is refused unless ``AUTHORIZATION.txt`` exists in the
protocol folder and quotes the protocol's current SHA-256 and the authorization sentence. No model is fitted, tuned or re-thresholded here;
no label is passed to any model (labels are read afterwards, only to compute metrics).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PROTOCOL = "experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md"
AUTH = "experiments/final_heldout_protocol/AUTHORIZATION.txt"
AUTH_PHRASE = "I authorize one-time evaluation of the frozen models"
SPLITS = {"val": {"days": (122, 151), "features": "data/features/val_features.parquet"},
          "heldout": {"days": (152, 182), "features": "data/features/test_features.parquet"}}
SEEDS = (42, 123, 2024)
TABULAR = {"logistic_regression": "experiments/logistic_regression", "random_forest": "experiments/random_forest", "xgboost": "experiments/xgboost"}
B_FULL = "experiments/comparison_B_graph_features/outputs/arms/B_full"
STATIC = {"graphsage": ("experiments/graphsage_robust/seed_{s}", "sage"), "gat": ("experiments/gat/seed_{s}", "gat"),
          "static_schedule_matched": ("experiments/temporal_ablation/static_schedule_matched/seed_{s}", "sage")}
TEMPORAL = {"temporal_graphsage": ("experiments/temporal_graphsage/seed_{s}", "temporal"),
            "memory_ablated": ("experiments/temporal_ablation/memory_ablated_temporal/seed_{s}", "meanmem")}
FAMILIES = ("tabular", "b", "static", "temporal")


class AuthorizationError(PermissionError):
    """The held-out period cannot be used without the user's recorded authorization."""


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def require_authorization(split: str, root: pathlib.Path = ROOT) -> dict:
    if split not in SPLITS:
        raise ValueError(f"unknown split '{split}'")
    if split == "val":
        return {"split": "val", "authorization_required": False}
    p = root / AUTH
    if not p.exists():
        raise AuthorizationError(f"the held-out period is sealed: {AUTH} does not exist. It is created only by the user, quoting the protocol's SHA-256 and "
                                 f"the sentence beginning '{AUTH_PHRASE}' (see {PROTOCOL} section 9). Nothing was read.")
    proto_hash = sha256_file(root / PROTOCOL)
    text = p.read_text(encoding="utf-8")
    if proto_hash not in text or AUTH_PHRASE not in text:
        raise AuthorizationError(f"{AUTH} does not quote the current protocol hash {proto_hash} together with the authorization sentence; refusing to read held-out data.")
    return {"split": "heldout", "authorization_required": True, "authorization_sha256": sha256_file(p), "protocol_sha256": proto_hash}


def frozen_registry() -> dict:
    """Every artifact the protocol freezes, with the path that must exist."""
    reg = {}
    for name, d in TABULAR.items():
        reg[name] = {"family": "tabular", "dir": d, "files": ["model.joblib", "feature_builder/feature_names.json", "metrics_val.json"]}
    reg["B_full"] = {"family": "b", "dir": B_FULL, "files": ["model.joblib", "feature_columns.json", "metrics_val.json"]}
    for name, (pat, kind) in {**STATIC, **TEMPORAL}.items():
        for s in SEEDS:
            reg[f"{name}_s{s}"] = {"family": "static" if name in STATIC else "temporal", "dir": pat.format(s=s), "files": ["model_checkpoint.pt", "metrics_val.json"]}
    return reg


def recorded_threshold(model_dir: pathlib.Path) -> float:
    m = json.loads((model_dir / "metrics_val.json").read_text(encoding="utf-8"))
    if "pooled" in m and "threshold_selection" in m:                    # comparison B arms
        return float(m["threshold_selection"]["threshold"])
    if "pooled" in m:                                                  # neural runs
        return float(m["pooled"]["threshold"])
    return float(m["threshold"])                                      # tabular comparison A


def evaluate_frame_metrics(y, score, segment, day, threshold: float) -> dict:
    """Same metric code as the Phase 12 table (src/evidence/table.evaluate_frame)."""
    from src.evidence import table as T
    frame = pd.DataFrame({"isFraud": np.asarray(y), "y_score": np.asarray(score), "segment": np.asarray(segment), "day": np.asarray(day)})
    return T.evaluate_frame(frame, threshold, windows=False)


# ------------------------------------------------------------------------------------------------ scoring by family
LABEL = "isFraud"


def read_features(path) -> pd.DataFrame:
    """Feature frame with the label column projected away at read time: no model input frame ever holds a label."""
    import pyarrow.parquet as pq
    names = [n for n in pq.read_schema(path).names if n != LABEL]
    return pq.read_table(path, columns=names).to_pandas()


def _tabular_scores(cfg, split: str, root: pathlib.Path) -> tuple:
    import joblib
    from src.models.lr_features import LogisticRegressionFeatureBuilder
    from src.models.tree_features import TreeFeatureBuilder
    df = read_features(root / SPLITS[split]["features"])
    ids = df["TransactionID"].to_numpy()
    out = {}
    for name, d in TABULAR.items():
        model = joblib.load(root / d / "model.joblib")
        if name == "logistic_regression":
            X = LogisticRegressionFeatureBuilder.load(cfg, root / d / "feature_builder").transform(df)
        else:
            X = TreeFeatureBuilder.load(cfg, root / d / "feature_builder").to_matrix(df, for_model="rf" if name == "random_forest" else "xgb")
        out[name] = pd.DataFrame({"TransactionID": ids, "y_score": model.predict_proba(X)[:, 1]})
    return out, df


def _entity_arrays(raw: dict, n: int) -> dict:
    ent = {}
    for rel, key in (("charged_to", "pa"), ("used_card", "card"), ("via_network", "net"), ("from_device", "dev")):
        src, dst, _ = raw["relations"][rel]
        a = np.full(n, -1, dtype=np.int64)
        m = src < n
        a[src[m]] = dst[m]
        ent[key] = a
    return ent


def _b_scores(cfg, split: str, root: pathlib.Path, raw: dict, model_dir: str = B_FULL, name: str = "B_full") -> dict:
    import joblib
    from src.features import graph_features as G
    from src.models.tree_features import TreeFeatureBuilder
    hi = SPLITS[split]["days"][1]
    tr = read_features(root / "data/features/train_features.parquet")
    va = read_features(root / "data/features/val_features.parquet")
    frames = [tr, va] + ([read_features(root / SPLITS["heldout"]["features"])] if split == "heldout" else [])
    full = pd.concat(frames, ignore_index=True)
    n = len(full)
    if not np.array_equal(full["TransactionID"].to_numpy(), raw["txn_id"][:n]):
        raise RuntimeError("feature frames and graph arrays are not aligned by TransactionID")
    ent = _entity_arrays(raw, n)
    gf = G.compute_graph_features(full["TransactionDT"].to_numpy(dtype=np.int64), full["TransactionAmt"].to_numpy(dtype=np.float64), ent)
    builder = TreeFeatureBuilder.load(cfg, root / TABULAR["xgboost"] / "feature_builder")
    part = full.iloc[len(tr) + (0 if split == "val" else len(va)):]
    start = len(tr) if split == "val" else len(tr) + len(va)
    Xa = builder.to_matrix(part, for_model="xgb")
    X = np.hstack([Xa, gf.to_numpy(dtype=np.float32)[start:]])
    cols = json.loads((root / model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    assert X.shape[1] == len(cols) == 510 and cols == list(builder.feature_names_) + G.feature_names()
    model = joblib.load(root / model_dir / "model.joblib")
    return {name: pd.DataFrame({"TransactionID": part["TransactionID"].to_numpy(), "y_score": model.predict_proba(X)[:, 1]})}


def _static_scores(cfg, split: str, root: pathlib.Path, dev, names, table=None) -> dict:
    import torch
    from sklearn.preprocessing import StandardScaler
    from src.graph.rolling_eval import rolling_predict
    from src.models.gat import HeteroGAT
    from src.models.graphsage import HeteroSAGE
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    day = graph["transaction"].day.numpy()
    x = graph["transaction"].x.numpy()
    scaler = StandardScaler().fit(x[(day >= cfg.split.train_day_min) & (day <= cfg.split.train_day_max)])
    graph["transaction"].x = torch.tensor(scaler.transform(x), dtype=torch.float32)
    graph = graph.to(dev)
    dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    lo, hi = SPLITS[split]["days"]
    table = STATIC if table is None else table
    out = {}
    for name in names:
        pat, kind = table[name]
        for s in SEEDS:
            model = (HeteroSAGE(graph.metadata(), dims, hidden=64, dropout=0.3) if kind == "sage"
                     else HeteroGAT(graph.metadata(), dims, hidden=64, heads=4, dropout=0.3)).to(dev)
            model.load_state_dict(torch.load(root / pat.format(s=s) / "model_checkpoint.pt", map_location=dev))
            ids, sc, _ = rolling_predict(model, graph, list(range(lo, hi + 1)))
            out[f"{name}_s{s}"] = pd.DataFrame({"TransactionID": ids, "y_score": sc})
            del model
            torch.cuda.empty_cache()
    return out


def _temporal_scores(cfg, split: str, root: pathlib.Path, dev, raw: dict, names, table=None) -> dict:
    import torch
    from src.graph.temporal import build_temporal_dataset
    from src.models.temporal_graphsage import TemporalGraphSAGE, rolling_temporal_predict
    from src.models.temporal_meanmem import TemporalGraphSAGEMeanMemory
    lo, hi = SPLITS[split]["days"]
    ds = build_temporal_dataset(raw, hi, dev, cfg, allow_test=(split == "heldout"), with_labels=False)
    table = TEMPORAL if table is None else table
    out = {}
    for name in names:
        pat, kind = table[name]
        for s in SEEDS:
            dim = raw["X"].shape[1]
            model = (TemporalGraphSAGE(dim) if kind == "temporal" else TemporalGraphSAGEMeanMemory(dim, hidden=64, dropout=0.3)).to(dev)
            model.load_state_dict(torch.load(root / pat.format(s=s) / "model_checkpoint.pt", map_location=dev))
            ids, sc, _ = rolling_temporal_predict(model, ds, lo, hi)
            out[f"{name}_s{s}"] = pd.DataFrame({"TransactionID": ids, "y_score": sc})
            del model
            torch.cuda.empty_cache()
    return out


def truncate_raw(raw: dict, max_day: int) -> dict:
    """Drop every row with day > max_day immediately after loading (as the Phase 9 loader does), so later-period rows are never used."""
    n = int(np.searchsorted(raw["day"], max_day, side="right"))
    rel = {}
    for k, (src, dst, et) in raw["relations"].items():
        m = src < n
        rel[k] = (src[m], dst[m], et[m])
    return {"day": raw["day"][:n], "X": raw["X"][:n], "y": raw["y"][:n], "txn_id": raw["txn_id"][:n], "relations": rel, "n_entities": raw["n_entities"]}


def run(split: str, families=FAMILIES, root: pathlib.Path = ROOT) -> dict:
    """Score the frozen models on ``split``. Returns {"scores": {model_id: DataFrame}, "labels": DataFrame(TransactionID, isFraud, day, segment), "auth": dict}."""
    auth = require_authorization(split, root)
    import torch
    from src.config import load_config
    from src.graph.schema import compute_pseudo_account_segments
    from src.graph.temporal import load_raw_arrays
    cfg = load_config()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw, graph_cpu, _ = load_raw_arrays(cfg)
    seg = compute_pseudo_account_segments(graph_cpu)
    lo, hi = SPLITS[split]["days"]
    raw = truncate_raw(raw, hi)
    seg = seg[:len(raw["day"])]
    m = (raw["day"] >= lo) & (raw["day"] <= hi)
    labels = pd.DataFrame({"TransactionID": raw["txn_id"][m], "isFraud": raw["y"][m], "day": raw["day"][m], "segment": seg[m]})
    scores = {}
    if "tabular" in families:
        t, _ = _tabular_scores(cfg, split, root)
        scores.update(t)
    if "b" in families:
        scores.update(_b_scores(cfg, split, root, raw))
    if "static" in families:
        scores.update(_static_scores(cfg, split, root, dev, list(STATIC)))
    if "temporal" in families:
        scores.update(_temporal_scores(cfg, split, root, dev, raw, list(TEMPORAL)))
    return {"scores": scores, "labels": labels, "auth": auth}
