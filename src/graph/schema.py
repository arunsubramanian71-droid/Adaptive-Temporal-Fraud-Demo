"""Graph schema: entity-key derivation and config loading.

The schema itself lives in configs/graph.yaml as data (same pattern as
configs/data.yaml) — this module just loads it and computes the actual key
columns on a dataframe. See that file's comments for the Phase 1 numbers
justifying every included/excluded entity type.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yaml

from src.config import Config, PROJECT_ROOT

GRAPH_CONFIG_PATH = PROJECT_ROOT / "configs" / "graph.yaml"


def load_graph_config(path: pathlib.Path = GRAPH_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def add_derived_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Adds the one derived column entity keys depend on: d1anchor = day -
    D1, the stable per-account origin date (Phase 1 Section 10). Does not
    mutate the input."""
    df = df.copy()
    day = (df["TransactionDT"] // cfg.day_seconds).astype(np.int32)
    df["day"] = day
    df["d1anchor"] = day - df["D1"].fillna(0)
    return df


def entity_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Builds a single string key from one or more columns, NaN-propagating:
    a row missing ANY component column gets NaN (no edge for that row) —
    this is what gives netproxy/device_fingerprint their measured Phase 1
    coverage percentages (23.6% / 12.0%), rather than silently coercing
    missing components to a placeholder that would create false shared
    entities between otherwise-unrelated transactions.
    """
    if any(c not in df.columns for c in columns):
        raise KeyError(f"missing column(s) for entity key: {[c for c in columns if c not in df.columns]}")
    all_present = df[columns].notna().all(axis=1)
    parts = [df[c].astype("string") for c in columns]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key.where(all_present)  # NaN where any component was missing


def build_all_entity_keys(df: pd.DataFrame, graph_cfg: dict) -> dict[str, pd.Series]:
    """Returns {entity_type_name: key_series} for every entity type in the
    schema, aligned to df's index."""
    keys = {}
    for name, spec in graph_cfg["entity_types"].items():
        keys[name] = entity_key(df, spec["columns"])
    return keys


def compute_pseudo_account_segments(graph) -> np.ndarray:
    """Same per-transaction, continuously-accumulating, strictly-prior
    methodology as Phase 3's historical.py / Phase 4's build_graph.py,
    Section H, applied to an already-built HeteroData graph. Returns a
    3-way segment label per transaction (aligned to graph['transaction']
    row order): 'has_edge_warm', 'has_edge_cold', or 'no_pseudo_account_edge'.

    Deliberately 3-way, not a 2-way warm/cold split — an earlier version of
    the Phase 5 GraphSAGE script lumped "has an edge but no prior history"
    together with "has NO pseudo_account edge at all" (missing addr1/D1,
    11.1% of rows) into one "cold_start" bucket. That no-edge group is
    strongly correlated with ProductCD='C' and has a 12.5% base fraud rate
    (vs. ~2% for the other two groups, Phase 1 Section 6) — conflating it
    with genuine cold-start inflated that bucket's apparent PR-AUC by
    picking up an easy, unrelated signal from node features rather than
    from graph structure. See the Phase 5 report, Section O.
    """
    day = graph["transaction"].day.detach().cpu().numpy()
    rel = next(et for et in graph.edge_types if et[0] == "transaction" and et[2] == "pseudo_account")
    ei = graph[rel].edge_index.detach().cpu().numpy()
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

    segment = np.full(len(day), "no_pseudo_account_edge", dtype=object)
    segment[has_edge & warm] = "has_edge_warm"
    segment[has_edge & ~warm] = "has_edge_cold"
    return segment
