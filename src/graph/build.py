"""Heterogeneous graph construction and temporal masking.

Design summary (see reports/phase4_graph_report.md for the full rationale):

  - ONE HeteroData object holds every transaction (all three chronological
    splits) and every entity node. Edges are static, observed facts ("this
    transaction used this card") that don't change with time, so they all
    belong in one graph object — nothing here is a leakage channel by
    itself.
  - The leakage control is `edge_time` (the source transaction's own day
    index, 1-182) stored on every edge, plus `apply_time_mask(graph, D)`,
    which returns a NEW graph containing only edges with edge_time <= D.
    This needs no compiled sampler (pyg-lib / torch-sparse are explicitly
    not installed — see Phase 4's environment constraint): masking is a
    boolean comparison over a fixed-size tensor, tractable at any graph
    size that fits in memory at all.
  - Entity nodes carry NO precomputed feature vector (just a constant
    placeholder) — their representation is meant to come entirely from
    time-masked message passing over neighbouring transactions. This is a
    deliberate leakage-avoidance choice: any precomputed entity statistic
    (e.g. "this card's total degree") would itself need the exact same
    backward-looking discipline as Phase 3's historical features, and
    skipping it here removes an entire class of possible mistakes.
  - `apply_time_mask` supports an ARBITRARY day cutoff, not just the three
    split boundaries — Phase 4 builds and validates this mechanism; the
    correct Phase 5 evaluation protocol is a per-day (or finer) rolling
    cutoff, not one coarse per-split cutoff (documented in the Phase 4
    report, Section F).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from src.config import Config
from src.graph.schema import build_all_entity_keys


def build_hetero_graph(df: pd.DataFrame, cfg: Config, gcfg: dict, txn_features: np.ndarray) -> tuple[HeteroData, dict]:
    """df must already have `day` and entity-key component columns (from
    src.graph.schema.add_derived_columns) and be in a stable, arbitrary row
    order — transaction node index i corresponds to df.iloc[i].

    Returns (graph, entity_id_maps) where entity_id_maps[name] is a
    {key_string: node_index} dict, saved separately for reproducibility
    (Phase 4 requirement 18) since it's needed to map new data onto the
    same entity nodes later.
    """
    data = HeteroData()
    n_txn = len(df)

    data["transaction"].x = torch.tensor(txn_features, dtype=torch.float32)
    data["transaction"].y = torch.tensor(df["isFraud"].to_numpy(), dtype=torch.float32)
    data["transaction"].transaction_id = torch.tensor(df["TransactionID"].to_numpy(), dtype=torch.int64)
    data["transaction"].day = torch.tensor(df["day"].to_numpy(), dtype=torch.int32)

    day_arr = df["day"].to_numpy()
    keys = build_all_entity_keys(df, gcfg)
    entity_id_maps: dict[str, dict] = {}

    for name, spec in gcfg["entity_types"].items():
        key = keys[name]
        present = key.notna().to_numpy()
        cats = sorted(key[present].unique().tolist())  # sorted -> reproducible node indexing across rebuilds
        id_map = {c: i for i, c in enumerate(cats)}
        entity_id_maps[name] = id_map
        n_entity = len(cats)

        data[name].x = torch.full((n_entity, gcfg["entity_feature_dim"]), 1.0, dtype=torch.float32)
        data[name].num_nodes = n_entity

        src_idx = np.nonzero(present)[0]
        dst_idx = key.to_numpy()[present]
        dst_idx = np.array([id_map[k] for k in dst_idx], dtype=np.int64)

        rel_name = next(r for r, spec2 in gcfg["edge_types"].items() if spec2["entity"] == name)
        edge_index = torch.tensor(np.stack([src_idx.astype(np.int64), dst_idx]), dtype=torch.int64)
        edge_time = torch.tensor(day_arr[src_idx], dtype=torch.int32)
        data["transaction", rel_name, name].edge_index = edge_index
        data["transaction", rel_name, name].edge_time = edge_time

    data = T.ToUndirected()(data)  # copies edge_time onto the auto-generated reverse relations too
    return data, entity_id_maps


def apply_time_mask(graph: HeteroData, as_of_day: int) -> HeteroData:
    """Returns a NEW HeteroData containing only edges with edge_time <=
    as_of_day, for every edge type (forward and reverse alike). Node sets
    and features are unchanged — nodes aren't removed, only the edges that
    would let information flow through them from the future. This is the
    entire temporal-leakage control mechanism; no sampler is involved."""
    masked = HeteroData()
    for nt in graph.node_types:
        for k, v in graph[nt].items():
            masked[nt][k] = v
    for et in graph.edge_types:
        store = graph[et]
        m = store.edge_time <= as_of_day
        masked[et].edge_index = store.edge_index[:, m]
        masked[et].edge_time = store.edge_time[m]
    return masked
