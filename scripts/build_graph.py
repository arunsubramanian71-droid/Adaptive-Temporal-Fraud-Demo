"""Construct the heterogeneous graph and save all artifacts.

Run only after scripts/estimate_graph_schema.py has validated the schema
(Phase 4 requirements 3-4). Builds one HeteroData covering all 590,540
labeled transactions + 4 entity node types, with time-stamped edges and no
precomputed entity features — see src/graph/build.py's module docstring.
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg

from src.config import load_config
from src.features.preprocessing import assert_features_exclude_blocklist
from src.graph.build import apply_time_mask, build_hetero_graph
from src.graph.schema import add_derived_columns, load_graph_config
from src.models.tree_features import TreeFeatureBuilder

RANDOM_STATE = 42


def load_full_labeled_frame(cfg) -> pd.DataFrame:
    parts = [pd.read_parquet(cfg.features_dir / f"{name}_features.parquet") for name in ("train", "val", "test")]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    t_start = time.time()

    cfg = load_config()
    gcfg = load_graph_config()
    graph_dir = cfg.processed_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading full labeled frame (all 3 splits, chronological) ===")
    df = load_full_labeled_frame(cfg)
    df = add_derived_columns(df, cfg)
    print(f"  {len(df):,} transactions, day range {int(df['day'].min())}-{int(df['day'].max())}")

    print("\n=== 2. Fitting transaction-feature encoder on TRAIN only (Phase 3 tree_features, NaN-imputed 'rf' variant) ===")
    train_only = df[(df["day"] >= cfg.split.train_day_min) & (df["day"] <= cfg.split.train_day_max)]
    t0 = time.time()
    builder = TreeFeatureBuilder(cfg=cfg).fit(train_only)
    assert_features_exclude_blocklist(builder.feature_names_, cfg)
    txn_features = builder.to_matrix(df, for_model="rf")  # NaN-free: required for a GNN linear projection layer
    t_feat = time.time() - t0
    n_nan = np.isnan(txn_features).sum()
    print(f"  feature dim: {txn_features.shape[1]}  |  built in {t_feat:.1f}s  |  NaN count: {n_nan} (must be 0)")
    assert n_nan == 0, "transaction node features must be NaN-free for GNN linear layers"

    print("\n=== 3. Constructing HeteroData ===")
    t0 = time.time()
    graph, entity_id_maps = build_hetero_graph(df, cfg, gcfg, txn_features)
    t_build = time.time() - t0
    print(f"  built in {t_build:.1f}s")
    print(f"  node types: {graph.node_types}")
    print(f"  edge types: {graph.edge_types}")
    for nt in graph.node_types:
        print(f"    {nt:<20} {graph[nt].num_nodes:>7,} nodes  x_dim={graph[nt].x.shape[1]}")
    for et in graph.edge_types:
        print(f"    {str(et):<55} {graph[et].edge_index.shape[1]:>9,} edges")
    total_nodes = sum(graph[nt].num_nodes for nt in graph.node_types)
    total_edges = sum(graph[et].edge_index.shape[1] for et in graph.edge_types)
    print(f"  TOTAL: {total_nodes:,} nodes, {total_edges:,} edges (incl. reverse)")

    print("\n=== 4. Sanity: temporal masking mechanism (three split-boundary cutoffs) ===")
    for name, day in [("TRAIN end", cfg.split.train_day_max), ("VAL end", cfg.split.val_day_max),
                       ("TEST end", cfg.split.test_day_max)]:
        t0 = time.time()
        masked = apply_time_mask(graph, day)
        dt = time.time() - t0
        edges_at_cutoff = sum(masked[et].edge_index.shape[1] for et in masked.edge_types)
        print(f"  cutoff day={day:>3} ({name:>9}): {edges_at_cutoff:>9,} edges visible  (masked in {dt*1000:.1f}ms)")

    print("\n=== 5. Cold-start / warm-start counts per split, per entity type ===")
    print("  Definition matches Phase 3's historical.py EXACTLY (per-transaction, continuously-accumulating,")
    print("  strictly-prior history — including earlier-in-same-period transactions), generalized here from")
    print("  2 to all 4 graph entity types, so these numbers are directly comparable to the Phase 3 report.")
    from src.graph.schema import build_all_entity_keys
    keys = build_all_entity_keys(df, gcfg)  # df is already in chronological row order
    warm_flags = {}
    for name, key in keys.items():
        present = key.notna()
        prior_count = key.groupby(key).cumcount()  # NaN for rows whose key itself is NaN
        warm_flags[name] = present & (prior_count.fillna(0) > 0)

    cold_start_stats = {}
    for split_name, (lo, hi) in [("train", (cfg.split.train_day_min, cfg.split.train_day_max)),
                                  ("val", (cfg.split.val_day_min, cfg.split.val_day_max)),
                                  ("test", (cfg.split.test_day_min, cfg.split.test_day_max))]:
        split_mask = (df["day"] >= lo) & (df["day"] <= hi)
        stats = {}
        for name, key in keys.items():
            present = key.notna() & split_mask
            warm = warm_flags[name] & split_mask
            n_edge = int(present.sum())
            n_warm = int(warm.sum())
            stats[name] = {"n_transactions_with_this_edge": n_edge, "n_warm_start": n_warm, "n_cold_start": n_edge - n_warm}
        cold_start_stats[split_name] = stats
        print(f"  {split_name}:")
        for name, s in stats.items():
            pct_warm = 100 * s["n_warm_start"] / s["n_transactions_with_this_edge"] if s["n_transactions_with_this_edge"] else float("nan")
            print(f"    {name:<20} edges={s['n_transactions_with_this_edge']:>7,}  warm={s['n_warm_start']:>7,} ({pct_warm:.1f}%)  cold={s['n_cold_start']:>7,}")

    print("\n=== 6. Saving artifacts ===")
    graph_path = graph_dir / "hetero_graph.pt"
    torch.save(graph, graph_path)
    print(f"  wrote {graph_path} ({graph_path.stat().st_size/1e6:.1f} MB)")

    for name, id_map in entity_id_maps.items():
        p = graph_dir / f"entity_id_map_{name}.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(id_map, fh)
        print(f"  wrote {p} ({len(id_map):,} entries)")

    builder.save(graph_dir / "transaction_feature_builder")
    print(f"  wrote {graph_dir / 'transaction_feature_builder'}/")

    schema_record = {
        "graph_config_path": "configs/graph.yaml",
        "random_state": RANDOM_STATE,
        "node_types": {nt: {"num_nodes": int(graph[nt].num_nodes), "feature_dim": int(graph[nt].x.shape[1])}
                       for nt in graph.node_types},
        "edge_types": {str(et): int(graph[et].edge_index.shape[1]) for et in graph.edge_types},
        "split": {"train_days": [cfg.split.train_day_min, cfg.split.train_day_max],
                  "val_days": [cfg.split.val_day_min, cfg.split.val_day_max],
                  "test_days": [cfg.split.test_day_min, cfg.split.test_day_max]},
        "transaction_feature_dim": txn_features.shape[1],
        "transaction_feature_names": builder.feature_names_,
    }
    with open(graph_dir / "graph_schema.json", "w", encoding="utf-8") as fh:
        json.dump(schema_record, fh, indent=1)
    print(f"  wrote {graph_dir / 'graph_schema.json'}")

    with open(graph_dir / "cold_start_stats.json", "w", encoding="utf-8") as fh:
        json.dump(cold_start_stats, fh, indent=1)
    print(f"  wrote {graph_dir / 'cold_start_stats.json'}")

    metadata = {
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": pyg.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "config_path": "configs/data.yaml",
        "graph_config_path": "configs/graph.yaml",
        "feature_build_seconds": round(t_feat, 2),
        "graph_construction_seconds": round(t_build, 2),
        "total_script_seconds": round(time.time() - t_start, 2),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }
    with open(graph_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=1)
    print(f"  wrote {graph_dir / 'metadata.json'}")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
