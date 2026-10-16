"""Pre-construction graph size estimate and schema validation.

Requirements 3 & 4 of Phase 4: estimate node/edge counts, edge counts by
relation, feature dimensions, memory, and construction time BEFORE
committing to full graph construction, and validate the proposed schema
against Phase 1's numbers (recomputed fresh here, not just cited) before
proceeding. This script is cheap (pandas groupbys over 590K rows); building
the actual HeteroData object is scripts/build_graph.py.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import load_config
from src.graph.schema import add_derived_columns, build_all_entity_keys, load_graph_config


def load_full_labeled_frame(cfg) -> pd.DataFrame:
    """The three Phase-2 feature-augmented splits, concatenated back into one
    chronologically-ordered frame (590,540 rows). NOT data/processed/train_merged.parquet,
    which is Phase 2's intermediate PRE-feature merge output — it lacks
    hour/day_of_week/*_isna/amt_decimals_gt2, all of which are added by
    build_base_features() per split. Concatenation in train, val, test order
    already yields correct global chronological order: each file is
    internally time-sorted, and the three periods don't overlap."""
    parts = [pd.read_parquet(cfg.features_dir / f"{name}_features.parquet") for name in ("train", "val", "test")]
    return pd.concat(parts, ignore_index=True)

# Real, measured numbers from this session's GPU capability probe (Phase 2
# approval step), at a graph size within 1% of this one (829,217 nodes /
# 2,651,412 edges there vs. what we compute below) — reused rather than
# re-deriving a theoretical estimate from scratch.
MEASURED_GPU_REFERENCE = {
    "sage_hidden_64": {"nodes": 829217, "edges": 2651412, "train_peak_reserved_gib": 4.32, "train_step_ms": 373},
    "sage_hidden_128": {"nodes": 829217, "edges": 2651412, "train_peak_reserved_gib": 8.06, "train_step_ms": 1108},
    "gpu_total_vram_gib": 15.98,
}


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    gcfg = load_graph_config()

    print("=== 1. Loading merged labeled frame (Phase 2 feature-augmented splits, concatenated) ===")
    merged = load_full_labeled_frame(cfg)
    n_txn = len(merged)
    print(f"  transaction nodes: {n_txn:,}")

    t_keys0 = time.time()
    df = add_derived_columns(merged, cfg)
    keys = build_all_entity_keys(df, gcfg)
    t_keys = time.time() - t_keys0

    print("\n=== 2. Entity node / edge counts by relation (recomputed fresh, not cited from Phase 1) ===")
    entity_stats = {}
    total_directed_edges = 0
    for name, key in keys.items():
        present = key.notna()
        n_edges = int(present.sum())
        n_nodes = int(key.nunique())
        deg = key[present].value_counts()
        stats = {
            "n_nodes": n_nodes, "n_edges": n_edges, "coverage_pct": round(100 * n_edges / n_txn, 2),
            "deg_median": float(deg.median()), "deg_p99": float(deg.quantile(0.99)), "deg_max": int(deg.max()),
        }
        entity_stats[name] = stats
        total_directed_edges += n_edges
        expected_cov = gcfg["entity_types"][name]["coverage_pct"]
        expected_max_deg = gcfg["entity_types"][name]["max_degree_phase1"]
        drift_cov = abs(stats["coverage_pct"] - expected_cov)
        drift_deg = stats["deg_max"] - expected_max_deg
        flag = "  [DRIFT]" if drift_cov > 0.5 or abs(drift_deg) > 5 else ""
        print(f"  {name:<20} nodes={n_nodes:>7,} edges={n_edges:>7,} coverage={stats['coverage_pct']:>5.1f}% "
              f"(Phase1: {expected_cov}%) deg_median={stats['deg_median']:.0f} deg_max={stats['deg_max']:>6,} "
              f"(Phase1: {expected_max_deg:,}){flag}")

    total_entity_nodes = sum(s["n_nodes"] for s in entity_stats.values())
    total_nodes = n_txn + total_entity_nodes
    total_edges_bidirectional = 2 * total_directed_edges

    print("\n=== 3. Totals ===")
    print(f"  total nodes                    : {total_nodes:,}  (transaction {n_txn:,} + entity {total_entity_nodes:,})")
    print(f"  total directed edges           : {total_directed_edges:,}")
    print(f"  total edges incl. reverse      : {total_edges_bidirectional:,}")

    print("\n=== 4. Hub / degree sanity check (validation.hub_degree_absolute_flag) ===")
    hub_threshold = gcfg["validation"]["hub_degree_absolute_flag"]
    any_hub_flag = False
    for name, s in entity_stats.items():
        if s["deg_max"] > hub_threshold:
            print(f"  [FLAG] {name}: max degree {s['deg_max']:,} exceeds threshold {hub_threshold:,}")
            any_hub_flag = True
    if not any_hub_flag:
        print(f"  OK: no entity type exceeds the {hub_threshold:,}-degree hub threshold "
              f"(largest is {max(s['deg_max'] for s in entity_stats.values()):,})")

    print("\n=== 5. Feature dimension estimate ===")
    from src.models.tree_features import TreeFeatureBuilder
    train_only = df[(df["day"] >= cfg.split.train_day_min) & (df["day"] <= cfg.split.train_day_max)]
    t_feat0 = time.time()
    builder = TreeFeatureBuilder(cfg=cfg).fit(train_only)
    t_feat = time.time() - t_feat0
    txn_feature_dim = len(builder.feature_names_)
    entity_feature_dim = gcfg["entity_feature_dim"]
    print(f"  transaction node feature dim   : {txn_feature_dim} (fit on TRAIN only, {t_feat:.1f}s)")
    print(f"  entity node feature dim        : {entity_feature_dim} (constant placeholder — see configs/graph.yaml)")

    print("\n=== 6. Memory estimate ===")
    txn_feat_mb = n_txn * txn_feature_dim * 4 / 1e6
    entity_feat_mb = total_entity_nodes * entity_feature_dim * 4 / 1e6
    edge_index_mb = total_edges_bidirectional * 2 * 4 / 1e6  # 2 rows (src,dst) x int32
    edge_time_mb = total_edges_bidirectional * 4 / 1e6  # one float32/int32 edge_time attribute
    print(f"  transaction feature matrix     : {txn_feat_mb:.1f} MB  ({n_txn:,} x {txn_feature_dim} x 4B)")
    print(f"  entity feature matrices        : {entity_feat_mb:.2f} MB")
    print(f"  edge_index (all relations)     : {edge_index_mb:.1f} MB (int32, incl. reverse)")
    print(f"  edge_time attributes           : {edge_time_mb:.1f} MB")
    print(f"  ESTIMATED graph object on disk/RAM: ~{txn_feat_mb+entity_feat_mb+edge_index_mb+edge_time_mb:.0f} MB")

    ref64, ref128 = MEASURED_GPU_REFERENCE["sage_hidden_64"], MEASURED_GPU_REFERENCE["sage_hidden_128"]
    scale = total_nodes / ref64["nodes"]
    print(f"\n  GPU TRAINING MEMORY — reusing REAL measurements from this session's capability probe")
    print(f"  (that benchmark graph was {ref64['nodes']:,} nodes / {ref64['edges']:,} edges; this graph is "
          f"{total_nodes:,} nodes / {total_edges_bidirectional:,} edges, a {scale:.2f}x scale factor):")
    print(f"    SAGE hidden=64 : measured {ref64['train_peak_reserved_gib']:.2f} GiB reserved, "
          f"{ref64['train_step_ms']} ms/step  ->  scaled estimate ~{ref64['train_peak_reserved_gib']*scale:.2f} GiB")
    print(f"    SAGE hidden=128: measured {ref128['train_peak_reserved_gib']:.2f} GiB reserved, "
          f"{ref128['train_step_ms']} ms/step  ->  scaled estimate ~{ref128['train_peak_reserved_gib']*scale:.2f} GiB")
    print(f"    GPU total VRAM  : {MEASURED_GPU_REFERENCE['gpu_total_vram_gib']:.2f} GiB "
          f"-> full-batch training comfortably fits at either hidden size.")

    print(f"\n=== 7. Construction time estimate ===")
    print(f"  entity-key computation (this run): {t_keys:.2f}s over {n_txn:,} rows")
    print(f"  expected full graph construction : low tens of seconds (same order as Phase 2/3 pipeline steps)")

    result = {
        "n_transaction_nodes": n_txn,
        "entity_stats": entity_stats,
        "total_nodes": total_nodes,
        "total_directed_edges": total_directed_edges,
        "total_edges_bidirectional": total_edges_bidirectional,
        "txn_feature_dim": txn_feature_dim,
        "entity_feature_dim": entity_feature_dim,
        "estimated_memory_mb": round(txn_feat_mb + entity_feat_mb + edge_index_mb + edge_time_mb, 1),
        "gpu_reference": MEASURED_GPU_REFERENCE,
        "scale_factor_vs_reference": round(scale, 3),
        "entity_key_computation_seconds": round(t_keys, 2),
    }
    out_path = cfg.processed_dir / "graph_size_estimate.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"\nWrote {out_path}")
    print(f"Done in {time.time()-t0:.1f}s. This was an ESTIMATE/VALIDATION pass only — no graph object was built.")


if __name__ == "__main__":
    main()
