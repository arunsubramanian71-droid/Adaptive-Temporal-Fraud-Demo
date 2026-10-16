"""Run all graph validation checks (Phase 4 requirement 12) against the
built graph. Exits nonzero if any check fails.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.graph import validate as V
from src.graph.schema import load_graph_config


def main() -> int:
    cfg = load_config()
    gcfg = load_graph_config()
    graph_dir = cfg.processed_dir / "graph"

    graph = torch.load(graph_dir / "hetero_graph.pt", weights_only=False)
    entity_id_maps = {}
    for name in gcfg["entity_types"]:
        with open(graph_dir / f"entity_id_map_{name}.json", "r", encoding="utf-8") as fh:
            entity_id_maps[name] = json.load(fh)
    with open(cfg.processed_dir / "graph_size_estimate.json", "r", encoding="utf-8") as fh:
        estimate = json.load(fh)
    expected_edges = {name: s["n_edges"] for name, s in estimate["entity_stats"].items()}

    results = []

    def run(name, fn, *args):
        try:
            ok, detail = fn(*args)
            results.append((name, "PASS" if ok else "FAIL", detail))
        except Exception as e:
            results.append((name, "ERROR", f"{type(e).__name__}: {e}"))

    run("no_duplicate_edges", V.check_no_duplicate_edges, graph)
    run("self_loops", V.check_self_loops, graph)
    run("temporal_consistency", V.check_temporal_consistency, graph)
    run("temporal_masking_mechanism", V.check_temporal_masking_mechanism, graph,
        [cfg.split.train_day_max, cfg.split.val_day_max, cfg.split.test_day_max, 60, 1])
    run("extreme_degree", V.check_extreme_degree, graph, gcfg["validation"]["hub_degree_absolute_flag"])
    run("no_isolated_transactions", V.check_no_isolated_transactions, graph)
    run("entity_id_maps_fully_used", V.check_entity_id_maps_fully_used, graph, entity_id_maps)
    run("edge_count_matches_estimate", V.check_edge_count_matches_coverage, graph, expected_edges)

    print("=== Graph validation results ===")
    n_pass = 0
    for name, status, detail in results:
        print(f"[{status:5s}] {name} — {detail}")
        n_pass += status == "PASS"
    print(f"\n{n_pass}/{len(results)} checks passed.")

    out_path = cfg.processed_dir / "graph" / "validation_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump([{"check": n, "status": s, "detail": d} for n, s, d in results], fh, indent=1)
    print(f"Wrote {out_path}")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
