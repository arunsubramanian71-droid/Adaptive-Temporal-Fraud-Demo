"""Graph validation checks (Phase 4 requirement 12).

Plain functions returning (ok: bool, detail: str), following the same
pattern as src/data/validate.py — runnable without pytest via
scripts/validate_graph.py.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.graph.build import apply_time_mask


def check_no_duplicate_edges(graph: HeteroData) -> tuple[bool, str]:
    bad = []
    for et in graph.edge_types:
        ei = graph[et].edge_index
        pairs = ei.T.numpy()
        n_unique = len(np.unique(pairs, axis=0))
        if n_unique != pairs.shape[0]:
            bad.append(f"{et}: {pairs.shape[0] - n_unique} duplicate edges")
    return (not bad), "; ".join(bad) if bad else "no duplicate (src,dst) pairs in any relation"


def check_self_loops(graph: HeteroData) -> tuple[bool, str]:
    """Every edge type here is Transaction<->Entity (bipartite): src and dst
    index spaces belong to different node types, so a 'self-loop' in the
    graph-theoretic sense (src==dst node) is structurally impossible, not
    merely absent. Reported explicitly rather than silently skipped, per
    the requirement to check for self-loops 'where inappropriate'."""
    for et in graph.edge_types:
        if et[0] == et[2]:
            return False, f"UNEXPECTED same-type relation {et} — this schema should be purely bipartite"
    return True, "all relations are Transaction<->Entity (bipartite by construction); a same-node-type self-loop is not structurally possible here"


def check_temporal_consistency(graph: HeteroData) -> tuple[bool, str]:
    """edge_time must exactly equal the SOURCE transaction's own day, for
    every forward (transaction->entity) edge — it is derived directly from
    it, so any mismatch would indicate a construction bug."""
    bad = []
    txn_day = graph["transaction"].day
    for et in graph.edge_types:
        if et[0] != "transaction":
            continue  # reverse edges: source is the entity, day isn't defined for it
        ei, et_time = graph[et].edge_index, graph[et].edge_time
        expected = txn_day[ei[0]]
        if not torch.equal(expected.to(et_time.dtype), et_time):
            n_bad = int((expected.to(et_time.dtype) != et_time).sum())
            bad.append(f"{et}: {n_bad} edges where edge_time != source transaction's day")
    return (not bad), "; ".join(bad) if bad else "edge_time exactly matches source transaction day on every forward edge"


def check_temporal_masking_mechanism(graph: HeteroData, cutoffs: list[int]) -> tuple[bool, str]:
    """The actual leakage-prevention mechanism, exercised end-to-end: for
    each cutoff D, mask the graph and confirm (a) no edge with edge_time > D
    survives, and (b) every transaction with day > D has zero edges in the
    masked graph (an over-D transaction must be completely unreachable from
    the graph's perspective at that cutoff)."""
    txn_day = graph["transaction"].day
    problems = []
    for D in cutoffs:
        g = apply_time_mask(graph, D)
        max_time = max((int(g[et].edge_time.max()) for et in g.edge_types if g[et].edge_index.numel() > 0), default=-1)
        if max_time > D:
            problems.append(f"cutoff={D}: masked graph still contains edge_time={max_time}")
        future_txn = (txn_day > D).nonzero(as_tuple=True)[0]
        if len(future_txn) > 0:
            future_set = set(future_txn.tolist())
            leaked = 0
            for et in g.edge_types:
                if et[0] != "transaction":
                    continue
                srcs = set(g[et].edge_index[0].tolist())
                leaked += len(srcs & future_set)
            if leaked:
                problems.append(f"cutoff={D}: {leaked} future-transaction source-edges survived masking")
    ok = not problems
    detail = "; ".join(problems) if problems else f"verified at cutoffs {cutoffs}: zero future edges, zero reachable future transactions"
    return ok, detail


def check_extreme_degree(graph: HeteroData, hub_threshold: int) -> tuple[bool, str]:
    flags = []
    for et in graph.edge_types:
        if et[0] != "transaction":
            continue
        dst = et[2]
        n_entity = graph[dst].num_nodes
        deg = torch.bincount(graph[et].edge_index[1], minlength=n_entity)
        max_deg = int(deg.max()) if len(deg) else 0
        if max_deg > hub_threshold:
            flags.append(f"{dst}: max degree {max_deg} exceeds {hub_threshold}")
    return (not flags), "; ".join(flags) if flags else f"no entity node type exceeds the {hub_threshold}-degree hub threshold"


def check_no_isolated_transactions(graph: HeteroData) -> tuple[bool, str]:
    """Every transaction has card1 (100% coverage, Phase 1) so every
    transaction node should have at least the used_card edge."""
    n_txn = graph["transaction"].num_nodes
    has_edge = torch.zeros(n_txn, dtype=torch.bool)
    for et in graph.edge_types:
        if et[0] == "transaction":
            has_edge[graph[et].edge_index[0]] = True
    n_isolated = int((~has_edge).sum())
    return n_isolated == 0, (f"{n_isolated} transaction nodes have zero entity edges (unexpected — card1 has 100% coverage)"
                              if n_isolated else "0 isolated transaction nodes (every transaction has >=1 entity edge)")


def check_entity_id_maps_fully_used(graph: HeteroData, entity_id_maps: dict[str, dict]) -> tuple[bool, str]:
    """Every entity node index must be the destination of at least one
    edge — by construction (id maps are built FROM the observed keys) this
    should always hold; checked as a construction-code sanity guard."""
    problems = []
    for name, id_map in entity_id_maps.items():
        rel = next(et for et in graph.edge_types if et[0] == "transaction" and et[2] == name)
        used = torch.zeros(len(id_map), dtype=torch.bool)
        used[graph[rel].edge_index[1]] = True
        n_unused = int((~used).sum())
        if n_unused:
            problems.append(f"{name}: {n_unused} node indices never referenced by an edge")
    return (not problems), "; ".join(problems) if problems else "every entity node index is the destination of >=1 edge"


def check_edge_count_matches_coverage(graph: HeteroData, expected_edges: dict[str, int]) -> tuple[bool, str]:
    """Cross-check: the built graph's forward-edge counts must exactly
    match the pre-construction estimate (scripts/estimate_graph_schema.py)
    for the same entity types — confirms the two code paths agree."""
    problems = []
    for et in graph.edge_types:
        if et[0] != "transaction":
            continue
        name = et[2]
        n = graph[et].edge_index.shape[1]
        if name in expected_edges and n != expected_edges[name]:
            problems.append(f"{name}: built {n:,} edges, estimate predicted {expected_edges[name]:,}")
    return (not problems), "; ".join(problems) if problems else "built edge counts match the pre-construction estimate exactly, for every relation"


ALL_CHECKS = [
    "check_no_duplicate_edges",
    "check_self_loops",
    "check_temporal_consistency",
    "check_extreme_degree",
    "check_no_isolated_transactions",
    "check_entity_id_maps_fully_used",
]
