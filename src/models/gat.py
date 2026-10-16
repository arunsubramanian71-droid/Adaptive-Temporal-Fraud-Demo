"""GAT model — the Phase 6B experiment.

Deliberately dimension-matched to src/models/graphsage.py's HeteroSAGE:
same hidden=64 throughout, same 2-layer depth, same dropout, same
classifier head. The only architectural difference is the aggregation
mechanism (learned attention weights vs. GraphSAGE's mean/sum pooling) —
this is what makes "does attention help" a fair question to ask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


class HeteroGAT(torch.nn.Module):
    """Per-type input projection -> 2 HeteroConv/GATConv layers -> dropout
    -> linear head on the transaction embedding only.

    Layer 1 concatenates heads (standard GAT design, Veličković et al.):
    heads * per_head_dim = hidden, so the *total* representation width
    matches GraphSAGE's hidden=64 exactly after this layer, not
    heads-times-wider. Layer 2 averages heads (also standard — the final
    layer of a GAT typically averages rather than concatenates), producing
    hidden=64 directly into the classifier head.

    `add_self_loops=False` is required here, not a stylistic choice: every
    relation in this schema is genuinely bipartite (transaction and each
    entity type are disjoint node sets sharing no index space), so a
    self-loop is not a well-defined operation for GATConv on a
    tuple-shaped (src_dim, dst_dim) input — this exact combination was
    already verified working in this environment during the Phase 2 ROCm
    capability probe.
    """

    def __init__(self, metadata, feature_dims: dict[str, int], hidden: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads}) for exact-width concat")
        per_head = hidden // heads

        self.proj = torch.nn.ModuleDict({nt: torch.nn.Linear(feature_dims[nt], hidden) for nt in metadata[0]})
        self.conv1 = HeteroConv({
            et: GATConv((hidden, hidden), per_head, heads=heads, concat=True, dropout=dropout, add_self_loops=False)
            for et in metadata[1]
        }, aggr="sum")
        self.conv2 = HeteroConv({
            et: GATConv((hidden, hidden), hidden, heads=heads, concat=False, dropout=dropout, add_self_loops=False)
            for et in metadata[1]
        }, aggr="sum")
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        h = {k: F.relu(self.proj[k](v)) for k, v in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        return self.head(h["transaction"]).squeeze(-1)
