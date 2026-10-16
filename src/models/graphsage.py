"""GraphSAGE model (the official Phase 5 experiment) and its non-relational
control (Phase 5 ablation item B).

Kept deliberately small — 2 layers, hidden=64 — per the instruction not to
reach for a large architecture before a small one has been tried.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroSAGE(torch.nn.Module):
    """Per-type input projection (node types have different raw feature
    dimensions: transaction=470, every entity type=1 — Phase 4, Section B)
    -> 2 HeteroConv/SAGEConv layers -> dropout -> linear head on the
    transaction embedding only (fraud labels exist at the transaction level,
    Phase 4 requirement 6)."""

    def __init__(self, metadata, feature_dims: dict[str, int], hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.proj = torch.nn.ModuleDict({nt: torch.nn.Linear(feature_dims[nt], hidden) for nt in metadata[0]})
        self.conv1 = HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in metadata[1]}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in metadata[1]}, aggr="sum")
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        h = {k: F.relu(self.proj[k](v)) for k, v in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        return self.head(h["transaction"]).squeeze(-1)


class MLPControl(torch.nn.Module):
    """Non-relational control (Phase 5 ablation item B): identical hidden
    width (64), depth (2 hidden layers), dropout, loss, and training
    protocol as HeteroSAGE, but with NO graph structure at all — a plain
    MLP on the exact same 470-dim transaction feature vector.

    This isolates the graph's marginal contribution while holding feature
    representation, capacity, and training recipe fixed. See the Phase 5
    report for why this was chosen over zeroing out edges inside the same
    HeteroConv architecture (SAGEConv's neighbor-aggregation term is
    ill-defined / degenerate for a node with zero neighbors, which every
    entity node would have under an edges-removed variant of the same
    model — a separate, equal-capacity MLP is the methodologically clean
    choice, not a patched version of the graph model)."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        return self.head(h).squeeze(-1)
