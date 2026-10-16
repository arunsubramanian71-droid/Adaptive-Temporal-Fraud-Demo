"""Rolling per-day temporal evaluation — the mechanism the Phase 4 report
identified as mandatory: a single coarse per-split graph mask is NOT
sufficient, because it would let an early-in-split transaction see
structural information from a later-in-split transaction through a shared
entity, within the same forward pass.

Protocol (day granularity, matching every other temporal decision in this
project — split boundaries, TransactionDT's own derived resolution): to
score transactions on day t, mask the graph to `edge_time <= t` (Phase 4's
`apply_time_mask`) and keep only the output for transaction nodes whose OWN
day equals t. This is exact, not approximate, at day granularity:
information_time <= transaction_time becomes literally t <= t for same-day
information and (t' < t) <= t for anything earlier — the "<=" in the
project's leakage rule is satisfied without truncation. The one accepted
coarsening is that two transactions on the SAME day may see each other's
structural (non-label) contribution within that day's pass; finer-than-day
resolution was not pursued because TransactionDT's absolute clock alignment
is itself only established at day granularity (Phase 1, Section 9).

The graph should already be resident on the target device before calling
this — masking a GPU-resident graph stays entirely on-device (no repeated
host<->device transfer across the rolling loop), which is what makes 30-61
passes per evaluation cheap (Phase 4's feasibility benchmark measured ~85ms
per forward-only pass on this graph).
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.graph.build import apply_time_mask


@torch.no_grad()
def rolling_predict(model: torch.nn.Module, graph: HeteroData, days: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (transaction_id, score, day) arrays, concatenated across all
    `days`, each covering only that day's own transactions."""
    model.eval()
    txn_day = graph["transaction"].day
    txn_ids = graph["transaction"].transaction_id
    out_ids, out_scores, out_days = [], [], []
    for t in days:
        g_t = apply_time_mask(graph, t)
        logits = model(g_t.x_dict, g_t.edge_index_dict)
        day_idx = (txn_day == t).nonzero(as_tuple=True)[0]
        scores = torch.sigmoid(logits[day_idx]).detach().cpu().numpy()
        ids = txn_ids[day_idx].detach().cpu().numpy()
        out_ids.append(ids)
        out_scores.append(scores)
        out_days.append(np.full(len(day_idx), t))
    return np.concatenate(out_ids), np.concatenate(out_scores), np.concatenate(out_days)
