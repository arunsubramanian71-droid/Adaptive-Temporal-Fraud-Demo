"""Temporal GraphSAGE with per-entity recurrent (GRU) memory — Phase 7.

Mechanism (one step per day d, strictly in chronological order):

  1. GraphSAGE layer 1 (transaction -> entity), on the DAY-d snapshot only:
       m_e = SAGE( day-d transactions' projected features ; root = entity's
                   PREVIOUS memory state )
  2. Recurrent state update (GRU), for entities touched on day d:
       s_e(d) = GRU( [m_e, elapsed-time encoding] , s_e(d-1) )
     Entities not touched on day d keep their state unchanged.
  3. GraphSAGE layer 2 (entity -> transaction), day-d transactions read the
     UPDATED states of the entities they connect to (summed over the four
     validated relations, exactly as the static model sums relations), then
     a linear head produces the fraud logit.

Why this is a genuine temporal GNN and not "GraphSAGE + a timestamp
feature": the entity state is a function of the ORDERED sequence of all
earlier snapshots. Two histories with the same set of past transactions in
a different order/timing produce different states; earlier graph state
influences later predictions only through this recurrence; and the model can
learn forgetting/recency through the GRU gates and the elapsed-time input.
The static GraphSAGE reference instead mean-pools an entity's whole history
uniformly, with no notion of order.

CAUSALITY: `step` reads data only via `ds.day_slice(d)` and writes state only
for entities touched on day d. No labels enter the model or its state at any
point (labels are used only in the loss/metrics, outside this module), so no
fraud label from any time can leak into a state. Same-day mutual visibility
(day-d transactions see day-d snapshot messages) matches the static rolling
protocol's day granularity and satisfies information_time <= prediction_time
at that granularity.

Truncated BPTT: gradients flow through state updates within a training
window; `detach_state` cuts the graph at window boundaries while the state
VALUES carry forward unchanged (no reset at any boundary).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.graph.temporal import ENTITY_TYPES, RELATIONS, TemporalDataset

MAX_DAY_FOR_GAP_NORM = 200.0


class TemporalState:
    """Per-entity memory + last-active day. Constructed empty at day 0."""

    def __init__(self, S: dict, last_day: dict):
        self.S = S
        self.last_day = last_day

    def clone(self) -> "TemporalState":
        return TemporalState({k: v.clone() for k, v in self.S.items()},
                             {k: v.clone() for k, v in self.last_day.items()})


class TemporalGraphSAGE(torch.nn.Module):
    def __init__(self, txn_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.proj_txn = torch.nn.Linear(txn_dim, hidden)
        # Learned initial state for never-seen entities (prior for cold-start).
        self.h0 = torch.nn.ParameterDict({ent: torch.nn.Parameter(torch.zeros(hidden)) for ent in ENTITY_TYPES})
        self.msg = torch.nn.ModuleDict({rel: SAGEConv((hidden, hidden), hidden) for rel in RELATIONS})
        self.gru = torch.nn.ModuleDict({ent: torch.nn.GRUCell(hidden + 2, hidden) for ent in ENTITY_TYPES})
        self.read = torch.nn.ModuleDict({rel: SAGEConv((hidden, hidden), hidden) for rel in RELATIONS})
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    # ------------------------------------------------------------------ state
    def init_state(self, ds: TemporalDataset) -> TemporalState:
        dev = ds.X.device
        S = {ent: torch.zeros(n, self.hidden, device=dev) for ent, n in ds.n_entities.items()}
        last = {ent: torch.full((n,), -1, dtype=torch.long, device=dev) for ent, n in ds.n_entities.items()}
        return TemporalState(S, last)

    @staticmethod
    def detach_state(state: TemporalState) -> None:
        for k in state.S:
            state.S[k] = state.S[k].detach()

    # ------------------------------------------------------------------- step
    def step(self, state: TemporalState, ds: TemporalDataset, d: int, predict: bool = True):
        x_day, rels = ds.day_slice(d)
        hx = F.relu(self.proj_txn(x_day))
        n_t = hx.size(0)
        new_states = {}
        for rel, ent in RELATIONS.items():
            e = rels[rel]
            ids = e.ids
            last = state.last_day[ent][ids]
            is_new = last < 0
            prev = torch.where(is_new.unsqueeze(1), self.h0[ent].unsqueeze(0), state.S[ent][ids])
            if ids.numel() > 0:
                m = self.msg[rel]((hx, prev), torch.stack([e.src_local, e.dst_local]))
                m = self.dropout(F.relu(m))
                gap = (d - last).clamp(min=0).to(m.dtype)
                log_gap = torch.log1p(gap) / math.log(MAX_DAY_FOR_GAP_NORM) * (~is_new).to(m.dtype)
                s_new = self.gru[ent](torch.cat([m, log_gap.unsqueeze(1), is_new.to(m.dtype).unsqueeze(1)], dim=1), prev)
            else:
                s_new = prev
            new_states[rel] = s_new
            # causal state write: only entities touched today, only from today's snapshot
            if torch.is_grad_enabled():
                state.S[ent] = state.S[ent].index_put((ids,), s_new)
            else:
                state.S[ent][ids] = s_new
            state.last_day[ent][ids] = d
        if not predict:
            return None
        z = 0
        for rel in RELATIONS:
            e = rels[rel]
            z = z + self.read[rel]((new_states[rel], hx), torch.stack([e.dst_local, e.src_local]))
        return self.head(F.relu(z)).squeeze(-1)


@torch.no_grad()
def rolling_temporal_predict(model: TemporalGraphSAGE, ds: TemporalDataset, first_pred_day: int, last_pred_day: int):
    """Causal rolling evaluation: process days 1..last_pred_day strictly in
    order with a fresh state built by the CURRENT parameters (eval mode);
    record predictions for days first_pred_day..last_pred_day. The state is
    carried, unreset, across the train->validation boundary."""
    model.eval()
    state = model.init_state(ds)
    ids_l, sc_l, day_l = [], [], []
    for d in ds.days:
        if d > last_pred_day:
            break
        logits = model.step(state, ds, d, predict=d >= first_pred_day)
        if logits is not None:
            ids_l.append(ds.ids_for_day(d))
            sc_l.append(torch.sigmoid(logits).cpu().numpy())
            day_l.append(torch.full((logits.numel(),), d).numpy())
    import numpy as np
    return np.concatenate(ids_l), np.concatenate(sc_l), np.concatenate(day_l)
