"""Memory-ablated temporal control — Phase 7A arm C.

Same causal streaming pipeline, snapshots, layer-2 read-out, heads and training regime as
`TemporalGraphSAGE` (Phase 7), but the recurrent GRU memory is replaced by an ORDER-AGNOSTIC
cumulative-mean memory (see experiments/temporal_ablation/DESIGN_BEFORE_EXECUTION.md):

  u_t      = W_r hx_t                          per-transaction message (== SAGEConv.lin_l applied before the mean)
  Sum_e(d) = Sum_e(d-1) + sum_{t in day d, t~e} u_t          running sum   } state row = [Sum | Cnt]
  Cnt_e(d) = Cnt_e(d-1) + #{t in day d, t~e}                 running count }
  mbar_e   = Sum_e(d) / Cnt_e(d)               mean over every message from ALL transactions <= d
  s_e      = MLP_e(Dropout(ReLU(mbar_e)))      parameter-matched replacement for the GRU cell

Removed relative to the GRU model: recurrence/gating, learned initial state h0, elapsed-time
input, and state-as-root in layer 1. The memory content is a permutation-invariant function of
the multiset of past messages (up to float rounding): it does not depend on the order or timing
of past events. No labels enter the state. `last_day` is kept only for interface parity.

CAUSALITY: identical access discipline to Phase 7 — `step` reads only `ds.day_slice(d)` and
writes state only for entities touched on day d.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.graph.temporal import ENTITY_TYPES, RELATIONS, TemporalDataset
from src.models.temporal_graphsage import TemporalState, rolling_temporal_predict  # noqa: F401  (re-exported)


class TemporalGraphSAGEMeanMemory(torch.nn.Module):
    def __init__(self, txn_dim: int, hidden: int = 64, dropout: float = 0.3, mem_hidden: int = 192):
        super().__init__()
        self.hidden = hidden
        self.proj_txn = torch.nn.Linear(txn_dim, hidden)
        self.msg = torch.nn.ModuleDict({rel: torch.nn.Linear(hidden, hidden) for rel in RELATIONS})
        self.mem_mlp = torch.nn.ModuleDict({
            ent: torch.nn.Sequential(torch.nn.Linear(hidden, mem_hidden), torch.nn.ReLU(), torch.nn.Linear(mem_hidden, hidden))
            for ent in ENTITY_TYPES})
        self.read = torch.nn.ModuleDict({rel: SAGEConv((hidden, hidden), hidden) for rel in RELATIONS})
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    # ------------------------------------------------------------------ state
    def init_state(self, ds: TemporalDataset) -> TemporalState:
        dev = ds.X.device
        S = {ent: torch.zeros(n, self.hidden + 1, device=dev) for ent, n in ds.n_entities.items()}   # [sum(64) | count(1)]
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
        new_s = {}
        for rel, ent in RELATIONS.items():
            e = rels[rel]
            ids = e.ids
            if ids.numel() > 0:
                u = self.msg[rel](hx[e.src_local])                                              # (E, H)
                msg = torch.cat([u, torch.ones(u.size(0), 1, device=u.device, dtype=u.dtype)], dim=1)
                contrib = torch.zeros(ids.numel(), self.hidden + 1, device=u.device, dtype=u.dtype).index_add(0, e.dst_local, msg)
                new = state.S[ent][ids] + contrib                                                # cumulative sum/count, today included
                mbar = new[:, :self.hidden] / new[:, self.hidden:].clamp(min=1.0)
                s = self.mem_mlp[ent](self.dropout(F.relu(mbar)))
                if torch.is_grad_enabled():
                    state.S[ent] = state.S[ent].index_put((ids,), new)
                else:
                    state.S[ent][ids] = new
                state.last_day[ent][ids] = d
            else:
                s = state.S[ent][ids][:, :self.hidden]
            new_s[rel] = s
        if not predict:
            return None
        z = 0
        for rel in RELATIONS:
            e = rels[rel]
            z = z + self.read[rel]((new_s[rel], hx), torch.stack([e.dst_local, e.src_local]))
        return self.head(F.relu(z)).squeeze(-1)
