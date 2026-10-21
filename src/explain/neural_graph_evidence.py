"""Counterfactual graph evidence for the Temporal GraphSAGE model (experiments/neural_graph_evidence/PREREGISTRATION_NEURAL_EVIDENCE.md).

Two counterfactuals on one day of the model's stream, both on a CLONE of the state at the start of the day, never on the live stream:
  * link removal: delete one transaction's edge for one relation (an entity left without edges that day is dropped from the day's entity list);
  * memory reset: mark one entity as never seen (the model then uses its learned initial state).
No label is an argument anywhere. Everything runs under ``torch.no_grad`` with the model in eval mode.
"""
from __future__ import annotations

import torch

from src.graph.temporal import RELATIONS, DayRel


class DayProxy:
    """Stands in for the dataset for one day: ``step`` only ever calls ``day_slice`` (and ``init_state`` reads ``X.device``)."""

    def __init__(self, ds, day: int, rels: dict):
        self._ds, self._day, self._rels = ds, day, rels
        self.X = ds.X

    def day_slice(self, d: int):
        if d != self._day:
            raise ValueError(f"proxy built for day {self._day}, asked for day {d}")
        x, _ = self._ds.day_slice(d)
        return x, self._rels


def remove_edge(rel: DayRel, pos: int) -> DayRel:
    """The same day-relation without the edge(s) of transaction ``pos``; entities left without any edge are dropped from ``ids``."""
    keep = rel.src_local != pos
    glob = rel.ids[rel.dst_local][keep]
    ids, inv = torch.unique(glob, return_inverse=True)
    return DayRel(src_local=rel.src_local[keep], dst_local=inv, ids=ids)


def entity_of(rel: DayRel, pos: int):
    """Global id of the entity transaction ``pos`` links to in this relation, or None if it has no edge."""
    m = (rel.src_local == pos).nonzero(as_tuple=True)[0]
    if m.numel() == 0:
        return None
    return int(rel.ids[rel.dst_local[m[0]]])


@torch.no_grad()
def day_logits(model, ds, state, d: int, rels: dict | None = None):
    """Logits for all transactions of day ``d`` computed on a clone of ``state`` (the argument is not modified)."""
    clone = state.clone()
    x, base_rels = ds.day_slice(d)
    proxy = DayProxy(ds, d, rels if rels is not None else base_rels)
    return model.step(clone, proxy, d, predict=True)


@torch.no_grad()
def counterfactuals_for(model, ds, state, d: int, pos: int) -> dict:
    """For the transaction at position ``pos`` of day ``d``: base logit and, per relation present, the link-removal and memory-reset effects.

    Returns {"base": float, "relations": {rel: {"present": bool, "entity": int|None, "entity_new": bool|None, "removal_effect": float|nan, "reset_effect": float|nan}}}.
    """
    nan = float("nan")
    _, rels = ds.day_slice(d)
    base = float(day_logits(model, ds, state, d)[pos])
    out = {}
    for name, ent in RELATIONS.items():
        eid = entity_of(rels[name], pos)
        if eid is None:
            out[name] = {"present": False, "entity": None, "entity_new": None, "removal_effect": nan, "reset_effect": nan}
            continue
        new_rels = dict(rels)
        new_rels[name] = remove_edge(rels[name], pos)
        removal = float(day_logits(model, ds, state, d, new_rels)[pos]) - base
        was_new = bool(state.last_day[ent][eid] < 0)
        reset_state = state.clone()
        reset_state.last_day[ent][eid] = -1
        reset = float(day_logits(model, ds, reset_state, d)[pos]) - base
        out[name] = {"present": True, "entity": eid, "entity_new": was_new, "removal_effect": removal, "reset_effect": reset}
    return {"base": base, "relations": out}
