"""Per-day snapshot dataset for the Phase 7 temporal GNN.

The Phase 4 graph is the single source of truth (same node types, relations,
transaction features). This module only re-slices it into 1-day snapshots so
a recurrent model can process time strictly in order.

CAUSALITY BY API DESIGN
-----------------------
The only way the model can read data is `TemporalDataset.day_slice(d)`, which
returns day-d transactions' features and day-d edges and nothing else. Any
information from earlier days reaches day d only through the model's
recurrent state, which is itself only ever updated from day-slices <= d. Labels
are not reachable through `day_slice` at all (they are accessed separately,
by the training loop and the metrics code only).

TEST IS NEVER LOADED
--------------------
`build_temporal_dataset` refuses `max_day > val_day_max` unless the caller
explicitly passes allow_test=True. Phase 7 never does: TEST rows are not
moved to the GPU and are not used to fit anything, consistent with Phases
3-6 (TEST stays sealed until a designated final evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.config import Config

# relation name -> entity node type (Phase 4 schema, unchanged)
RELATIONS: dict[str, str] = {
    "charged_to": "pseudo_account",
    "used_card": "card",
    "via_network": "netproxy",
    "from_device": "device_fingerprint",
}
ENTITY_TYPES: tuple[str, ...] = tuple(RELATIONS.values())


@dataclass
class DayRel:
    src_local: torch.Tensor  # (E,) transaction position within the day
    dst_local: torch.Tensor  # (E,) index into `ids`
    ids: torch.Tensor        # (m,) global ids of entities touched that day


class TemporalDataset:
    def __init__(self, X, y, txn_id, txn_day, day_ptr, rel, n_entities, max_day):
        self.X = X
        self.y = y
        self.txn_id = txn_id
        self.txn_day = txn_day
        self.day_ptr = day_ptr
        self.rel = rel
        self.n_entities = n_entities
        self.max_day = max_day
        self.days = sorted(day_ptr)
        self.access_log: list[int] | None = None

    def day_slice(self, d: int):
        """The model's ONLY data accessor. Returns (features of day-d
        transactions, day-d edges). Logged when access_log is enabled so the
        leakage suite can prove each step touched only its own day."""
        if self.access_log is not None:
            self.access_log.append(d)
        lo, hi = self.day_ptr[d]
        return self.X[lo:hi], self.rel[d]

    def labels(self, d: int) -> torch.Tensor:
        """Training loop / metrics only — never called by the model."""
        lo, hi = self.day_ptr[d]
        return self.y[lo:hi]

    def ids_for_day(self, d: int) -> np.ndarray:
        lo, hi = self.day_ptr[d]
        return self.txn_id[lo:hi]


def load_raw_arrays(cfg: Config):
    """Loads the Phase 4 graph (read-only) and returns numpy arrays plus the
    CPU graph object (needed for the cold/warm/no-edge segment definition).
    The StandardScaler is fit on TRAIN-period transaction rows only — the
    identical procedure Phases 5/6A/6B used."""
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    day = graph["transaction"].day.numpy().astype(np.int64)
    x = graph["transaction"].x.numpy()
    train_mask = (day >= cfg.split.train_day_min) & (day <= cfg.split.train_day_max)
    scaler = StandardScaler().fit(x[train_mask])
    X = scaler.transform(x).astype(np.float32)

    relations = {}
    for rel, ent in RELATIONS.items():
        et = ("transaction", rel, ent)
        ei = graph[et].edge_index.numpy()
        relations[rel] = (ei[0].astype(np.int64), ei[1].astype(np.int64),
                          graph[et].edge_time.numpy().astype(np.int64))
    raw = {
        "day": day,
        "X": X,
        "y": graph["transaction"].y.numpy().astype(np.float32),
        "txn_id": graph["transaction"].transaction_id.numpy(),
        "relations": relations,
        "n_entities": {ent: int(graph[ent].num_nodes) for ent in ENTITY_TYPES},
    }
    return raw, graph, scaler


def build_temporal_dataset(raw: dict, max_day: int, device, cfg: Config, allow_test: bool = False,
                           with_labels: bool = True) -> TemporalDataset:
    if max_day > cfg.split.val_day_max and not allow_test:
        raise ValueError(f"max_day={max_day} exceeds the last VALIDATION day ({cfg.split.val_day_max}); "
                         f"TEST days must not be loaded in Phase 7 (pass allow_test=True only for a designated final evaluation)")
    day = raw["day"]
    if not np.all(np.diff(day) >= 0):
        raise ValueError("transactions are not in chronological order")
    n_keep = int(np.searchsorted(day, max_day, side="right"))
    day_k = day[:n_keep]
    days = np.unique(day_k)
    lo_arr = np.searchsorted(day_k, days, side="left")
    hi_arr = np.searchsorted(day_k, days, side="right")
    day_ptr = {int(d): (int(lo), int(hi)) for d, lo, hi in zip(days, lo_arr, hi_arr)}

    rel_by_day: dict[int, dict[str, DayRel]] = {int(d): {} for d in days}
    dev = device
    for rel, (src, dst, etime) in raw["relations"].items():
        if not (np.all(np.diff(etime) >= 0) and np.array_equal(etime, day[src])):
            raise ValueError(f"relation {rel}: edge_time is not the chronologically-ordered source-transaction day")
        n_e = int(np.searchsorted(etime, max_day, side="right"))
        src_k, dst_k, et_k = src[:n_e], dst[:n_e], etime[:n_e]
        e_lo = np.searchsorted(et_k, days, side="left")
        e_hi = np.searchsorted(et_k, days, side="right")
        for d, lo, hi in zip(days, e_lo, e_hi):
            d = int(d)
            s_local = src_k[lo:hi] - day_ptr[d][0]
            ids, inv = np.unique(dst_k[lo:hi], return_inverse=True)
            rel_by_day[d][rel] = DayRel(
                src_local=torch.from_numpy(s_local).to(dev),
                dst_local=torch.from_numpy(inv.astype(np.int64)).to(dev),
                ids=torch.from_numpy(ids.astype(np.int64)).to(dev),
            )

    X = torch.from_numpy(raw["X"][:n_keep]).to(dev)
    y = torch.from_numpy(raw["y"][:n_keep]).to(dev) if with_labels else None
    return TemporalDataset(X, y, raw["txn_id"][:n_keep], day_k, day_ptr, rel_by_day, raw["n_entities"], max_day)


def snapshot_stats(ds: TemporalDataset) -> dict:
    """Per-day active nodes/edges — the numbers the feasibility estimate is
    based on (Phase 7 requirement: estimate snapshot sizes before training)."""
    rows = []
    for d in ds.days:
        lo, hi = ds.day_ptr[d]
        r = {"day": d, "transactions": hi - lo}
        for rel in RELATIONS:
            dr = ds.rel[d][rel]
            r[f"{rel}_edges"] = int(dr.src_local.numel())
            r[f"{rel}_entities"] = int(dr.ids.numel())
        rows.append(r)
    keys = [k for k in rows[0] if k != "day"]
    return {"n_days": len(rows),
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
            "max": {k: int(np.max([r[k] for r in rows])) for k in keys}}
