"""Real-data arrays for Phase 9, truncated to days <= 151 at load time.

Convention (design section 5, unchanged from Phases 7-8): arrays are obtained through the existing `load_raw_arrays`, which
reads the Phase 4 graph file; rows with day > 151 are dropped immediately and never used, inspected, or reported.
A derived cache (days <= 151 only) is written under experiments/phase9_coordinated_fraud/cache/ so later steps do not
re-read the shared graph file.

`GraphView` is the label-free view handed to discovery/scoring code. Labels (`y`) live only on `Arrays` and are used by the
benchmark generator and by retrospective evaluation code - never by discovery.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from src.coordination.frozen import MAX_DAY
from src.graph.temporal import RELATIONS

ENT_KEYS = {"charged_to": "pa", "used_card": "card", "via_network": "net", "from_device": "dev"}


@dataclass
class GraphView:
    """Label-free, model-free structural view used by candidate generation."""
    day: np.ndarray
    card: np.ndarray
    net: np.ndarray
    dev: np.ndarray
    n_net: int
    n_dev: int

    @property
    def n(self) -> int:
        return len(self.day)


@dataclass
class Arrays:
    day: np.ndarray
    txn_id: np.ndarray
    card: np.ndarray
    pa: np.ndarray
    net: np.ndarray
    dev: np.ndarray
    n_entities: dict          # keys: pseudo_account, card, netproxy, device_fingerprint
    y: np.ndarray | None = None
    X: np.ndarray | None = None

    def view(self) -> GraphView:
        return GraphView(self.day, self.card, self.net, self.dev,
                         int(self.n_entities["netproxy"]), int(self.n_entities["device_fingerprint"]))

    def to_raw(self) -> dict:
        """Dict in the layout expected by build_temporal_dataset (relations rebuilt from per-transaction entity arrays)."""
        rel = {}
        for r, k in ENT_KEYS.items():
            a = getattr(self, k)
            src = np.nonzero(a >= 0)[0].astype(np.int64)
            rel[r] = (src, a[src].astype(np.int64), self.day[src].astype(np.int64))
        return {"day": self.day.astype(np.int64), "X": self.X, "y": self.y, "txn_id": self.txn_id,
                "relations": rel, "n_entities": dict(self.n_entities)}


def _cache_dir(cfg) -> pathlib.Path:
    d = cfg.experiments_dir / "phase9_coordinated_fraud" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_real(cfg, with_X: bool = False) -> Arrays:
    cd = _cache_dir(cfg)
    npz = cd / "real_le151.npz"
    xpath = cd / "real_le151_X.npy"
    if not npz.exists() or (with_X and not xpath.exists()):
        from src.graph.temporal import load_raw_arrays
        raw, graph, _ = load_raw_arrays(cfg)
        del graph
        n = int(np.searchsorted(raw["day"], MAX_DAY, side="right"))
        ent = {}
        for rel, k in ENT_KEYS.items():
            src, dst, _ = raw["relations"][rel]
            a = np.full(n, -1, dtype=np.int64)
            m = src < n
            a[src[m]] = dst[m]
            ent[k] = a
        np.savez(npz, day=raw["day"][:n].astype(np.int64), txn_id=raw["txn_id"][:n].astype(np.int64), y=raw["y"][:n],
                 card=ent["card"], pa=ent["pa"], net=ent["net"], dev=ent["dev"],
                 n_pa=raw["n_entities"]["pseudo_account"], n_card=raw["n_entities"]["card"],
                 n_net=raw["n_entities"]["netproxy"], n_dev=raw["n_entities"]["device_fingerprint"])
        np.save(xpath, raw["X"][:n].astype(np.float32))
        del raw
    z = np.load(npz)
    ne = {"pseudo_account": int(z["n_pa"]), "card": int(z["n_card"]), "netproxy": int(z["n_net"]), "device_fingerprint": int(z["n_dev"])}
    X = np.load(xpath, mmap_mode="r") if with_X else None
    if X is not None:
        X = np.ascontiguousarray(X)
    a = Arrays(day=z["day"], txn_id=z["txn_id"], card=z["card"], pa=z["pa"], net=z["net"], dev=z["dev"], n_entities=ne, y=z["y"], X=X)
    assert a.day.max() <= MAX_DAY
    return a
