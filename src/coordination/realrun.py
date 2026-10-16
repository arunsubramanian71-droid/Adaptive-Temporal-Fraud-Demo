"""Exploratory real-data pipeline wrapper (Phase 9 scope deviation: Tier 0 / E0 only; see exploratory_realdata/BANNER.md).

The single entry point `run_discovery` is the code path used BOTH by the leakage validations (S2/S3/S4) and by the Tier 0 run, so the
validated function is exactly the one that produces the findings. It accepts an `Arrays` object but touches only `arrays.view()`
(day, card, net, dev): no labels, no model risk, no risk-based components (risk=None, sets=("comm",)).
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

from src.coordination.data import Arrays
from src.coordination.nullmodel import load_tables
from src.coordination.pipeline import discover
from src.coordination.runner import exp_dir, load_locked, require_gates
from src.coordination.scoring import ArrivalTable

EXPLORATORY_DIR = "exploratory_realdata"

BANNER = ("Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 "
          "protocol halted at S6. Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under "
          "the original protocol.")


def out_dir(cfg):
    d = exp_dir(cfg) / EXPLORATORY_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_locked_context(cfg):
    """G0 + timing pilot must have passed; the locked configuration and the null tables must be unmodified since locking."""
    gates = require_gates(cfg)
    locked = load_locked(cfg)
    p = exp_dir(cfg) / "null_tables.pkl"
    if hashlib.sha256(p.read_bytes()).hexdigest() != locked["null_tables_sha256"]:
        raise RuntimeError("null tables differ from the locked configuration's recorded hash")
    return locked, load_tables(p), gates


def run_discovery(arrays: Arrays, tables, W: int, gamma: float, seed: int, days, table: ArrivalTable | None = None) -> dict:
    """Label-free S+T discovery (candidates -> S, T percentiles -> shared tracker). Uses only arrays.view()."""
    v = arrays.view()
    if table is None:
        table = ArrivalTable(v)
    return discover(v, table, tables, W, gamma, seed=seed, days=list(days), risk=None, sets=("comm",))


def hash_output(out: dict) -> str:
    """Canonical SHA-256 of every community-day record and the full tracker state."""
    h = hashlib.sha256()
    for r in out["comm"]:
        h.update(struct.pack("<qqq", int(r["day"]), int(r["idx"]), int(r["track"])))
        h.update(np.ascontiguousarray(r["cards"], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(r["infra"], dtype=np.int64).tobytes())
        h.update(struct.pack("<dddd", float(r["q"]), float(r["u_s"]), float(r["t_raw"]), float(r["u_t"])))
    tr = out["_trackers"]["comm"]
    for tid in sorted(tr.tracks):
        t = tr.tracks[tid]
        h.update(repr((t.tid, t.first_day, t.last_day, t.parent, t.flags, t.days)).encode())
    return h.hexdigest()
