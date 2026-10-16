"""Shared I/O helpers for Phase 9 scripts: gates, locked configuration, benchmark persistence (manifest sealed, row groups sealed)."""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from src.coordination import frozen as F
from src.coordination.benchmark import manifest_json, sha256_text
from src.coordination.data import Arrays
from src.coordination.g0 import require_g0

EXP = "phase9_coordinated_fraud"


def exp_dir(cfg) -> pathlib.Path:
    d = cfg.experiments_dir / EXP
    d.mkdir(parents=True, exist_ok=True)
    return d


def require_gates(cfg) -> dict:
    """G0 must have passed (hard gate) and the timing pilot must have passed before any benchmark/discovery work."""
    d = exp_dir(cfg)
    g0 = require_g0(d, cfg)
    tp = json.load(open(d / "timing_pilot.json", encoding="utf-8"))
    if not tp.get("passed"):
        raise RuntimeError("timing pilot gate tripped: STOP and request a rescoping decision")
    return {"g0": g0["passed"], "timing_pilot": tp["passed"]}


def locked_config_path(cfg) -> pathlib.Path:
    return exp_dir(cfg) / "locked_config.json"


def load_locked(cfg) -> dict:
    p = locked_config_path(cfg)
    if not p.exists():
        raise RuntimeError("locked configuration missing: the evaluation benchmark must not be generated or evaluated (design 11.6)")
    rec = json.load(open(p, encoding="utf-8"))
    h = (exp_dir(cfg) / "locked_config.sha256").read_text().strip()
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
        raise RuntimeError("locked configuration was modified after locking")
    return rec


def bench_dir(cfg, kind: str, seed: int) -> pathlib.Path:
    d = exp_dir(cfg) / f"benchmark_{kind}" / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_benchmark(cfg, kind: str, seed: int, aug: Arrays, x_src, manifest: dict, is_planted, row_group) -> dict:
    d = bench_dir(cfg, kind, seed)
    # detector-visible arrays: NO labels, NO ground truth
    np.savez(d / "arrays.npz", day=aug.day, txn_id=aug.txn_id, card=aug.card, pa=aug.pa, net=aug.net, dev=aug.dev, x_src=x_src,
             n_pa=aug.n_entities["pseudo_account"], n_card=aug.n_entities["card"], n_net=aug.n_entities["netproxy"],
             n_dev=aug.n_entities["device_fingerprint"])
    sealed = d / "sealed"
    sealed.mkdir(exist_ok=True)
    txt = manifest_json(manifest)
    (sealed / "manifest_sealed.json").write_text(txt, encoding="utf-8")
    (sealed / "manifest.sha256").write_text(sha256_text(txt))
    np.savez(sealed / "row_truth.npz", is_planted=is_planted, row_group=row_group)
    return {"manifest_sha256": sha256_text(txt), "n_rows": int(len(aug.day)), "n_planted": int(is_planted.sum())}


def load_benchmark_arrays(cfg, kind: str, seed: int, real: Arrays) -> tuple[Arrays, np.ndarray]:
    """Detector-side loader: reads only arrays.npz (never the sealed directory)."""
    z = np.load(bench_dir(cfg, kind, seed) / "arrays.npz")
    ne = {"pseudo_account": int(z["n_pa"]), "card": int(z["n_card"]), "netproxy": int(z["n_net"]), "device_fingerprint": int(z["n_dev"])}
    a = Arrays(day=z["day"], txn_id=z["txn_id"], card=z["card"], pa=z["pa"], net=z["net"], dev=z["dev"], n_entities=ne, y=None, X=None)
    return a, z["x_src"]


def load_sealed(cfg, kind: str, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluation-side loader (opens the sealed ground truth)."""
    d = bench_dir(cfg, kind, seed) / "sealed"
    txt = (d / "manifest_sealed.json").read_text(encoding="utf-8")
    assert sha256_text(txt) == (d / "manifest.sha256").read_text().strip(), "manifest hash mismatch (modified after sealing)"
    z = np.load(d / "row_truth.npz")
    return json.loads(txt), z["is_planted"], z["row_group"]
