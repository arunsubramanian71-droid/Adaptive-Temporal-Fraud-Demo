"""Causal risk stream with the frozen-parameter Phase 8 shared-init checkpoint (design 7.3, F10).

Only experiments/phase8_adaptive_drift/shared_init/model_checkpoint_initial_seed{S}.pt is allowed (fit on TRAIN days 1-111; never saw
VAL labels, VAL-based selection, or any update). The dataset is built without labels (`with_labels=False`); the model is Phase 7's
TemporalGraphSAGE, unmodified. Outputs: per-transaction logits, within-day rank percentiles, and end-of-day `card` entity states.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from scipy.stats import rankdata

from src.coordination.data import Arrays
from src.coordination.frozen import MAX_DAY
from src.coordination.pipeline import RiskData
from src.graph.temporal import build_temporal_dataset
from src.models.temporal_graphsage import TemporalGraphSAGE


def checkpoint_path(cfg, seed: int):
    p = cfg.experiments_dir / "phase8_adaptive_drift" / "shared_init" / f"model_checkpoint_initial_seed{seed}.pt"
    assert p.exists() and "shared_init" in str(p)
    return p


def ckpt_sha256(cfg, seed: int) -> str:
    return hashlib.sha256(open(checkpoint_path(cfg, seed), "rb").read()).hexdigest()


def risk_stream(arrays: Arrays, cfg, seed: int, device, snapshot_days: set) -> RiskData:
    raw = arrays.to_raw()
    ds = build_temporal_dataset(raw, MAX_DAY, device, cfg, with_labels=False)
    model = TemporalGraphSAGE(raw["X"].shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint_path(cfg, seed), map_location=device))
    model.eval()
    n = len(arrays.day)
    logit = np.zeros(n, dtype=np.float32)
    pct = np.zeros(n, dtype=np.float64)
    snaps = {}
    with torch.no_grad():
        state = model.init_state(ds)
        for d in ds.days:
            lg = model.step(state, ds, d, predict=True)
            lo, hi = ds.day_ptr[d]
            v = lg.detach().cpu().numpy()
            logit[lo:hi] = v
            pct[lo:hi] = rankdata(v, method="average") / max(len(v), 1)
            if d in snapshot_days:
                snaps[d] = state.S["card"].detach().cpu().numpy().astype(np.float32).copy()
    return RiskData(pct=pct, logit=logit, snaps=snaps)
