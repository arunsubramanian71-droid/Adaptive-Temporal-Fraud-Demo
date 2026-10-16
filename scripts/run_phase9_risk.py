"""Phase 9 causal risk stream with the frozen-parameter Phase 8 shared-init checkpoint (design 7.3).

  python -u scripts/run_phase9_risk.py --kind eval|clean --seed S [--device cpu|cuda]
kind=clean: the real data (days <= 151, no planting); kind=eval: the augmented eval benchmark. Saves per-transaction logits and
within-day rank percentiles plus end-of-day card states for days 119..151. No labels are used (dataset built with_labels=False).
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import load_config
from src.coordination.data import get_real
from src.coordination.risk import ckpt_sha256, risk_stream
from src.coordination.runner import exp_dir, load_benchmark_arrays, require_gates

SNAP_FROM = 119


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["eval", "clean"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cfg = load_config()
    require_gates(cfg)
    real = get_real(cfg, with_X=True)
    if a.kind == "clean":
        arrays = real
    else:
        arrays, x_src = load_benchmark_arrays(cfg, "eval", a.seed, real)
        arrays.X = real.X[x_src]
    t0 = time.time()
    risk = risk_stream(arrays, cfg, a.seed, torch.device(a.device), snapshot_days=set(range(SNAP_FROM, 152)))
    out = exp_dir(cfg) / "risk"
    out.mkdir(exist_ok=True)
    p = out / f"risk_{a.kind}_seed{a.seed}.npz"
    np.savez(p, pct=risk.pct, logit=risk.logit, snap_days=np.array(sorted(risk.snaps)),
             snaps=np.stack([risk.snaps[d] for d in sorted(risk.snaps)]).astype(np.float32))
    rec = {"kind": a.kind, "seed": a.seed, "device": a.device, "seconds": round(time.time() - t0, 1), "checkpoint_sha256": ckpt_sha256(cfg, a.seed),
           "n_rows": int(len(arrays.day))}
    with open(out / f"risk_{a.kind}_seed{a.seed}.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
