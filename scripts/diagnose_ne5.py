"""Post-result diagnosis of the failed registered criterion NE5(a) of the neural graph evidence (experiments/neural_graph_evidence/PREREGISTRATION_NEURAL_EVIDENCE.md).

NE5(a) demanded that resetting the memory of an entity that was never seen is EXACTLY zero. On the GPU run 13 of 102 such resets were nonzero (largest 9.5e-07).
This script asks whether that is arithmetic noise or a behaviour of the code, on validation data only, label-free, without touching the recorded outputs:
  1. CPU: the same counterfactuals for the never-seen entities on the CPU, where kernels are deterministic: if the code is right the effect is exactly zero.
  2. GPU: the same identity case twice (a no-op perturbation) to measure the arithmetic noise floor of a perturbation that changes nothing.
The registered result and the registered rule are NOT changed; this is a labelled diagnostic written after the failure.

Usage:  .venv/Scripts/python.exe -B scripts/diagnose_ne5.py
Writes experiments/neural_graph_evidence/diagnostics/NE5_diagnosis.json (+ sidecar).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.explain import neural_graph_evidence as NG  # noqa: E402
from src.graph.temporal import RELATIONS, build_temporal_dataset, load_raw_arrays  # noqa: E402
from src.models.temporal_graphsage import TemporalGraphSAGE  # noqa: E402

OUT = ROOT / "experiments" / "neural_graph_evidence" / "diagnostics"
CKPT = ROOT / "experiments/temporal_graphsage/seed_42/model_checkpoint.pt"
SEL = ROOT / "experiments/phase11_explainable_evidence/xgb_explanations_val/selection_val.csv"


def run_device(dev, raw, cfg, sel):
    ds = build_temporal_dataset(raw, cfg.split.val_day_max, dev, cfg, with_labels=False)
    model = TemporalGraphSAGE(ds.X.shape[1]).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    model.eval()
    state = model.init_state(ds)
    reset_new, noop = [], []
    with torch.no_grad():
        for d in ds.days:
            if d > 151:
                break
            if d >= 122:
                day_ids = ds.ids_for_day(d)
                pos_of = {int(t): i for i, t in enumerate(day_ids)}
                _, rels = ds.day_slice(d)
                for tid in sel.loc[sel["day"] == d, "TransactionID"]:
                    pos = pos_of[int(tid)]
                    base = NG.day_logits(model, ds, state, d)
                    base2 = NG.day_logits(model, ds, state, d)
                    noop.append(float(base2[pos] - base[pos]))
                    for name, ent in RELATIONS.items():
                        eid = NG.entity_of(rels[name], pos)
                        if eid is None or not bool(state.last_day[ent][eid] < 0):
                            continue
                        rs = state.clone()
                        rs.last_day[ent][eid] = -1                      # already -1: the state is identical
                        reset_new.append(float(NG.day_logits(model, ds, rs, d)[pos] - base[pos]))
            model.step(state, ds, d, predict=False)
    return np.array(reset_new), np.array(noop)


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    sel = pd.read_csv(SEL)
    raw, _g, _ = load_raw_arrays(cfg)
    res = {}
    for name, dev in (("cpu", torch.device("cpu")), ("gpu", torch.device("cuda:0") if torch.cuda.is_available() else None)):
        if dev is None:
            continue
        r, n = run_device(dev, raw, cfg, sel)
        res[name] = {"n_reset_never_seen": int(len(r)), "n_nonzero": int((r != 0).sum()), "max_abs_reset_effect": float(np.abs(r).max()) if len(r) else None,
                     "n_noop_pairs": int(len(n)), "noop_n_nonzero": int((n != 0).sum()), "noop_max_abs": float(np.abs(n).max()) if len(n) else None}
        print(name, res[name], flush=True)
    cpu_exact = res["cpu"]["n_nonzero"] == 0 and res["cpu"]["noop_n_nonzero"] == 0
    verdict = ("NUMERICAL PRECISION (GPU arithmetic is not bit-reproducible): on the CPU every never-seen reset is exactly zero and so is the no-op; on the GPU the identical computation done twice differs at the same float rounding scale" if cpu_exact and res.get("gpu", {}).get("noop_n_nonzero", 0) > 0 else
               "CODE BEHAVIOUR OR CRITERION: see the numbers (not explained by GPU noise alone)" if not cpu_exact else
               "NOT DETERMINED: the GPU no-op showed no noise in this run; the CPU result is exact")
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "device_results": res, "cpu_exact_zero": bool(cpu_exact), "verdict": verdict,
           "registered_result": "NE5(a) FAILED as registered on the GPU run; recorded outputs and the registered rule are unchanged", "seconds": round(time.time() - t0, 1),
           "note": "post-result diagnostic; label-free; validation days <= 151 only; no held-out data"}
    OUT.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(rec, indent=1) + "\n").encode("utf-8")
    (OUT / "NE5_diagnosis.json").write_bytes(data)
    (OUT / "NE5_diagnosis.sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
