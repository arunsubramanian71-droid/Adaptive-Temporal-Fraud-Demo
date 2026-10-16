"""Phase 7 diagnostic: why did the first GPU leakage run show ~1e-4 logit
differences on PAST days when the future was corrupted?

Three discriminating experiments (seed-42 trained checkpoint, days <= 151):

 A. GPU, NO perturbation: build the dataset twice from identical raw arrays
    and compare. If this alone reproduces the ~1e-4 discrepancy, it is not
    caused by future data (there is none to differ).
 B. GPU, corrupt ONLY day 151, compare days <= 150. Structurally the model
    cannot read day 151 while processing day <= 150 (each step reads only its
    own day; verified by the access-pattern check), so any difference here
    is numerical, not informational.
 C. CPU (deterministic kernels), full future-corruption at several cutoffs:
    the requirement is BITWISE equality on all past days - no tolerance.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import load_config
from src.graph import temporal_checks as TC
from src.graph.temporal import build_temporal_dataset, load_raw_arrays
from src.models.temporal_graphsage import TemporalGraphSAGE

MAX_DAY, FIRST = 151, 122


def prob_diff(a, b, days):
    sig = lambda x: 1 / (1 + np.exp(-x.astype(np.float64)))
    return float(max(np.max(np.abs(sig(a[d]) - sig(b[d]))) for d in days))


def main():
    cfg = load_config()
    raw, graph, _ = load_raw_arrays(cfg)
    ckpt = cfg.experiments_dir / "temporal_graphsage" / "seed_42" / "model_checkpoint.pt"
    out = {}

    # ---------------- A & B on GPU ----------------
    gpu = torch.device("cuda:0")
    m = TemporalGraphSAGE(raw["X"].shape[1]).to(gpu)
    m.load_state_dict(torch.load(ckpt, map_location=gpu))
    ds1 = build_temporal_dataset(raw, MAX_DAY, gpu, cfg)
    base = TC.stream_logits(m, ds1, FIRST, MAX_DAY)
    days = sorted(base)
    ds2 = build_temporal_dataset(raw, MAX_DAY, gpu, cfg)  # identical data, separately allocated
    rebuilt = TC.stream_logits(m, ds2, FIRST, MAX_DAY)
    a = TC._max_abs_diff(base, rebuilt, days)
    out["A_gpu_rebuild_identical_data"] = {"max_abs_dlogit": a, "max_abs_dprob": prob_diff(base, rebuilt, days)}
    print(f"[A] GPU, NO perturbation, dataset rebuilt from identical arrays: max|dlogit| = {a:.3e}  "
          f"(prob-space {out['A_gpu_rebuild_identical_data']['max_abs_dprob']:.2e})", flush=True)
    del ds2

    pert = TC.perturb_future(raw, 150, np.random.default_rng(0))  # only day 151 corrupted
    dsp = build_temporal_dataset(pert, MAX_DAY, gpu, cfg)
    o = TC.stream_logits(m, dsp, FIRST, MAX_DAY)
    past = [d for d in days if d <= 150]
    b = TC._max_abs_diff(base, o, past)
    out["B_gpu_only_day151_corrupted_compare_days_le_150"] = {"max_abs_dlogit": b, "max_abs_dprob": prob_diff(base, o, past)}
    print(f"[B] GPU, corrupt ONLY day 151, compare days <= 150: max|dlogit| = {b:.3e}  "
          f"(prob-space {out['B_gpu_only_day151_corrupted_compare_days_le_150']['max_abs_dprob']:.2e})", flush=True)
    del dsp, ds1
    del m
    torch.cuda.empty_cache()

    # ---------------- C on CPU (deterministic) ----------------
    cpu = torch.device("cpu")
    mc = TemporalGraphSAGE(raw["X"].shape[1])
    mc.load_state_dict(torch.load(ckpt, map_location=cpu))
    dsc = build_temporal_dataset(raw, MAX_DAY, cpu, cfg)
    base_c = TC.stream_logits(mc, dsc, FIRST, MAX_DAY)
    base_c2 = TC.stream_logits(mc, dsc, FIRST, MAX_DAY)
    rebuilt_c = TC.stream_logits(mc, build_temporal_dataset(raw, MAX_DAY, cpu, cfg), FIRST, MAX_DAY)
    dc = sorted(base_c)
    out["C_cpu_repeat_same_ds"] = TC._max_abs_diff(base_c, base_c2, dc)
    out["C_cpu_rebuild_identical"] = TC._max_abs_diff(base_c, rebuilt_c, dc)
    print(f"[C0] CPU, repeat on same dataset: max|dlogit| = {out['C_cpu_repeat_same_ds']:.3e};  "
          f"rebuilt dataset: {out['C_cpu_rebuild_identical']:.3e}", flush=True)
    out["C_cpu_future_corruption"] = {}
    for c in [122, 130, 140, 150]:
        pc = TC.perturb_future(raw, c, np.random.default_rng(c))
        o = TC.stream_logits(mc, build_temporal_dataset(pc, MAX_DAY, cpu, cfg), FIRST, MAX_DAY)
        past = [d for d in dc if d <= c]
        fut = [d for d in dc if d > c]
        dp = TC._max_abs_diff(base_c, o, past)
        df = TC._max_abs_diff(base_c, o, fut) if fut else float("nan")
        out["C_cpu_future_corruption"][str(c)] = {"max_abs_dlogit_days_le_cutoff": dp, "max_abs_dlogit_days_gt_cutoff": df}
        print(f"[C] CPU, corrupt everything after day {c}: days <= {c}: max|dlogit| = {dp:.3e} "
              f"({'BITWISE IDENTICAL' if dp == 0.0 else 'differs'});  days > {c}: {df:.3e} (perturbation effective)", flush=True)

    path = cfg.experiments_dir / "temporal_graphsage" / "leakage_validation" / "numerics_diagnosis.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
