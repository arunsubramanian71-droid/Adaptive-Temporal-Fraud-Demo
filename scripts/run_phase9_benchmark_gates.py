"""Phase 9 benchmark validity gates G1, G2, G4 on the EVAL benchmark (design 11.7). Evaluated BEFORE any detector output exists.
Opens the sealed manifest (generator-side validation only; no detector output has been produced yet - checked by file absence).

  G1: alpha=0 planted transactions with within-day risk percentile > 0.99 (top 1%)  <= 3%
  G2: alpha=1 top-1% fraction >= 5 x the alpha=0 fraction
  G4: mean |delta risk percentile| of REAL transactions between the clean and augmented streams <= 0.5 percentile points
(G3, G5 were checked at generation time; G0 before everything.) Any failure -> hard stop S6.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.data import get_real
from src.coordination.runner import exp_dir, load_benchmark_arrays, load_sealed, require_gates


def main() -> int:
    cfg = load_config()
    require_gates(cfg)
    d = exp_dir(cfg)
    assert not list((d / "discovery").glob("eval_*")) if (d / "discovery").exists() else True, "gates must run before any detector output exists"
    real = get_real(cfg)
    res = {}
    ok_all = True
    for S in F.SEEDS:
        man, is_pl, rg = load_sealed(cfg, "eval", S)
        arrays, _ = load_benchmark_arrays(cfg, "eval", S, real)
        ra = np.load(d / "risk" / f"risk_eval_seed{S}.npz")
        rc = np.load(d / "risk" / f"risk_clean_seed{S}.npz")
        pct = ra["pct"]
        gmeta = {g["group_id"]: g for g in man["groups"]}
        def frac(pred):
            sel = np.array([(r >= 0) and pred(gmeta[int(r)]) for r in rg])
            return float((pct[sel] > 0.99).mean()), int(sel.sum())
        f0, n0 = frac(lambda g: g["kind"] == "coordinated" and g["regime_alpha"] == 0.0)
        f05, n05 = frac(lambda g: g["kind"] == "coordinated" and g["regime_alpha"] == 0.5)
        f1, n1 = frac(lambda g: g["kind"] == "coordinated" and g["regime_alpha"] == 1.0)
        fd, nd = frac(lambda g: g["kind"] == "decoy")
        # G4: real rows only; align by transaction id
        real_aug = ~is_pl
        id_aug = arrays.txn_id[real_aug]
        clean_by_id = dict(zip(real.txn_id.tolist(), rc["pct"].tolist()))
        in_eval = (arrays.day[real_aug] >= 122)
        dif = np.abs(pct[real_aug][in_eval] - np.array([clean_by_id[i] for i in id_aug[in_eval].tolist()])) * 100.0
        g1 = f0 <= F.G1_MAX_TOP1
        g2 = (f1 >= F.G2_MIN_RATIO * f0) and f1 > 0
        g4 = float(dif.mean()) <= F.G4_MAX_DELTA_PCT_POINTS
        res[S] = {"alpha0_top1_fraction": f0, "n_alpha0_txns": n0, "alpha05_top1_fraction": f05, "n_alpha05_txns": n05,
                  "alpha1_top1_fraction": f1, "n_alpha1_txns": n1, "decoy_top1_fraction": fd, "n_decoy_txns": nd,
                  "G1_pass": bool(g1), "G2_pass": bool(g2), "G2_ratio": (f1 / f0) if f0 > 0 else None,
                  "G4_mean_abs_delta_pct_points": float(dif.mean()), "G4_pass": bool(g4), "G4_max_abs_delta_pct_points": float(dif.max()),
                  "baseline_top1_by_construction": 0.01}
        ok_all &= bool(g1 and g2 and g4)
        print(f"seed {S}: G1 {g1} (alpha0 top1 {f0:.3f}, n={n0}) | G2 {g2} (alpha1 {f1:.3f}; ratio {res[S]['G2_ratio']}) | G4 {g4} (mean |d| {dif.mean():.3f} pts) | alpha0.5 {f05:.3f} decoy {fd:.3f}", flush=True)
    res["all_passed"] = bool(ok_all)
    with open(d / "eval_benchmark_G1_G2_G4.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return 0 if ok_all else 3


if __name__ == "__main__":
    sys.exit(main())
