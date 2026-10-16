"""READ-ONLY diagnosis of the G4 failure (design hard stop S6). Does not regenerate anything and does not change any gate.
Decomposes the clean-vs-augmented risk-percentile difference of REAL transactions into
  (a) the as-defined quantity (percentile within day, planted rows included in the denominator),
  (b) the same quantity after ranking real rows only (planted rows removed from the denominator) -> isolates state interference,
  (c) raw logit differences of real rows.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.stats import rankdata

from src.config import load_config
from src.coordination.data import get_real
from src.coordination.runner import exp_dir, load_benchmark_arrays, load_sealed

cfg = load_config()
d = exp_dir(cfg)
real = get_real(cfg)
out = {}
for S in (42, 123, 2024):
    man, is_pl, rg = load_sealed(cfg, "eval", S)
    arrays, _ = load_benchmark_arrays(cfg, "eval", S, real)
    ra, rc = np.load(d / "risk" / f"risk_eval_seed{S}.npz"), np.load(d / "risk" / f"risk_clean_seed{S}.npz")
    clean = dict(zip(real.txn_id.tolist(), zip(rc["pct"].tolist(), rc["logit"].tolist())))
    ra_pct, ra_logit = ra["pct"], ra["logit"]
    idx = np.nonzero((~is_pl) & (arrays.day >= 122))[0]
    ids = arrays.txn_id[idx]
    c_pct = np.array([clean[i][0] for i in ids.tolist()])
    c_logit = np.array([clean[i][1] for i in ids.tolist()])
    a_pct, a_logit = ra_pct[idx], ra_logit[idx]
    d_as_defined = np.abs(a_pct - c_pct) * 100
    # (b) re-rank real rows only, within day, in the augmented stream
    a_pct_realonly = np.zeros(len(idx))
    for day in np.unique(arrays.day[idx]):
        m = arrays.day[idx] == day
        a_pct_realonly[m] = rankdata(a_logit[m], method="average") / m.sum()
    d_real_only = np.abs(a_pct_realonly - c_pct) * 100
    dl = np.abs(a_logit - c_logit)
    pl_pct = ra_pct[is_pl & (arrays.day >= 122)]
    out[S] = {"n_real_rows": int(len(idx)), "as_defined_mean_abs_pct_points": float(d_as_defined.mean()),
              "real_only_rerank_mean_abs_pct_points": float(d_real_only.mean()), "real_only_rerank_max": float(d_real_only.max()),
              "mean_abs_logit_diff": float(dl.mean()), "frac_rows_logit_unchanged": float((dl == 0).mean()),
              "frac_rows_logit_changed_gt_1e-3": float((dl > 1e-3).mean()), "planted_share_of_day_rows": float(is_pl.sum() / max((arrays.day >= 122).sum(), 1)),
              "planted_mean_within_day_pct": float(pl_pct.mean()), "planted_median_within_day_pct": float(np.median(pl_pct))}
    print(S, json.dumps(out[S]))
with open(pathlib.Path(__file__).with_name("g4_diagnosis.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
