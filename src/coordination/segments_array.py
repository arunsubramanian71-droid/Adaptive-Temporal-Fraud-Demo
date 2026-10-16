"""Array-level equivalent of src.graph.schema.compute_pseudo_account_segments (hard gate G0).

Semantics replicated exactly: rows are processed in stable day order; a transaction is `has_edge_warm` iff its
pseudo-account key occurred in an EARLIER row (strictly earlier day, or earlier row on the same day, in array order);
`has_edge_cold` if it has a key with no earlier occurrence; `no_pseudo_account_edge` if the key is missing (< 0).
Inputs are plain arrays - no graph object, no labels.
"""
from __future__ import annotations

import numpy as np

COLD, WARM, NOEDGE = "has_edge_cold", "has_edge_warm", "no_pseudo_account_edge"


def segments_from_arrays(day: np.ndarray, pa_key: np.ndarray) -> np.ndarray:
    day = np.asarray(day)
    key = np.asarray(pa_key, dtype=np.int64)
    n = len(day)
    present = key >= 0
    order = np.argsort(day, kind="stable")
    p_sorted = present[order]
    k_sorted = key[order]
    idx = np.nonzero(p_sorted)[0]
    prior = np.zeros(n, dtype=np.int64)
    if idx.size:
        ks = k_sorted[idx]
        o2 = np.argsort(ks, kind="stable")          # groups equal keys, preserving chronological row order
        ks2 = ks[o2]
        new_grp = np.r_[True, ks2[1:] != ks2[:-1]]
        grp_start = np.maximum.accumulate(np.where(new_grp, np.arange(ks2.size), 0))
        occ = np.arange(ks2.size) - grp_start       # 0 for the first occurrence of a key
        tmp = np.empty(ks.size, dtype=np.int64)
        tmp[o2] = occ
        prior[idx] = tmp
    warm_sorted = p_sorted & (prior > 0)
    warm = np.zeros(n, dtype=bool)
    warm[order] = warm_sorted
    seg = np.full(n, NOEDGE, dtype=object)
    seg[present & warm] = WARM
    seg[present & ~warm] = COLD
    return seg
