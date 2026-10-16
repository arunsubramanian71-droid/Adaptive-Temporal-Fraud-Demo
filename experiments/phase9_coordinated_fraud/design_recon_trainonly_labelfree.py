"""PROVENANCE RECORD ONLY — not Phase 9 implementation.

This is the exact computation that was run inline (heredoc) during the Phase 9 DESIGN stage to
obtain label-free, TRAIN-period-only structural facts used for the feasibility estimates in
DESIGN_BEFORE_EXECUTION.md (Section 17). It reads: graph structure (entity ids, edge_time via
transaction day) for transactions with day <= 121 only. It does NOT read labels (`y`), does not
load any model, and touches no VAL/TEST rows. Output: design_recon_trainonly_labelfree.json.

Saved afterwards so the numbers are reproducible; Phase 9 execution must re-derive anything it
needs through its own scripted, tested code rather than importing this file.
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from src.config import load_config

cfg = load_config()
g = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
day = g["transaction"].day.numpy()
n = len(day)
RELS = {"charged_to": "pseudo_account", "used_card": "card", "via_network": "netproxy", "from_device": "device_fingerprint"}
ent = {}
for rel, e in RELS.items():
    ei = g[("transaction", rel, e)].edge_index.numpy()
    a = np.full(n, -1, dtype=np.int64)
    a[ei[0]] = ei[1]
    ent[e] = a
TR = day <= cfg.split.train_day_max  # label-free, TRAIN-period only (days <= 121)
out = {"scope": "label-free, TRAIN-period only (days<=121); no labels, no model, no VAL/TEST rows read",
       "n_txn_train": int(TR.sum())}
nn = {e: int(g[e].num_nodes) for e in RELS.values()}

prof = {}
for e, a in ent.items():
    m = TR & (a >= 0)
    ids = a[m]
    cnt = np.bincount(ids, minlength=nn[e])
    act = cnt[cnt > 0]
    d = {"txns_with_edge": int(m.sum()), "active_entities": int((cnt > 0).sum()),
         "degree_median": float(np.median(act)), "degree_p99": float(np.percentile(act, 99)), "degree_max": int(act.max())}
    for other in ("card", "pseudo_account"):
        if other == e:
            continue
        mm = m & (ent[other] >= 0)
        pair = np.unique(a[mm] * np.int64(1 << 24) + ent[other][mm])
        per_e = np.bincount((pair >> 24).astype(np.int64), minlength=nn[e])
        pe = per_e[per_e > 0]
        d[f"distinct_{other}_per_entity"] = {
            "median": float(np.median(pe)), "p90": float(np.percentile(pe, 90)), "p99": float(np.percentile(pe, 99)),
            "max": int(pe.max()), "share_entities_with_>=2": float((pe >= 2).mean()), "share_with_>=3": float((pe >= 3).mean())}
    prof[e] = d
out["entity_profile"] = prof

off = {}
o = 0
for e in RELS.values():
    off[e] = o
    o += nn[e]
N = o
rows, cols, pair_counts = [], [], {}
names = list(RELS.values())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        A, B = names[i], names[j]
        mm = TR & (ent[A] >= 0) & (ent[B] >= 0)
        p = np.unique(ent[A][mm] * np.int64(1 << 24) + ent[B][mm])
        pair_counts[f"{A}--{B}"] = int(p.size)
        rows.append((p >> 24) + off[A])
        cols.append((p & ((1 << 24) - 1)) + off[B])
out["distinct_entity_pairs"] = pair_counts
r = np.concatenate(rows)
c = np.concatenate(cols)


def comps(r, c, keep):
    msk = keep[r] & keep[c]
    A = sp.coo_matrix((np.ones(msk.sum()), (r[msk], c[msk])), shape=(N, N)).tocsr()
    k, lab = connected_components(A, directed=False)
    deg = np.asarray(((A + A.T) > 0).sum(1)).ravel()
    active = deg > 0
    szs = np.bincount(lab[active])
    szs = szs[szs > 0]
    return {"active_nodes": int(active.sum()), "n_components": int(len(szs)), "giant_component_nodes": int(szs.max()),
            "giant_share": float(szs.max() / active.sum()), "components_size>=3": int((szs >= 3).sum()),
            "components_size>=10": int((szs >= 10).sum())}


txdeg = np.zeros(N, dtype=np.int64)
for e in names:
    m = TR & (ent[e] >= 0)
    txdeg[off[e]:off[e] + nn[e]] = np.bincount(ent[e][m], minlength=nn[e])
res = {"all_entities": comps(r, c, np.ones(N, bool))}
for cap in (5000, 1000, 200):
    res[f"hubs_removed_txdeg>{cap}"] = comps(r, c, txdeg <= cap)
    res[f"hubs_removed_txdeg>{cap}"]["entities_removed"] = int((txdeg > cap).sum())
out["cooccurrence_components"] = res

win = {}
for e in ("netproxy", "device_fingerprint"):
    a = ent[e]
    for W in (3, 7, 14):
        starts = range(1, cfg.split.train_day_max - W + 2, W)
        cc, n3c, n3p = [], [], []
        for s in starts:
            m = TR & (a >= 0) & (day >= s) & (day < s + W)
            if m.sum() == 0:
                continue
            dfw = pd.DataFrame({"e": a[m], "card": ent["card"][m], "pa": ent["pseudo_account"][m]})
            gc = dfw.groupby("e")["card"].nunique()
            gp = dfw[dfw.pa >= 0].groupby("e")["pa"].nunique()
            n3c.append(int((gc >= 3).sum()))
            n3p.append(int((gp >= 3).sum()))
            cc.append(int(gc.size))
        win[f"{e}_W{W}d"] = {"windows": len(cc), "mean_active_entities_per_window": float(np.mean(cc)),
                             "mean_entities_with_>=3_distinct_cards": float(np.mean(n3c)),
                             "mean_entities_with_>=3_distinct_pseudo_accounts": float(np.mean(n3p))}
out["windowed_shared_infrastructure"] = win

k = (ent["pseudo_account"] >= 0).astype(int) + (ent["netproxy"] >= 0) + (ent["device_fingerprint"] >= 0)
out["txn_count_by_n_noncard_entities_train"] = {int(i): int(((k == i) & TR).sum()) for i in range(4)}
json.dump(out, open("experiments/phase9_coordinated_fraud/design_recon_trainonly_labelfree.json", "w"), indent=1)
print(json.dumps(out, indent=1))
