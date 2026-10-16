"""Phase 9 required validations S2 / S3 / S4 for the exploratory real-data pipeline (design sections 5, 19, 20).

Validated object: `src.coordination.realrun.run_discovery` - the exact function used by the Tier 0 run - on the real clean arrays
(days <= 151), for every locked (W, gamma) configuration (W in {3, 7, 14}, gamma = 0.5 as locked) and every seed (42, 123, 2024).

  S2 label invariance (design 5, 20): discovery outputs bitwise identical when the labels are (a) the real labels, (b) random bits,
     (c) absent, (d) a poisoned object that raises on ANY access; plus static check that discovery modules reference no labels,
     no model risk, no benchmark, no recovery metrics; and that no record carries a risk component.
  S3 future invariance (design 5, 20): outputs for days <= c bitwise identical when ALL data after day c is (a) replaced by random
     entity ids / days, (b) replaced by an adversarial infrastructure burst, (c) deleted. Cutoffs c in {60, 100, 121, 136, 150}.
     Positive control: corrupting rows INSIDE the compared range must change the hash (proves the test can detect a difference).
  S4 determinism (design 19, 20): the same configuration twice in-process and once in a fresh interpreter (random PYTHONHASHSEED)
     gives identical output hashes.
Hash-only comparison: no discovery content is inspected. Any failure -> nonzero exit, failure artifact preserved, STOP.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.data import Arrays, get_real
from src.coordination.realrun import BANNER, hash_output, load_locked_context, out_dir, run_discovery
from src.coordination.scoring import ArrivalTable

ROOT = pathlib.Path(__file__).resolve().parent.parent
CUTOFFS = (60, 100, 121, 136, 150)
N_DAYS = 10
DISCOVERY_MODULES = ("graphs", "scoring", "pipeline", "tracking", "nullmodel", "segments_array", "realrun", "realdesc")


class Poison:
    """Stands in for the label array: any use (indexing, conversion, iteration, attribute) raises."""
    def _boom(self, *a, **k):
        raise AssertionError("discovery touched the label object")
    __array__ = __getitem__ = __len__ = __iter__ = __getattr__ = __bool__ = __eq__ = __add__ = _boom


KEEP = object()


def clone(a: Arrays, y=KEEP, X=None) -> Arrays:
    yy = (a.y.copy() if a.y is not None else None) if y is KEEP else y
    return Arrays(day=a.day.copy(), txn_id=a.txn_id.copy(), card=a.card.copy(), pa=a.pa.copy(), net=a.net.copy(), dev=a.dev.copy(),
                  n_entities=dict(a.n_entities), y=yy, X=X)


def corrupt_after(a: Arrays, c: int, seed: int, mode: str) -> Arrays:
    """mode 'random': random entity ids/days/labels after day c; 'burst': adversarial burst on one infrastructure entity."""
    rng = np.random.default_rng(seed)
    b = clone(a)
    f = b.day > c
    n = int(f.sum())
    ne = b.n_entities
    if n == 0:
        return b
    if mode == "random":
        b.day[f] = rng.integers(c + 1, F.MAX_DAY + 1, size=n)
        b.card[f] = rng.integers(0, ne["card"], size=n)
        b.pa[f] = rng.integers(-1, ne["pseudo_account"], size=n)
        b.net[f] = rng.integers(-1, ne["netproxy"], size=n)
        b.dev[f] = rng.integers(-1, ne["device_fingerprint"], size=n)
        b.y[f] = rng.integers(0, 2, size=n)
    elif mode == "burst":
        b.day[f] = c + 1
        b.card[f] = rng.permutation(ne["card"])[np.arange(n) % ne["card"]]
        b.net[f] = 0
        b.dev[f] = 0
        b.y[f] = 1.0
    else:
        raise ValueError(mode)
    return b


def truncate_after(a: Arrays, c: int) -> Arrays:
    k = a.day <= c
    return Arrays(day=a.day[k], txn_id=a.txn_id[k], card=a.card[k], pa=a.pa[k], net=a.net[k], dev=a.dev[k], n_entities=dict(a.n_entities),
                  y=None if a.y is None else a.y[k], X=None)


def corrupt_inside(a: Arrays, c: int, seed: int) -> Arrays:
    """Positive control: corrupt the last 3 days INSIDE the compared range (days c-2..c)."""
    rng = np.random.default_rng(seed)
    b = clone(a)
    f = (b.day >= c - 2) & (b.day <= c)
    n = int(f.sum())
    b.card[f] = rng.integers(0, b.n_entities["card"], size=n)
    b.net[f] = rng.integers(-1, b.n_entities["netproxy"], size=n)
    b.dev[f] = rng.integers(-1, b.n_entities["device_fingerprint"], size=n)
    return b


def static_checks() -> dict:
    res = {}
    bad_tokens = ["isFraud", ".y[", "arrays.y", "real.y", "coordination.benchmark", "coordination.metrics", "coordination.risk", "RiskData"]
    for mod in DISCOVERY_MODULES:
        p = ROOT / "src" / "coordination" / f"{mod}.py"
        if not p.exists():
            res[mod] = {"exists": False, "violations": ["module missing"]}
            continue
        src = p.read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", src, flags=re.S)
        viol = [t for t in bad_tokens if t in code]
        # pipeline.py legitimately contains the (unused here) risk code path and its own RiskData type; scoring has R components.
        if mod in ("pipeline",):
            viol = [t for t in viol if t not in ("RiskData",)]
        if "benchmark" in code.replace("benchmark-specific", "") and mod != "realrun":
            viol.append("benchmark")
        res[mod] = {"exists": True, "violations": viol}
    # GraphView (the only object discovery receives) must carry no label/risk fields
    from src.coordination.data import GraphView
    res["GraphView_fields"] = sorted(GraphView.__dataclass_fields__.keys())
    res["GraphView_has_no_label_field"] = not any(k in ("y", "label", "labels", "risk", "pct", "logit") for k in GraphView.__dataclass_fields__)
    res["passed"] = all(v["violations"] == [] for k, v in res.items() if isinstance(v, dict) and "violations" in v) and res["GraphView_has_no_label_field"]
    return res


def child_mode(W: int, seed: int) -> int:
    cfg = load_config()
    locked, tables, _ = load_locked_context(cfg)
    real = get_real(cfg)
    g = locked["gamma_per_W"][str(W)]
    out = run_discovery(real, tables[(W, g)], W, g, seed, range(F.EVAL_PERIOD[0], F.EVAL_PERIOD[1] + 1))
    print("HASH", hash_output(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, type=int, metavar=("W", "SEED"))
    a = ap.parse_args()
    if a.child:
        return child_mode(*a.child)

    t0 = time.time()
    cfg = load_config()
    locked, tables, gates = load_locked_context(cfg)
    real = get_real(cfg)
    rec = {"banner": BANNER, "validated_function": "src.coordination.realrun.run_discovery", "gates": gates,
           "locked_gamma_per_W": locked["gamma_per_W"], "cutoffs": list(CUTOFFS), "days_compared_per_cutoff": N_DAYS,
           "S2": {}, "S3": {}, "S4": {}}
    failures = []
    Ws = F.WINDOWS
    cfgs = [(W, float(locked["gamma_per_W"][str(W)])) for W in Ws]
    tab = {W: tables[(W, g)] for W, g in cfgs}

    # ------------------------------------------------------------------ S2
    rec["S2"]["static"] = static_checks()
    if not rec["S2"]["static"]["passed"]:
        failures.append("S2 static check")
    days_all = range(F.EVAL_PERIOD[0], F.EVAL_PERIOD[1] + 1)
    variants = {"real_labels": clone(real), "random_bits": clone(real, y=np.random.default_rng(7).integers(0, 2, size=len(real.day)).astype(np.float32)),
                "labels_absent": clone(real, y=None), "poisoned_labels": clone(real, y=Poison())}
    tables_v = {k: ArrivalTable(v.view()) for k, v in variants.items()}
    s2 = {}
    for W, g in cfgs:
        for S in F.SEEDS:
            hs = {}
            for k, v in variants.items():
                out = run_discovery(v, tab[W], W, g, S, days_all, table=tables_v[k])
                if any(("u_rp" in r or "u_rh" in r) for r in out["comm"]):
                    failures.append(f"S2 risk key present W{W} seed{S}")
                hs[k] = hash_output(out)
            same = len(set(hs.values())) == 1
            s2[f"W{W}_seed{S}"] = {"hash": hs["real_labels"], "n_records": len(out["comm"]), "variants_identical": same}
            if not same:
                failures.append(f"S2 W{W} seed{S}")
    rec["S2"]["dynamic"] = s2
    rec["S2"]["passed"] = rec["S2"]["static"]["passed"] and all(v["variants_identical"] for v in s2.values())
    print(f"S2 {'PASS' if rec['S2']['passed'] else 'FAIL'} ({len(s2)} configs x 4 label variants) {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ S3
    s3 = {}
    for c in CUTOFFS:
        variants3 = {"original": real, "random_future": corrupt_after(real, c, 100 + c, "random"),
                     "burst_future": corrupt_after(real, c, 200 + c, "burst"), "truncated": truncate_after(real, c)}
        tabs3 = {k: ArrivalTable(v.view()) for k, v in variants3.items()}
        ctrl = corrupt_inside(real, c, 300 + c)
        tab_ctrl = ArrivalTable(ctrl.view())
        days_c = range(c - N_DAYS + 1, c + 1)
        for W, g in cfgs:
            for S in F.SEEDS:
                hs = {k: hash_output(run_discovery(v, tab[W], W, g, S, days_c, table=tabs3[k])) for k, v in variants3.items()}
                h_ctrl = hash_output(run_discovery(ctrl, tab[W], W, g, S, days_c, table=tab_ctrl))
                same = len(set(hs.values())) == 1
                detects = h_ctrl != hs["original"]
                s3[f"c{c}_W{W}_seed{S}"] = {"identical_across_future_corruptions": same, "positive_control_detects_change": detects,
                                            "hash": hs["original"]}
                if not same:
                    failures.append(f"S3 c={c} W{W} seed{S}")
                if not detects:
                    failures.append(f"S3 positive control did not detect change c={c} W{W} seed{S}")
        print(f"  S3 cutoff {c} done {time.time()-t0:.0f}s", flush=True)
    rec["S3"]["results"] = s3
    rec["S3"]["passed"] = all(v["identical_across_future_corruptions"] and v["positive_control_detects_change"] for v in s3.values())
    print(f"S3 {'PASS' if rec['S3']['passed'] else 'FAIL'} ({len(s3)} configs)", flush=True)

    # ------------------------------------------------------------------ S4
    s4 = {}
    tr = ArrivalTable(real.view())
    for W, g in cfgs:
        for S in F.SEEDS:
            h1 = hash_output(run_discovery(real, tab[W], W, g, S, days_all, table=tr))
            h2 = hash_output(run_discovery(real, tab[W], W, g, S, days_all))           # fresh arrival table as well
            s4[f"W{W}_seed{S}"] = {"hash1": h1, "hash2": h2, "in_process_identical": h1 == h2}
    # fresh interpreter (random PYTHONHASHSEED) for W=7 and W=3/14, seed 42
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    for W, g in cfgs:
        S = 42
        r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "--child", str(W), str(S)], capture_output=True, text=True, env=env, cwd=str(ROOT))
        h3 = next((ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("HASH")), None)
        e = s4[f"W{W}_seed{S}"]
        e["fresh_process_hash"] = h3
        e["fresh_process_identical"] = h3 == e["hash1"]
        if r.returncode != 0:
            e["child_stderr_tail"] = r.stderr[-500:]
    rec["S4"]["results"] = s4
    rec["S4"]["passed"] = all(v["in_process_identical"] and v.get("fresh_process_identical", True) for v in s4.values())
    for k, v in s4.items():
        if not (v["in_process_identical"] and v.get("fresh_process_identical", True)):
            failures.append(f"S4 {k}")
    print(f"S4 {'PASS' if rec['S4']['passed'] else 'FAIL'}", flush=True)

    rec["all_passed"] = rec["S2"]["passed"] and rec["S3"]["passed"] and rec["S4"]["passed"]
    rec["failures"] = failures
    rec["elapsed_s"] = round(time.time() - t0, 1)
    od = out_dir(cfg)
    name = "validation_S2_S3_S4.json" if rec["all_passed"] else "validation_S2_S3_S4_FAILED.json"
    with open(od / name, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(f"VALIDATION {'PASSED' if rec['all_passed'] else 'FAILED: ' + str(failures)} -> {od / name}", flush=True)
    return 0 if rec["all_passed"] else 4


if __name__ == "__main__":
    sys.exit(main())
