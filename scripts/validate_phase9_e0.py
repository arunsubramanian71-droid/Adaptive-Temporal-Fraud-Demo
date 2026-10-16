"""E0 validation (addendum section 5): unit tests, null-generator property tests P0-P6, S2/S3/S4 extensions, Tier-0 integrity.

Hash-only on real data: no alert count, A, tau or tail proportion is computed on real data. Validation null histories use replicate keys
k >= 1,000,000 (the experiment uses 0..99). The driver logic (`run_seed`) is exercised ONLY on a synthetic toy dataset.
Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6.
Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.
Any failure -> nonzero exit and a *_FAILED.json record; STOP and report.
"""
import itertools
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from scipy.stats import chisquare

import validate_phase9_leakage as V
from src.config import load_config
from src.coordination import frozen as F
from src.coordination.data import Arrays, get_real
from src.coordination.e0 import (ADDENDUM, BANNED_REPORT_WORDS, EVAL_DAYS, GAMMA_E0, K_E0, REHIST_START, TAIL_LABEL, TAU_N_DAYS, TAU_TARGET, VALIDATION_K0,
                                 W_E0, alert_counts, record_scores, render_report, rewired_arrays, run_null, run_observed, run_seed, select_tau,
                                 sha256_file, source_hashes, tail_proportion, tau_days)
from src.coordination.graphs import rewire, window_rows
from src.coordination.lint import lint_files, lint_text
from src.coordination.nullmodel import NullTables, calibration_days
from src.coordination.pipeline import record_score
from src.coordination.realrun import BANNER, hash_output, load_locked_context, out_dir, run_discovery
from src.coordination.scoring import ArrivalTable

CUTS = (125, 136, 150)
K0 = VALIDATION_K0


def sha_arr(*arrs) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def check(rec: dict, name: str, ok: bool, detail=None):
    rec[name] = {"passed": bool(ok)} | ({"detail": detail} if detail is not None else {})
    return bool(ok)


def edges_equal(a, b) -> bool:
    return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("card", "infra", "day", "kind"))


def canon(e, lo, hi):
    m = (e.day >= lo) & (e.day <= hi)
    arr = np.stack([e.kind[m].astype(np.int64), e.day[m], e.infra[m], e.card[m]], 1)
    return arr[np.lexsort(arr.T[::-1])]


def toy_arrays(seed=2025) -> Arrays:
    rng = np.random.default_rng(seed)
    days = np.sort(np.repeat(np.arange(100, 152), 120))
    n = len(days)
    net = np.where(rng.random(n) < 0.5, rng.integers(0, 30, n), -1)
    dev = np.where(rng.random(n) < 0.4, rng.integers(0, 20, n), -1)
    return Arrays(day=days.astype(np.int64), txn_id=np.arange(n, dtype=np.int64), card=rng.integers(0, 300, n).astype(np.int64),
                  pa=rng.integers(0, 200, n).astype(np.int64), net=net.astype(np.int64), dev=dev.astype(np.int64),
                  n_entities={"pseudo_account": 200, "card": 300, "netproxy": 30, "device_fingerprint": 20}, y=np.zeros(n, dtype=np.float32))


def toy_tables() -> NullTables:
    nt = NullTables(W_E0, GAMMA_E0)
    for b in range(len(F.SIZE_BUCKETS)):
        nt.s[("comm", b)] = np.sort(np.linspace(-5, 60, 200))
        nt.t[("comm", b)] = np.sort(np.linspace(0, 5, 200))
    return nt


def static_e0() -> dict:
    res = {}
    bad = ["isFraud", ".y[", "arrays.y", "real.y", "coordination.benchmark", "coordination.metrics", "coordination.risk", "RiskData"]
    for f in ("src/coordination/e0.py", "scripts/run_phase9_e0.py", "scripts/run_phase9_e0_tau.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", src, flags=re.S)
        viol = [t for t in bad if t in code]
        if f.endswith("e0.py") and "src/" in f and "benchmark" in code.lower():
            viol.append("benchmark")
        res[f] = viol
    res["passed"] = all(v == [] for k, v in res.items() if k != "passed")
    return res


def child(argv) -> int:
    cfg = load_config()
    locked, tables, _ = load_locked_context(cfg)
    real = get_real(cfg)
    nt = tables[(W_E0, GAMMA_E0)]
    kind, S = argv[0], int(argv[1])
    if kind == "null_arrays":
        a = rewired_arrays(real, S, K0)
        print("HASH", sha_arr(a.net, a.dev))
    elif kind == "obs":
        print("HASH", hash_output(run_observed(real, nt, S, ArrivalTable(real.view()))))
    elif kind == "null":
        print("HASH", hash_output(run_null(real, nt, S, K0, ArrivalTable(real.view()))))
    elif kind == "tau":
        print("HASH", hash_output(run_discovery(real, nt, W_E0, GAMMA_E0, S, tau_days(), table=ArrivalTable(real.view()))))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2:])
    t0 = time.time()
    cfg = load_config()
    od = out_dir(cfg)
    add = ROOT / ADDENDUM
    if sha256_file(add) != (od / "E0_ADDENDUM_post_Tier0_exploratory.sha256").read_text().strip():
        raise RuntimeError("addendum hash mismatch: nothing may be implemented or validated against a modified addendum")
    forbidden = [p.name for p in list(od.glob("e0_tau_seed*")) + [od / "e0_results.json", od / "E0_REPORT.md"] if p.exists()]
    assert not forbidden, f"E0 outputs must not exist before validation: {forbidden}"
    locked, tables, gates = load_locked_context(cfg)
    nt = tables[(W_E0, GAMMA_E0)]
    real = get_real(cfg)
    tier0 = json.loads((od / "tier0_summary.json").read_text(encoding="utf-8"))
    R = {"banner": BANNER, "addendum_sha256": sha256_file(add), "source_sha256": source_hashes(), "gates": gates,
         "env": {"python": platform.python_version(), "numpy": np.__version__}, "validation_replicate_key_start": K0}
    fails = []

    def rec(group, name, ok, detail=None):
        R.setdefault(group, {})
        if not check(R[group], name, ok, detail):
            fails.append(f"{group}.{name}")

    # ---------------------------------------------------------------- unit tests (synthetic only)
    s = select_tau([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2)
    rec("unit", "tau_simple", s["feasible"] and s["tau"] == 0.3 and s["n_records_ge_tau"] == 4 and s["realized_rate_per_day"] == 2.0)
    s = select_tau([0.9, 0.9, 0.9, 0.5, 0.5], 1)
    rec("unit", "tau_degenerate_top_tie_mass", (not s["feasible"]) and "highest distinct score" in s["reason"])
    s = select_tau([0.9, 0.8, 0.8, 0.5], 1)
    rec("unit", "tau_jump_slack_and_tie_alert", s["feasible"] and s["tau"] == 0.9 and s["n_records_tied_at_tau"] == 1 and s["realized_rate_per_day"] == 1.0)
    s = select_tau([0.4, 0.6], 5)
    rec("unit", "tau_all_below_target_uses_min_score", s["feasible"] and s["tau"] == 0.4 and s["n_records_ge_tau"] == 2)
    rec("unit", "tau_empty_infeasible", not select_tau([], 30)["feasible"])
    a, b = 0.3, 0.7
    r_ = {"u_s": a, "u_t": b}
    rec("unit", "score_float_identity", record_score(r_, ("s", "t")) == 0.5 * (a + b) == float(np.mean([a, b])))
    rng = np.random.default_rng(5)
    ok = True
    for _ in range(200):
        sc = np.round(rng.random(rng.integers(1, 200)), 2)
        nd = int(rng.integers(1, 40))
        sel = select_tau(sc, nd)
        if sel["feasible"]:
            vals = np.unique(sc)
            ge = lambda v: int((sc >= v).sum())
            lower = vals[vals < sel["tau"]]
            ok &= ge(sel["tau"]) <= 2.0 * nd and (lower.size == 0 or ge(lower[-1]) > 2.0 * nd)
        else:
            ok &= int((sc >= sc.max()).sum()) > 2.0 * nd
    rec("unit", "tau_is_smallest_admissible_random_cases", ok)
    td = tau_days()
    rec("unit", "tau_days", len(td) == TAU_N_DAYS == 30 and td == sorted(set(td)) and min(td) >= 89 and max(td) <= 121
        and not (set(td) & set(calibration_days())) and (set(range(89, 122)) & set(calibration_days())) == {96, 103, 118}, td)
    tp = tail_proportion(3, [1, 2, 3, 4])
    rec("unit", "tail_ties_count_as_extreme", tp["n_null_ge_obs"] == 2 and abs(tp["tail_proportion"] - 0.6) < 1e-12 and abs(tp["min_attainable"] - 0.2) < 1e-12)
    rec("unit", "tail_bounds", tail_proportion(10, [1, 2, 3, 4])["tail_proportion"] == 0.2 and tail_proportion(0, [1, 2, 3, 4])["tail_proportion"] == 1.0)
    rec("unit", "tail_min_for_K100", abs(tail_proportion(10 ** 6, np.zeros(100))["tail_proportion"] - 1 / 101) < 1e-12)

    # ---------------------------------------------------------------- toy end-to-end of the driver logic (synthetic data only)
    toy, tnt = toy_arrays(), toy_tables()
    ttab = ArrivalTable(toy.view())
    r1 = run_seed(toy, tnt, 42, 0.5, ttab, K=3, k_start=K0)
    r2 = run_seed(toy, tnt, 42, 0.5, ttab, K=3, k_start=K0)
    r3 = run_seed(toy, tnt, 42, 0.5, ttab, K=2, k_start=K0 + 1)
    rec("toy_driver", "deterministic", r1 == r2)
    rec("toy_driver", "batch_and_order_independent", r1["A_null"][1:] == r3["A_null"])
    rec("toy_driver", "shapes_and_bounds", len(r1["obs_counts_per_day"]) == 30 and len(r1["A_null"]) == 3 and r1["A_obs"] == sum(r1["obs_counts_per_day"])
        and 1 / 4 <= r1["tail"]["tail_proportion"] <= 1.0 and r1["A_obs"] >= 0)
    rec("toy_driver", "toy_has_candidates", int(sum(r1["obs_counts_per_day"])) >= 0 and len(run_observed(toy, tnt, 42, ttab)["comm"]) > 0)
    fake_tau = {42: {"tau": 0.5, "realized_rate_per_day": 1.9, "n_records_tied_at_tau": 1}}
    rep = render_report({42: r1}, fake_tau)
    low = rep.lower()
    rec("report", "banner_and_labels", rep.startswith("> " + BANNER) and TAIL_LABEL in rep and "not independent replications" in rep
        and "not evidence of fraud" in rep)
    rec("report", "no_threshold_or_verdict_language", not [w for w in BANNED_REPORT_WORDS if w in low], [w for w in BANNED_REPORT_WORDS if w in low])
    rec("report", "terminology_lint_clean", not lint_text(rep), lint_text(rep))

    # ---------------------------------------------------------------- null-generator property tests
    v_real = real.view()
    null42 = rewired_arrays(real, 42, K0)
    v_null = null42.view()
    ok0 = True
    for d in (122, 137, 151):
        exp = rewire(window_rows(v_real, d, 1), np.random.default_rng(np.random.SeedSequence([F.PERM_SEED_BASE + 42, 1, K0, d])))
        ok0 &= edges_equal(window_rows(v_null, d, 1), exp)
    rec("P0", "assembly_equals_rewire_on_window_rows", ok0)
    wa, wb = window_rows(v_null, 130, 7), window_rows(v_null, 131, 7)
    rec("P1", "overlapping_windows_share_identical_rows", np.array_equal(canon(wa, 125, 130), canon(wb, 125, 130)))
    exp127 = rewire(window_rows(v_real, 127, 1), np.random.default_rng(np.random.SeedSequence([F.PERM_SEED_BASE + 42, 1, K0, 127])))
    rec("P1", "window_rows_equal_the_single_rewired_day", np.array_equal(canon(wa, 127, 127), canon(exp127, 127, 127)) and np.array_equal(canon(wb, 127, 127), canon(exp127, 127, 127)))
    in_r = (real.day >= REHIST_START) & (real.day <= F.MAX_DAY)
    n_net, n_dev = int(real.n_entities["netproxy"]), int(real.n_entities["device_fingerprint"])
    only_infra = (np.array_equal(null42.card, real.card) and np.array_equal(null42.day, real.day) and np.array_equal(null42.pa, real.pa)
                  and np.array_equal(null42.txn_id, real.txn_id) and np.array_equal(null42.net >= 0, real.net >= 0) and np.array_equal(null42.dev >= 0, real.dev >= 0))
    rec("P2", "only_infrastructure_values_change", only_infra)
    outside = (~in_r) | (real.card < 0)
    rec("P2", "rows_outside_range_or_without_card_untouched", np.array_equal(null42.net[outside], real.net[outside]) and np.array_equal(null42.dev[outside], real.dev[outside]))
    ok2 = True
    for d in range(REHIST_START, F.MAX_DAY + 1):
        m = (real.day == d) & (real.card >= 0)
        for a, b in ((real.net, null42.net), (real.dev, null42.dev)):
            mm = m & (a >= 0)
            ok2 &= np.array_equal(np.sort(a[mm]), np.sort(b[mm]))
    rec("P2", "per_day_per_kind_infrastructure_multiset_preserved", ok2)
    rec("P3", "ids_stay_in_their_kind_range", bool(((null42.net[in_r & (null42.net >= 0)] < n_net).all()) and ((null42.dev[in_r & (null42.dev >= 0)] < n_dev).all())))
    changed = float(((null42.net != real.net) & in_r).sum() / max(int((in_r & (real.net >= 0)).sum()), 1))
    rec("P2", "rewiring_changes_a_material_share_of_rows", changed > 0.05, changed)
    for c in CUTS:
        rc = V.corrupt_after(real, c, 900 + c, "random")
        rb = V.corrupt_after(real, c, 950 + c, "burst")
        tr = V.truncate_after(real, c)
        base = rewired_arrays(real, 42, K0)
        n_c = int((real.day <= c).sum())               # real rows are in day order, so rows with day <= c are the prefix
        okc = True
        for x in (rc, rb, tr):
            nx = rewired_arrays(x, 42, K0)
            okc &= np.array_equal(nx.net[:n_c], base.net[:n_c]) and np.array_equal(nx.dev[:n_c], base.dev[:n_c])
        okc &= bool((real.day[:n_c] <= c).all()) and bool((real.day[n_c:] > c).all())
        rec("P4", f"causality_cutoff_{c}", okc)
    a1, a2 = rewired_arrays(real, 42, K0), rewired_arrays(real, 42, K0)
    b1 = rewired_arrays(real, 42, K0 + 1)
    c_first = rewired_arrays(real, 42, K0 + 5)
    c_second = rewired_arrays(real, 42, K0 + 5)
    rec("P5", "same_key_same_history", np.array_equal(a1.net, a2.net) and np.array_equal(a1.dev, a2.dev))
    rec("P5", "different_k_different_history", not np.array_equal(a1.net, b1.net))
    rec("P5", "different_seed_different_history", not np.array_equal(a1.net, rewired_arrays(real, 123, K0).net))
    rec("P5", "order_independence", np.array_equal(c_first.net, c_second.net) and np.array_equal(b1.net, rewired_arrays(real, 42, K0 + 1).net))
    perm_toy = Arrays(day=np.full(4, 140, dtype=np.int64), txn_id=np.arange(4, dtype=np.int64), card=np.arange(4, dtype=np.int64), pa=np.arange(4, dtype=np.int64),
                      net=np.array([10, 11, 12, 13], dtype=np.int64), dev=np.full(4, -1, dtype=np.int64),
                      n_entities={"pseudo_account": 4, "card": 4, "netproxy": 20, "device_fingerprint": 5}, y=None)
    counts: dict = {}
    for j in range(12000):
        key = tuple(rewired_arrays(perm_toy, 42, K0 + j).net.tolist())
        counts[key] = counts.get(key, 0) + 1
    obs_c = np.array([counts.get(p, 0) for p in itertools.permutations([10, 11, 12, 13])])
    chi = chisquare(obs_c)
    rec("P6", "small_case_uniformity_chi_square", len(counts) == 24 and chi.pvalue > 1e-3, {"chi2": float(chi.statistic), "p": float(chi.pvalue), "n_arrangements": len(counts)})
    print(f"unit/toy/property tests done {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- S2: label invariance (hash-only), observed / null / tau paths
    variants = {"real_labels": V.clone(real), "random_bits": V.clone(real, y=np.random.default_rng(7).integers(0, 2, size=len(real.day)).astype(np.float32)),
                "labels_absent": V.clone(real, y=None), "poisoned_labels": V.clone(real, y=V.Poison())}
    tabs = {k: ArrivalTable(v.view()) for k, v in variants.items()}
    s2 = {}
    for S in F.SEEDS:
        h_obs, h_null, h_tau = {}, {}, {}
        for k, v in variants.items():
            h_obs[k] = hash_output(run_observed(v, nt, S, tabs[k]))
            h_null[k] = hash_output(run_null(v, nt, S, K0, tabs[k]))
            h_tau[k] = hash_output(run_discovery(v, nt, W_E0, GAMMA_E0, S, tau_days(), table=tabs[k]))
        s2[S] = {"obs": len(set(h_obs.values())) == 1, "null": len(set(h_null.values())) == 1, "tau": len(set(h_tau.values())) == 1,
                 "obs_hash": h_obs["real_labels"],
                 "matches_tier0_output_hash": h_obs["real_labels"] == tier0["configs"][f"W{W_E0}_seed{S}"]["output_hash"]}
    rec("S2", "label_variants_identical_all_paths", all(v["obs"] and v["null"] and v["tau"] for v in s2.values()), {str(k): {a: b for a, b in v.items() if a != "obs_hash"} for k, v in s2.items()})
    rec("integrity", "observed_path_reproduces_tier0_output_hash", all(v["matches_tier0_output_hash"] for v in s2.values()))
    st = static_e0()
    rec("S2", "static_module_check_e0", st["passed"], st)
    print(f"S2 done {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- S3: future invariance
    s3 = {}
    for c in CUTS:
        days_c = [d for d in EVAL_DAYS if d <= c]
        vv = {"original": real, "random_future": V.corrupt_after(real, c, 100 + c, "random"), "burst_future": V.corrupt_after(real, c, 200 + c, "burst"),
              "truncated": V.truncate_after(real, c)}
        tb = {k: ArrivalTable(v.view()) for k, v in vv.items()}
        ctrl = V.corrupt_inside(real, c, 300 + c)
        tctrl = ArrivalTable(ctrl.view())
        for S in F.SEEDS:
            ho = {k: hash_output(run_observed(v, nt, S, tb[k], days_c)) for k, v in vv.items()}
            hn = {k: hash_output(run_null(v, nt, S, K0, tb[k], days_c)) for k, v in vv.items()}
            co = hash_output(run_observed(ctrl, nt, S, tctrl, days_c)) != ho["original"]
            cn = hash_output(run_null(ctrl, nt, S, K0, tctrl, days_c)) != hn["original"]
            s3[f"c{c}_S{S}"] = {"obs_identical": len(set(ho.values())) == 1, "null_identical": len(set(hn.values())) == 1, "control_obs": co, "control_null": cn}
        print(f"  S3 cutoff {c} {time.time() - t0:.0f}s", flush=True)
    vv = {"original": real, "random_future": V.corrupt_after(real, 121, 421, "random"), "burst_future": V.corrupt_after(real, 121, 521, "burst"), "truncated": V.truncate_after(real, 121)}
    tb = {k: ArrivalTable(v.view()) for k, v in vv.items()}
    ctrl = V.corrupt_inside(real, 121, 621)
    tctrl = ArrivalTable(ctrl.view())
    for S in F.SEEDS:
        ht = {k: hash_output(run_discovery(v, nt, W_E0, GAMMA_E0, S, tau_days(), table=tb[k])) for k, v in vv.items()}
        ct = hash_output(run_discovery(ctrl, nt, W_E0, GAMMA_E0, S, tau_days(), table=tctrl)) != ht["original"]
        s3[f"tau_c121_S{S}"] = {"identical": len(set(ht.values())) == 1, "control": ct}
    ok3 = all((v.get("obs_identical", True) and v.get("null_identical", True) and v.get("identical", True)) and v.get("control_obs", True) and v.get("control_null", True) and v.get("control", True)
              for v in s3.values())
    rec("S3", "future_invariance_observed_null_tau_with_positive_controls", ok3, s3)
    print(f"S3 done {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- S4: determinism (in-process and fresh interpreter)
    ttab = ArrivalTable(real.view())
    s4 = {}
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}

    def fresh(kind, S):
        r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "--child", kind, str(S)], capture_output=True, text=True, env=env, cwd=str(ROOT))
        return next((ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("HASH")), None), r.returncode, r.stderr[-300:]

    for kind in ("null_arrays", "obs", "null", "tau"):
        S = 42
        if kind == "null_arrays":
            h1 = sha_arr(*(lambda a: (a.net, a.dev))(rewired_arrays(real, S, K0)))
            h2 = sha_arr(*(lambda a: (a.net, a.dev))(rewired_arrays(real, S, K0)))
        elif kind == "obs":
            h1, h2 = (hash_output(run_observed(real, nt, S, ttab)) for _ in range(2))
        elif kind == "null":
            h1, h2 = (hash_output(run_null(real, nt, S, K0, ttab)) for _ in range(2))
        else:
            h1, h2 = (hash_output(run_discovery(real, nt, W_E0, GAMMA_E0, S, tau_days(), table=ttab)) for _ in range(2))
        h3, rc, err = fresh(kind, S)
        s4[kind] = {"in_process_identical": h1 == h2, "fresh_process_identical": h3 == h1, "returncode": rc}
    rec("S4", "determinism_in_process_and_fresh_interpreter", all(v["in_process_identical"] and v["fresh_process_identical"] for v in s4.values()), s4)

    # ---------------------------------------------------------------- hygiene
    new_files = [ROOT / "src/coordination/e0.py", ROOT / "scripts/run_phase9_e0.py", ROOT / "scripts/run_phase9_e0_tau.py", ROOT / "scripts/validate_phase9_e0.py", add]
    lint = {k: v for k, v in lint_files(new_files).items() if v}
    rec("hygiene", "terminology_and_identifier_lint_clean", not lint, lint)
    comp_ok = True
    for f in ("scripts/run_phase9_e0.py", "scripts/run_phase9_e0_tau.py"):
        try:
            compile((ROOT / f).read_text(encoding="utf-8"), f, "exec")
        except SyntaxError:
            comp_ok = False
    rec("hygiene", "drivers_compile_but_were_not_executed", comp_ok)
    still = [p.name for p in list(od.glob("e0_tau_seed*")) + [od / "e0_results.json", od / "E0_REPORT.md"] if p.exists()]
    rec("hygiene", "no_tau_or_e0_outputs_were_produced", not still, still)

    R["all_passed"] = not fails
    R["failures"] = fails
    R["source_sha256"] = source_hashes()
    R["elapsed_s"] = round(time.time() - t0, 1)
    name = "validation_E0.json" if not fails else "validation_E0_FAILED.json"
    (od / name).write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(f"E0 VALIDATION {'PASSED' if not fails else 'FAILED: ' + str(fails)} -> {od / name}", flush=True)
    return 0 if not fails else 4


if __name__ == "__main__":
    sys.exit(main())
