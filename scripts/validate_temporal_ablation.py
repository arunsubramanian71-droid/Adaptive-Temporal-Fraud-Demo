"""Phase 7A leakage / validity suite for the two new arms.

  --arm C : temporal leakage suite (src/graph/temporal_checks.py, unchanged) on the trained memory-ablated
            models + order-agnostic-memory test (with the GRU model D as positive control) + Phase 4 graph suite.
  --arm B : interventional checks of the static rolling protocol on the trained schedule-matched models.

  --device cpu  : deterministic -> exact equality required
  --device cuda : run-to-run noise-floor tolerance (measured on identical repeated input)
TEST is never scored or used; TEST rows exist inside the static Phase 4 graph but are unreachable through
edge_time masking - check `static_masked_graph_isolates_future_nodes` verifies that for every scored day.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.graph import temporal_checks as TC
from src.graph.build import apply_time_mask
from src.graph.rolling_eval import rolling_predict
from src.graph.temporal import build_temporal_dataset, load_raw_arrays
from src.models.graphsage import HeteroSAGE
from src.models.temporal_graphsage import TemporalGraphSAGE
from src.models.temporal_meanmem import TemporalGraphSAGEMeanMemory

SEEDS = [42, 123, 2024]
CUTOFFS_C = [122, 130, 140, 150]
CUTOFFS_B = [122, 136, 151]


# ------------------------------------------------------------------ arm C extras
@torch.no_grad()
def check_order_agnostic(model_cls, ckpt, raw, dev, cfg, last_day: int, expect_invariant: bool):
    """Process days 1..last_day in chronological order vs a random permutation; compare final entity state.
    C: must agree up to float32 summation rounding (relative). D (positive control): must differ substantially."""
    ds = build_temporal_dataset(raw, last_day, dev, cfg)
    model = model_cls(raw["X"].shape[1]).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    st = model.init_state(ds)
    for d in ds.days:
        model.step(st, ds, d, predict=False)
    order = list(ds.days)
    np.random.RandomState(7).shuffle(order)
    st2 = model.init_state(ds)
    for d in order:
        model.step(st2, ds, d, predict=False)
    rel, absd = 0.0, 0.0
    for k in st.S:
        a, b = st.S[k], st2.S[k]
        absd = max(absd, float((a - b).abs().max()))
        rel = max(rel, float(((a - b).abs() / (a.abs().max().clamp(min=1e-6))).max()))
    ok = rel < 1e-5 if expect_invariant else rel > 1e-2
    del ds
    return ok, (f"state after chronological vs shuffled day order ({model_cls.__name__}, days 1-{last_day}): "
                f"max|diff|={absd:.3e}, max relative diff (to per-type state scale)={rel:.3e} -> "
                f"{'order-agnostic (float rounding only)' if expect_invariant else 'order-DEPENDENT (positive control)'}")


def run_arm_c(args, cfg, dev, out, results):
    base = cfg.experiments_dir / "temporal_ablation" / "memory_ablated_temporal"
    print("=== 1. Phase 4 graph validation suite (unchanged, re-run) ===", flush=True)
    proc = subprocess.run([sys.executable, "-u", "scripts/validate_graph.py"], capture_output=True, text=True,
                          cwd=str(pathlib.Path(__file__).resolve().parent.parent))
    text = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines() if "offload-arch" not in l)
    (out / f"graph_validation_output_armC_{args.device}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    results.append(("graph_validation_suite_8_of_8", proc.returncode == 0, "scripts/validate_graph.py exit code 0" if proc.returncode == 0 else "FAILED"))

    raw, graph, scaler = load_raw_arrays(cfg)
    max_day, first_pred = cfg.split.val_day_max, cfg.split.val_day_min
    ds = build_temporal_dataset(raw, max_day, dev, cfg)
    print("\n=== 2. Dataset-level checks ===", flush=True)
    for name, fn in [("no_test_data_loaded", lambda: TC.check_no_test_data(ds, raw, cfg)),
                     ("scaler_fit_on_train_only", lambda: TC.check_scaler_train_only(raw, cfg)),
                     ("edge_consumption_no_future_edges", lambda: TC.check_edge_consumption(raw, dev, cfg, max_day))]:
        ok, detail = fn()
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    del ds

    print("\n=== 3. Order-agnostic memory test (CPU-independent property; run on the current device) ===", flush=True)
    ck42 = base / "seed_42" / "model_checkpoint.pt"
    ok, d = check_order_agnostic(TemporalGraphSAGEMeanMemory, ck42, raw, dev, cfg, 40, True)
    results.append(("order_agnostic_memory_C_seed42", ok, d)); print(f"  [{'PASS' if ok else 'FAIL'}] {d}", flush=True)
    ck_d = cfg.experiments_dir / "temporal_graphsage" / "seed_42" / "model_checkpoint.pt"
    ok, d = check_order_agnostic(TemporalGraphSAGE, ck_d, raw, dev, cfg, 40, False)
    results.append(("order_dependent_memory_D_positive_control", ok, d)); print(f"  [{'PASS' if ok else 'FAIL'}] {d}", flush=True)

    print("\n=== 4. Temporal-state checks on trained arm-C models ===", flush=True)
    for seed in SEEDS:
        model = TemporalGraphSAGEMeanMemory(raw["X"].shape[1]).to(dev)
        model.load_state_dict(torch.load(base / f"seed_{seed}" / "model_checkpoint.pt", map_location=dev))
        model.eval()
        print(f"\n  --- seed {seed} (trained checkpoint) ---", flush=True)
        checks = [
            ("future_invariance", lambda: TC.check_future_invariance(model, raw, dev, cfg, max_day, first_pred, CUTOFFS_C)),
            ("past_sensitivity_history_used", lambda: TC.check_past_sensitivity(model, raw, dev, cfg, max_day, first_pred)),
            ("train_val_boundary_state_continuity", lambda: TC.check_boundary_continuity(model, raw, dev, cfg, max_day, cfg.split.train_day_max)),
            ("per_step_access_only_own_day", lambda: TC.check_access_pattern(model, raw, dev, cfg, max_day, first_pred)),
            ("labels_never_reach_model_or_state", lambda: TC.check_labels_not_used(model, raw, dev, cfg, max_day, first_pred)),
        ]
        for name, fn in checks:
            ok, detail = fn()
            results.append((f"seed{seed}:{name}", ok, detail))
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ arm B checks
@torch.no_grad()
def rolling_logits(model, graph, days):
    """Line-for-line the same protocol as src.graph.rolling_eval.rolling_predict but returns raw logits."""
    model.eval()
    txn_day = graph["transaction"].day
    out = {}
    for t in days:
        g_t = apply_time_mask(graph, t)
        logits = model(g_t.x_dict, g_t.edge_index_dict)
        out[t] = logits[(txn_day == t).nonzero(as_tuple=True)[0]].detach().cpu().numpy()
    return out


def perturb_graph_future(graph, cutoff, seed=0):
    """Return a copy of `graph` whose FUTURE (day > cutoff) is corrupted: transaction features replaced by large noise,
    labels flipped, and the entity endpoint of every future edge (edge_time > cutoff) randomly reassigned."""
    g = graph.clone()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    day = g["transaction"].day.cpu()
    fut = (day > cutoff).to(g["transaction"].x.device)
    x = g["transaction"].x.clone()
    x[fut] = torch.randn(int(fut.sum()), x.size(1), generator=gen).to(x.device) * 5.0
    g["transaction"].x = x
    g["transaction"].y = 1 - g["transaction"].y
    for et in g.edge_types:
        ei = g[et].edge_index.clone()
        m = (g[et].edge_time > cutoff)
        ent_row = 1 if et[0] == "transaction" else 0
        n_ent = g[et[2] if et[0] == "transaction" else et[0]].num_nodes
        k = int(m.sum())
        if k:
            ei[ent_row, m] = torch.randint(0, n_ent, (k,), generator=gen).to(ei.device)
        g[et].edge_index = ei
    return g


def perturb_graph_past(graph, lo, hi, seed=1):
    g = graph.clone()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    day = g["transaction"].day.cpu()
    sel = ((day >= lo) & (day <= hi)).to(g["transaction"].x.device)
    x = g["transaction"].x.clone()
    x[sel] = torch.randn(int(sel.sum()), x.size(1), generator=gen).to(x.device) * 5.0
    g["transaction"].x = x
    return g


def _diff(a, b, days):
    return max(float(np.max(np.abs(a[d] - b[d]))) for d in days)


def run_arm_b(args, cfg, dev, out, results):
    from train_graphsage_robustness import build_scaled_graph
    base = cfg.experiments_dir / "temporal_ablation" / "static_schedule_matched"
    graph, scaler = build_scaled_graph(cfg)
    graph = graph.to(dev)
    val_days = list(range(cfg.split.val_day_min, cfg.split.val_day_max + 1))
    txn_day = graph["transaction"].day

    print("=== static-protocol checks (dataset level) ===", flush=True)
    # scaler fit on TRAIN only
    raw_g = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    x = raw_g["transaction"].x.numpy(); dd = raw_g["transaction"].day.numpy()
    tr = (dd >= cfg.split.train_day_min) & (dd <= cfg.split.train_day_max)
    ref_tr = StandardScaler().fit(x[tr]); ref_all = StandardScaler().fit(x)
    same_tr = np.allclose(scaler.mean_, ref_tr.mean_) and np.allclose(scaler.scale_, ref_tr.scale_)
    differs_all = not np.allclose(scaler.mean_, ref_all.mean_, rtol=1e-6, atol=1e-8)
    ok = bool(same_tr and differs_all)
    results.append(("scaler_fit_on_train_only", ok, f"scaler equals TRAIN-only refit: {same_tr}; differs from an all-data fit (detectable): {differs_all}"))
    print(f"  [{'PASS' if ok else 'FAIL'}] scaler_fit_on_train_only: {results[-1][2]}", flush=True)
    del raw_g, x

    # window coverage: training targets are exactly TRAIN transactions, chronological disjoint 18 windows
    from train_static_schedule_matched import make_windows
    wins = make_windows(txn_day, cfg.split.train_day_max)
    idx_all = torch.cat([w[2] for w in wins])
    n_train = int((txn_day <= cfg.split.train_day_max).sum())
    disjoint = idx_all.numel() == torch.unique(idx_all).numel()
    covers = idx_all.numel() == n_train and int(txn_day[idx_all].max()) <= cfg.split.train_day_max
    chrono = all(wins[i][1] < wins[i + 1][0] for i in range(len(wins) - 1))
    ok = bool(len(wins) == 18 and disjoint and covers and chrono)
    results.append(("training_windows_train_only_disjoint_chronological", ok,
                    f"{len(wins)} windows; {idx_all.numel():,} targets == {n_train:,} TRAIN txns; max target day {int(txn_day[idx_all].max())}; disjoint={disjoint}; chronological={chrono}"))
    print(f"  [{'PASS' if ok else 'FAIL'}] training_windows: {results[-1][2]}", flush=True)
    tg = apply_time_mask(graph, cfg.split.train_day_max)
    mx = max(int(tg[et].edge_time.max()) for et in tg.edge_types)
    ok = mx <= cfg.split.train_day_max
    results.append(("training_graph_has_no_edges_after_train_end", ok, f"max edge_time in TRAIN-masked graph = {mx} (train ends {cfg.split.train_day_max})"))
    print(f"  [{'PASS' if ok else 'FAIL'}] training graph: {results[-1][2]}", flush=True)

    # future nodes isolated in every scored day's masked graph
    bad = []
    for t in val_days:
        g_t = apply_time_mask(graph, t)
        fut = txn_day > t
        for et in g_t.edge_types:
            ei = g_t[et].edge_index
            tx_row = 0 if et[0] == "transaction" else 1
            if bool(fut[ei[tx_row]].any()):
                bad.append((t, et))
    ok = not bad
    results.append(("static_masked_graph_isolates_future_nodes", ok,
                    f"for all {len(val_days)} scored days t, no edge touches any transaction with day > t (incl. all TEST rows present in the file) - " + ("OK" if ok else f"violations {bad[:3]}")))
    print(f"  [{'PASS' if ok else 'FAIL'}] {results[-1][0]}: {results[-1][2]}", flush=True)
    scored_max = max(val_days)
    ok = scored_max == cfg.split.val_day_max
    results.append(("scored_days_are_validation_only", ok, f"scored days {min(val_days)}..{scored_max}; TEST starts {cfg.split.test_day_min}; TEST never scored"))
    print(f"  [{'PASS' if ok else 'FAIL'}] scored_days_are_validation_only: {results[-1][2]}", flush=True)

    for seed in SEEDS:
        print(f"\n  --- seed {seed} (trained checkpoint) ---", flush=True)
        fdims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
        model = HeteroSAGE(graph.metadata(), fdims).to(dev)
        model.load_state_dict(torch.load(base / f"seed_{seed}" / "model_checkpoint.pt", map_location=dev))
        model.eval()
        base_lg = rolling_logits(model, graph, val_days)
        base2 = rolling_logits(model, graph, val_days)
        noise = _diff(base_lg, base2, val_days)
        tol = 0.0 if dev.type == "cpu" else 4 * noise
        # protocol equivalence with the production rolling_predict
        ids, sc, _ = rolling_predict(model, graph, val_days)
        p_local = np.concatenate([1 / (1 + np.exp(-base_lg[t].astype(np.float64))) for t in val_days])
        eq = float(np.max(np.abs(p_local - sc)))
        ok = eq < 1e-6
        results.append((f"seed{seed}:rolling_logits_equals_rolling_predict", ok, f"max|p_local - rolling_predict| = {eq:.2e}"))
        print(f"  [{'PASS' if ok else 'FAIL'}] rolling_logits_equals_rolling_predict: {results[-1][2]}", flush=True)
        for c in CUTOFFS_B:
            gp = perturb_graph_future(graph, c)
            days_c = [t for t in val_days if t <= c]
            pert = rolling_logits(model, gp, days_c)
            dmax = _diff(base_lg, pert, days_c)
            ok = dmax <= tol
            results.append((f"seed{seed}:static_future_invariance_cutoff{c}", ok,
                            f"corrupting features+edges+labels of every day > {c} (TEST included): max|dlogit| on days {days_c[0]}..{c} = {dmax:.3e} "
                            f"(noise floor {noise:.3e}; {'bitwise identical' if dmax == 0 else 'within noise floor' if ok else 'EXCEEDS noise floor'})"))
            print(f"  [{'PASS' if ok else 'FAIL'}] future_invariance cutoff {c}: {results[-1][2]}", flush=True)
            del gp
        gp = perturb_graph_past(graph, cfg.split.train_day_max - 5, cfg.split.train_day_max)
        pert = rolling_logits(model, gp, val_days)
        dmax = _diff(base_lg, pert, val_days)
        ok = dmax > 1e-3
        results.append((f"seed{seed}:static_past_sensitivity_positive_control", ok, f"corrupting days {cfg.split.train_day_max-5}..{cfg.split.train_day_max} changes later predictions: max|dlogit| = {dmax:.3e} (test can detect dependence)"))
        print(f"  [{'PASS' if ok else 'FAIL'}] past_sensitivity: {results[-1][2]}", flush=True)
        del gp, model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B", "C"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()
    cfg = load_config()
    out = cfg.experiments_dir / "temporal_ablation" / "leakage_validation"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    print(f"arm {args.arm}, device {dev} ({'deterministic - exact equality' if dev.type == 'cpu' else 'GPU - noise-floor tolerance'})", flush=True)
    results = []
    (run_arm_c if args.arm == "C" else run_arm_b)(args, cfg, dev, out, results)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} checks passed.", flush=True)
    path = out / f"leakage_validation_arm{args.arm}_{args.device}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{"check": n, "passed": bool(ok), "detail": d} for n, ok, d in results], fh, indent=1)
    print(f"Wrote {path}", flush=True)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
