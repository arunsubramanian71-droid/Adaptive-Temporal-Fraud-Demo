"""Temporal-state leakage checks for the Phase 7 temporal GNN.

Each check returns (ok: bool, detail: str). The strongest evidence here is
interventional, not inspectional: rather than only arguing that no future
information is read, we CHANGE the future (features, edges, labels) and
verify that predictions for earlier days are unchanged, and we CHANGE the
past and verify later predictions DO change (a positive control proving both
that the tests can detect dependence and that the recurrent memory is
genuinely used).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.graph.temporal import RELATIONS, TemporalDataset, build_temporal_dataset
from src.models.temporal_graphsage import TemporalGraphSAGE


@torch.no_grad()
def stream_logits(model: TemporalGraphSAGE, ds: TemporalDataset, first_pred_day: int, last_day: int) -> dict[int, np.ndarray]:
    """Process days strictly in order; return raw logits per predicted day."""
    model.eval()
    state = model.init_state(ds)
    out = {}
    for d in ds.days:
        if d > last_day:
            break
        logits = model.step(state, ds, d, predict=d >= first_pred_day)
        if logits is not None:
            out[d] = logits.detach().cpu().numpy()
    return out


def _max_abs_diff(a: dict[int, np.ndarray], b: dict[int, np.ndarray], days: list[int]) -> float:
    if not days:
        return 0.0
    return float(max(np.max(np.abs(a[d] - b[d])) for d in days))


def _assert_same_layout(a: np.ndarray, b: np.ndarray) -> None:
    """A perturbed feature array must have the SAME memory layout as the
    baseline. sklearn's transform() yields a Fortran-ordered array here, and
    a plain ndarray.copy() silently converts to C order; the GPU then picks a
    different GEMM path and returns last-bit-different logits (3 ulps at the
    largest outlier logit). That produced spurious 'leak' failures in the
    first validation run, which were harness artifacts, not information flow."""
    if (a.flags["C_CONTIGUOUS"], a.flags["F_CONTIGUOUS"]) != (b.flags["C_CONTIGUOUS"], b.flags["F_CONTIGUOUS"]):
        raise AssertionError("perturbed feature array has a different memory layout than the baseline")


def noise_floor(model, ds, first_pred_day: int, last_day: int, n_repeats: int = 3) -> float:
    """Empirical run-to-run difference on IDENTICAL input (GPU scatter
    atomics). Exactly 0.0 on deterministic (CPU) execution."""
    runs = [stream_logits(model, ds, first_pred_day, last_day) for _ in range(n_repeats + 1)]
    days = sorted(runs[0])
    return max(_max_abs_diff(runs[0], r, days) for r in runs[1:])


def perturb_future(raw: dict, cutoff: int, rng: np.random.Generator) -> dict:
    """Corrupts EVERYTHING dated after `cutoff`: transaction features -> large
    noise, labels flipped, and every later edge re-pointed to a random
    entity. If any of that could reach a prediction at day <= cutoff, the
    prediction changes."""
    r = copy.copy(raw)
    late = raw["day"] > cutoff
    X = raw["X"].copy(order="K")  # order="K" preserves the memory layout (see _assert_same_layout)
    X[late] = rng.normal(0.0, 5.0, size=(int(late.sum()), X.shape[1])).astype(np.float32)
    _assert_same_layout(X, raw["X"])
    r["X"] = X
    y = raw["y"].copy()
    y[late] = 1.0 - y[late]
    r["y"] = y
    rels = {}
    for rel, (src, dst, et) in raw["relations"].items():
        n_ent = raw["n_entities"][RELATIONS[rel]]
        dst2 = dst.copy()
        m = et > cutoff
        dst2[m] = rng.integers(0, n_ent, size=int(m.sum()))
        rels[rel] = (src, dst2, et)
    r["relations"] = rels
    return r


def perturb_past(raw: dict, lo_day: int, hi_day: int, rng: np.random.Generator) -> dict:
    """Corrupts transaction features and entity assignments within
    [lo_day, hi_day] (an EARLIER period) — positive control."""
    r = copy.copy(raw)
    m_t = (raw["day"] >= lo_day) & (raw["day"] <= hi_day)
    X = raw["X"].copy(order="K")
    X[m_t] = rng.normal(0.0, 5.0, size=(int(m_t.sum()), X.shape[1])).astype(np.float32)
    _assert_same_layout(X, raw["X"])
    r["X"] = X
    rels = {}
    for rel, (src, dst, et) in raw["relations"].items():
        n_ent = raw["n_entities"][RELATIONS[rel]]
        dst2 = dst.copy()
        m = (et >= lo_day) & (et <= hi_day)
        dst2[m] = rng.integers(0, n_ent, size=int(m.sum()))
        rels[rel] = (src, dst2, et)
    r["relations"] = rels
    return r


def check_future_invariance(model, raw, device, cfg, max_day: int, first_pred_day: int, cutoffs: list[int], seed: int = 0):
    """For each cutoff c: predictions on days first_pred_day..c must be
    unchanged when everything after c is corrupted. Also reports the
    run-to-run noise floor (GPU scatter atomics) and confirms the corruption
    really does change predictions AFTER c (i.e. the perturbation is not a
    no-op)."""
    rng = np.random.default_rng(seed)
    ds0 = build_temporal_dataset(raw, max_day, device, cfg)
    base = stream_logits(model, ds0, first_pred_day, max_day)
    all_days = sorted(base)
    noise = noise_floor(model, ds0, first_pred_day, max_day, n_repeats=3)
    del ds0
    tol = 4.0 * noise  # noise == 0.0 (deterministic device) => EXACT bitwise equality is required
    lines, ok_all = [f"run-to-run noise floor (identical inputs, 4 runs) = {noise:.3e}; pass tolerance = {tol:.3e}"
                     + (" (deterministic: exact equality required)" if noise == 0.0 else "")], True
    for c in cutoffs:
        pert = perturb_future(raw, c, rng)
        dsp = build_temporal_dataset(pert, max_day, device, cfg)
        out = stream_logits(model, dsp, first_pred_day, max_day)
        del dsp
        past_days = [d for d in all_days if d <= c]
        fut_days = [d for d in all_days if d > c]
        d_past = _max_abs_diff(base, out, past_days)
        d_fut = _max_abs_diff(base, out, fut_days) if fut_days else float("nan")
        effective = (not fut_days) or d_fut > max(100 * tol, 1e-3)
        ok = d_past <= tol and effective
        ok_all &= ok
        lines.append(f"cutoff={c}: max|dlogit| on days <= {c} = {d_past:.3e} ({'bitwise identical' if d_past == 0.0 else ('unchanged within noise floor' if d_past <= tol else 'CHANGED - LEAK')}); "
                     f"on days > {c} = {d_fut:.3e} ({'perturbation is effective' if effective else 'perturbation had no effect - test uninformative'})")
    return ok_all, "; ".join(lines)


def check_past_sensitivity(model, raw, device, cfg, max_day: int, first_pred_day: int, seed: int = 1):
    """Positive control / temporal-mechanism check: corrupt the ~5 days
    immediately BEFORE the prediction window; predictions inside the window
    must change (the recurrent memory really carries earlier information
    forward)."""
    rng = np.random.default_rng(seed)
    ds0 = build_temporal_dataset(raw, max_day, device, cfg)
    base = stream_logits(model, ds0, first_pred_day, max_day)
    del ds0
    pert = perturb_past(raw, first_pred_day - 6, first_pred_day - 1, rng)
    dsp = build_temporal_dataset(pert, max_day, device, cfg)
    out = stream_logits(model, dsp, first_pred_day, max_day)
    del dsp
    days = sorted(base)
    early = _max_abs_diff(base, out, days[:3])
    overall = _max_abs_diff(base, out, days)
    ok = overall > 1e-3
    return ok, (f"corrupting days {first_pred_day-6}..{first_pred_day-1} changes later predictions: max|dlogit| = {overall:.3e} "
                f"(first 3 predicted days: {early:.3e}) — {'earlier state DOES influence later predictions (recurrent memory is used)' if ok else 'NO influence — memory unused'}")


def check_boundary_continuity(model, raw, device, cfg, max_day: int, boundary_day: int):
    """State at the boundary: continuing a stream vs. pausing at
    `boundary_day` (state deep-copied) and resuming must give identical
    later predictions — the boundary neither resets nor contaminates state."""
    ds = build_temporal_dataset(raw, max_day, device, cfg)
    model.eval()
    with torch.no_grad():
        cont = model.init_state(ds)
        out_c = {}
        for d in ds.days:
            lg = model.step(cont, ds, d, predict=d > boundary_day)
            if lg is not None:
                out_c[d] = lg.cpu().numpy()
        st = model.init_state(ds)
        out_s = {}
        for d in ds.days:
            if d == boundary_day + 1:
                st = st.clone()  # pause/resume across the boundary
            lg = model.step(st, ds, d, predict=d > boundary_day)
            if lg is not None:
                out_s[d] = lg.cpu().numpy()
    diff = _max_abs_diff(out_c, out_s, sorted(out_c))
    noise = noise_floor(model, ds, boundary_day + 1, max_day, n_repeats=3)
    tol = 4.0 * noise
    ok = diff <= tol
    del ds
    return ok, (f"continuous stream vs pause/clone/resume at day {boundary_day}: max|dlogit| = {diff:.3e} "
                f"(noise floor {noise:.3e}; {'bitwise identical' if diff == 0.0 else 'within noise floor' if ok else 'EXCEEDS noise floor'}) "
                f"- state carried unchanged across the boundary")


def check_access_pattern(model, raw, device, cfg, max_day: int, first_pred_day: int):
    """Instrumented run: every step d may read ONLY day d's slice."""
    ds = build_temporal_dataset(raw, max_day, device, cfg)
    ds.access_log = []
    model.eval()
    with torch.no_grad():
        state = model.init_state(ds)
        bad = []
        for d in ds.days:
            n0 = len(ds.access_log)
            model.step(state, ds, d, predict=d >= first_pred_day)
            touched = set(ds.access_log[n0:])
            if touched != {d}:
                bad.append((d, sorted(touched)))
    del ds
    return (not bad), ("each step d read exactly one data slice: day d (no other day's features/edges accessed)"
                       if not bad else f"steps that read other days: {bad[:3]}")


def check_labels_not_used(model, raw, device, cfg, max_day: int, first_pred_day: int):
    """Model + state never touch labels: a dataset built WITHOUT labels
    (y=None) yields identical predictions."""
    ds_a = build_temporal_dataset(raw, max_day, device, cfg, with_labels=True)
    a = stream_logits(model, ds_a, first_pred_day, max_day)
    a2 = stream_logits(model, ds_a, first_pred_day, max_day)
    del ds_a
    ds_b = build_temporal_dataset(raw, max_day, device, cfg, with_labels=False)
    assert ds_b.y is None
    b = stream_logits(model, ds_b, first_pred_day, max_day)
    del ds_b
    days = sorted(a)
    noise = _max_abs_diff(a, a2, days)
    diff = _max_abs_diff(a, b, days)
    ok = diff <= 4 * noise
    return ok, f"labels removed entirely from the dataset: max|dlogit| = {diff:.3e} (noise floor {noise:.3e}) — no label reaches model or state"


def check_edge_consumption(raw, device, cfg, max_day: int):
    """Edges consumed by steps 1..c are exactly the edges with edge_time <= c
    — none later, none missing — for every relation."""
    ds = build_temporal_dataset(raw, max_day, device, cfg)
    problems = []
    for rel, (src, dst, et) in raw["relations"].items():
        cum = 0
        for d in ds.days:
            n = int(ds.rel[d][rel].src_local.numel())
            cum += n
            expect = int(np.searchsorted(et, d, side="right"))
            if cum != expect:
                problems.append(f"{rel} day {d}: consumed {cum} vs edge_time<=d has {expect}")
                break
            lo, hi = ds.day_ptr[d]
            sl = ds.rel[d][rel].src_local
            if n and (int(sl.min()) < 0 or int(sl.max()) >= hi - lo):
                problems.append(f"{rel} day {d}: source index outside that day's transactions")
                break
    del ds
    return (not problems), ("; ".join(problems) if problems else
                            f"for every day c<= {max_day} and every relation, edges consumed by steps 1..c == edges with edge_time<=c (0 future edges, 0 missing)")


def check_no_test_data(ds: TemporalDataset, raw, cfg: Config):
    n_allowed = int(np.searchsorted(raw["day"], cfg.split.val_day_max, side="right"))
    refused = False
    try:
        build_temporal_dataset(raw, cfg.split.val_day_max + 1, ds.X.device, cfg)
    except ValueError:
        refused = True
    ok = ds.max_day <= cfg.split.val_day_max and ds.X.shape[0] <= n_allowed and refused
    return ok, (f"dataset max_day={ds.max_day} (<= last VAL day {cfg.split.val_day_max}); {ds.X.shape[0]:,} transaction rows on device "
                f"(<= {n_allowed:,} rows with day<=VAL end); builder refuses TEST days: {refused}")


def check_scaler_train_only(raw, cfg: Config):
    day, X = raw["day"], raw["X"]
    tr = day <= cfg.split.train_day_max
    va = (day >= cfg.split.val_day_min) & (day <= cfg.split.val_day_max)
    m_tr = np.abs(X[tr].mean(0)).max()
    s_tr = np.abs(X[tr].std(0)[X[tr].std(0) > 0] - 1).max()
    m_va = np.abs(X[va].mean(0)).max()
    ok = m_tr < 1e-3 and s_tr < 1e-2 and m_va > 1e-2
    return ok, (f"TRAIN rows: max|mean|={m_tr:.2e}, max|std-1|={s_tr:.2e} (standardized); VAL rows: max|mean|={m_va:.2e} (NOT centred — "
                f"scaler was not fit on VAL)")
