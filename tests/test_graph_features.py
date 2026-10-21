"""Comparison B: unit tests for src/features/graph_features.py on synthetic data (no real data, no labels, no TEST).
Covers BA2, BA3 and the wording part of BA9 of experiments/comparison_B_graph_features/PREREGISTRATION_B.md."""
from __future__ import annotations

import inspect
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files  # noqa: E402
from src.features import graph_features as G  # noqa: E402

DAY, WEEK = G.DAY, G.WEEK


def make_data(n=400, seed=0, span_days=12, n_ent=6, p_missing=0.25, with_ties=True):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, span_days * DAY, n)
    if with_ties:
        t[: n // 5] = (t[: n // 5] // 3600) * 3600          # many exact ties
        t[n // 5: n // 5 + 40] = DAY * 3                    # a large tie block
    ents = {}
    for i, r in enumerate(G.RELATIONS):
        e = rng.integers(0, n_ent + i, n).astype(np.int64)
        e[rng.random(n) < p_missing] = -1
        ents[r] = e
    amt = np.round(rng.lognormal(3.5, 1.0, n), 2)
    return t.astype(np.int64), amt, ents


def brute(t, amt, ents):
    """Independent O(n^2) definition of the 40 features (PREREGISTRATION_B.md section 4); window bounds are inclusive at t - width."""
    n = len(t)
    la = np.log1p(amt)
    rows = []
    for i in range(n):
        d = {}
        for r in G.RELATIONS:
            e = ents[r][i]
            d[f"g_{r}_has_edge"] = 1.0 if e >= 0 else 0.0
            prior = [j for j in range(n) if e >= 0 and ents[r][j] == e and t[j] < t[i]]
            nan = float("nan")
            d[f"g_{r}_prior_n"] = float(len(prior)) if e >= 0 else nan
            d[f"g_{r}_prior_n_1d"] = float(sum(t[j] >= t[i] - DAY for j in prior)) if e >= 0 else nan
            d[f"g_{r}_prior_n_7d"] = float(sum(t[j] >= t[i] - WEEK for j in prior)) if e >= 0 else nan
            d[f"g_{r}_secs_since_last"] = float(t[i] - max(t[j] for j in prior)) if prior else nan
            m = float(np.mean([la[j] for j in prior])) if prior else nan
            d[f"g_{r}_prior_amt_logmean"] = m
            d[f"g_{r}_amt_dev"] = float(la[i] - m) if prior else nan
            for s in G.RELATIONS:
                if s != r:
                    d[f"g_{r}_prior_distinct_{s}"] = float(len({ents[s][j] for j in prior if ents[s][j] >= 0})) if e >= 0 else nan
        rows.append(d)
    return rows


AMOUNT_KEYS = ("prior_amt_logmean", "amt_dev")


def same(a, b, approx=False):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) if approx else a == b


def assert_frames_equal(df, rows):
    for i, d in enumerate(rows):
        for k, v in d.items():
            assert same(float(df[k].iat[i]), v, approx=any(a in k for a in AMOUNT_KEYS)), (i, k, df[k].iat[i], v)


# ------------------------------------------------------------------------------------------------ spec
def test_feature_inventory_matches_the_preregistration():
    names = G.feature_names()
    assert len(names) == 40 and len(set(names)) == 40
    groups = G.ablation_groups()
    assert set(groups) == {"drop_pa", "drop_card", "drop_net", "drop_dev", "drop_time_windows"}
    for r in G.RELATIONS:
        g = groups[f"drop_{r}"]
        assert len(g) == 13 and len(set(g)) == 13 and set(g) <= set(names), (r, g)
        assert f"g_{r}_has_edge" in g
        assert all(f"prior_distinct_{r}" in n or n.startswith(f"g_{r}_") for n in g)
    assert len(groups["drop_time_windows"]) == 12


# ------------------------------------------------------------------------------------------------ BA3: brute-force equivalence
def test_equals_brute_force_with_ties_missing_edges_and_windows():
    for seed in range(4):
        t, amt, ents = make_data(seed=seed)
        assert_frames_equal(G.compute_graph_features(t, amt, ents), brute(t, amt, ents))


def test_equals_brute_force_with_dense_entities_and_no_missing():
    t, amt, ents = make_data(n=250, seed=9, n_ent=3, p_missing=0.0, span_days=40)
    assert_frames_equal(G.compute_graph_features(t, amt, ents), brute(t, amt, ents))


def test_window_edges_are_inclusive_at_exactly_one_day_and_one_week():
    t = np.array([0, 100, DAY + 100, DAY + 101, WEEK + 100, WEEK + 101])
    ents = {r: np.zeros(len(t), dtype=np.int64) for r in G.RELATIONS}
    f = G.compute_graph_features(t, np.ones(len(t)), ents)
    # row 2 (t = DAY+100): the prior row at t = 100 is exactly DAY earlier -> inside the 1-day window; the row at t = 0 is not
    assert f["g_pa_prior_n"].iat[2] == 2 and f["g_pa_prior_n_1d"].iat[2] == 1
    # row 3 (t = DAY+101): the prior row at 100 is DAY+1 earlier -> outside; row at DAY+100 inside
    assert f["g_pa_prior_n_1d"].iat[3] == 1
    # row 4 (t = WEEK+100): the row at t = 100 is exactly one week earlier -> inside the 7-day window
    assert f["g_pa_prior_n_7d"].iat[4] == 4 - 1 + 0 or f["g_pa_prior_n_7d"].iat[4] >= 1
    assert f["g_pa_prior_n_7d"].iat[4] == 3 and f["g_pa_prior_n_7d"].iat[5] == 3


def test_ties_in_time_are_never_prior_to_each_other():
    t = np.array([50, 50, 50, 60])
    ents = {r: np.zeros(4, dtype=np.int64) for r in G.RELATIONS}
    f = G.compute_graph_features(t, np.array([1.0, 2.0, 3.0, 4.0]), ents)
    assert f["g_card_prior_n"].tolist() == [0, 0, 0, 3]
    assert f["g_card_secs_since_last"].iloc[:3].isna().all() and f["g_card_secs_since_last"].iat[3] == 10


def test_no_edge_rows_have_nan_and_zero_flag_and_do_not_count_as_prior():
    t = np.array([10, 20, 30])
    ents = {"pa": np.array([0, -1, 0]), "card": np.array([0, 0, 0]), "net": np.array([-1, -1, -1]), "dev": np.array([-1, 0, -1])}
    f = G.compute_graph_features(t, np.ones(3), ents)
    assert f["g_pa_has_edge"].tolist() == [1, 0, 1] and math.isnan(f["g_pa_prior_n"].iat[1])
    assert f["g_pa_prior_n"].iat[2] == 1                   # the row without an edge is not a prior neighbour
    assert f["g_net_has_edge"].sum() == 0 and f["g_net_prior_n"].isna().all()
    assert f["g_card_prior_distinct_dev"].iat[2] == 1      # only the row with a dev edge contributes
    assert f["g_card_prior_distinct_net"].iat[2] == 0


# ------------------------------------------------------------------------------------------------ BA2: causality
def test_future_rows_never_change_earlier_features():
    t, amt, ents = make_data(n=500, seed=3, span_days=20)
    full = G.compute_graph_features(t, amt, ents)
    cut = 8 * DAY
    keep = t < cut
    part = G.compute_graph_features(t[keep], amt[keep], {k: v[keep] for k, v in ents.items()})
    a, b = full[keep].reset_index(drop=True), part.reset_index(drop=True)
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_changing_every_later_row_leaves_earlier_features_bit_identical():
    t, amt, ents = make_data(n=400, seed=5, span_days=16)
    base = G.compute_graph_features(t, amt, ents)
    rng = np.random.default_rng(1)
    cut = 6 * DAY
    later = t >= cut
    t2, amt2 = t.copy(), amt.copy()
    t2[later] = rng.integers(cut, 16 * DAY, later.sum())
    amt2[later] = rng.lognormal(4, 2, later.sum())
    ents2 = {k: np.where(later, rng.integers(-1, 12, len(t)), v) for k, v in ents.items()}
    changed = G.compute_graph_features(t2, amt2, ents2)
    e, c = base[~later].to_numpy(), changed[~later].to_numpy()
    assert np.array_equal(e, c, equal_nan=True)


def test_row_order_of_the_input_does_not_matter():
    t, amt, ents = make_data(n=300, seed=7)
    f = G.compute_graph_features(t, amt, ents)
    perm = np.random.default_rng(2).permutation(len(t))
    g = G.compute_graph_features(t[perm], amt[perm], {k: v[perm] for k, v in ents.items()})
    assert np.array_equal(f.to_numpy()[perm], g.to_numpy(), equal_nan=True)


def test_function_has_no_label_argument_and_the_module_never_names_a_label():
    params = list(inspect.signature(G.compute_graph_features).parameters)
    assert params == ["time", "amount", "entities"], params
    src = pathlib.Path(G.__file__).read_text(encoding="utf-8")
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    assert not re.search(r"isFraud|label|y_true|y_score|fraud", code, re.I), "label-like token in code"


# ------------------------------------------------------------------------------------------------ errors and wording
def test_invalid_inputs_raise_clear_errors():
    t, amt, ents = make_data(n=20)
    for bad, msg in ((lambda: G.compute_graph_features(t, amt[:-1], ents), "amount"),
                     (lambda: G.compute_graph_features(t, amt, {k: v for k, v in ents.items() if k != "dev"}), "keys"),
                     (lambda: G.compute_graph_features(t, amt, {**ents, "dev": ents["dev"][:-1]}), "entity array"),
                     (lambda: G.compute_graph_features(t + G.TIME_LIMIT, amt, ents), "times must lie")):
        try:
            bad()
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(f"expected ValueError mentioning {msg}")


def test_empty_input_returns_empty_frame_with_all_columns():
    f = G.compute_graph_features(np.array([], dtype=np.int64), np.array([]), {r: np.array([], dtype=np.int64) for r in G.RELATIONS})
    assert len(f) == 0 and list(f.columns) == G.feature_names()


def test_wording_lint_on_graph_features_sources():
    files = [ROOT / "src" / "features" / "graph_features.py", ROOT / "tests" / "test_graph_features.py",
             ROOT / "experiments" / "comparison_B_graph_features" / "PREREGISTRATION_B.md"]
    bad = {str(k): v for k, v in lint_files(files).items() if v}
    assert bad == {}, bad


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
