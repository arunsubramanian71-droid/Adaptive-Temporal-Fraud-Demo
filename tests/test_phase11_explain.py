"""Phase 11 first pass: toy/synthetic tests for src/explain (no real data, no TEST, no labels used for anything but fitting the toy model).

Covers acceptance tests A1-A6, A8 and A9 of experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md on synthetic inputs."""
from __future__ import annotations

import ast
import inspect
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files, lint_text  # noqa: E402  (read-only use of the Phase 9 wording lint)
from src.explain import cards as C  # noqa: E402
from src.explain import xgb_contrib as X  # noqa: E402


# ------------------------------------------------------------------------------------------------ toy XGBoost model
def _toy_model(seed=0, n=700, f=6, early=True):
    rng = np.random.default_rng(seed)
    Xm = rng.normal(size=(n, f)).astype(np.float32)
    Xm[rng.random(Xm.shape) < 0.15] = np.nan                       # NaN stays native
    z = np.nan_to_num(Xm[:, 0]) - 0.7 * np.nan_to_num(Xm[:, 1]) + rng.normal(scale=2.0, size=n)
    y = (z > 0).astype(int)
    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.3, tree_method="hist", n_jobs=1, random_state=seed)
    if early:
        m = xgb.XGBClassifier(early_stopping_rounds=8, eval_metric="logloss", **kw)
        m.fit(Xm[:500], y[:500], eval_set=[(Xm[500:], y[500:])], verbose=False)
    else:
        m = xgb.XGBClassifier(**kw).fit(Xm, y)
    return m, Xm


def test_A1_contributions_sum_to_the_margin_on_toy_models():
    for seed in range(4):
        for early in (True, False):
            m, Xm = _toy_model(seed, early=early)
            contrib, bias, margin = X.contributions(m, Xm)
            assert contrib.shape == (len(Xm), Xm.shape[1]) and bias.shape == margin.shape == (len(Xm),)
            err = np.abs(contrib.sum(axis=1) + bias - margin)
            assert err.max() <= 1e-4, (seed, early, err.max())


def test_A2_margin_matches_the_models_own_probability_and_uses_the_early_stopping_range():
    m, Xm = _toy_model(1, early=True)
    contrib, bias, margin = X.contributions(m, Xm)
    p = X.sigmoid(margin)
    assert np.abs(p - m.predict_proba(Xm)[:, 1]).max() <= 1e-6
    # the early-stopping range matters: using all trees would give different margins (the test is meaningful only then)
    n_trees = len(m.get_booster().get_dump())
    assert n_trees > m.best_iteration + 1
    all_trees = m.get_booster().predict(xgb.DMatrix(Xm), output_margin=True)
    assert np.abs(all_trees - margin).max() > 1e-3


def test_A3_batch_invariance():
    m, Xm = _toy_model(2)
    contrib, bias, margin = X.contributions(m, Xm)
    for i in (0, 7, 123, 499):
        c1, b1, m1 = X.contributions(m, Xm[i:i + 1])
        assert np.abs(c1[0] - contrib[i]).max() <= 1e-5 and abs(b1[0] - bias[i]) <= 1e-5 and abs(m1[0] - margin[i]) <= 1e-5


def test_top_features_orders_by_absolute_contribution_and_keeps_nan_as_null():
    names = [f"f{i}" for i in range(5)]
    c = np.array([0.1, -0.9, 0.9, 0.0, -0.1])
    v = np.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float32)
    out = X.top_features(c, names, v, k=3)
    assert [o["feature"] for o in out] == ["f1", "f2", "f0"]           # tie |0.9| broken by lower column index
    assert out[0]["value"] is None and out[1]["value"] == 3.0


# ------------------------------------------------------------------------------------------------ selection
def _synthetic_day_table(n_per_day=40, seed=5, ties=True):
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(122, 152), n_per_day)
    ids = rng.permutation(np.arange(3_000_000, 3_000_000 + len(days)))
    sc = rng.random(len(days))
    if ties:
        sc[np.arange(len(sc)) % 7 == 0] = 0.5                           # many exact ties
    return ids, days, sc


def test_A4_selection_has_exact_count_groups_and_is_reproducible():
    ids, days, sc = _synthetic_day_table()
    a = X.select_transactions(ids, days, sc)
    b = X.select_transactions(ids, days, sc)
    assert a.equals(b)
    assert len(a) == 300 and a["TransactionID"].is_unique
    assert (a.groupby("day").size() == 10).all() and a["day"].between(122, 151).all()
    assert (a["group"] == "top_score").sum() == 150 and (a["group"] == "uniform_draw").sum() == 150
    for d, g in a.groupby("day"):
        top = set(g.loc[g["group"] == "top_score", "TransactionID"])
        uni = set(g.loc[g["group"] == "uniform_draw", "TransactionID"])
        assert top.isdisjoint(uni) and len(top) == 5 and len(uni) == 5


def test_selection_top_group_is_the_highest_scores_with_ties_by_lower_id():
    ids, days, sc = _synthetic_day_table()
    a = X.select_transactions(ids, days, sc)
    for d in (122, 130, 151):
        m = days == d
        order = np.lexsort((ids[m], -sc[m]))
        want = ids[m][order[:5]].tolist()
        got = a[(a["day"] == d) & (a["group"] == "top_score")].sort_values("rank_in_group")["TransactionID"].tolist()
        assert got == want


def test_selection_ties_at_the_top_boundary_are_broken_by_lower_transaction_id():
    ids, days, sc = _synthetic_day_table(ties=False)
    m = days == 140
    tied = np.flatnonzero(m)[:8]                                         # 8 transactions share the day's maximum score
    sc = sc.copy()
    sc[tied] = 2.0
    a = X.select_transactions(ids, days, sc)
    got = a[(a["day"] == 140) & (a["group"] == "top_score")].sort_values("rank_in_group")["TransactionID"].tolist()
    assert got == sorted(ids[tied].tolist())[:5]                         # the 5 lowest ids among the tied maxima, in id order
    uni = set(a[(a["day"] == 140) & (a["group"] == "uniform_draw")]["TransactionID"])
    assert uni.isdisjoint(got) and len(uni) == 5


def test_selection_day_guard_rejects_test_period_rows_even_when_every_val_day_is_populated():
    ids, days, sc = _synthetic_day_table()
    extra_ids = np.arange(9_000_000, 9_000_020)
    for bad_day in (152, 121, 0):
        args = (np.concatenate([ids, extra_ids]), np.concatenate([days, np.full(20, bad_day)]), np.concatenate([sc, np.random.default_rng(0).random(20)]))
        try:
            X.select_transactions(*args)
        except ValueError as e:
            assert "outside" in str(e) and "sealed" in str(e)
        else:
            raise AssertionError(f"day {bad_day} accepted")


def test_selection_uniform_group_follows_the_documented_generator_and_seed_matters():
    ids, days, sc = _synthetic_day_table()
    a = X.select_transactions(ids, days, sc, seed=40042)
    d = 137
    m = days == d
    top = a[(a["day"] == d) & (a["group"] == "top_score")]["TransactionID"].to_numpy()
    rest = np.sort(ids[m][~np.isin(ids[m], top)])
    rng = np.random.default_rng(np.random.SeedSequence([40042, d]))
    want = np.sort(rest[rng.choice(len(rest), size=5, replace=False)]).tolist()
    got = a[(a["day"] == d) & (a["group"] == "uniform_draw")].sort_values("rank_in_group")["TransactionID"].tolist()
    assert got == want
    b = X.select_transactions(ids, days, sc, seed=40043)
    assert not a[a["group"] == "uniform_draw"]["TransactionID"].equals(b[b["group"] == "uniform_draw"]["TransactionID"])
    assert a[a["group"] == "top_score"].equals(b[b["group"] == "top_score"])      # the top group does not depend on the seed


def test_selection_is_independent_of_row_order():
    ids, days, sc = _synthetic_day_table()
    a = X.select_transactions(ids, days, sc)
    perm = np.random.default_rng(1).permutation(len(ids))
    assert a.equals(X.select_transactions(ids[perm], days[perm], sc[perm]))


def test_selection_rejects_days_outside_val_and_too_small_days_and_bad_input():
    ids, days, sc = _synthetic_day_table()
    for bad_days in (days + 1, days - 1):
        try:
            X.select_transactions(ids, bad_days, sc)
        except ValueError:
            pass
        else:
            raise AssertionError("days outside 122-151 accepted")
    small = days != 130
    for args in ((ids[small], days[small], sc[small]), (ids, days, np.where(np.arange(len(sc)) == 3, np.nan, sc)),
                 (np.concatenate([ids, ids[:1]]), np.concatenate([days, days[:1]]), np.concatenate([sc, sc[:1]]))):
        try:
            X.select_transactions(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid selection input accepted")


def test_A5_selection_and_explanation_functions_take_no_label_argument():
    allowed = {"transaction_id", "day", "score", "seed", "n_top", "n_draw", "day_range"}
    assert set(inspect.signature(X.select_transactions).parameters) == allowed
    assert set(inspect.signature(X.contributions).parameters) == {"model", "X"}
    forbidden = ("label", "isfraud", "target", "y_true", "fraud")
    for fn in (X.select_transactions, X.contributions, X.top_features):
        assert not any(any(f in p.lower() for f in forbidden) for p in inspect.signature(fn).parameters), fn


def test_A5_static_scan_label_column_name_appears_only_as_the_exclusion_constant():
    tree = ast.parse((ROOT / "src" / "explain" / "xgb_contrib.py").read_text(encoding="utf-8"))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and "isfraud" in n.value.lower()]
    assert len(hits) == 1                                              # the _NEVER_LOADED tuple
    assigned = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_NEVER_LOADED" for t in n.targets)]
    assert len(assigned) == 1


# ------------------------------------------------------------------------------------------------ loaders
def _write_val_like(dirpath, days=range(122, 152), per_day=12, extra_day=None):
    rng = np.random.default_rng(3)
    rows = []
    for d in list(days) + ([extra_day] if extra_day else []):
        for k in range(per_day):
            rows.append({"TransactionID": d * 1000 + k, "isFraud": int(rng.random() < 0.5), "TransactionDT": d * 86400 + k * 100,
                         "f1": float(rng.normal()), "f2": float(rng.normal())})
    df = pd.DataFrame(rows)
    p = pathlib.Path(dirpath) / "val_like.parquet"
    df.to_parquet(p)
    return p, df


def test_A5_loaders_never_return_the_label_column():
    with tempfile.TemporaryDirectory() as td:
        p, df = _write_val_like(td)
        ids = X.load_ids_days(p)
        assert list(ids.columns) == ["TransactionID", "day"] and ids["day"].between(122, 151).all()
        rows = X.load_feature_rows(p, [124001, 122003, 151011])
        assert "isFraud" not in rows.columns and rows["TransactionID"].tolist() == [124001, 122003, 151011]
        preds = pd.DataFrame({"TransactionID": df["TransactionID"], "isFraud": df["isFraud"], "y_score": 0.3, "y_pred": 0})
        pp = pathlib.Path(td) / "predictions_like.parquet"
        preds.to_parquet(pp)
        s = X.load_scores(pp)
        assert list(s.columns) == ["TransactionID", "y_score"]


def test_A6_loaders_reject_test_files_and_days_outside_the_val_window():
    with tempfile.TemporaryDirectory() as td:
        p, _ = _write_val_like(td, extra_day=152)
        try:
            X.load_ids_days(p)
        except ValueError:
            pass
        else:
            raise AssertionError("day 152 (TEST) accepted")
        q = pathlib.Path(td) / "test_features.parquet"
        q.write_bytes(p.read_bytes())
        for fn in (X.load_ids_days, X.load_scores):
            try:
                fn(q)
            except ValueError:
                pass
            else:
                raise AssertionError("TEST-named file accepted")
        try:
            X.load_feature_rows(q, [1])
        except ValueError:
            pass
        else:
            raise AssertionError("TEST-named file accepted")


def test_A6_val_window_constant_matches_the_frozen_project_split():
    from src.config import load_config
    s = load_config().split
    assert (s.val_day_min, s.val_day_max) == X.VAL_DAYS and s.train_day_max == X.VAL_DAYS[0] - 1


# ------------------------------------------------------------------------------------------------ community cards (synthetic CSV frames)
def _tier0_frame(seed=0, n_tracks=40, n_days=30):
    rng = np.random.default_rng(seed)
    rows = []
    for tr in range(n_tracks):
        start = int(rng.integers(122, 152))
        length = int(rng.integers(1, 12))
        for d in range(start, min(152, start + length)):
            us, ut = float(rng.random()), float(rng.random())
            r = {"day": d, "track": tr, "n_cards": int(rng.integers(3, 80)), "n_infra": int(rng.integers(1, 30)), "q": 1.0, "q_recomputed": 1.0,
                 "w_in": 2.0, "edges_in": 4, "density": float(rng.random() / 10), "hub_share": float(rng.random()), "conductance": float(rng.random()),
                 "u_s": us, "t_raw": 1.0, "u_t": ut, "score_st": (us + ut) / 2.0, "n_member_txn": int(rng.integers(3, 200)), "p90_window_degree": 5.0}
            for c in C.SEGMENT_COLUMNS:
                r[c] = int(rng.integers(0, 50))
            rows.append(r)
    return pd.DataFrame(rows)[C.EXPECTED_COLUMNS]


def test_cards_group_selection_follows_section_16_rules_and_is_deterministic():
    df = _tier0_frame()
    tau = 0.7
    g = C.select_groups(df, tau, seed=42)
    assert g == C.select_groups(df, tau, seed=42)
    t = C.track_table(df).set_index("track")
    assert len(g["a_top_peak_score"]) == 10 and len(set(g["a_top_peak_score"])) == 10
    top10 = sorted(t.index, key=lambda k: (-t.loc[k, "peak_score_st"], k))[:10]
    assert g["a_top_peak_score"] == top10
    assert g["b_largest"] == sorted(t.index, key=lambda k: (-t.loc[k, "peak_n_cards"], k))[:5]
    alerted = C.alerted_tracks(df, tau)
    assert set(g["c_alerted_sample"]) <= set(alerted) and len(g["c_alerted_sample"]) == min(5, len(alerted))
    rng = np.random.default_rng(40000 + 42)
    assert g["c_alerted_sample"] == sorted(alerted[i] for i in rng.choice(len(alerted), size=5, replace=False))
    assert g["c_alerted_sample"] != C.select_groups(df, tau, seed=123)["c_alerted_sample"]
    # group (d): parameter-free rank-sum of peak u_s and peak hub share
    order_s = sorted(t.index, key=lambda k: (-t.loc[k, "peak_u_s"], k))
    order_h = sorted(t.index, key=lambda k: (-t.loc[k, "peak_hub_share"], k))
    key = {k: order_s.index(k) + order_h.index(k) for k in t.index}
    assert g["d_hub_dominated_high_s"] == sorted(t.index, key=lambda k: (key[k], k))[:5]


def test_cards_alerted_rule_includes_ties_at_tau_and_handles_no_alert():
    df = _tier0_frame()
    tau = float(df["score_st"].iloc[3])
    assert int(df["track"].iloc[3]) in C.alerted_tracks(df, tau)
    assert C.select_groups(df, 10.0, seed=42)["c_alerted_sample"] == []


def test_A9_cards_use_only_existing_fields_and_list_what_is_unavailable():
    df = _tier0_frame()
    g = C.select_groups(df, 0.7, 42)
    m = C.memberships(g)
    assert list(m) == sorted(m) and all(set(v) <= set(C.GROUPS) for v in m.values())
    cards = [C.build_card(df, tr, gs, 0.7) for tr, gs in m.items()]
    allowed_timeline = set(C.TIMELINE_COLUMNS) | {"alerted"}
    for c in cards:
        assert c["standing_statement"] == C.STANDING and c["unavailable_fields"] == list(C.UNAVAILABLE_FIELDS)
        assert all(set(r) == allowed_timeline for r in c["timeline"])
        assert set(c["member_transaction_composition_txn_days"]) == set(C.SEGMENT_COLUMNS)
        rows = df[df["track"] == c["track"]]
        assert c["first_observed_day"] == rows["day"].min() and c["last_observed_day"] == rows["day"].max()
        assert c["left_censored"] == (rows["day"].min() == 122) and c["right_censored"] == (rows["day"].max() == 151)
        assert c["n_alerted_days"] == int((rows["score_st"] >= 0.7).sum())
        for k, v in c["member_transaction_composition_txn_days"].items():
            assert v == int(rows[k].sum())
    assert any("member card identities" in f for f in cards[0]["unavailable_fields"])


def test_cards_reject_unexpected_columns_days_outside_val_and_duplicates():
    df = _tier0_frame()
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "t.csv"
        df.assign(isFraud=0).to_csv(p, index=False)
        for bad in (df.assign(isFraud=0), df.assign(day=df["day"] + 100), pd.concat([df, df.iloc[:1]])):
            bad.to_csv(p, index=False)
            try:
                C.load_tier0(p)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid Tier 0 file accepted")
        df.to_csv(p, index=False)
        assert len(C.load_tier0(p)) == len(df)
    assert C.a2_max_abs_difference(df) < 1e-12


def test_A8_wording_lint_clean_on_sources_and_rendered_cards():
    df = _tier0_frame()
    g = C.select_groups(df, 0.7, 42)
    m = C.memberships(g)
    cards = [C.build_card(df, tr, gs, 0.7) for tr, gs in m.items()]
    md = C.render_markdown(cards, {"seed": 42, "W": 7, "tau": 0.7, "groups": g})
    assert lint_text(md) == [] and C.STANDING in md and C.NOT_ESTABLISHED in md
    files = sorted((ROOT / "src" / "explain").glob("*.py")) + [pathlib.Path(__file__)]
    assert lint_files(files) == {} or all(not v for v in lint_files(files).values()), lint_files(files)


def test_new_package_imports_nothing_from_phase_modules_or_test_data():
    for f in (ROOT / "src" / "explain").glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom):
                assert "coordination" not in (node.module or "") and "evolution" not in (node.module or ""), f
        assert "test_features" not in src, f


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
