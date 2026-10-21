"""Graph evidence for model B: tests on a small synthetic XGBoost model (no real data, no labels from the project, no TEST)."""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files  # noqa: E402
from src.explain import graph_evidence as GE  # noqa: E402
from src.explain.xgb_contrib import contributions  # noqa: E402
from src.features import graph_features as G  # noqa: E402

COLS = ["t0", "t1", *G.feature_names()]
_C = {}


def fitted():
    if "m" not in _C:
        rng = np.random.default_rng(0)
        n = 1500
        X = rng.normal(size=(n, len(COLS))).astype(np.float32)
        for r in G.RELATIONS:
            X[:, COLS.index(f"g_{r}_has_edge")] = (rng.random(n) < 0.6).astype(np.float32)
        # the label depends on a device feature and a card feature only (a known relation structure)
        y = ((X[:, COLS.index("g_dev_prior_n")] * X[:, COLS.index("g_dev_has_edge")] + 0.5 * X[:, COLS.index("g_card_prior_n")] + 0.3 * X[:, 0]
              + rng.normal(scale=0.3, size=n)) > 0.6).astype(int)
        clf = xgb.XGBClassifier(n_estimators=40, max_depth=3, tree_method="hist", n_jobs=1, random_state=0).fit(X, y)
        _C.update(m=clf, X=X)
    return _C["m"], _C["X"]


def test_relation_assignment_follows_the_preregistration():
    assert GE.relation_of_feature("g_dev_prior_n") == "dev" and GE.relation_of_feature("g_card_prior_distinct_pa") == "card"
    assert GE.relation_of_feature("TransactionAmt") is None and GE.relation_of_feature("g_x_prior_n") is None
    rc = GE.relation_columns(COLS)
    assert {r: len(v) for r, v in rc.items()} == {"pa": 10, "card": 10, "net": 10, "dev": 10}


def test_relation_sums_plus_tabular_plus_bias_equal_the_margin():
    clf, X = fitted()
    contrib, bias, m = contributions(clf, X[:200])
    infl = GE.relation_influence(contrib, COLS)
    total = infl.sum(axis=1).to_numpy() + bias
    assert np.max(np.abs(total - m)) < 1e-4


def test_perturbation_marks_the_relation_absent_and_changes_nothing_else():
    clf, X = fitted()
    Xp = GE.perturb_relation(X[:50], COLS, "dev")
    rc = set(GE.relation_columns(COLS)["dev"])
    for i, c in enumerate(COLS):
        if i in rc:
            assert np.all(np.isnan(Xp[:, i])) or (c == "g_dev_has_edge" and np.all(Xp[:, i] == 0))
        else:
            assert np.array_equal(Xp[:, i], X[:50, i], equal_nan=True)
    assert not np.shares_memory(Xp, X)
    try:
        GE.perturb_relation(X[:5], COLS, "email")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown relation accepted")


def test_effects_are_nan_without_an_edge_and_removing_an_irrelevant_relation_matters_less():
    clf, X = fitted()
    eff = GE.perturbation_effects(clf, X, COLS)
    present = GE.present_relations(X, COLS)
    assert eff.isna().to_numpy().tolist() == (~present).tolist()
    assert eff["dev"].abs().mean() > eff["net"].abs().mean(), "the relation the label depends on should matter more than an unused one"


def test_pick_relations_is_deterministic_seeded_and_only_picks_present_relations():
    clf, X = fitted()
    contrib, _, _ = contributions(clf, X[:100])
    infl = GE.relation_influence(contrib, COLS)[list(G.RELATIONS)].abs()
    present = GE.present_relations(X[:100], COLS)
    ids = np.arange(9_000_000, 9_000_100)
    t1, r1 = GE.pick_relations(ids, present, infl)
    t2, r2 = GE.pick_relations(ids, present, infl)
    assert t1 == t2 and r1 == r2
    for k in range(100):
        if present[k].any():
            assert present[k, G.RELATIONS.index(t1[k])] and present[k, G.RELATIONS.index(r1[k])]
            best = max((infl.iloc[k][G.RELATIONS[j]], -j) for j in range(4) if present[k, j])
            assert t1[k] == G.RELATIONS[-best[1]]
        else:
            assert t1[k] is None and r1[k] is None
    t3, r3 = GE.pick_relations(ids, present, infl, seed=1)
    assert t3 == t1 and r3 != r1


def test_sanity_check_reports_the_comparison_it_made():
    clf, X = fitted()
    contrib, _, _ = contributions(clf, X)
    infl = GE.relation_influence(contrib, COLS)[list(G.RELATIONS)].abs()
    present = GE.present_relations(X, COLS)
    top, rnd = GE.pick_relations(np.arange(len(X)), present, infl)
    s = GE.perturbation_sanity(GE.perturbation_effects(clf, X, COLS), present, top, rnd)
    assert s["n_rows_with_two_or_more_relations"] > 100 and s["top_at_least_random"] is True, s


def test_module_has_no_label_arguments_and_lint_is_clean():
    src = (ROOT / "src" / "explain" / "graph_evidence.py").read_text(encoding="utf-8")
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    assert not re.search(r"isFraud|label|y_true|\by\b", code, re.I)
    bad = {str(k): v for k, v in lint_files([ROOT / "src" / "explain" / "graph_evidence.py", ROOT / "tests" / "test_graph_evidence.py",
                                              ROOT / "experiments" / "comparison_B_graph_features" / "GRAPH_EVIDENCE_PREREG.md"]).items() if v}
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
