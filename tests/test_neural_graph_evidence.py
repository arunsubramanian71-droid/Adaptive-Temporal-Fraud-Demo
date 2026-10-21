"""Neural graph evidence: tests of the counterfactual functions on a small synthetic dataset and an untrained Temporal GraphSAGE (NE4, NE5 logic).
No real data, no checkpoint, no label."""
from __future__ import annotations

import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.coordination.lint import lint_files  # noqa: E402
from src.explain import neural_graph_evidence as NG  # noqa: E402
from src.graph.temporal import RELATIONS, DayRel, TemporalDataset  # noqa: E402
from src.models.temporal_graphsage import TemporalGraphSAGE  # noqa: E402

_C = {}


def build():
    """Two days. Day 1: transactions 0,1 (card 0 shared; netproxy 0 only for tx 0). Day 2: transactions 0,1,2 (card 0 for tx 0 and 1, card 1 for tx 2; pseudo-account 0 for tx 0 only;
    device none; netproxy 0 for tx 2)."""
    if "b" in _C:
        return _C["b"]
    torch.manual_seed(0)
    x = torch.randn(5, 4)
    n_ent = {"pseudo_account": 3, "card": 3, "netproxy": 3, "device_fingerprint": 3}

    def rel(src, dst):
        src_t = torch.tensor(src, dtype=torch.long)
        ids, inv = torch.unique(torch.tensor(dst, dtype=torch.long), return_inverse=True) if dst else (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
        return DayRel(src_t if dst else torch.zeros(0, dtype=torch.long), inv, ids)

    day1 = {"charged_to": rel([], []), "used_card": rel([0, 1], [0, 0]), "via_network": rel([0], [0]), "from_device": rel([], [])}
    day2 = {"charged_to": rel([0], [0]), "used_card": rel([0, 1, 2], [0, 0, 1]), "via_network": rel([2], [0]), "from_device": rel([], [])}
    ds = TemporalDataset(x, None, np.arange(5), np.array([1, 1, 2, 2, 2]), {1: (0, 2), 2: (2, 5)}, {1: day1, 2: day2}, n_ent, 2)
    model = TemporalGraphSAGE(4).eval()
    state = model.init_state(ds)
    with torch.no_grad():
        model.step(state, ds, 1, predict=False)          # state at the start of day 2
    _C["b"] = (model, ds, state)
    return _C["b"]


def snapshot(ds, state):
    return ([(k, v.src_local.clone(), v.dst_local.clone(), v.ids.clone()) for k, v in ds.rel[2].items()],
            {k: v.clone() for k, v in state.S.items()}, {k: v.clone() for k, v in state.last_day.items()})


def same_snapshot(a, b):
    return all(x[0] == y[0] and all(torch.equal(p, q) for p, q in zip(x[1:], y[1:])) for x, y in zip(a[0], b[0])) and \
        all(torch.equal(a[1][k], b[1][k]) for k in a[1]) and all(torch.equal(a[2][k], b[2][k]) for k in a[2])


def test_base_logit_equals_a_direct_step_and_state_is_not_mutated():
    model, ds, state = build()
    before = snapshot(ds, state)
    clone = state.clone()
    with torch.no_grad():
        direct = model.step(clone, ds, 2, predict=True)
    mine = NG.day_logits(model, ds, state, 2)
    assert torch.equal(direct, mine)
    assert same_snapshot(before, snapshot(ds, state)), "the live state or the dataset changed"


def test_remove_edge_deletes_exactly_that_transactions_edge_and_drops_orphaned_entities():
    _, ds, _ = build()
    card = ds.rel[2]["used_card"]                  # edges: tx0->card0, tx1->card0, tx2->card1
    r = NG.remove_edge(card, 2)                    # card1 is only linked to tx2: it must leave `ids`
    assert r.src_local.tolist() == [0, 1] and r.ids.tolist() == [0] and r.dst_local.tolist() == [0, 0]
    r = NG.remove_edge(card, 0)                    # card0 keeps tx1
    assert r.src_local.tolist() == [1, 2] and sorted(r.ids.tolist()) == [0, 1]
    assert [r.ids[i].item() for i in r.dst_local] == [0, 1]
    none = NG.remove_edge(ds.rel[2]["from_device"], 0)
    assert none.src_local.numel() == 0 and none.ids.numel() == 0
    assert NG.entity_of(card, 1) == 0 and NG.entity_of(card, 2) == 1 and NG.entity_of(ds.rel[2]["charged_to"], 1) is None


def test_removal_changes_only_through_the_targets_link_and_leaves_other_edges_identical():
    model, ds, state = build()
    _, rels = ds.day_slice(2)
    new = dict(rels)
    new["used_card"] = NG.remove_edge(rels["used_card"], 0)
    for name in RELATIONS:
        if name != "used_card":
            assert new[name] is rels[name]
    assert rels["used_card"].src_local.tolist() == [0, 1, 2]
    base = NG.day_logits(model, ds, state, 2)
    pert = NG.day_logits(model, ds, state, 2, new)
    assert not torch.equal(base[0], pert[0]), "removing the target's own link must change its score"


def test_counterfactuals_structure_presence_and_memory_semantics():
    model, ds, state = build()
    out = NG.counterfactuals_for(model, ds, state, 2, 0)           # tx0 on day 2: pseudo_account 0 (new), card 0 (seen on day 1), netproxy none, device none
    assert set(out["relations"]) == set(RELATIONS) and math.isfinite(out["base"])
    r = out["relations"]
    assert r["charged_to"]["present"] and r["charged_to"]["entity_new"] is True and r["charged_to"]["reset_effect"] == 0.0, "reset of a never-seen entity must change nothing"
    assert r["used_card"]["present"] and r["used_card"]["entity_new"] is False and r["used_card"]["reset_effect"] != 0.0, "reset of a seen entity must change the score"
    assert not r["via_network"]["present"] and math.isnan(r["via_network"]["removal_effect"]) and r["via_network"]["entity"] is None
    assert not r["from_device"]["present"]
    assert r["used_card"]["removal_effect"] != 0.0 and r["charged_to"]["removal_effect"] != 0.0
    # tx2 links to card 1 (never seen) and netproxy 0 (seen on day 1 through tx0)
    o2 = NG.counterfactuals_for(model, ds, state, 2, 2)["relations"]
    assert o2["used_card"]["entity_new"] is True and o2["used_card"]["reset_effect"] == 0.0
    assert o2["via_network"]["entity_new"] is False and o2["via_network"]["reset_effect"] != 0.0


def test_counterfactuals_never_mutate_the_dataset_or_the_state_and_are_deterministic():
    model, ds, state = build()
    before = snapshot(ds, state)
    a = NG.counterfactuals_for(model, ds, state, 2, 1)
    assert same_snapshot(before, snapshot(ds, state))
    b = NG.counterfactuals_for(model, ds, state, 2, 1)
    assert a == b or all((a["relations"][k][f] == b["relations"][k][f]) or (isinstance(a["relations"][k][f], float) and math.isnan(a["relations"][k][f]) and math.isnan(b["relations"][k][f]))
                         for k in a["relations"] for f in a["relations"][k])


def test_proxy_refuses_another_day_and_module_has_no_label_argument():
    model, ds, state = build()
    proxy = NG.DayProxy(ds, 2, ds.rel[2])
    try:
        proxy.day_slice(1)
    except ValueError as e:
        assert "day 2" in str(e)
    else:
        raise AssertionError("proxy served a different day")
    src = (ROOT / "src" / "explain" / "neural_graph_evidence.py").read_text(encoding="utf-8")
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    assert not re.search(r"isFraud|label|y_true|\by\b|\.y\b", code, re.I)
    bad = {str(k): v for k, v in lint_files([ROOT / "src/explain/neural_graph_evidence.py", ROOT / "tests/test_neural_graph_evidence.py",
                                              ROOT / "experiments/neural_graph_evidence/PREREGISTRATION_NEURAL_EVIDENCE.md"]).items() if v}
    assert bad == {}, bad


# ------------------------------------------------------------------------------------------------ recorded outputs
def _sha(p):
    import hashlib
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def test_recorded_outputs_follow_the_preregistration_and_report_the_failed_criterion():
    import json
    import pandas as pd
    e = ROOT / "experiments" / "neural_graph_evidence"
    out = e / "outputs"
    pre = e / "PREREGISTRATION_NEURAL_EVIDENCE.md"
    assert _sha(pre) == (e / "PREREGISTRATION_NEURAL_EVIDENCE.sha256").read_text().strip()
    man = json.loads((out / "manifest_neural_evidence.json").read_text(encoding="utf-8"))
    assert man["preregistration_sha256"] == _sha(pre) and man["test_data_read"] is False
    for k, h in man["outputs_sha256"].items():
        assert _sha(out / k) == h, k
    for k, h in man["inputs_sha256"].items():
        assert _sha(ROOT / k) == h, k
    import os
    assert os.stat(pre).st_mtime < os.stat(ROOT / "src/explain/neural_graph_evidence.py").st_mtime, "the code must postdate the pre-registration"
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    a = pd.read_csv(out / "relation_effects.csv")
    sel = pd.read_csv(ROOT / "experiments/phase11_explainable_evidence/xgb_explanations_val/selection_val.csv")
    assert a["TransactionID"].tolist() == sel["TransactionID"].tolist() == sorted(a["TransactionID"].tolist()) or set(a["TransactionID"]) == set(sel["TransactionID"])
    assert len(a) == 300 and a["day"].between(122, 151).all()
    rels = list(RELATIONS)
    new = np.concatenate([a.loc[a[f"entity_new_{r}"] == 1, f"reset_effect_{r}"].to_numpy() for r in rels])
    seen = np.concatenate([a.loc[a[f"entity_new_{r}"] == 0, f"reset_effect_{r}"].to_numpy() for r in rels])
    assert s["NE5_sanity"]["n_reset_new"] == len(new) and s["NE5_sanity"]["n_reset_seen"] == len(seen)
    assert s["NE5_sanity"]["a_reset_of_never_seen_entities_exactly_zero"] == bool(np.all(new == 0.0))
    assert s["acceptance"]["NE5"] is False and abs(new).max() < 1e-5, "the registered NE5(a) failure must stay recorded, and the residual must be at float noise level"
    assert s["NE1_reproduction"]["pass"] and s["NE1_reproduction"]["max_abs_probability_difference_vs_recorded"] <= 1e-4
    rep = (out / "NEURAL_EVIDENCE_REPORT.md").read_text(encoding="utf-8")
    assert "NE5" in rep and "**FAIL**" in rep and "not changed" in rep
    assert _sha(out / "NEURAL_EVIDENCE_REPORT.md") == (out / "NEURAL_EVIDENCE_REPORT.sha256").read_text().strip()
    assert lint_files([out / "NEURAL_EVIDENCE_REPORT.md"])[str(out / "NEURAL_EVIDENCE_REPORT.md")] == []


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
