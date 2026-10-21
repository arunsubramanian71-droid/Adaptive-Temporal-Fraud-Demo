"""Counterfactual graph evidence for the Temporal GraphSAGE checkpoint (seed 42) on the 300 Phase 11 validation transactions.

Protocol: experiments/neural_graph_evidence/PREREGISTRATION_NEURAL_EVIDENCE.md. Label-free (dataset built without labels), days <= 151 only, no held-out file.
Writes only into experiments/neural_graph_evidence/outputs/.

Usage:  .venv/Scripts/python.exe -B scripts/run_neural_graph_evidence.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import pathlib
import platform
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import run_comparison_b as RB  # noqa: E402  (guarded Reader; main() is not run on import)
from src.config import load_config  # noqa: E402
from src.explain import neural_graph_evidence as NG  # noqa: E402
from src.graph.temporal import RELATIONS, build_temporal_dataset, load_raw_arrays  # noqa: E402
from src.models.temporal_graphsage import TemporalGraphSAGE  # noqa: E402

EXP = "experiments/neural_graph_evidence"
OUT = ROOT / EXP / "outputs"
PREREG, PREREG_SIDECAR = f"{EXP}/PREREGISTRATION_NEURAL_EVIDENCE.md", f"{EXP}/PREREGISTRATION_NEURAL_EVIDENCE.sha256"
SELECTION = "experiments/phase11_explainable_evidence/xgb_explanations_val/selection_val.csv"
SELECTION_HASH = "623d1ee2a0c6995e7c940565cd3c5d38a3bce076776de6a7976600fb440be963"          # recorded in Phase 11's manifest_val.json
CKPT = "experiments/temporal_graphsage/seed_42/model_checkpoint.pt"
PRED = "experiments/temporal_graphsage/seed_42/predictions_val.parquet"
ALLOWED = [PREREG, PREREG_SIDECAR, SELECTION, CKPT, PRED]


class Reader(RB.Reader):
    """The shared guard refuses every .pt file; this script may load exactly one: the project's own Temporal GraphSAGE seed-42 checkpoint."""

    def _check(self, rel, kind):
        rel_n = str(rel).replace("\\", "/")
        if rel_n == CKPT and rel_n in self.allowed:
            self.log.append((rel_n, kind))
            return self.root / rel_n
        return super()._check(rel, kind)


def one_pass(model, ds, sel):
    state = model.init_state(ds)
    rows = []
    with torch.no_grad():
        for d in ds.days:
            if d > 151:
                break
            if d >= 122:
                day_ids = ds.ids_for_day(d)
                pos_of = {int(t): i for i, t in enumerate(day_ids)}
                for _, r in sel[sel["day"] == d].iterrows():
                    cf = NG.counterfactuals_for(model, ds, state, d, pos_of[int(r["TransactionID"])])
                    row = {"TransactionID": int(r["TransactionID"]), "day": d, "group": r["group"], "base_logit": cf["base"], "base_score": 1.0 / (1.0 + math.exp(-cf["base"]))}
                    for name in RELATIONS:
                        c = cf["relations"][name]
                        row[f"present_{name}"] = int(c["present"])
                        row[f"entity_new_{name}"] = (int(c["entity_new"]) if c["present"] else None)
                        row[f"removal_effect_{name}"] = c["removal_effect"]
                        row[f"reset_effect_{name}"] = c["reset_effect"]
                    rows.append(row)
            model.step(state, ds, d, predict=False)         # advance the real stream (labels never involved)
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    reader = Reader(ROOT, allowed=ALLOWED, extra=())
    pre_hash = reader.sha256(PREREG)
    assert pre_hash == reader.read_text(PREREG_SIDECAR).strip(), "pre-registration does not match its sidecar"
    sel_hash = reader.sha256(SELECTION)
    sel = pd.read_csv(reader.path(SELECTION, "csv"))
    ne2 = {"selection_hash_equals_phase11_record": sel_hash == SELECTION_HASH, "n_ids": int(len(sel)), "ids_unique": bool(sel["TransactionID"].is_unique),
           "days_in_122_151": bool(sel["day"].between(122, 151).all()), "ten_per_day": bool((sel.groupby("day").size() == 10).all() and sel["day"].nunique() == 30)}
    ck_hash = reader.sha256(CKPT)
    cfg = load_config()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw, _graph, _ = load_raw_arrays(cfg)
    ds = build_temporal_dataset(raw, cfg.split.val_day_max, dev, cfg, with_labels=False)       # truncated at day 151, no labels
    assert ds.y is None and ds.max_day == 151
    del raw, _graph
    model = TemporalGraphSAGE(ds.X.shape[1]).to(dev)
    model.load_state_dict(torch.load(reader.path(CKPT, "checkpoint"), map_location=dev))
    model.eval()
    a = one_pass(model, ds, sel)
    b = one_pass(model, ds, sel)
    effect_cols = [c for c in a.columns if c.startswith(("removal_effect_", "reset_effect_", "base_logit"))]
    rep = float(np.nanmax(np.abs(a[effect_cols].to_numpy(dtype=float) - b[effect_cols].to_numpy(dtype=float))))
    pred = pd.read_parquet(reader.path(PRED, "parquet"), columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
    ne1 = float(np.max(np.abs(a["base_score"].to_numpy() - pred.loc[a["TransactionID"]].to_numpy())))

    removal = [a[f"removal_effect_{n}"].to_numpy(dtype=float) for n in RELATIONS]
    present = [a[f"present_{n}"].to_numpy() == 1 for n in RELATIONS]
    n_removals = int(sum(p.sum() for p in present))
    changed = int(sum((np.abs(r[p]) > 1e-6).sum() for r, p in zip(removal, present)))
    new_flags = [a[f"entity_new_{n}"].to_numpy(dtype=float) for n in RELATIONS]
    reset = [a[f"reset_effect_{n}"].to_numpy(dtype=float) for n in RELATIONS]
    new_resets = np.concatenate([r[(f == 1)] for r, f in zip(reset, new_flags)])
    seen_resets = np.concatenate([r[(f == 0)] for r, f in zip(reset, new_flags)])
    ne5 = {"a_reset_of_never_seen_entities_exactly_zero": bool(len(new_resets) > 0 and np.all(new_resets == 0.0)), "n_reset_new": int(len(new_resets)),
           "b_reset_of_seen_entities_changes_score_in_majority": bool(len(seen_resets) > 0 and (np.abs(seen_resets) > 1e-6).mean() > 0.5), "n_reset_seen": int(len(seen_resets)),
           "share_reset_seen_changed": float((np.abs(seen_resets) > 1e-6).mean()) if len(seen_resets) else None,
           "c_link_removal_changes_score_in_majority": bool(n_removals > 0 and changed / n_removals > 0.5), "n_removals": n_removals, "share_removal_changed": changed / max(n_removals, 1)}
    top = pd.Series([max((n for n, p, r in zip(RELATIONS, present, removal) if p[i]), key=lambda n: abs(a[f"removal_effect_{n}"].iat[i]), default=None) for i in range(len(a))])
    summary_rel = {}
    for n, p, r, f, rs in zip(RELATIONS, present, removal, new_flags, reset):
        seen = (f == 0)
        summary_rel[n] = {"share_with_edge": float(p.mean()), "mean_abs_removal_effect_when_present": float(np.abs(r[p]).mean()) if p.any() else None,
                          "mean_abs_reset_effect_entity_seen": float(np.abs(rs[seen]).mean()) if seen.any() else None, "share_entity_new_when_present": float(np.nanmean(f[p])) if p.any() else None,
                          "share_of_transactions_where_largest_removal_effect": float((top == n).mean())}
    summary = {"NE1_reproduction": {"max_abs_probability_difference_vs_recorded": ne1, "bound": 1e-4, "pass": ne1 <= 1e-4}, "NE2_selection": ne2,
               "NE3_label_free": {"dataset_built_without_labels": True, "dataset_max_day": 151, "label_in_script": False},
               "NE5_sanity": ne5, "NE7_second_run_max_abs_difference_in_effects": rep,
               "by_relation": summary_rel, "n_transactions": int(len(a)), "statement": "Descriptive counterfactual evidence about the Temporal GraphSAGE seed-42 model on 300 pre-registered validation transactions; VAL-tuned; "
                                                                                    "not evidence of fraud, causation or coordination."}
    summary["acceptance"] = {"NE1": summary["NE1_reproduction"]["pass"], "NE2": all(v for v in ne2.values() if isinstance(v, bool)), "NE3": True,
                             "NE5": bool(ne5["a_reset_of_never_seen_entities_exactly_zero"] and ne5["b_reset_of_seen_entities_changes_score_in_majority"] and ne5["c_link_removal_changes_score_in_majority"]),
                             "NE7_repeatability": rep <= 1e-5}

    def wbytes(name, data: bytes):
        (OUT / name).write_bytes(data)
        h = hashlib.sha256(data).hexdigest()
        (OUT / (name.rsplit(".", 1)[0] + ".sha256")).write_bytes(h.encode("ascii"))
        return h

    out = a.copy()
    out["largest_removal_relation"] = top
    hashes = {"relation_effects.csv": wbytes("relation_effects.csv", out.to_csv(index=False, float_format="%.9g", lineterminator="\n").encode("utf-8")),
              "summary.json": wbytes("summary.json", (json.dumps(summary, indent=1) + "\n").encode("utf-8"))}
    manifest = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "preregistration_sha256": pre_hash,
                "inputs_sha256": {SELECTION: sel_hash, CKPT: ck_hash, PRED: reader.sha256(PRED)}, "outputs_sha256": hashes, "io_audit_guarded_reads": reader.audit(),
                "graph_loaded_through": "src.graph.temporal.load_raw_arrays (the project loader; dataset truncated at day 151 and built without labels)", "test_data_read": False,
                "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}, "device": str(dev), "seconds": round(time.time() - t0, 1)}
    wbytes("manifest_neural_evidence.json", (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(summary["acceptance"]), "| NE1 max diff", f"{ne1:.2e}", "| repeat diff", f"{rep:.2e}")
    print(json.dumps(ne5, indent=0)[:600])
    return 0 if all(summary["acceptance"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
