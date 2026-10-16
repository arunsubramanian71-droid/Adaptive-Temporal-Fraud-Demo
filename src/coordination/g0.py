"""Hard gate G0 (design section 9): array-level segmentation equivalence on explicitly day-truncated TRAIN+VAL data.

TEST-free by construction: only train_features.parquet and val_features.parquet are opened (recorded in the gate record);
the shared Phase 4 graph file (hetero_graph.pt, which contains TEST rows) is NOT loaded. The reference is produced by the
existing, unmodified add_derived_columns / build_hetero_graph / compute_pseudo_account_segments applied to the truncated
frame (zero-feature placeholder: segmentation uses only day and pseudo-account edges).
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from src.coordination.frozen import MAX_DAY
from src.coordination.segments_array import COLD, NOEDGE, WARM, segments_from_arrays

GATE_FILE = "segmentation_equivalence_gate.json"
SAVED_SEEDS = (42, 123, 2024)


def sha256_file(p, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def toy_cases() -> list[tuple[str, bool, str]]:
    out = []

    def chk(name, day, key, expect):
        got = list(segments_from_arrays(np.array(day), np.array(key)))
        ok = got == expect
        out.append((name, ok, "" if ok else f"got {got} expected {expect}"))

    chk("first_occurrence_repeat_same_day_missing_key", [1, 1, 2, 2, 3, 3, 3], [5, 5, -1, 6, 5, 6, 7],
        [COLD, WARM, NOEDGE, COLD, WARM, WARM, COLD])
    chk("unsorted_days_use_stable_day_order", [2, 1, 1], [9, 9, 9], [WARM, COLD, WARM])
    chk("planted_style_large_synthetic_id", [1, 2, 2], [10**9 + 3, 10**9 + 3, 10**9 + 4], [COLD, WARM, COLD])
    chk("all_missing", [1, 2], [-1, -1], [NOEDGE, NOEDGE])
    chk("empty", [], [], [])
    return out


def run_g0(cfg, out_dir) -> dict:
    from src.graph.build import build_hetero_graph
    from src.graph.schema import add_derived_columns, build_all_entity_keys, compute_pseudo_account_segments, load_graph_config

    t0 = time.time()
    rec: dict = {"gate": "G0", "design": "experiments/phase9_coordinated_fraud/DESIGN_BEFORE_EXECUTION.md section 9",
                 "criteria": {}, "files_opened": [], "passed": False}
    files = [cfg.features_dir / "train_features.parquet", cfg.features_dir / "val_features.parquet"]
    for f in files:
        assert "test" not in f.name
    rec["files_opened"] = [str(f) for f in files]
    rec["input_sha256"] = {f.name: sha256_file(f) for f in files}
    rec["shared_graph_file_loaded"] = False
    rec["test_file_opened"] = False

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = add_derived_columns(df, cfg)
    n = len(df)
    day = df["day"].to_numpy()
    c1 = {"n_rows": int(n), "expected": int(cfg.expected_counts["split_rows"]["train"] + cfg.expected_counts["split_rows"]["val"]),
          "max_day": int(day.max()), "min_day": int(day.min()), "day_nondecreasing": bool(np.all(np.diff(day) >= 0))}
    c1["passed"] = bool(c1["n_rows"] == c1["expected"] == 501214 and c1["max_day"] <= MAX_DAY and c1["day_nondecreasing"])
    rec["criteria"]["1_test_free_input"] = c1
    if not c1["passed"]:
        rec["elapsed_s"] = time.time() - t0
        return _write(rec, out_dir)

    gcfg = load_graph_config()
    graph, _ = build_hetero_graph(df, cfg, gcfg, np.zeros((n, 1), dtype=np.float32))
    ref = np.asarray(compute_pseudo_account_segments(graph), dtype=object)
    del graph
    pa_key = build_all_entity_keys(df, gcfg)["pseudo_account"]
    codes, _ = pd.factorize(pa_key)                       # -1 where the key is missing
    codes = codes.astype(np.int64)
    arr = segments_from_arrays(day, codes)

    mism = np.nonzero(ref != arr)[0]
    counts = {s: {"reference": int((ref == s).sum()), "array_level": int((arr == s).sum())} for s in (COLD, WARM, NOEDGE)}
    rec["criteria"]["2_3_exact_match"] = {"n_compared": int(n), "n_mismatch": int(mism.size), "label_counts": counts,
                                          "passed": bool(mism.size == 0 and all(v["reference"] == v["array_level"] for v in counts.values()))}

    tid = df["TransactionID"].to_numpy()
    val_mask = (day >= 122) & (day <= 151)
    ref_by_id = pd.Series(ref[val_mask], index=tid[val_mask])
    ok4, detail4 = True, {}
    for s in SAVED_SEEDS:
        sp = cfg.experiments_dir / "temporal_graphsage" / f"seed_{s}" / "predictions_val.parquet"
        pv = pd.read_parquet(sp, columns=["TransactionID", "segment"])
        merged = ref_by_id.reindex(pv["TransactionID"].to_numpy())
        n_missing = int(merged.isna().sum())
        n_bad = int((merged.to_numpy() != pv["segment"].to_numpy()).sum())
        detail4[f"seed{s}"] = {"n_saved": int(len(pv)), "n_missing_in_reference": n_missing, "n_mismatch": n_bad,
                               "sha256": sha256_file(sp)}
        ok4 &= (len(pv) == int(val_mask.sum()) and n_missing == 0 and n_bad == 0)
    detail4["n_val_reference"] = int(val_mask.sum())
    detail4["passed"] = bool(ok4)
    rec["criteria"]["4_tie_to_saved_phase7_val_labels"] = detail4

    fi = {}
    ok5 = True
    for c in (60, 121, 136):
        m = int((day <= c).sum())
        sub = segments_from_arrays(day[:m], codes[:m])
        bad = int((sub != arr[:m]).sum())
        fi[f"cutoff_{c}"] = {"n_rows": m, "n_mismatch": bad}
        ok5 &= bad == 0
    fi["passed"] = bool(ok5)
    rec["criteria"]["5_future_independence"] = fi

    dd = pd.DataFrame({"day": day, "k": codes})
    dd = dd[dd["k"] >= 0]
    later_same_day = dd.groupby(["day", "k"]).cumcount() > 0
    rows = dd.index[later_same_day.to_numpy()].to_numpy()
    s6 = {"n_later_same_day_duplicates": int(rows.size),
          "n_not_warm_reference": int((ref[rows] != WARM).sum()), "n_not_warm_array": int((arr[rows] != WARM).sum())}
    s6["passed"] = bool(rows.size > 0 and s6["n_not_warm_reference"] == 0 and s6["n_not_warm_array"] == 0)
    rec["criteria"]["6_same_day_semantics"] = s6

    toys = toy_cases()
    rec["criteria"]["7_toy_cases"] = {"cases": [{"name": a, "passed": b, "detail": c} for a, b, c in toys], "passed": all(b for _, b, _ in toys)}

    rec["passed"] = all(v["passed"] for v in rec["criteria"].values())
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["segmentation_source_sha256"] = hashlib.sha256(open("src/coordination/segments_array.py", "rb").read()).hexdigest()
    return _write(rec, out_dir)


def _write(rec, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / GATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def require_g0(out_dir, cfg=None) -> dict:
    """Generators call this: refuse to run unless a passing gate record exists (and input hashes still match)."""
    p = out_dir / GATE_FILE
    if not p.exists():
        raise RuntimeError("G0 gate record missing: benchmark generation must not proceed (design S9)")
    rec = json.load(open(p, encoding="utf-8"))
    if not rec.get("passed"):
        raise RuntimeError("G0 did not pass: benchmark generation must not proceed (design S9)")
    if cfg is not None:
        for name, h in rec["input_sha256"].items():
            if sha256_file(cfg.features_dir / name) != h:
                raise RuntimeError(f"G0 input file changed since the gate passed: {name}")
    return rec
