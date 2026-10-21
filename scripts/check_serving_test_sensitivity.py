"""Mutation-sensitivity check for tests/test_serving.py (criterion P13-8): inject known defects into an in-memory copy of src/serving/app.py and verify
that the tests fail for each. No file is modified; the synthetic demo bundle is the only model used; no research data and no TEST file is read.

Usage:  .venv/Scripts/python.exe -B scripts/check_serving_test_sensitivity.py
Writes experiments/phase13_serving_dashboard/test_results/test_sensitivity.json
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys
import types

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import tests.test_serving as TS  # noqa: E402

SRC = (ROOT / "src" / "serving" / "app.py").read_text(encoding="utf-8")
OUT = ROOT / "experiments" / "phase13_serving_dashboard" / "test_results"

MUTANTS = [
    ("M01 request size limit removed", "if n > MAX_BODY:", "if False:"),
    ("M02 history and new swapped in the score call", "res = S.score(self.bundle, old, new, explain=True, top_k=5)", "res = S.score(self.bundle, new, old, explain=True, top_k=5)"),
    ("M03 non-local bind allowed", "if host not in LOCAL_HOSTS and not allow_remote:", "if False:"),
    ("M04 held-out names no longer rejected by the router", ' or "test" in low:', " or False:"),
    ("M05 path traversal guard removed", 'if ".." in path or "\\\\" in path or "test" in low:', "if False:"),
    ("M06 explanations returned for every row", "if bool(r.alert) and r.top_features:", "if True:"),
    ("M07 stack trace returned to the client", 'return self._error(500, "internal error")', "return self._error(500, traceback.format_exc())"),
    ("M08 community output gains a member list", "out = {k: card[k] for k in CARD_KEYS if k in card}", 'out = {k: card[k] for k in CARD_KEYS if k in card}; out["member_ids"] = card.get("members", [])'),
    ("M09 candidate disclaimers dropped", '"standing_statement": c["standing_statement"], "not_established": c["not_established"],', '"standing_statement": "", "not_established": "",'),
    ("M10 evidence values rounded", '"pr_auc": f("pr_auc_mean"),', '"pr_auc": round(f("pr_auc_mean") or 0, 2),'),
    ("M11 sample rows served without a demo bundle", "if services.sample is None:", "if False:"),
    ("M12 content security header removed", 'self.send_header("Content-Security-Policy", "default-src \'none\'; script-src \'unsafe-inline\'; style-src \'unsafe-inline\'; connect-src \'self\'; img-src \'self\' data:")', "pass"),
    ("M13 scoring reads a file path named by the client", 'raise ApiError(404, "no scoring bundle is loaded (start the server with --demo or --bundle DIR)")\n        if not isinstance(payload, dict)', 'raise ApiError(404, "no scoring bundle is loaded (start the server with --demo or --bundle DIR)")\n        open(str(payload.get("path", "")) or "nul") if isinstance(payload, dict) and payload.get("path") else None\n        if not isinstance(payload, dict)'),
    ("M14 non-alert rows get a null explanation entry that crashes", 'item["top_features"] = json.loads(r.top_features)', 'item["top_features"] = json.loads(r.top_features)\n            item["labels_used"] = True'),
]


EQUIVALENT = {"M04": "defence in depth: no route serves files, so unknown paths already return 404; the guard has no observable effect with the current route table",
              "M05": "defence in depth: same reason as M04"}


def make_module(source: str, name: str):
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / "src" / "serving" / "app.py")
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)  # noqa: S102 - in-memory copy of the project's own source
    return mod


def run_tests(mod):
    original_app, original_svc = TS.APP, TS._C.get("svc")
    base = original_svc
    TS.APP = mod
    TS._C["svc"] = mod.Services(bundle=base.bundle, synthetic_demo=True, sample=base.sample)
    failed = []
    try:
        for name in [k for k, v in vars(TS).items() if k.startswith("test_") and callable(v)]:
            try:
                getattr(TS, name)()
            except BaseException:  # noqa: BLE001
                failed.append(name)
    finally:
        TS.APP, TS._C["svc"] = original_app, original_svc
    return failed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TS.services()
    unmutated = run_tests(make_module(SRC, "serving_unmutated"))
    rows, undetected = [], []
    for label, old, new in MUTANTS:
        if SRC.count(old) != 1:
            rows.append({"mutant": label, "status": "NOT APPLIED", "detail": f"pattern found {SRC.count(old)} times"})
            undetected.append(label)
            continue
        try:
            failed = run_tests(make_module(SRC.replace(old, new), "serving_mutant"))
        except Exception as e:  # noqa: BLE001
            failed = [f"module error: {type(e).__name__}: {e}"]
        key = label.split()[0]
        if not failed and key in EQUIVALENT:
            rows.append({"mutant": label, "status": "equivalent", "reason": EQUIVALENT[key], "failing_tests": []})
            print(f"[equivalent] {label}", flush=True)
            continue
        rows.append({"mutant": label, "status": "detected" if failed else "UNDETECTED", "failing_tests": failed})
        if not failed:
            undetected.append(label)
        print(f"[{'detected' if failed else 'UNDETECTED'}] {label} -> {len(failed)} failing test(s)", flush=True)
    rec = {"purpose": "mutation sensitivity of tests/test_serving.py", "run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
           "unmutated_failures": unmutated, "n_mutants": len(MUTANTS), "n_detected": sum(r["status"] == "detected" for r in rows), "n_equivalent": sum(r["status"] == "equivalent" for r in rows), "mutants": rows,
           "scope_statement": "In-memory mutants only; no file modified; synthetic bundle only. Detection shows these tests fail on these defects, not that no other defect exists."}
    (OUT / "test_sensitivity.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"unmutated failures: {unmutated or 'none'}; detected {rec['n_detected']}, equivalent {rec['n_equivalent']}, of {rec['n_mutants']}")
    return 0 if not unmutated and not undetected else 1


if __name__ == "__main__":
    sys.exit(main())
