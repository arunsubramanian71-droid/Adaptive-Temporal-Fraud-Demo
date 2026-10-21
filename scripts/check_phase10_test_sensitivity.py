"""Test-sensitivity (mutation) check for the Phase 10 core tests.

Each mutant is an in-memory text edit of ``src/evolution/tracker.py`` (the file on disk is never modified). A mutant is DETECTED if at
least one Phase 10 core test fails against it. The unmutated module must pass everything. This checks the tests, not the research.
Usage:  .venv/Scripts/python.exe -B scripts/check_phase10_test_sensitivity.py
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import importlib
import io
import json
import pathlib
import sys
import types

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TRACKER = ROOT / "src" / "evolution" / "tracker.py"
OUT = ROOT / "experiments" / "phase10_community_evolution" / "core_implementation"
SUITES = ("tests.test_phase10_toys", "tests.test_phase10_properties", "tests.test_phase10_compat")

MUTANTS = {
    "M1 identity key drops the overlap term": ("return (-pairs[k][3], -pairs[k][0], k[0], k[1])      # Jaccard, overlap, lineage id, canonical rank",
                                                "return (-pairs[k][3], k[0], k[1])"),
    "M2 re-link window g_r instead of g_r + 1": ("dormant = [l for l in live if 2 <= t - l.last_day <= self.g_r + 1]",
                                                  "dormant = [l for l in live if 2 <= t - l.last_day <= self.g_r]"),
    "M3 newcomer majority non-strict": ("2 * n_new > sizes[r]", "2 * n_new >= sizes[r]"),
    "M4 declaration one day late": ("and t - l.last_day == self.g_d]", "and t - l.last_day == self.g_d + 1]"),
    "M5 R-A violated (re-link into components)": ("elig = [r for r in range(n) if r not in comp_of_rank and r not in sel]",
                                                   "elig = [r for r in range(n) if r not in sel]"),
    "M6 growth boundary strict": ("if b * q >= a * (q + p):", "if b * q > a * (q + p):"),
    "M7 identity threshold strict": ("J >= self.theta_id, kf >= self.theta_ev, kb >= self.theta_ev)\n        id_edges",
                                     "J > self.theta_id, kf >= self.theta_ev, kb >= self.theta_ev)\n        id_edges"),
    "M8 re-link key puts gap before Jaccard": ("key=lambda k: (-cand[k][3], -cand[k][0], cand[k][6], k[0], k[1])",
                                                "key=lambda k: (cand[k][6], -cand[k][3], -cand[k][0], k[0], k[1])"),
    "M9 reorganized targets = whole component": (
        "recs.append(LineageEnd(lid, t, reason, tuple(nodes[e[1]] for e in own)",
        "recs.append(LineageEnd(lid, t, reason, tuple(nodes[r2] for r2 in sorted({e[1] for e in q_edges if comp_of_lid.get(e[0]) == comp_of_lid.get(lid)}))"),
    "M10 contraction boundary strict": ("if b * (q + p) <= a * q:", "if b * (q + p) < a * q:"),
    "M11 legacy split flag dropped": ('flag_appends.append((first_of_rank[r], ("split", t)))', "pass"),
    "M12 closed lineages counted as dormant": ("closed = [l for l in live if t - l.last_day > self.g_r + 1]", "closed = []"),
    "M13 compat key uses the overlap term": ("return (-pairs[k][3], k[0], k[1])                    # legacy Phase 9 key",
                                              "return (-pairs[k][3], -pairs[k][0], k[0], k[1])"),
    "M14 legacy merged flag once per lineage": ('flag_appends.append((lid, ("merged", t)))',
                                                 'flag_appends.append((lid, ("merged", t))) if (lid, ("merged", t)) not in flag_appends else None'),
    "M15 legacy known_at off by one": ('"disappearance_known_at": (l.last_day + self.g_d) if ended else None,',
                                        '"disappearance_known_at": (l.last_day + self.g_d + 1) if ended else None,'),
    "M16 legacy parent pointer not set": ("parent=legacy_parent.get(r),", "parent=None,"),
    "M17 classification active boundary": ("elif l.last_day == T:", "elif l.last_day >= T - 1:"),
    "M18 merge-product origin swapped": ('"many_to_one": "born_merge_product"', '"many_to_one": "born_split_child"'),
}


def load(code: str) -> None:
    for m in [k for k in sys.modules if k.startswith("src.evolution") or k.startswith("tests.")]:
        del sys.modules[m]
    mod = types.ModuleType("src.evolution.tracker")
    mod.__file__ = "mutant"
    sys.modules["src.evolution.tracker"] = mod
    exec(compile(code, "mutant", "exec"), mod.__dict__)
    importlib.import_module("src.evolution")


def failing_tests() -> list[str]:
    failed = []
    for name in SUITES:
        m = importlib.import_module(name)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            m._run_all()
        failed += [l.split(":")[0].replace("[FAIL] ", "") for l in buf.getvalue().splitlines() if l.startswith("[FAIL]")]
    return failed


def main() -> int:
    src = TRACKER.read_text(encoding="utf-8")
    load(src)
    base = failing_tests()
    rows = []
    for label, (old, new) in MUTANTS.items():
        if src.count(old) != 1:
            rows.append({"mutant": label, "status": "pattern_not_unique", "failing_tests": []})
            continue
        load(src.replace(old, new))
        f = failing_tests()
        rows.append({"mutant": label, "status": "detected" if f else "NOT_DETECTED", "failing_tests": f})
    detected = sum(r["status"] == "detected" for r in rows)
    record = {
        "purpose": "Test-sensitivity (mutation) check of the Phase 10 core tests; in-memory mutants only, no project file modified.",
        "run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "tracker_sha256": hashlib.sha256(TRACKER.read_bytes()).hexdigest(),
        "unmutated_failures": base,
        "summary": {"mutants": len(rows), "detected": detected},
        "mutants": rows,
        "interpretation": "A detected mutant shows that at least one test fails when that specific defect is injected. "
                          "It does not show that no other defect exists.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "core_test_sensitivity.json").write_bytes((json.dumps(record, indent=1, ensure_ascii=True) + "\n").encode("ascii"))
    print(f"unmutated failures: {base}\nmutants detected: {detected}/{len(rows)}")
    for r in rows:
        print(f"  {r['status']:>16}  {r['mutant']}")
    return 0 if (not base and detected == len(rows)) else 1


if __name__ == "__main__":
    sys.exit(main())
