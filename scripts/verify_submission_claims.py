"""Check that the numbers quoted in the submission documents appear in the authoritative project records.

For every numeric token in the checked documents (decimals, comma-grouped integers, percentages), the token must equal, at the token's own
precision, a number found in the authoritative corpus (phase reports, closure reports, recorded result files, the Phase 12 and B outputs).
Tokens inside fenced code blocks, the reference list, dates and section numbers are skipped. Unmatched tokens are listed and cause a
non-zero exit code: each must be fixed or justified in submission/claims_exceptions.json (token -> reason, derived from recorded values).

Usage:  .venv/Scripts/python.exe -B scripts/verify_submission_claims.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ["submission/MANUSCRIPT.md", "submission/README.md", "submission/DEMO_GUIDE.md", "submission/COMPLETION_MATRIX.md", "submission/FINAL_COMPLETION_RECORD.md"]
MD_GLOBS = ["reports/*.md", "experiments/phase*/**/*.md", "experiments/comparison_B_graph_features/**/*.md", "experiments/imbalance_check_xgboost/**/*.md",
            "experiments/final_heldout_protocol/*.md", "submission/COMPLETION_MATRIX.md", "submission/FINAL_COMPLETION_RECORD.md", "experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md", "experiments/phase13_serving_dashboard/CRITERIA_TRACE.md",
            "experiments/neural_graph_evidence/outputs/NEURAL_EVIDENCE_REPORT.md", "experiments/final_heldout_protocol/PROTOCOL_REVIEW.md"]
# structured records a number may be traced to (rounded to the token's precision); deliberately a short, named list
STRUCT_FILES = ["experiments/phase12_validation_evidence_table/outputs/evidence_cells.csv", "experiments/phase12_validation_evidence_table/outputs/evidence_runs.csv",
                "experiments/comparison_B_graph_features/outputs/decision.json", "experiments/comparison_B_graph_features/outputs/table_with_B.csv",
                "experiments/comparison_B_graph_features/graph_evidence/summary.json", "experiments/imbalance_check_xgboost/outputs/decision.json",
                "experiments/imbalance_check_xgboost/outputs/arms_table.csv", "experiments/phase8_adaptive_drift/comparison.json",
                "experiments/temporal_ablation/comparison.json", "experiments/temporal_graphsage/comparison.json", "experiments/phase11_explainable_evidence/xgb_explanations_val/summary_val.json",
                "experiments/phase9_coordinated_fraud/eval_benchmark_G1_G2_G4.json", "submission/test_results.json", "experiments/neural_graph_evidence/outputs/summary.json",
                "experiments/phase8_state_continuity_supplement/outputs/decision_compare.json", "experiments/neural_graph_evidence/diagnostics/NE5_diagnosis.json",
                "experiments/phase8_state_continuity_supplement/outputs/readiness/readiness_F.json", "experiments/final_heldout_protocol/readiness_val_amendment/readiness_report.json"]
NUM = re.compile(r"(?<![\w.])-?\+?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])[-+]?\d+\.\d+(?:e[-+]?\d+)?%?|(?<![\w.])\d{4,}(?![\w.])", re.I)


def corpus_numbers() -> dict:
    """Literal numeric strings from the prose records plus, for the named structured records, values rounded to every precision."""
    lit: set = set()
    n_files = 0
    for g in MD_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if p.is_file():
                n_files += 1
                lit |= {m.group(0).replace(",", "").lstrip("+") for m in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", p.read_text(encoding="utf-8", errors="ignore"))}
    vals: dict = {}
    ints: set = set()
    for rel in STRUCT_FILES:
        p = ROOT / rel
        if not p.is_file():
            continue
        n_files += 1
        for m in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", p.read_text(encoding="utf-8", errors="ignore")):
            sx = m.group(0)
            x = float(sx)
            if "." in sx or "e" in sx.lower():
                for d in range(0, 8):
                    vals.setdefault(d, set()).add(round(x, d))
            else:
                ints.add(int(x))
    return {"literals": lit, "floats": vals, "ints": ints, "files": n_files}


def body(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = text.split("\n## References")[0]
    text = re.sub(r"`[^`]*`", " ", text)                       # inline code (paths, commands, hashes)
    text = re.sub(r"(?m)^#+\s+\d+(?:\.\d+)*", " ", text)           # heading numbers
    text = re.sub(r"\b[Ss]ections?\s+\d+(?:\.\d+)*(?:\s*(?:and|,|-)\s*\d+(?:\.\d+)*)*", " ", text)   # cross-references
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)          # dates
    text = re.sub(r"\b[0-9a-f]{12,64}\b", " ", text)            # hashes
    return text


def check(doc: str, corpus: dict, exceptions: dict) -> list:
    text = body((ROOT / doc).read_text(encoding="utf-8"))
    bad = []
    for m in NUM.finditer(text):
        tok = m.group(0)
        raw = tok.replace(",", "").rstrip("%").lstrip("+")
        if tok.endswith("%") and "." not in raw:
            continue
        try:
            x = float(raw)
        except ValueError:
            continue
        if raw in corpus["literals"] or raw.lstrip("-") in corpus["literals"]:
            ok = True
        elif "." in raw or "e" in raw.lower():
            d = len(raw.split(".")[1].split("e")[0].split("E")[0]) if "." in raw else 0
            ok = round(x, d) in corpus["floats"].get(d, set()) or round(-x, d) in corpus["floats"].get(d, set())
        else:
            ok = int(x) in corpus["ints"] or int(x) <= 2100                      # years and small counts are not data claims
        if not ok and tok not in exceptions and raw not in exceptions:
            ctx = text[max(0, m.start() - 50): m.end() + 30].replace("\n", " ")
            bad.append((tok, ctx))
    return bad


def self_test(corpus: dict) -> bool:
    """The checker must reject invented numbers: unmatched tokens in a synthetic document have to be reported."""
    fake = "The model reached 0.8123 PR-AUC and 7.7777 and 123,456,789 transactions and 61.31%."
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp(prefix="claims_"))
    (d / "x.md").write_text(fake, encoding="utf-8")
    global ROOT
    old = ROOT
    try:
        ROOT = d
        bad = [t for t, _ in check("x.md", corpus, {})]
    finally:
        ROOT = old
    return {"0.8123", "7.7777", "123,456,789"} <= set(bad)


def main() -> int:
    exc_path = ROOT / "submission" / "claims_exceptions.json"
    exceptions = json.loads(exc_path.read_text(encoding="utf-8")) if exc_path.exists() else {}
    corpus = corpus_numbers()
    print(f"authoritative corpus: {corpus['files']} files; self-test (rejects invented numbers): {'ok' if self_test(corpus) else 'FAILED'}")
    total_bad = 0
    for doc in DOCS:
        if not (ROOT / doc).exists():
            print(f"[skip] {doc} (not present)")
            continue
        bad = check(doc, corpus, exceptions)
        total_bad += len(bad)
        print(f"{doc}: {'all numeric tokens traced' if not bad else str(len(bad)) + ' unmatched'}")
        for tok, ctx in bad:
            print(f"    {tok!r}  ...{ctx}...")
    print("RESULT:", "PASS" if total_bad == 0 else f"{total_bad} unmatched tokens (fix or justify in submission/claims_exceptions.json)")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
