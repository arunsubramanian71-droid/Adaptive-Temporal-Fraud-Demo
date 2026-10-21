"""Phase 12: build the consolidated validation-evidence table from stored VAL predictions (no training, scoring or prediction).

Scope record: experiments/phase12_validation_evidence_table/SCOPE_AND_ACCEPTANCE.md. Reads only the pre-registered 165-file source set
through a guarded reader; recomputes metrics with the unmodified src/models/metrics.py; writes nothing unless every gate passes.
Usage:  .venv/Scripts/python.exe -B scripts/run_phase12_table.py
"""
from __future__ import annotations

import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evidence import build as B  # noqa: E402
from src.evidence import sources as S  # noqa: E402
from src.evidence import table as T  # noqa: E402

OUT = ROOT / "experiments" / "phase12_validation_evidence_table" / "outputs"


def main() -> int:
    try:
        res = B.build(ROOT, OUT)
    except (T.BuildBlocked, S.SourceError) as e:
        print(f"BLOCKED: {e}")
        return 3
    print({k: v for k, v in res.items() if k not in ("outputs", "cross_bad")})
    for item in res["cross_bad"][:10]:
        print("cross-check discrepancy:", item)
    print("written:", sorted(res["outputs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
