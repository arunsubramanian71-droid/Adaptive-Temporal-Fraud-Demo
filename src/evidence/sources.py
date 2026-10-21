"""Phase 12: the pre-registered source set, its fingerprint and a guarded reader.

Scope record: experiments/phase12_validation_evidence_table/SCOPE_AND_ACCEPTANCE.md (section 3, AC1). Only the 165 declared files may be
opened; every open is logged. TEST-named paths, checkpoints, feature files and the excluded folders are refused. Nothing here trains,
scores or predicts anything; all data are existing stored validation predictions and JSON records.
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pandas as pd
import pyarrow.parquet as pq

SEEDS = (42, 123, 2024)
E = "experiments/"
ARMS_F = ("frozen", "adaptive", "replay_control")
_GNN3 = ("predictions_val.parquet", "metrics_val.json", "config.json")
_TAB4 = _GNN3 + ("metadata.json",)
SCOPE_MD = "experiments/phase12_validation_evidence_table/SCOPE_AND_ACCEPTANCE.md"
SCOPE_SIDECAR = "experiments/phase12_validation_evidence_table/SCOPE_AND_ACCEPTANCE.sha256"
EXCLUDED_FRAGMENTS = ("gat_300epoch", "experiments/graphsage/", "experiments/graphsage_control_mlp/", "/pilot/")
GROUP_ORDER = ("A", "A_reference_historical_control", "C", "C_control_mlp", "D", "E", "E_controls_7A", "F", "cross_check_only")
EXPECTED_COUNTS = {"A": 12, "A_reference_historical_control": 4, "C": 10, "C_control_mlp": 3, "D": 10, "E": 10, "E_controls_7A": 20, "F": 93,
                   "cross_check_only": 3}


class SourceError(RuntimeError):
    """A read outside the declared source set, or a refused path."""


def declared_groups() -> dict:
    g: dict = {k: [] for k in GROUP_ORDER}
    for d in ("logistic_regression", "random_forest", "xgboost"):
        g["A"] += [f"{E}{d}/{f}" for f in _TAB4]
    g["A_reference_historical_control"] += [f"{E}xgboost_historical_control/{f}" for f in _TAB4]
    for s in SEEDS:
        g["C"] += [f"{E}graphsage_robust/seed_{s}/{f}" for f in _GNN3]
    g["C"].append(f"{E}graphsage_robust/summary.json")
    g["C_control_mlp"] += [f"{E}graphsage_robust/mlp_control_extended/{f}" for f in _GNN3]
    for s in SEEDS:
        g["D"] += [f"{E}gat/seed_{s}/{f}" for f in _GNN3]
    g["D"].append(f"{E}gat/summary.json")
    for s in SEEDS:
        g["E"] += [f"{E}temporal_graphsage/seed_{s}/{f}" for f in _GNN3]
    g["E"].append(f"{E}temporal_graphsage/summary.json")
    g["cross_check_only"].append(f"{E}temporal_graphsage/comparison.json")
    for arm in ("static_schedule_matched", "memory_ablated_temporal"):
        for s in SEEDS:
            g["E_controls_7A"] += [f"{E}temporal_ablation/{arm}/seed_{s}/{f}" for f in _GNN3]
        g["E_controls_7A"].append(f"{E}temporal_ablation/{arm}/summary.json")
    g["cross_check_only"].append(f"{E}temporal_ablation/comparison.json")
    for arm in ARMS_F:
        for s in SEEDS:
            for w in range(1, 6):
                g["F"].append(f"{E}phase8_adaptive_drift/{arm}/seed_{s}/predictions_val_W{w}.parquet")
                g["F"].append(f"{E}phase8_adaptive_drift/{arm}/seed_{s}/metrics_val_W{w}.json")
    g["F"] += [f"{E}phase8_adaptive_drift/shared_init/hparams_seed{s}.json" for s in SEEDS]
    g["cross_check_only"].append(f"{E}phase8_adaptive_drift/comparison.json")
    return g


def declared_paths() -> list:
    paths = [p for v in declared_groups().values() for p in v]
    assert len(paths) == len(set(paths))
    return sorted(paths)


def fingerprint_text(hashes: dict) -> str:
    """Lines '<posix path>\\t<sha256>\\n', sorted by path (scope record, section 3)."""
    return "".join(f"{p}\t{hashes[p]}\n" for p in sorted(hashes))


def fingerprint(hashes: dict) -> str:
    return hashlib.sha256(fingerprint_text(hashes).encode("utf-8")).hexdigest()


class GuardedReader:
    """Every read goes through here; paths outside ``allowed`` (plus ``extra``) raise ``SourceError`` and every open is logged."""

    def __init__(self, root, allowed=None, extra=(SCOPE_MD, SCOPE_SIDECAR)):
        self.root = pathlib.Path(root)
        self.allowed = set(declared_paths() if allowed is None else allowed)
        self.extra = set(extra)
        self.log: list = []

    def _check(self, rel, kind):
        rel = str(rel).replace("\\", "/")
        if rel not in self.allowed and rel not in self.extra:
            raise SourceError(f"not in the declared source set: {rel}")
        low = rel.lower()
        if "test" in low.replace("latest", "") or low.endswith((".pt", ".joblib", ".pth", ".ckpt")) or any(f in low for f in EXCLUDED_FRAGMENTS):
            raise SourceError(f"refused path: {rel}")
        self.log.append((rel, kind))
        return self.root / rel

    def sha256(self, rel) -> str:
        return hashlib.sha256(self._check(rel, "hash").read_bytes()).hexdigest()

    def hash_all(self) -> dict:
        return {p: self.sha256(p) for p in sorted(self.allowed)}

    def read_parquet(self, rel, columns=None) -> pd.DataFrame:
        return pq.read_table(self._check(rel, "parquet"), columns=columns).to_pandas()

    def read_json(self, rel):
        import json
        return json.loads(self._check(rel, "json").read_text(encoding="utf-8"))

    def read_text(self, rel) -> str:
        return self._check(rel, "text").read_text(encoding="utf-8")

    def audit(self) -> dict:
        opened = sorted({p for p, _ in self.log})
        kinds: dict = {}
        for p, k in self.log:
            kinds.setdefault(k, set()).add(p)
        return {"n_opened_unique": len(opened), "opened": opened, "kinds": {k: len(v) for k, v in sorted(kinds.items())},
                "n_open_events": len(self.log)}


# ------------------------------------------------------------------------------------------------ scope record parsing
_FP_RE = re.compile(r"Value at the time of writing:\s*`([0-9a-f]{64})`")


def parse_scope_fingerprint(text: str) -> str:
    m = _FP_RE.search(text)
    if not m:
        raise SourceError("fingerprint not found in the scope record")
    return m.group(1)


def parse_comparability(text: str) -> list:
    """Rows of the section-5 table of the scope record: (comparison, label, reason). Labels are read from the record, not re-typed."""
    start = text.index("## 5. Comparability pre-registration")
    end = text.index("## 6.", start)
    rows = []
    for ln in text[start:end].splitlines():
        if not ln.startswith("|") or set(ln.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] == "Comparison":
            continue
        rows.append((cells[0].replace("**", ""), cells[1].replace("**", ""), cells[2]))
    return rows
