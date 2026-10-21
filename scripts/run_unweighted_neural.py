"""Unweighted-loss runs of GraphSAGE (C) and Temporal GraphSAGE (E): experiments/imbalance_check_xgboost/PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md.

Reuses the existing training scripts unmodified. Only the loss weighting is overridden: the scripts' binary cross-entropy is replaced, for the
duration of the run, by the same function without ``pos_weight`` (effective positive-class weight 1.0). Outputs go to new folders; no existing
experiment folder is written. Days <= 151 only (the training scripts never load later days for scoring).

Usage:  .venv/Scripts/python.exe -B scripts/run_unweighted_neural.py --family static|temporal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

PREREG = ROOT / "experiments" / "imbalance_check_xgboost" / "PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md"
FAMILIES = {"static": ("train_graphsage_robustness", "unweighted_graphsage"), "temporal": ("train_temporal_graphsage", "unweighted_temporal_graphsage")}


class _UnweightedF:
    """Stand-in for torch.nn.functional inside a training module: every function is the original except the weighted cross-entropy."""

    def __getattr__(self, name):
        return getattr(torch.nn.functional, name)

    @staticmethod
    def binary_cross_entropy_with_logits(logits, target, pos_weight=None, **kw):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, **kw)     # pos_weight deliberately dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    a = ap.parse_args()
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == PREREG.with_suffix(".sha256").read_text().strip(), "pre-registration does not match its sidecar"
    mod_name, exp_name = FAMILIES[a.family]
    mod = __import__(mod_name)
    out = ROOT / "experiments" / exp_name
    if out.exists() and any(out.iterdir()):
        print(f"error: {out} is not empty; refusing to overwrite", file=sys.stderr)
        return 2
    mod.EXPERIMENT_NAME = exp_name
    mod.F = _UnweightedF()
    assert mod.F.binary_cross_entropy_with_logits(torch.zeros(3), torch.ones(3), pos_weight=torch.tensor(27.0)).item() == torch.nn.functional.binary_cross_entropy_with_logits(torch.zeros(3), torch.ones(3)).item()
    if a.family == "temporal":
        sys.argv = [mod_name + ".py"]            # the script's own argument parser must see no flags (no --pilot)
    mod.main()
    for cfg_path in sorted(out.glob("seed_*/config.json")):
        c = json.loads(cfg_path.read_text(encoding="utf-8"))
        c["scale_pos_weight_computed_but_overridden"] = c.pop("scale_pos_weight")
        c["effective_pos_weight"] = 1.0
        c["note"] = "unweighted follow-up (PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md): the training script's binary cross-entropy was run without pos_weight"
        cfg_path.write_text(json.dumps(c, indent=1), encoding="utf-8")
    (out / "UNWEIGHTED_RUN.json").write_text(json.dumps({"preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(), "family": a.family,
                                                         "effective_pos_weight": 1.0, "base_script": f"scripts/{mod_name}.py"}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
