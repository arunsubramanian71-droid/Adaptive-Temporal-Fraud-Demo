"""Torch-free import compatibility for the demonstration.

The frozen Phase 9 modules ``src/coordination/data.py`` and ``graphs.py`` need only one constant from ``src/graph/temporal.py`` (the relation names), but
that file imports PyTorch at the top. On a machine without PyTorch (the core ``requirements.txt`` environment) the demonstration would then fail at import.
This module registers a minimal stand-in for ``src.graph.temporal`` that carries the same ``RELATIONS`` mapping, read from the real file's source without
importing it, and raises a clear ImportError for anything else. It does nothing when PyTorch is installed, and it never modifies a frozen file.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types

TEMPORAL = pathlib.Path(__file__).resolve().parent.parent / "graph" / "temporal.py"


def read_relations() -> dict:
    tree = ast.parse(TEMPORAL.read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "RELATIONS":
            return ast.literal_eval(node.value)
    raise RuntimeError("RELATIONS not found in src/graph/temporal.py")


def make_stub() -> types.ModuleType:
    stub = types.ModuleType("src.graph.temporal")
    stub.RELATIONS = read_relations()
    stub.ENTITY_TYPES = tuple(stub.RELATIONS.values())
    stub.__file__ = str(TEMPORAL)
    stub.__doc__ = "Torch-free stand-in (see src/demo/_compat.py): only RELATIONS and ENTITY_TYPES."

    def _missing(name):
        if name.startswith("__"):                     # import machinery probes dunder names (for example __path__)
            raise AttributeError(name)
        raise ImportError(f"'{name}' needs PyTorch, which is not installed; install requirements-gnn.txt to use the neural-network code")
    stub.__getattr__ = _missing
    return stub


def install_if_needed() -> bool:
    """Register the stand-in when PyTorch is absent and the real module is not already loaded. Returns True if installed."""
    if "src.graph.temporal" in sys.modules or importlib.util.find_spec("torch") is not None:
        return False
    sys.modules["src.graph.temporal"] = make_stub()
    return True


install_if_needed()
