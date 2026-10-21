"""Content fingerprint of the project code a readiness gate exercised (protocol section 6: the held-out run must use the code that the validation gate validated).

A readiness record stores the SHA-256 of every project file under ``src/`` and ``scripts/`` that was loaded when the gate ran. Before any held-out read, the runner recomputes those
hashes and refuses if any file changed or is missing. Content hashes (not file times) are used, so a fresh clone, whose checkout sets file times in arbitrary order, is judged by
the same rule as the working copy; ``.gitattributes`` keeps every file byte-for-byte as committed.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CODE_DIRS = ("src", "scripts")


def file_sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def loaded_project_code(root: pathlib.Path = ROOT) -> dict:
    """{relative path: SHA-256} of every loaded module file under src/ or scripts/ (the running script included)."""
    files = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = pathlib.Path(f).resolve()
        try:
            rel = p.relative_to(root.resolve())
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in CODE_DIRS and p.suffix == ".py":
            files.add(rel.as_posix())
    return {f: file_sha256(root / f) for f in sorted(files)}


def changed_code(recorded: dict, root: pathlib.Path = ROOT) -> list:
    """Files of a recorded fingerprint that are now missing or have different content."""
    return sorted(f for f, h in recorded.items() if not (root / f).is_file() or file_sha256(root / f) != h)
