"""Phase 11 output helpers: deterministic UTF-8/LF files with a bare SHA-256 sidecar (project convention)."""
from __future__ import annotations

import hashlib
import json
import pathlib


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_bytes_with_sidecar(path, data: bytes) -> str:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    h = hashlib.sha256(data).hexdigest()
    path.with_suffix(".sha256").write_bytes(h.encode("ascii"))
    return h


def write_text(path, text: str) -> str:
    return write_bytes_with_sidecar(path, text.replace("\r\n", "\n").encode("utf-8"))


def write_json(path, obj) -> str:
    return write_text(path, json.dumps(obj, indent=1, ensure_ascii=True, allow_nan=False) + "\n")


def write_csv(path, df) -> str:
    return write_bytes_with_sidecar(path, df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
