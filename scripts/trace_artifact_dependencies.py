"""Trace which untracked (withheld) project files each named test opens.

Runs the given tests (module::test) in the full project directory, where the withheld artifacts exist, with an audit hook that records every file the test opens under the
project root. Files that are not tracked by git are the test's artifact dependencies. This only observes: no file is modified and no test is changed. Tests that check existence
only (without opening) are not seen by this tool and are declared by hand in tests/artifact_requirements.json.

Usage:  .venv/Scripts/python.exe -B scripts/trace_artifact_dependencies.py tests.test_x::test_y [...]   (prints JSON)
"""
from __future__ import annotations

import importlib
import io
import json
import pathlib
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OPENED: set = set()
ACTIVE = [False]


def hook(event, args):
    if ACTIVE[0] and event == "open" and isinstance(args[0], (str, bytes, pathlib.Path)):
        try:
            OPENED.add(str(pathlib.Path(args[0] if not isinstance(args[0], bytes) else args[0].decode()).resolve()))
        except Exception:  # noqa: BLE001
            pass


def main(argv) -> int:
    tracked = set(subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True).stdout.split("\0"))
    sys.addaudithook(hook)
    out = {}
    for spec in argv:
        module, test = spec.split("::")
        mod = importlib.import_module(module)
        OPENED.clear()
        ACTIVE[0] = True
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                getattr(mod, test)()
            status = "pass"
        except BaseException as e:  # noqa: BLE001
            status = f"fail: {type(e).__name__}"
        ACTIVE[0] = False
        rel = []
        for p in OPENED:
            try:
                r = pathlib.Path(p).relative_to(ROOT).as_posix()
            except ValueError:
                continue
            if r.startswith((".venv/", ".git/")) or "__pycache__" in r or r.endswith(".pyc") or not pathlib.Path(p).is_file():
                continue
            if r not in tracked:
                rel.append(r)
        withheld = sorted(set(rel))
        out[spec] = {"status_in_full_directory": status, "withheld_files_opened": withheld}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
