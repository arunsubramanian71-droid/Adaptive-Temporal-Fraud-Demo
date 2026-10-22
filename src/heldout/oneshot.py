"""One-shot rule for the held-out output folders (HELDOUT_EVALUATION_PROTOCOL.md section 7; AMENDMENT_1.md section 7).

A held-out run may be repeated only if it crashed before any held-out score was written. Every file a held-out runner writes into its output folder holds held-out
scores or metrics (prediction files, metric tables, the manifest, the completion flag), so a folder that holds any file at all counts as executed. Folders alone hold
nothing. Only existence is checked here; no file is opened.
"""
from __future__ import annotations

import pathlib

FOLDERS = ("heldout_results", "heldout_results_amendment", "heldout_results_f")        # base protocol, Amendment 1 H/U, Amendment 1 F


class OneShotError(PermissionError):
    """The held-out output folder already holds output; the run is one-shot."""


def held_out_outputs(folder) -> list:
    """Relative paths of every file under ``folder`` (empty if the folder is absent)."""
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def require_unused(folder, what: str) -> None:
    """Raise before anything is read if ``folder`` already holds held-out output."""
    files = held_out_outputs(folder)
    if files:
        raise OneShotError(f"{what}: {pathlib.Path(folder).name}/ already holds {len(files)} held-out output file(s), e.g. {files[0]}. The held-out run is one-shot; "
                           "a repeat is allowed only after a crash that wrote no held-out score (protocol section 7). Nothing was read.")


def folders_with_output(protocol_folder) -> list:
    """Names of the held-out output folders (of ``FOLDERS``) that hold any file."""
    return [f for f in FOLDERS if held_out_outputs(pathlib.Path(protocol_folder) / f)]
