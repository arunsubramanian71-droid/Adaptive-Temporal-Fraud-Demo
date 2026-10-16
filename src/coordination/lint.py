"""Terminology lint (design section 17): real Phase 9 outputs never call anything a detected/confirmed fraud ring, criminal network,
gang, syndicate, mule ring/network, organised fraud. Exactly two disclaimers are whitelisted. Synthetic constructs never use "ring"
in identifiers or file names (they are "planted coordinated groups" / "decoy groups").
"""
from __future__ import annotations

import pathlib
import re

WHITELIST = (
    "candidate coordinated community; not a confirmed fraud ring",
    "this does not establish that ieee-cis contains fraud rings",
)
PATTERNS = {
    "fraud_ring": r"fraud[\s_-]+rings?",
    "ring_members": r"ring[\s_-]+members?",
    "criminal_network": r"criminal[\s_-]+network",
    "gang": r"\bgangs?\b",
    "syndicate": r"syndicate",
    "mule": r"\bmule[\s_-]+(ring|network)s?",
    "organised_fraud": r"organi[sz]ed[\s_-]+(fraud|crime)",
}
IDENT_RING = re.compile(r"\b[a-z0-9_]*ring_[a-z0-9_]+\b|\b[a-z0-9_]+_rings?\b", re.I)
IDENT_ALLOWED = re.compile(r"(rewiring|scoring|string|during|ensuring|clustering|filtering|offering|hiring|monitoring|"
                           r"wiring|engineering|numbering|ordering|triggering|referring|appending|stirring|spring|bearing|steering|"
                           r"tuning|caring|sharing|pairing|declaring|preparing|comparing|covering|transferring|recurring|occurring|"
                           r"inferring|requiring|measuring|centering|entering|gathering|rendering|rolling|loading|nodes|ties|"
                           r"coloring|boring)", re.I)


def lint_text(text: str, is_design_doc: bool = False) -> list[str]:
    t = text.lower()
    for w in WHITELIST:
        t = t.replace(w, " ")
    bad = []
    for name, pat in PATTERNS.items():
        for m in re.finditer(pat, t):
            bad.append(f"{name}: ...{t[max(0, m.start() - 30):m.end() + 30]!r}...")
    return bad


def lint_identifiers(text: str) -> list[str]:
    """Flags code identifiers/file names that use 'ring' as a word component (e.g. ring_id, group_ring)."""
    bad = []
    for m in re.finditer(r"\b[A-Za-z0-9_]*(?:^|_)rings?(?:_[A-Za-z0-9_]+)?\b|\b[A-Za-z0-9_]+_rings?\b", text):
        s = m.group(0)
        if "ring" in s.lower() and not IDENT_ALLOWED.search(s):
            bad.append(s)
    return bad


def lint_files(paths) -> dict:
    out = {}
    for p in paths:
        p = pathlib.Path(p)
        txt = p.read_text(encoding="utf-8", errors="ignore")
        v = lint_text(txt)
        if p.suffix == ".py":
            v += ["identifier: " + s for s in lint_identifiers(txt)]
        v += ["filename: " + p.name] if re.search(r"(^|[_\W])rings?([_\W]|$)", p.stem, re.I) else []
        out[str(p)] = v
    return out
