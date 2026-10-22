"""The root README and the handoff guides only name files that exist (or are documented withheld artifacts), their table of contents works, and they
keep the project's wording rule. Added 2026-10-22 with the root README, the collaborator guide and the artifact inventory. No data is read."""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files  # noqa: E402

DOCS = ["README.md", "submission/COLLABORATOR_GUIDE.md", "submission/ARTIFACT_INVENTORY.md", "submission/PROFESSOR_DEFENSE_GUIDE.md", "submission/DEMO_GUIDE.md"]
MUST_NOT_EXIST = {"experiments/final_heldout_protocol/AUTHORIZATION.txt"}           # named only to say that it must not be created
PATH = re.compile(r"`((?:src|scripts|tests|experiments|submission|reports|configs|data)/[^`\s]*|requirements[^`\s]*\.txt|README\.md)`")


def _gitignore_match(rel: str) -> bool:
    """Fallback for a copy without .git (GitHub's "Download ZIP"): the plain patterns of .gitignore, matched with fnmatch (no negations are needed here)."""
    import fnmatch
    parts = rel.split("/")
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        pat = line.strip()
        if not pat or pat.startswith(("#", "!")):
            continue
        pat = pat.rstrip("/").lstrip("/")
        if "/" in pat:                                                   # path pattern (fnmatch's * also crosses '/', which covers **)
            if fnmatch.fnmatch(rel, pat) or rel.startswith(pat + "/"):
                return True
        elif any(fnmatch.fnmatch(p, pat) for p in parts):                 # a bare name matches any path component
            return True
    return False


def _ignored(rel: str) -> bool:
    """A withheld artifact: git ignores it (.gitignore or .git/info/exclude); without .git, by the .gitignore patterns."""
    if not (ROOT / ".git").exists():
        return _gitignore_match(rel)
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=ROOT).returncode == 0


def referenced_paths(text: str) -> list:
    out = []
    for m in PATH.finditer(text):
        p = m.group(1).rstrip(".,;:)")
        if any(ch in p for ch in "*{<$") or p.endswith("/"):         # glob, brace or placeholder patterns, folders written with a trailing slash
            continue
        out.append(p)
    return sorted(set(out))


def test_every_named_file_exists_or_is_a_documented_withheld_artifact():
    missing = {}
    for doc in DOCS:
        for p in referenced_paths((ROOT / doc).read_text(encoding="utf-8")):
            if p in MUST_NOT_EXIST:
                assert not (ROOT / p).exists(), p
                continue
            if not (ROOT / p).exists() and not _ignored(p):
                missing.setdefault(doc, []).append(p)
    assert missing == {}, missing


def test_root_readme_table_of_contents_links_match_its_headings():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    def slug(h):                                                        # GitHub's anchor rule: lower case, drop punctuation, spaces to hyphens
        return re.sub(r"\s", "-", re.sub(r"[^\w\- ]", "", h.strip().lower()))
    anchors = {slug(h) for h in re.findall(r"^#{2,3} (.+)$", text, re.M)}
    links = re.findall(r"\]\(#([^)]+)\)", text)
    assert links and all(l in anchors for l in links), sorted(set(links) - anchors)


def test_handoff_documents_keep_the_wording_rule_and_state_the_sealed_period():
    bad = {k: v for k, v in lint_files([ROOT / d for d in DOCS]).items() if v}
    assert bad == {}, bad
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "validation-period" in readme and "no model has been scored on it" in readme and "candidate" in readme
    assert "AUTHORIZATION.txt" in readme and "--split heldout" in readme                     # tells the reader what never to run


def test_the_quickstart_script_never_touches_the_held_out_period_or_deletes_files():
    ps = (ROOT / "scripts" / "quickstart_windows.ps1").read_text(encoding="utf-8")
    code = "\n".join(l for l in ps.splitlines() if not l.lstrip().startswith("#") and "<#" not in l)
    body = code[code.index("param("):]
    for forbidden in ("heldout", "AUTHORIZATION", "test_features", "Remove-Item", "rm ", "del ", "Invoke-WebRequest", "kaggle"):
        assert forbidden.lower() not in body.lower(), forbidden
    for step in ("requirements.txt", "check_setup.py", "run_core_tests.py", "demo_end_to_end.py"):
        assert step in body, step


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print("[PASS]", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print("[FAIL]", fn.__name__, repr(e))
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
