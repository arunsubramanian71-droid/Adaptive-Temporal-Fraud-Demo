"""Phase 13: tests of the local API and dashboard (experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md, criteria P13-1 to P13-8).
HTTP tests run a real server on an ephemeral 127.0.0.1 port. Scoring uses the SYNTHETIC demo bundle; no research data, model or TEST file is touched."""
from __future__ import annotations

import csv
import http.client
import json
import pathlib
import re
import sys
import threading

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.coordination.lint import lint_files, lint_text  # noqa: E402
from src.serving import app as APP  # noqa: E402
from src.serving.demo import build_demo_services  # noqa: E402

_C = {}


def services():
    if "svc" not in _C:
        _C["svc"] = build_demo_services()
    return _C["svc"]


class Running:
    """Context manager: a server on 127.0.0.1:<ephemeral port> in a background thread."""

    def __init__(self, svc=None, module=None):
        self.svc = svc if svc is not None else services()
        self.module = module or APP

    def __enter__(self):
        self.server = self.module.make_server(self.svc, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.t = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()

    def call(self, method, path, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
        h = {"Content-Type": "application/json", **(headers or {})}
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        text = r.read().decode("utf-8")
        c.close()
        ctype = r.getheader("Content-Type", "")
        return r.status, (json.loads(text) if "json" in ctype else text), r


def sample():
    return services().sample


# ------------------------------------------------------------------------------------------------ P13-1 startup and binding
def test_p13_1_binds_locally_only_unless_explicitly_allowed():
    for host in ("0.0.0.0", "192.168.1.5", "example.org"):
        try:
            APP.make_server(services(), host, 0)
        except ValueError as e:
            assert "non-local" in str(e)
        else:
            raise AssertionError(f"bound to {host}")
    s = APP.make_server(services(), "127.0.0.1", 0)
    assert s.server_address[0] == "127.0.0.1"
    s.server_close()


def test_p13_1_launcher_validates_arguments():
    import run_dashboard as RD
    assert RD.main(["--demo", "--bundle", "x"]) == 2
    assert RD.main(["--host", "0.0.0.0", "--port", "0"]) == 2
    assert RD.main(["--bundle", "C:/data/test_bundle"]) == 2           # held-out-named path refused
    assert RD.main(["--bundle", "definitely_missing_bundle_dir"]) == 2


# ------------------------------------------------------------------------------------------------ P13-2 HTTP behaviour
def test_p13_2_status_codes_and_json_errors():
    with Running() as s:
        assert s.call("GET", "/health")[:2] == (200, {"status": "ok"})
        st, body, _ = s.call("GET", "/nope")
        assert st == 404 and body["error"] and "Traceback" not in json.dumps(body)
        assert s.call("GET", "/api/score")[0] == 405 and s.call("POST", "/api/evidence", {})[0] == 405 and s.call("PUT", "/api/score", {})[0] == 405
        assert s.call("DELETE", "/health")[0] == 405 or s.call("DELETE", "/health")[0] == 404
        st, body, _ = s.call("POST", "/api/score", raw=b"{not json")
        assert st == 400 and "JSON" in body["error"]
        for bad in ({}, {"new": []}, {"new": "x"}, {"new": [1, 2]}, {"new": [{}], "history": "z"}):
            st, body, _ = s.call("POST", "/api/score", bad)
            assert st == 400, (bad, st, body)
        st, body, _ = s.call("POST", "/api/score", {"new": [{"TransactionID": 1}]})
        assert st == 400 and "missing required column" in body["error"], body
        st, body, _ = s.call("POST", "/api/score", {"new": [{}] * (APP.MAX_ROWS + 1)})
        assert st == 400 and "at most" in body["error"], (st, body)          # row limit (scope P13-2: more than 5,000 rows -> 400)
        big = b'{"new": [' + b" " * (APP.MAX_BODY + 10) + b"]}"
        assert s.call("POST", "/api/score", raw=big)[0] == 413
        st, _, r = s.call("GET", "/health")
        assert r.getheader("X-Content-Type-Options") == "nosniff" and "no-store" in r.getheader("Cache-Control") and "default-src 'none'" in r.getheader("Content-Security-Policy")


def test_p13_2_internal_errors_are_generic():
    svc = services()
    original = svc.status
    svc.status = lambda: (_ for _ in ()).throw(RuntimeError("secret internals"))
    try:
        with Running(svc) as s:
            st, body, _ = s.call("GET", "/api/status")
        assert st == 500 and body["error"] == "internal error" and "secret" not in json.dumps(body)
    finally:
        svc.status = original


# ------------------------------------------------------------------------------------------------ P13-3 scoring
def test_p13_3_scores_equal_the_library_and_are_causal_and_label_free():
    from src.inference import scorer as S
    import pandas as pd
    svc = services()
    smp = sample()
    with Running() as s:
        st, resp, _ = s.call("POST", "/api/score", {"history": smp["history"], "new": smp["new"]})
        assert st == 200 and resp["synthetic_demo"] is True and "not evidence of fraud" in resp["statement"]
        lib = S.score(svc.bundle, pd.DataFrame(smp["history"]), pd.DataFrame(smp["new"]))
        assert [r["TransactionID"] for r in resp["results"]] == lib["TransactionID"].tolist()
        assert np.array_equal(np.array([r["score"] for r in resp["results"]]), lib["score"].to_numpy())
        assert resp["n"] == len(smp["new"]) and resp["n_alerts"] == sum(r["alert"] for r in resp["results"])
        for r in resp["results"]:
            assert 0.0 <= r["score"] <= 1.0 and (("top_features" in r) == r["alert"])
            assert set(r) <= {"TransactionID", "score", "alert", "top_features"}, set(r)
        st2, extra, _ = s.call("POST", "/api/score", {"new": smp["new"][:3], "path": "C:/definitely/not/a/file.txt", "bundle": "elsewhere"})
        assert st2 == 200 and extra["n"] == 3, "unknown request keys must be ignored, never used as file paths"
        k = len(smp["new"]) // 2
        _, early, _ = s.call("POST", "/api/score", {"history": smp["history"], "new": smp["new"][:k]})
        assert [r["score"] for r in early["results"]] == [r["score"] for r in resp["results"][:k]], "early scores changed when later rows were removed"
        poisoned_new = [{**r, "isFraud": 1 - (i % 2)} for i, r in enumerate(smp["new"])]
        poisoned_hist = [{**r, "isFraud": 1} for r in smp["history"]]
        _, p, _ = s.call("POST", "/api/score", {"history": poisoned_hist, "new": poisoned_new})
        assert [r["score"] for r in p["results"]] == [r["score"] for r in resp["results"]], "posted labels changed the scores"
        _, nohist, _ = s.call("POST", "/api/score", {"new": smp["new"]})
        assert nohist["n"] == len(smp["new"])
        st, body, _ = s.call("POST", "/api/score", {"history": smp["new"], "new": smp["history"][:5]})
        assert st == 400 and "later than the earliest new" in body["error"]


def test_p13_3_model_endpoint_and_missing_bundle():
    with Running() as s:
        st, m, _ = s.call("GET", "/api/model")
        assert st == 200 and m["synthetic_demo"] is True and "TransactionID" in m["required_columns"] and "isFraud" not in m["required_columns"]
    with Running(APP.Services()) as s:
        assert s.call("GET", "/api/model")[0] == 404 and s.call("POST", "/api/score", {"new": [{}]})[0] == 404 and s.call("GET", "/api/sample")[0] == 404
        assert s.call("GET", "/api/status")[1]["scoring_bundle_loaded"] is False


# ------------------------------------------------------------------------------------------------ P13-4 evidence
def test_p13_4_evidence_values_equal_the_recorded_files():
    with Running() as s:
        st, e, _ = s.call("GET", "/api/evidence")
    assert st == 200 and "validation-period" in e["banner"] and "No held-out" in e["held_out"] and "distinct evaluation basis" in e["f_banner"]
    cells = list(csv.DictReader((ROOT / "experiments/phase12_validation_evidence_table/outputs/evidence_cells.csv").read_text(encoding="utf-8").splitlines()))
    assert len(e["phase12_cells"]) == len(cells)
    for api, r in zip(e["phase12_cells"], cells):
        assert api["model"] == r["model"] and api["cell"] == r["cell"] and api["pr_auc"] == (float(r["pr_auc_mean"]) if r["pr_auc_mean"] != "" else None) and api["basis"] == r["basis"]
    assert any(c["cell"] == "B" and c["status"] == "MISSING" for c in e["phase12_cells"]), "the historical B row must stay MISSING in the Phase 12 table"
    b = list(csv.DictReader((ROOT / "experiments/comparison_B_graph_features/outputs/table_with_B.csv").read_text(encoding="utf-8").splitlines()))
    assert [r["model"] for r in e["comparison_b_rows"]] == [r["model"] for r in b]
    assert [r["pr_auc"] for r in e["comparison_b_rows"]] == [float(r["pr_auc"]) for r in b]
    dec = json.loads((ROOT / "experiments/comparison_B_graph_features/outputs/decision.json").read_text(encoding="utf-8"))
    assert e["comparison_b_verdicts"]["H1_direct_B_vs_A"] == dec["H1_direct_B_vs_A"]["verdict"] and e["comparison_b_verdicts"]["H2_E_vs_B_sub"] == dec["H2_E_vs_B_sub"]["verdict"]
    un = json.loads((ROOT / "experiments/imbalance_check_xgboost/outputs/unweighted_decision.json").read_text(encoding="utf-8"))
    assert e["imbalance"]["unweighted_followup"]["H2"] == un["rule3_H2_unweighted_E"]["verdict"]


# ------------------------------------------------------------------------------------------------ P13-5 communities
def test_p13_5_communities_are_aggregate_only_candidates_with_disclaimers():
    with Running() as s:
        st, c, _ = s.call("GET", "/api/communities")
        raw = s.call("GET", "/api/communities")[2]
    assert st == 200 and c["n_cards"] == 24 == len(c["cards"])
    assert c["standing_statement"] == "candidate coordinated community; not a confirmed fraud ring" and c["not_established"] == "This does not establish that IEEE-CIS contains fraud rings."
    text = json.dumps(c)
    assert lint_text(text) == [], lint_text(text)
    forbidden = re.compile(r"member|card_id|identit|TransactionID|isFraud|label", re.I)
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(c["cards"])
    assert not [k for k in keys if forbidden.search(k) and k != "member_transaction_composition_txn_days" and k != "n_member_txn"], keys
    assert all(set(card) <= set(APP.CARD_KEYS) for card in c["cards"])
    src = json.loads((ROOT / "experiments/phase11_explainable_evidence/community_cards/cards_W7_seed42.json").read_text(encoding="utf-8"))
    assert [card["track"] for card in c["cards"]] == [card["track"] for card in src["cards"]]
    assert c["cards"][0]["peak"]["n_cards"] == src["cards"][0]["peak"]["n_cards"]


# ------------------------------------------------------------------------------------------------ P13-6 access control
def test_p13_6_no_path_traversal_no_sealed_names_and_only_listed_files_are_read():
    svc = APP.Services(bundle=services().bundle, synthetic_demo=True, sample=sample())
    with Running(svc) as s:
        for path in ("/../etc/passwd", "/..%2f..%2fconfigs/data.yaml", "/data/features/test_features.parquet", "/test_transaction.csv", "/api/test", "/static/../x", "/%2e%2e/secret",
                     "/experiments/xgboost/model.joblib", "/src/serving/app.py", "/configs/data.yaml"):
            assert s.call("GET", path)[0] == 404, path
        for p in ("/", "/api/evidence", "/api/communities", "/api/status", "/api/model", "/api/sample", "/health"):
            assert s.call("GET", p)[0] == 200, p
        s.call("POST", "/api/score", {"new": sample()["new"][:3]})
    allowed = set(APP.READABLE.values())
    assert set(svc.opened) <= allowed, set(svc.opened) - allowed
    assert not any("test" in o.lower().replace("latest", "") for o in svc.opened)
    assert not any(o.endswith((".pt", ".joblib", ".parquet", ".npz")) for o in svc.opened)


def test_p13_6_status_reports_heldout_artifacts_by_existence_only():
    with Running() as s:
        st = s.call("GET", "/api/status")[1]
    assert st["validation_period_only"] is True and st["held_out_evaluation_performed"] is False and st["held_out_authorization_present"] is False
    assert "BLOCKED" in st["phase12"] and "not established" in st["h5"] and "candidates only" in st["communities"]


def test_p13_6_status_and_note_follow_held_out_output_in_any_of_the_three_folders():
    """The note 'no held-out evaluation has been performed' must not outlive an authorized run: any file in any held-out output folder (base, Amendment 1 H/U, or F)
    turns the flag on and replaces the note. Existence only; nothing is read."""
    import shutil
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp(prefix="srv_"))
    try:
        svc = APP.Services(root=d)
        st = svc.status()
        assert st["held_out_evaluation_performed"] is False and st["held_out_note"] == APP.HELD_OUT_NOTE
        for folder in ("heldout_results", "heldout_results_amendment", "heldout_results_f"):
            p = d / "experiments" / "final_heldout_protocol" / folder
            p.mkdir(parents=True)
            assert svc.status()["held_out_evaluation_performed"] is False, folder        # an empty folder (crash before any score) holds no result
            (p / "x.parquet").write_bytes(b"x")
            st = svc.status()
            assert st["held_out_evaluation_performed"] is True and st["held_out_note"] != APP.HELD_OUT_NOTE, folder
            assert "not shown" in st["held_out_note"] and folder in st["held_out_note"], st["held_out_note"]
            shutil.rmtree(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ P13-7 dashboard page
def test_p13_7_dashboard_is_self_contained_safe_and_properly_worded():
    with Running() as s:
        st, page, r = s.call("GET", "/")
        assert st == 200 and r.getheader("Content-Type").startswith("text/html")
    assert "Fraud study dashboard" in page and "role=\"note\"" in page
    assert not re.search(r"(src|href)\s*=\s*[\"']https?://|@import|url\(\s*[\"']?https?:", page), "external resource reference"
    assert "innerHTML" not in page and "outerHTML" not in page and "document.write" not in page and "eval(" not in page, "unsafe DOM sink"
    assert page.count("textContent") >= 3 and "createTextNode" in page
    assert lint_text(page) == []
    for needle in ("validation", "candidates only", "not evidence of fraud", "SYNTHETIC"):
        assert needle in page, needle
    assert "api/evidence" in page and "api/communities" in page and "api/score" in page


def test_p13_8_wording_lint_on_all_serving_sources_and_docs():
    files = [ROOT / "src/serving/app.py", ROOT / "src/serving/demo.py", ROOT / "src/serving/static/dashboard.html", ROOT / "scripts/run_dashboard.py", ROOT / "tests/test_serving.py",
             ROOT / "experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md"]
    bad = {str(k): v for k, v in lint_files(files).items() if v}
    assert bad == {}, bad


def test_p13_8_json_endpoints_are_deterministic():
    with Running() as s:
        for p in ("/api/evidence", "/api/communities", "/api/status", "/api/model"):
            a = s.call("GET", p)[1]
            b = s.call("GET", p)[1]
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), p


def test_p13_10_no_new_dependency_is_imported():
    stdlib_or_project = {"csv", "http", "json", "math", "pathlib", "socketserver", "threading", "traceback", "numpy", "pandas", "__future__", "src"}
    for f in (ROOT / "src/serving/app.py", ROOT / "src/serving/demo.py", ROOT / "scripts/run_dashboard.py"):
        mods = set(re.findall(r"^(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", f.read_text(encoding="utf-8"), re.M)) - {"argparse", "sys"}
        assert mods <= stdlib_or_project, (f.name, mods - stdlib_or_project)
    assert not any(n in (ROOT / "requirements.txt").read_text(encoding="utf-8").lower() for n in ("fastapi", "streamlit", "flask", "uvicorn"))


# ------------------------------------------------------------------------------------------------ found in the 2026-10-22 browser and API pass
def test_invalid_identifiers_amounts_and_nesting_get_400_not_500():
    """Before 2026-10-22: a text or boolean TransactionID and a deeply nested body returned 500 'internal error'; a fractional id was reported back rounded;
    a NaN amount was accepted although it corrupts the amount features of later rows of the same entity."""
    new = sample()["new"][:3]
    with Running() as s:
        for bad_id in ("abc", 1.5, True, None):
            st, j, _ = s.call("POST", "/api/score", {"new": [dict(new[0], TransactionID=bad_id)] + new[1:]})
            assert st == 400 and "TransactionID must be integers" in j["error"], (bad_id, st, j)
        st, j, _ = s.call("POST", "/api/score", raw=json.dumps({"new": new}).replace('"TransactionAmt": ', '"TransactionAmt": NaN, "x": ', 1).encode())
        assert st == 400 and "TransactionAmt must be finite" in j["error"], j
        st, j, _ = s.call("POST", "/api/score", raw=b"[" * 100000 + b"]" * 100000)
        assert st == 400 and "nested too deeply" in j["error"], j
        st, j, _ = s.call("POST", "/api/score", {"new": new})
        assert st == 200 and [r["TransactionID"] for r in j["results"]] == [r["TransactionID"] for r in new]


def test_a_second_server_on_a_busy_port_is_refused_and_a_restart_on_that_port_works():
    """Before 2026-10-22, on Windows (SO_REUSEADDR) a second dashboard on the same port started without error and took over the port's connections."""
    with Running() as s:
        assert s.call("GET", "/health")[0] == 200
        try:
            dup = APP.make_server(services(), "127.0.0.1", s.port)
        except OSError:
            dup = None
        if dup is not None:
            dup.server_close()
            raise AssertionError("a second server bound a port that is already listening")
        port = s.port
    again = APP.make_server(services(), "127.0.0.1", port)                    # restart right after shutdown (connections in TIME_WAIT)
    again.server_close()


def test_status_reports_both_h5_readings():
    st = services().status()
    assert "not established" in st["h5"] and "Phase 9b" in st["h5"] and "H5a-9b" in st["h5"]


def _css_vars(page: str, dark: bool) -> dict:
    def parse(block):
        return {k: (v if len(v) == 7 else "#" + "".join(ch * 2 for ch in v[1:])) for k, v in re.findall(r"--([a-z]+):(#[0-9a-f]{6}|#[0-9a-f]{3})\b", block)}
    light = parse(re.search(r":root\{([^}]*)\}", page).group(1))
    if not dark:
        return light
    return {**light, **parse(re.search(r"prefers-color-scheme:dark\)\{:root\{([^}]*)\}", page).group(1))}


def _contrast(a: str, b: str) -> float:
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    la, lb = lum(a), lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_page_colours_meet_wcag_aa_contrast_in_both_schemes():
    """Before 2026-10-22 the dark scheme put white text on the light-blue accent (2.5:1) on the primary buttons."""
    page = (ROOT / APP.DASHBOARD).read_text(encoding="utf-8")
    assert "button.act{background:var(--accent);color:var(--onaccent)" in page
    for dark in (False, True):
        v = _css_vars(page, dark)
        for fg, bg in (("ink", "bg"), ("ink", "panel"), ("mute", "panel"), ("mute", "bg"), ("warn", "warnbg"), ("bad", "panel"), ("accent", "panel"), ("onaccent", "accent")):
            assert _contrast(v[fg], v[bg]) >= 4.5, (("dark" if dark else "light"), fg, bg, round(_contrast(v[fg], v[bg]), 2))


def test_page_tabs_live_regions_busy_guard_and_failure_banner():
    """Accessibility and interaction fixes verified in a real browser by scripts/verify_dashboard_browser.py; these source checks keep them from regressing."""
    page = (ROOT / APP.DASHBOARD).read_text(encoding="utf-8")
    for k in ("evidence", "score", "research", "communities", "status"):
        assert f'<section id="tab-{k}" role="tabpanel" aria-labelledby="btn-{k}"' in page, k
    assert '"aria-controls": "tab-" + k' in page and "ArrowRight" in page and "Home" in page and "End" in page
    for k in ("scoreInfo", "scoreOut", "researchInfo", "researchOut"):
        assert re.search(rf'id="{k}"[^>]*aria-live="polite"', page), k
    assert ":focus-visible{outline:2px solid var(--accent)" in page
    assert "async function busy(" in page and page.count("await busy(") == 3                 # double clicks send one request (sample, score, research)
    assert "The standing statement could not be loaded" in page                              # the banner never stays on 'Loading' after a failure
    assert '"Loading the synthetic sample..."' in page                                       # no stale 'Loaded ...' text while a new sample loads


def test_recorded_browser_run_passed_everywhere_and_publishes_no_validation_row():
    """scripts/verify_dashboard_browser.py (real Chrome, two viewports, two colour schemes) wrote this record; its screenshots are published, so the research tab
    is photographed with synthetic placeholder rows and no IEEE-CIS transaction id (2987000-3577539) may appear in the record."""
    import hashlib
    d = ROOT / "experiments" / "phase13_serving_dashboard" / "test_results" / "browser"
    data = (d / "browser_verification.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == (d / "browser_verification.sha256").read_text(encoding="ascii").strip()
    rec = json.loads(data)
    assert rec["summary"]["failed"] == 0 and rec["summary"]["not_run"] == 0 and rec["summary"]["passed"] == rec["summary"]["checks"] >= 100, rec["summary"]
    assert set(rec["viewports"]) == {"desktop", "phone"} and rec["colour_schemes"] == ["light", "dark"] and "Playwright" in rec["browser"]
    assert len(rec["screenshots_sha256"]) == 25
    for name, h in rec["screenshots_sha256"].items():
        assert hashlib.sha256((d / "screenshots" / name).read_bytes()).hexdigest() == h, name
    assert "synthetic placeholder rows" in rec["research_tab_screenshots"] and rec["research_rows_file_used"] is True
    assert re.findall(r"(?<![0-9a-f])(29[89]\d{4}|3[0-5]\d{5})(?![0-9a-f])", data.decode("utf-8")) == []
    harness = (ROOT / "scripts" / "verify_dashboard_browser.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+src\b", harness, re.M)          # runs in a separate environment; drives the documented server command
    assert 'placeholder = out / "_placeholder"' in harness and "page.set_input_files(\"#researchFile\", str(placeholder))" in harness


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
