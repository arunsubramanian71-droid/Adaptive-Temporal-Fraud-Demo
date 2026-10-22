"""Real-browser verification of the Phase 13 dashboard (optional; needs Playwright and an installed Chrome or Edge).

  python -m pip install playwright==1.64.0          # in any environment; no browser download is needed with --channel chrome/msedge
  python scripts/verify_dashboard_browser.py --research-rows PATH [--channel chrome] [--out-dir DIR]

The script starts the documented server command (`scripts/run_dashboard.py`, with the project interpreter `.venv/Scripts/python.exe` by default) in
separate processes, drives the real page in headless Chrome, and records every check, browser console error, failed request and screenshot.
It imports nothing from the project, so the project's own environment is not changed. PATH is a JSON file of validation rows written by
`scripts/make_research_sample.py` (outside the repository: transaction-level data are never committed); without it the research-scoring checks
are reported NOT RUN. Viewports: desktop 1366x900 and phone 390x844, light and dark colour schemes. Exit code 0 only if every executed check passed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "experiments" / "phase13_serving_dashboard" / "test_results" / "browser"
VIEWPORTS = {"desktop": {"width": 1366, "height": 900}, "phone": {"width": 390, "height": 844}}
TABS = ("evidence", "score", "research", "communities", "status")

CONTRAST_JS = """
(sel) => {
  const lum = (c) => { const m = c.match(/[\\d.]+/g).map(Number); const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
                       return 0.2126 * f(m[0]) + 0.7152 * f(m[1]) + 0.0722 * f(m[2]); };
  const bgOf = (e) => { for (let n = e; n; n = n.parentElement) { const b = getComputedStyle(n).backgroundColor; const m = b.match(/[\\d.]+/g);
                        if (m && (m.length < 4 || Number(m[3]) > 0)) return b; } return "rgb(255, 255, 255)"; };
  const out = [];
  document.querySelectorAll(sel).forEach((e) => { if (!e.offsetParent && e.tagName !== "BODY") return; const fg = getComputedStyle(e).color, bg = bgOf(e);
    const a = lum(fg), b = lum(bg); out.push({ text: (e.textContent || "").trim().slice(0, 40), fg, bg, ratio: (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05) }); });
  return out;
}
"""


class Server:
    def __init__(self, python: str, args: list, port: int = 0):
        self.cmd = [python, "-u", "-B", "scripts/run_dashboard.py", "--port", str(port), *args]
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.log = []
        t0 = time.time()
        while time.time() - t0 < 180:
            line = self.proc.stdout.readline()
            if line:
                self.log.append(line.rstrip())
            m = re.search(r"http://127\.0\.0\.1:(\d+)/", line or "")
            if m:
                self.port = int(m.group(1))
                self.url = f"http://127.0.0.1:{self.port}/"
                return
            if self.proc.poll() is not None:
                raise RuntimeError("server exited before it listened: " + " | ".join(self.log[-5:]))
        raise RuntimeError("server did not report a port within 180 s")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        rest = self.proc.stdout.read() if self.proc.stdout else ""
        self.log += [l for l in rest.splitlines() if l.strip()]
        return self.proc.returncode


class Recorder:
    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail=""):
        self.checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)[:400]})
        print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  ({str(detail)[:160]})" if detail and not ok else ""), flush=True)

    def not_run(self, name, why):
        self.checks.append({"check": name, "result": "NOT RUN", "detail": why})
        print(f"[NOT RUN] {name}: {why}", flush=True)


def watch(page, sink):
    page.on("console", lambda m: sink["console"].append({"type": m.type, "text": m.text[:300]}) if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: sink["pageerrors"].append(str(e)[:300]))
    page.on("requestfailed", lambda r: sink["requestfailed"].append({"url": r.url, "failure": r.failure}))
    page.on("response", lambda r: sink["responses"].append({"url": r.url, "status": r.status, "method": r.request.method}))


def ready(page):
    page.wait_for_selector("#evidence table", timeout=30000)
    page.wait_for_function("() => !document.getElementById('banner').textContent.startsWith('Loading')", timeout=30000)
    page.wait_for_selector("#communities .card", state="attached", timeout=30000)


def no_hscroll(page):
    return page.evaluate("() => ({doc: document.documentElement.scrollWidth, view: window.innerWidth})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-rows", default=None)
    ap.add_argument("--channel", default="chrome", choices=["chrome", "msedge", "chromium"])
    ap.add_argument("--python", default=str(ROOT / ".venv" / "Scripts" / "python.exe"), help="interpreter that runs the server (the project environment)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--full-page", action="store_true", help="full-page screenshots (for review; larger files). Default: what the viewport shows")
    a = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    out = pathlib.Path(a.out_dir)
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    R = Recorder()
    sink = {"console": [], "pageerrors": [], "requestfailed": [], "responses": []}
    rows_path = pathlib.Path(a.research_rows) if a.research_rows else None
    research_available = (ROOT / "experiments" / "xgboost" / "model.joblib").exists()
    t_start = time.time()
    servers = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=None if a.channel == "chromium" else a.channel, headless=True)
        version = browser.version
        srv = Server(a.python, ["--demo"] + (["--research-model"] if research_available else []))
        servers.append(srv)
        placeholder = None
        if research_available:
            # published screenshots never show IEEE-CIS rows (competition rule 7B): the research tab is photographed with synthetic placeholder rows
            # (identifiers 900000001..., every feature missing except a seeded amount); the real validation rows are used only for the functional checks
            import random
            import urllib.request
            with urllib.request.urlopen(srv.url + "api/research/model", timeout=60) as r:
                cols = json.loads(r.read())["required_input_columns"]
            rng = random.Random(7)
            placeholder = out / "_placeholder" / "synthetic_placeholder_rows.json"
            placeholder.parent.mkdir(parents=True, exist_ok=True)
            placeholder.write_text(json.dumps({"rows": [{**{c: None for c in cols}, "TransactionID": 900000001 + i, "TransactionAmt": round(rng.uniform(5, 500), 2)}
                                                        for i in range(12)]}), encoding="utf-8")
        try:
            # ---------------------------------------------------------------- layout and screenshots: 2 viewports x 2 colour schemes
            for vp_name, vp in VIEWPORTS.items():
                for scheme in ("light", "dark"):
                    ctx = browser.new_context(viewport=vp, color_scheme=scheme, device_scale_factor=1, is_mobile=(vp_name == "phone"), has_touch=(vp_name == "phone"))
                    page = ctx.new_page()
                    watch(page, sink)
                    page.goto(srv.url)
                    ready(page)
                    tag = f"{vp_name}_{scheme}"
                    for t in TABS:
                        page.click(f"#btn-{t}")
                        if t == "score":
                            page.click("#loadSample")
                            page.wait_for_function("() => document.getElementById('scoreInfo').textContent.startsWith('Loaded')", timeout=30000)
                            page.click("#doScore")
                            page.wait_for_selector("#scoreOut table", timeout=30000)
                        if t == "research" and placeholder is not None:
                            page.set_input_files("#researchFile", str(placeholder))
                            page.wait_for_function("() => document.getElementById('researchInfo').textContent.startsWith('Loaded')", timeout=30000)
                            page.click("#doResearch")
                            page.wait_for_selector("#researchOut table", timeout=60000)
                        dims = no_hscroll(page)
                        R.check(f"{tag}: tab '{t}' has no horizontal page scroll", dims["doc"] <= dims["view"], dims)
                        hints = page.evaluate("() => [...document.querySelectorAll('section.on .tablewrap')].map(w => [w.previousElementSibling.hidden, w.scrollWidth, w.clientWidth])")
                        if hints:
                            R.check(f"{tag}: tab '{t}' shows the sideways-scroll hint exactly under the tables wider than their frame",
                                    all(hidden == (sw <= cw + 1) for hidden, sw, cw in hints), hints)
                        page.screenshot(path=str(shots / f"{tag}_{t}.png"), full_page=a.full_page)
                    for sel, label, minimum in (("#banner", "banner text", 4.5), ("nav button", "tab labels", 4.5), ("button.act", "primary buttons", 4.5),
                                                ("button.sec", "secondary buttons", 4.5), (".muted", "muted text", 4.5), ("td", "table cells", 4.5), (".alert", "alert text", 4.5)):
                        for t in TABS:
                            page.click(f"#btn-{t}")
                            vals = page.evaluate(CONTRAST_JS, f"section.on {sel}" if sel not in ("#banner", "nav button") else sel)
                            worst = min((v["ratio"] for v in vals), default=None)
                            if worst is not None:
                                R.check(f"{tag}: contrast of {label} on tab '{t}' >= {minimum}:1", worst >= minimum, f"worst {worst:.2f} {[v for v in vals if v['ratio'] == worst][0]}")
                            if sel in ("#banner", "nav button"):
                                break
                    ctx.close()

            # ---------------------------------------------------------------- functional checks, desktop light
            ctx = browser.new_context(viewport=VIEWPORTS["desktop"])
            page = ctx.new_page()
            watch(page, sink)
            page.goto(srv.url)
            ready(page)
            ev = page.evaluate("() => fetch('/api/evidence').then(r => r.json())")
            R.check("evidence: one table row per recorded Phase 12 cell", page.locator("#evidence table").first.locator("tbody tr").count() == len(ev["phase12_cells"]),
                    page.locator("#evidence table").first.locator("tbody tr").count())
            R.check("evidence: banner carries the standing statement and the held-out note", ev["banner"] in page.inner_text("#banner") and ev["held_out"] in page.inner_text("#banner"))
            for t in TABS:
                page.click(f"#btn-{t}")
                vis = [x for x in TABS if page.is_visible(f"#tab-{x}")]
                sel = [x for x in TABS if page.get_attribute(f"#btn-{x}", "aria-selected") == "true"]
                R.check(f"navigation: clicking '{t}' shows only its section and marks only its tab selected", vis == [t] and sel == [t], f"visible {vis} selected {sel}")
            # keyboard: Tab reaches a tab button, focus is visible, arrow keys move between tabs
            page.goto(srv.url)
            ready(page)
            page.keyboard.press("Tab")                                             # keyboard focus, as a user would reach the tabs
            focus = page.evaluate("() => { const e = document.activeElement; const s = getComputedStyle(e); return {id: e.id, outline: s.outlineStyle + ' ' + s.outlineWidth, shadow: s.boxShadow}; }")
            R.check("keyboard: Tab reaches the selected tab and shows a visible focus indicator", focus["id"] == "btn-evidence"
                    and (focus["outline"].split()[0] not in ("none", "") or focus["shadow"] != "none"), focus)
            page.screenshot(path=str(shots / "keyboard_focus_on_tab.png"))
            page.keyboard.press("ArrowRight")
            moved = page.evaluate("() => document.activeElement.id")
            R.check("keyboard: ArrowRight moves to and selects the next tab (WAI-ARIA tabs pattern)", moved == "btn-score" and page.is_visible("#tab-score"), moved)
            page.keyboard.press("End")
            end = page.evaluate("() => document.activeElement.id")
            page.keyboard.press("ArrowRight")
            wrap = page.evaluate("() => document.activeElement.id")
            R.check("keyboard: End selects the last tab and ArrowRight wraps to the first", end == "btn-status" and wrap == "btn-evidence" and page.is_visible("#tab-evidence"), (end, wrap))
            roles = page.evaluate("() => [...document.querySelectorAll('[role=tab]')].map(b => [b.getAttribute('aria-controls'), (document.getElementById(b.getAttribute('aria-controls')) || {}).getAttribute ? document.getElementById(b.getAttribute('aria-controls')).getAttribute('role') : null])")
            R.check("accessibility: every tab names its panel (aria-controls) and the panel has role=tabpanel", all(c and r == "tabpanel" for c, r in roles), roles)
            live = page.evaluate("() => ['scoreInfo','scoreOut','researchInfo','researchOut'].map(i => document.getElementById(i).getAttribute('aria-live'))")
            R.check("accessibility: scoring messages and results are announced (aria-live)", all(v in ("polite", "assertive") for v in live), live)

            # score tab
            page.click("#btn-score")
            page.fill("#payload", '{"history": [], "new": []}')
            page.click("#doScore")
            R.check("score: no rows gives an explicit message and sends no request", "no rows to score" in page.inner_text("#scoreOut"), page.inner_text("#scoreOut"))
            page.fill("#payload", "{not json")
            page.click("#doScore")
            R.check("score: invalid JSON is reported", "not valid JSON" in page.inner_text("#scoreOut"))
            with page.expect_response("**/api/sample"):
                page.click("#loadSample")
            page.wait_for_function("() => document.getElementById('scoreInfo').textContent.startsWith('Loaded')")
            sample = json.loads(page.input_value("#payload"))
            n_before = len([r for r in sink["responses"] if r["url"].endswith("/api/score") and r["method"] == "POST"])
            page.click("#doScore")
            page.wait_for_selector("#scoreOut table")
            txt = page.inner_text("#scoreOut")
            n_rows = page.locator("#scoreOut tbody tr").count()
            R.check("score: one result row per posted row, labelled as the synthetic demonstration", n_rows == len(sample["new"]) and "SYNTHETIC DEMONSTRATION BUNDLE" in txt, n_rows)
            R.check("score: alerted rows list their largest contributions", all(("ALERT" not in r) or ("(" in r) for r in page.locator("#scoreOut tbody tr").all_inner_texts()))
            page.evaluate("() => { document.getElementById('doScore').click(); document.getElementById('doScore').click(); }")   # double click
            page.wait_for_selector("#scoreOut table")
            page.wait_for_timeout(1500)
            R.check("score: a double click yields exactly one result table", page.locator("#scoreOut table").count() == 1, page.locator("#scoreOut table").count())
            posts = len([r for r in sink["responses"] if r["url"].endswith("/api/score") and r["method"] == "POST"]) - n_before
            R.check("score: a double click sends at most one request while the first is in flight", posts <= 2, posts)
            bad = dict(sample)
            bad["new"] = [{k: v for k, v in r.items() if k != "TransactionAmt"} for r in sample["new"][:2]]
            page.fill("#payload", json.dumps(bad))
            page.click("#doScore")
            page.wait_for_selector("#scoreOut .alert")
            R.check("score: a server-side validation error is shown verbatim", "missing required column" in page.inner_text("#scoreOut"), page.inner_text("#scoreOut"))
            ids = dict(sample)
            ids["new"] = [{**sample["new"][0], "TransactionID": "abc"}]
            page.fill("#payload", json.dumps(ids))
            page.click("#doScore")
            page.wait_for_selector("#scoreOut .alert")
            R.check("score: a non-integer TransactionID is refused with an explanation (not an internal error)", "TransactionID" in page.inner_text("#scoreOut") and "internal" not in page.inner_text("#scoreOut"), page.inner_text("#scoreOut"))

            # research tab
            page.click("#btn-research")
            if not research_available:
                R.not_run("research tab: model loaded and scoring", "experiments/xgboost/model.joblib is not present (withheld artifact)")
            else:
                page.wait_for_function("() => !document.getElementById('researchModel').textContent.startsWith('Checking')")
                info = page.inner_text("#researchModel")
                R.check("research: the loaded model, its hash and threshold are shown", info.startswith("Loaded:") and "SHA-256" in info and "threshold" in info, info)
                page.fill("#researchPayload", '{"rows": []}')
                page.click("#doResearch")
                R.check("research: no rows gives an explicit message", "no rows" in page.inner_text("#researchOut").lower(), page.inner_text("#researchOut"))
                tmp = out / "_tmp"
                tmp.mkdir(exist_ok=True)
                (tmp / "not_json.json").write_text("this is not json", encoding="utf-8")
                (tmp / "no_rows.json").write_text('{"something": 1}', encoding="utf-8")
                page.set_input_files("#researchFile", str(tmp / "not_json.json"))
                page.wait_for_function("() => { const t = document.getElementById('researchInfo').textContent; return t.length > 0 && !t.startsWith('Reading'); }")
                R.check("research: a file that is not JSON is reported", "Could not read the file" in page.inner_text("#researchInfo"), page.inner_text("#researchInfo"))
                page.set_input_files("#researchFile", str(tmp / "no_rows.json"))
                page.wait_for_function("() => { const t = document.getElementById('researchInfo').textContent; return t.length > 0 && !t.startsWith('Reading'); }")
                R.check("research: a JSON file without rows is reported", "list of rows" in page.inner_text("#researchInfo"), page.inner_text("#researchInfo"))
                if rows_path and rows_path.exists():
                    rows = json.loads(rows_path.read_text(encoding="utf-8"))["rows"]
                    page.set_input_files("#researchFile", str(rows_path))
                    page.wait_for_function("() => document.getElementById('researchInfo').textContent.startsWith('Loaded')")
                    R.check("research: the chosen file is read in the browser", f"Loaded {len(rows)} rows" in page.inner_text("#researchInfo"), page.inner_text("#researchInfo"))
                    page.click("#doResearch")
                    page.wait_for_selector("#researchOut table", timeout=60000)
                    rtxt = page.inner_text("#researchOut")
                    R.check("research: one result row per posted row, labelled as the research model", page.locator("#researchOut tbody tr").count() == len(rows) and "RESEARCH MODEL" in rtxt,
                            page.locator("#researchOut tbody tr").count())
                    api = page.evaluate("(rows) => fetch('/api/research/score', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({rows})}).then(r => r.json())", rows)
                    shown = [t.split("\t")[2] for t in page.locator("#researchOut tbody tr").all_inner_texts()]
                    R.check("research: displayed scores equal the API's scores (4 decimals)", shown == [f"{x['score']:.4f}" for x in api["results"]], f"{len(shown)} rows compared")
                    labelled = [{**r, "isFraud": 1} for r in rows[:2]]
                    (tmp / "with_label.json").write_text(json.dumps({"rows": labelled}), encoding="utf-8")
                    page.set_input_files("#researchFile", str(tmp / "with_label.json"))
                    page.wait_for_function("() => document.getElementById('researchInfo').textContent.startsWith('Loaded 2')")
                    page.click("#doResearch")
                    page.wait_for_selector("#researchOut table")
                    R.check("research: label columns in the rows are reported as ignored", "were ignored" in page.inner_text("#researchOut"))
                    page.fill("#researchPayload", '{"rows": []}')                      # no validation row stays on the page (later screenshots)
                    page.evaluate("() => { document.getElementById('researchOut').textContent = ''; document.getElementById('researchInfo').textContent = ''; }")
                else:
                    R.not_run("research: scoring a file of validation rows", "no --research-rows file (write one with scripts/make_research_sample.py)")
                for f in tmp.iterdir():
                    f.unlink()
                tmp.rmdir()

            # communities and status
            page.click("#btn-communities")
            comm = page.evaluate("() => fetch('/api/communities').then(r => r.json())")
            multi = sum(1 for k in comm["cards"] if len(k.get("timeline", [])) >= 2)
            single = sum(1 for k in comm["cards"] if len(k.get("timeline", [])) == 1)
            R.check("communities: one card per recorded aggregate card; a line for multi-day timelines and a point for single-day ones",
                    page.locator("#communities .card").count() == comm["n_cards"] and page.locator("#communities .card svg polyline").count() == multi
                    and page.locator("#communities .card svg circle").count() == single, (page.locator("#communities .card").count(), multi, single))
            R.check("communities: candidate wording is shown", "candidate" in page.inner_text("#communities").lower())
            page.click("#btn-status")
            st = page.inner_text("#status")
            R.check("status: held-out evaluation performed = false and the research model is reported loaded", "held out evaluation performed\nfalse" in st and ("research model loaded\ntrue" in st or not research_available), st[:200])

            # error state: evidence endpoint fails -> the banner must not stay on 'Loading'
            ctx2 = browser.new_context(viewport=VIEWPORTS["desktop"])
            pg2 = ctx2.new_page()
            pg2.route("**/api/evidence", lambda route: route.fulfill(status=500, content_type="application/json", body='{"error": "internal error", "status": 500}'))
            pg2.goto(srv.url)
            pg2.wait_for_selector("#evidence .alert", timeout=30000)
            pg2.wait_for_timeout(500)
            R.check("error state: a failed evidence request is shown and the banner does not stay on 'Loading'", not pg2.inner_text("#banner").startswith("Loading"), pg2.inner_text("#banner"))
            pg2.screenshot(path=str(shots / "error_evidence_unavailable.png"), full_page=a.full_page)
            ctx2.close()

            # server stopped while the page is open, then restarted on the same port
            page.click("#btn-score")
            with page.expect_response("**/api/sample"):
                page.click("#loadSample")
            page.wait_for_function("() => document.getElementById('scoreInfo').textContent.startsWith('Loaded')")
            page.wait_for_function("() => !document.getElementById('doScore').disabled")
            port = srv.port
            srv.stop()
            page.click("#doScore")
            page.wait_for_selector("#scoreOut .alert", timeout=30000)
            R.check("server stopped: scoring reports the failure honestly", "Scoring failed" in page.inner_text("#scoreOut"), page.inner_text("#scoreOut"))
            page.screenshot(path=str(shots / "error_server_stopped.png"), full_page=a.full_page)
            srv = Server(a.python, ["--demo"], port=port)
            servers.append(srv)
            page.click("#doScore")
            page.wait_for_selector("#scoreOut table", timeout=30000)
            R.check("server restarted on the same port: the open page scores again", page.locator("#scoreOut table").count() == 1)
            ctx.close()
            sink_ignored = {"requestfailed_while_stopped": [r for r in sink["requestfailed"]]}
            sink["requestfailed"] = []
        finally:
            srv.stop()

        # ---------------------------------------------------------------- plain server: no bundle, no research model
        plain = Server(a.python, [])
        servers.append(plain)
        try:
            ctx = browser.new_context(viewport=VIEWPORTS["desktop"])
            page = ctx.new_page()
            watch(page, sink)
            page.goto(plain.url)
            ready(page)
            page.click("#btn-research")
            page.wait_for_function("() => !document.getElementById('researchModel').textContent.startsWith('Checking')")
            R.check("no research model: the tab says so and its button is disabled", "not loaded" in page.inner_text("#researchModel") and page.is_disabled("#doResearch"), page.inner_text("#researchModel"))
            page.click("#btn-score")
            page.click("#loadSample")
            page.wait_for_function("() => { const t = document.getElementById('scoreInfo').textContent; return t.length > 0 && !t.startsWith('Loading'); }")
            R.check("no bundle: the sample button explains that no sample exists", "no sample rows" in page.inner_text("#scoreInfo"), page.inner_text("#scoreInfo"))
            page.fill("#payload", json.dumps({"new": [{"TransactionID": 1, "TransactionDT": 1, "TransactionAmt": 1}]}))
            page.click("#doScore")
            page.wait_for_selector("#scoreOut .alert")
            R.check("no bundle: scoring reports that no bundle is loaded", "no scoring bundle" in page.inner_text("#scoreOut"), page.inner_text("#scoreOut"))
            page.screenshot(path=str(shots / "plain_server_score_unavailable.png"), full_page=a.full_page)
            page.click("#btn-research")
            page.screenshot(path=str(shots / "plain_server_research_unavailable.png"), full_page=a.full_page)
            ctx.close()
        finally:
            plain.stop()
        browser.close()

    if placeholder is not None:
        placeholder.unlink()
        placeholder.parent.rmdir()
    quantized = False
    try:                                                                     # optional: 256-colour PNGs are about 60 % smaller and stay legible
        from PIL import Image
        for f in shots.glob("*.png"):
            Image.open(f).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(f, optimize=True)
        quantized = True
    except ImportError:
        pass
    expected_4xx = [r for r in sink["responses"] if r["status"] >= 400]
    unexpected = [r for r in expected_4xx if not (r["url"].endswith(("/api/score", "/api/research/score", "/api/sample", "/api/research/model")))]
    R.check("browser: no JavaScript errors", not sink["pageerrors"] and not [c for c in sink["console"] if c["type"] == "error" and "Failed to load resource" not in c["text"]],
            sink["pageerrors"] + sink["console"])
    R.check("browser: no failed request apart from the induced ones", not sink["requestfailed"] and not unexpected, sink["requestfailed"] + unexpected)
    n_fail = sum(c["result"] == "FAIL" for c in R.checks)
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "browser": f"{a.channel} {version} (headless, Playwright)",
           "viewports": VIEWPORTS, "colour_schemes": ["light", "dark"], "full_page_screenshots": a.full_page, "screenshots_quantized_256_colours": quantized,
           "research_tab_screenshots": "synthetic placeholder rows (identifiers 900000001..., features missing); no IEEE-CIS row appears in any screenshot or in this record", "research_rows_file_used": bool(rows_path and rows_path.exists()),
           "summary": {"checks": len(R.checks), "passed": sum(c["result"] == "PASS" for c in R.checks), "failed": n_fail, "not_run": sum(c["result"] == "NOT RUN" for c in R.checks)},
           "checks": R.checks, "console_messages": sink["console"], "induced_request_failures": len(sink_ignored["requestfailed_while_stopped"]) if "sink_ignored" in dir() else None,
           "http_4xx_5xx_responses": [{"path": re.sub(r"^https?://[^/]+", "", r["url"]), "status": r["status"]} for r in expected_4xx],
           "server_commands": [" ".join(["<python>"] + s.cmd[1:]) for s in servers], "seconds": round(time.time() - t_start, 1),
           "screenshots_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(shots.glob("*.png"))},
           "statement": "Real-browser run of the documented server command; the demo bundle is synthetic and the research rows are validation rows (label removed) read from a file outside the repository."}
    data = (json.dumps(rec, indent=1) + "\n").encode("utf-8")
    (out / "browser_verification.json").write_bytes(data)
    (out / "browser_verification.sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(f"\n{rec['summary']['passed']}/{rec['summary']['checks']} checks passed, {n_fail} failed, {rec['summary']['not_run']} not run; record: {out / 'browser_verification.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
