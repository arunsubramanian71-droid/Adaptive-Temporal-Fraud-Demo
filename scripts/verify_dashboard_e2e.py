"""End-to-end check of the dashboard and API: starts the server exactly as documented (a separate process running scripts/run_dashboard.py --demo),
exercises every route, the scoring path with the synthetic sample, the error handling and the shutdown, and records what was observed.

  .venv/Scripts/python.exe -B scripts/verify_dashboard_e2e.py            (Windows)
  .venv/bin/python -B scripts/verify_dashboard_e2e.py                    (macOS / Linux)
  ... [--out FILE] [--record]

It checks the served HTML and JSON (and the launcher's refusals); it does not render the page. Rendering is checked separately (browser tools) and recorded
in experiments/phase13_serving_dashboard/test_results/browser_verification.json. Only the process started here is stopped.
Writes demo_output/verification/e2e_verification.json (or --out FILE). Only --record overwrites the tracked, dated record
experiments/phase13_serving_dashboard/test_results/e2e_verification.json (since 2026-10-23; scripts/verification_output.py says why).
"""
from __future__ import annotations

import datetime
import http.client
import json
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "phase13_serving_dashboard" / "test_results" / "e2e_verification.json"


def call(port, method, path, body=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
    c.request(method, path, body=data, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    text = r.read().decode("utf-8")
    c.close()
    return r.status, (json.loads(text) if "json" in r.getheader("Content-Type", "") else text)


def port_open(port) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launcher(args, timeout=60):
    p = subprocess.run([sys.executable, "-B", "scripts/run_dashboard.py", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stderr.strip()


def main(argv=None) -> int:
    from verification_output import output_path, write_record
    out = output_path(argv, OUT, __doc__)
    checks, t0 = [], time.time()

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})
        print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""), flush=True)

    # launcher refusals (start nothing)
    rc, err = launcher(["--host", "0.0.0.0"])
    check("launcher refuses a non-local host", rc == 2 and "non-local" in err, err)
    rc, err = launcher(["--bundle", "C:/data/test_bundle"])
    check("launcher refuses a held-out-named bundle path", rc == 2 and "sealed" in err, err)
    rc, err = launcher(["--demo", "--bundle", "x"])
    check("launcher refuses --demo together with --bundle", rc == 2, err)

    proc = subprocess.Popen([sys.executable, "-u", "-B", "scripts/run_dashboard.py", "--demo", "--port", "0"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, port, ready = [], None, threading.Event()

    def pump():
        nonlocal port
        for ln in proc.stdout:
            lines.append(ln.rstrip())
            m = re.search(r"http://127\.0\.0\.1:(\d+)/", ln)
            if m and "dashboard on" in ln:
                port = int(m.group(1))
                ready.set()
    threading.Thread(target=pump, daemon=True).start()
    try:
        check("server prints its address within 120 s", ready.wait(120), "\n".join(lines[-5:]))
        if port is None:
            raise RuntimeError("server did not start")
        check("server listens on the loopback address only", port_open(port) and not any(l for l in lines if "0.0.0.0" in l))
        st, html = call(port, "GET", "/")
        check("GET / returns the dashboard page", st == 200 and "Fraud study dashboard" in html and "validation" in html.lower())
        check("page has no external resource reference", not re.search(r"(src|href)\s*=\s*[\"']https?://", html))
        for path in ("/health", "/api/status", "/api/model", "/api/evidence", "/api/communities", "/api/sample"):
            st, body = call(port, "GET", path)
            check(f"GET {path} returns JSON 200", st == 200 and isinstance(body, dict), st)
        _, ev = call(port, "GET", "/api/evidence")
        check("evidence carries the validation-only banner and the held-out statement", "validation-period" in ev["banner"] and "No held-out" in ev["held_out"])
        _, cm = call(port, "GET", "/api/communities")
        check("communities: 24 aggregate cards with the candidate statements", cm["n_cards"] == 24 and cm["standing_statement"].startswith("candidate coordinated community"))
        _, smp = call(port, "GET", "/api/sample")
        st, res = call(port, "POST", "/api/score", {"history": smp["history"], "new": smp["new"]})
        check("scoring the synthetic sample works and is labelled synthetic", st == 200 and res["synthetic_demo"] is True and res["n"] == len(smp["new"]) and all(0 <= r["score"] <= 1 for r in res["results"]), res if st != 200 else "")
        check("alerted rows carry explanations, others do not", all(("top_features" in r) == r["alert"] for r in res["results"]))
        errors = {"unknown path -> 404": call(port, "GET", "/nope")[0] == 404, "wrong method on /api/score -> 405": call(port, "GET", "/api/score")[0] == 405,
                  "PUT -> 405": call(port, "PUT", "/api/score", {})[0] == 405, "invalid JSON -> 400": call(port, "POST", "/api/score", raw=b"{oops")[0] == 400,
                  "empty new list -> 400": call(port, "POST", "/api/score", {"new": []})[0] == 400, "missing columns -> 400": call(port, "POST", "/api/score", {"new": [{"TransactionID": 1}]})[0] == 400,
                  "more than 5,000 rows -> 400": call(port, "POST", "/api/score", {"new": [{}] * 5001})[0] == 400,
                  "body over 1 MB -> 413": call(port, "POST", "/api/score", raw=b'{"new": [' + b" " * 1_100_000 + b"]}")[0] == 413,
                  "held-out-named path -> 404": call(port, "GET", "/data/features/test_features.parquet")[0] == 404, "path traversal -> 404": call(port, "GET", "/../configs/data.yaml")[0] == 404}
        for k, v in errors.items():
            check(k, v)
        bodies = [call(port, "GET", "/nope")[1], call(port, "POST", "/api/score", raw=b"{oops")[1]]
        check("error bodies are JSON without a stack trace", all(isinstance(b, dict) and "Traceback" not in json.dumps(b) for b in bodies))
        check("server still answers after the error cases", call(port, "GET", "/health")[0] == 200)
    except Exception as e:  # noqa: BLE001
        check("end-to-end run completed without an exception", False, f"{type(e).__name__}: {e}")
    finally:
        pid = proc.pid
        proc.terminate()
        try:
            proc.wait(timeout=20)
            stopped = True
        except subprocess.TimeoutExpired:
            proc.kill()
            stopped = proc.wait(timeout=20) is not None
        time.sleep(0.5)
    check("shutdown: the server process exited", stopped and proc.poll() is not None, f"pid {pid}, exit code {proc.poll()}")
    check("shutdown: the port is closed", port is not None and not port_open(port))
    rec = {"date_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "command": "scripts/run_dashboard.py --demo --port 0 (separate process)",
           "scope": "HTTP-level end-to-end check of the served page, API, scoring path, error handling and shutdown; it does not render the page", "python": sys.version.split()[0],
           "seconds": round(time.time() - t0, 1), "checks": checks, "all_pass": all(c["pass"] for c in checks)}
    write_record(out, json.dumps(rec, indent=1))
    print(f"{sum(c['pass'] for c in checks)}/{len(checks)} checks passed; record: {out}")
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
