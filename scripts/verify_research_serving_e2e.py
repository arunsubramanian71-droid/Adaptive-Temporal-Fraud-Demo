"""Live end-to-end check of the research-model mode: start the real launcher with both scoring modes, score validation rows through HTTP, compare with direct
inference of the recorded model, confirm the two modes are labelled and never mixed, and shut the server down.

  .venv/Scripts/python.exe -B scripts/verify_research_serving_e2e.py

Needs the withheld artifacts (experiments/xgboost/model.joblib and data/features/val_features.parquet); exits 3 with a clear message if they are missing.
Validation rows only (label removed); no held-out file is opened. Writes experiments/phase13_serving_dashboard/test_results/research_e2e_verification.json.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import re
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from verify_dashboard_e2e import call, port_open  # noqa: E402

OUT = ROOT / "experiments" / "phase13_serving_dashboard" / "test_results" / "research_e2e_verification.json"
NEEDED = ("experiments/xgboost/model.joblib", "data/features/val_features.parquet")


def main() -> int:
    missing = [p for p in NEEDED if not (ROOT / p).exists()]
    if missing:
        print(f"NOT RUN: withheld artifact(s) missing: {missing}; regenerate them with scripts/build_processed_data.py and scripts/train_xgboost.py", file=sys.stderr)
        return 3
    checks, t0 = [], time.time()

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})
        print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""), flush=True)

    from src.heldout.evaluator import read_features
    from src.inference.research_model import ResearchModel
    df = read_features(ROOT / NEEDED[1]).head(60)
    m = ResearchModel()
    direct = m.model.predict_proba(m.builder.to_matrix(df, for_model="xgb"))[:, 1]
    rows = json.loads(df.replace({np.nan: None}).to_json(orient="records"))

    proc = subprocess.Popen([sys.executable, "-u", "-B", "scripts/run_dashboard.py", "--demo", "--research-model", "--port", "0"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, port, ready = [], None, threading.Event()

    def pump():
        nonlocal port
        for ln in proc.stdout:
            lines.append(ln.rstrip())
            mm = re.search(r"http://127\.0\.0\.1:(\d+)/", ln)
            if mm and "dashboard on" in ln:
                port = int(mm.group(1))
                ready.set()
    threading.Thread(target=pump, daemon=True).start()
    try:
        check("launcher with --demo --research-model prints its address within 180 s", ready.wait(180), "\n".join(lines[-5:]))
        if port is None:
            raise RuntimeError("server did not start")
        st, html = call(port, "GET", "/")
        check("dashboard page has the Research model tab and the demo tab", st == 200 and "tab-research" in html and "Score transactions (demo)" in html)
        st, s = call(port, "GET", "/api/status")
        check("status reports both modes loaded", st == 200 and s["research_model_loaded"] is True and s["synthetic_demo"] is True, s)
        st, info = call(port, "GET", "/api/research/model")
        check("research model info: correct artifact hash and recorded threshold", st == 200 and info["model_sha256"] == m.model_sha256 and info["threshold"] == m.threshold
              and info["synthetic_demo"] is False and info["uses_history"] is False, info.get("model_sha256"))
        st, r = call(port, "POST", "/api/research/score", {"rows": rows})
        api = np.array([x["score"] for x in r.get("results", [])])
        check("research scores over HTTP equal direct inference of the recorded model (60 validation rows)", st == 200 and len(api) == len(direct) and np.array_equal(api, direct),
              f"status {st}, max diff {float(np.abs(api - direct).max()) if len(api) == len(direct) else 'n/a'}")
        check("research responses are labelled research_model and not synthetic", r.get("mode") == "research_model" and r.get("synthetic_demo") is False)
        st, r1 = call(port, "POST", "/api/research/score", {"rows": [dict(x, isFraud=1) for x in rows]})
        check("labels in the request are ignored", st == 200 and [x["score"] for x in r1["results"]] == list(api) and r1["labels_in_request_ignored"] is True)
        st, smp = call(port, "GET", "/api/sample")
        st2, demo = call(port, "POST", "/api/score", {"history": smp["history"], "new": smp["new"]})
        check("the synthetic demo mode still answers and is labelled synthetic", st2 == 200 and demo["synthetic_demo"] is True and "mode" not in demo)
        st, bad = call(port, "POST", "/api/research/score", {"rows": [{"TransactionID": 1}]})
        check("an incomplete research row gets an actionable 400", st == 400 and "required input column" in bad["error"], bad)
        n_alert = sum(x["alert"] for x in r["results"])
        check("explanations accompany exactly the alerted rows", all(("top_features" in x) == x["alert"] for x in r["results"]), f"{n_alert} alerts")
    finally:
        pid = proc.pid
        proc.terminate()
        try:
            proc.wait(timeout=20)
            stopped = True
        except subprocess.TimeoutExpired:
            proc.kill()
            stopped = proc.wait(timeout=20) is not None
    check("shutdown: the server process exited", stopped and proc.poll() is not None, f"pid {pid}, exit code {proc.poll()}")
    t_exit, closed = time.time(), False
    while port is not None and time.time() - t_exit < 10:                 # the OS releases the socket shortly after the process exits (measured about 1.5 s)
        if not port_open(port):
            closed = True
            break
        time.sleep(0.25)
    check("shutdown: the port is closed within 10 s of the process exit", closed, f"{time.time() - t_exit:.2f} s")
    rec = {"date_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "command": "scripts/run_dashboard.py --demo --research-model --port 0 (separate process)",
           "scope": "HTTP-level end-to-end check of the research-model mode beside the synthetic demo mode; 60 validation rows (label removed); no held-out file",
           "python": sys.version.split()[0], "seconds": round(time.time() - t0, 1), "checks": checks, "all_pass": all(c["pass"] for c in checks)}
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"{sum(c['pass'] for c in checks)}/{len(checks)} checks passed")
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
