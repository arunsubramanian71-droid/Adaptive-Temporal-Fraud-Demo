"""Measure the serving path on synthetic data only (no research data exists on this machine; the held-out period is never touched).

Usage: <project .venv python> -B measure_performance.py <repo root> <out.json>
"""
from __future__ import annotations

import json
import pathlib
import platform
import resource
import statistics
import sys
import time

root = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(root))

t0 = time.perf_counter()
from src.serving.demo import build_demo_services  # noqa: E402
import_s = time.perf_counter() - t0
t0 = time.perf_counter()
svc = build_demo_services()
startup_s = time.perf_counter() - t0

sample = svc.sample
hist, new = sample["history"], sample["new"]


def timed(fn, repeats=7):
    xs = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t)
    return {"median_ms": round(1000 * statistics.median(xs), 1), "min_ms": round(1000 * min(xs), 1), "max_ms": round(1000 * max(xs), 1), "repeats": repeats}


from src.demo.synthetic import make_transactions  # noqa: E402
from src.serving.demo import _records  # noqa: E402
df = make_transactions(seed=1)
cols = svc.bundle.schema.required(False)
big_hist = _records(df[df["day"] <= 20], cols)[-3000:]          # history strictly earlier than every new row (the scorer refuses otherwise)
big_new = _records(df[df["day"] >= 21], cols)[:2000]

res = {
    "scope": "synthetic demonstration bundle and synthetic rows only; in-process calls of the same Services methods the HTTP handlers use (no network time)",
    "machine": f"{platform.platform()}, {platform.machine()}, Python {platform.python_version()}",
    "import_seconds": round(import_s, 2),
    "demo_startup_seconds_train_bundle_in_memory": round(startup_s, 2),
    "score_1_new_row_with_300_history_rows": timed(lambda: svc.score({"history": hist, "new": new[:1]})),
    "score_30_new_rows_with_300_history_rows": timed(lambda: svc.score({"history": hist, "new": new})),
    f"score_{len(big_new)}_new_rows_with_{len(big_hist)}_history_rows": timed(lambda: svc.score({"history": big_hist, "new": big_new}), repeats=3),
    "evidence_endpoint_payload": timed(svc.evidence),
    "communities_endpoint_payload": timed(svc.communities),
    "status_endpoint_payload": timed(svc.status),
    "peak_resident_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1),
    "rows_checked": {"history_big": len(big_hist), "new_big": len(big_new)},
    "limitations": "one machine, warm caches after the first call, synthetic feature distribution; research-model inference not measured (model withheld); the neural models are not served",
}
out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
