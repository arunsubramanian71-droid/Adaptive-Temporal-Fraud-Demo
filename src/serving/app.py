"""Local HTTP API and investigation dashboard (Phase 13, experiments/phase13_serving_dashboard/SCOPE_AND_ACCEPTANCE.md).

Standard library only (no new dependency). Serves recorded validation-period aggregates, aggregate-only candidate-community cards, a scoring
endpoint backed by ``src/inference/scorer.py`` (the synthetic demo bundle or an operator's bundle) and, when started with the research model, a separate
research-model endpoint backed by ``src/inference/research_model.py`` (the recorded comparison-A XGBoost, served exactly as evaluated). The two scoring
modes are never mixed and every response says which one produced it. It never reads a TEST-named path, a checkpoint or a transaction-level file; the only files it opens
are listed in ``READABLE`` and each open is logged in ``Services.opened``.
"""
from __future__ import annotations

import csv
import http.server
import json
import math
import pathlib
import socketserver
import threading
import traceback

from src.inference import scorer as S

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MAX_BODY = 1_000_000
MAX_ROWS = 5000
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
DASHBOARD = "src/serving/static/dashboard.html"
READABLE = {
    "evidence_cells": "experiments/phase12_validation_evidence_table/outputs/evidence_cells.csv",
    "evidence_table": "experiments/phase12_validation_evidence_table/outputs/evidence_table.json",
    "table_with_B": "experiments/comparison_B_graph_features/outputs/table_with_B.csv",
    "decision_B": "experiments/comparison_B_graph_features/outputs/decision.json",
    "decision_unweighted": "experiments/imbalance_check_xgboost/outputs/unweighted_decision.json",
    "decision_imbalance": "experiments/imbalance_check_xgboost/outputs/decision.json",
    "cards": "experiments/phase11_explainable_evidence/community_cards/cards_W7_seed42.json",
    "dashboard": DASHBOARD,
}
SCORE_NOTE = "A score is a model output on the posted rows. It is not evidence of fraud, causation or coordination."
HELD_OUT_NOTE = ("No held-out (days 152-182) evaluation has been performed; every recorded result is a validation-period result and most models were tuned on that period.")
HELD_OUT_DONE_NOTE = ("Held-out (days 152-182) output exists in {folders} of experiments/final_heldout_protocol; held-out results are not shown on this dashboard, which "
                      "serves validation-period results only (most models were tuned on that period). Read the held-out records in that folder.")

CARD_KEYS = ("track", "groups", "standing_statement", "first_observed_day", "last_observed_day", "n_observed_days", "left_censored", "right_censored", "peak",
             "n_alerted_days", "member_transaction_composition_txn_days", "timeline")
TIMELINE_KEYS = ("day", "n_cards", "n_infra", "density", "hub_share", "conductance", "u_s", "u_t", "score_st", "n_member_txn", "alerted")


def held_out_note(root: pathlib.Path) -> tuple:
    """(performed, note) from the existence of any file in the three held-out output folders (src/heldout/oneshot.py); nothing is opened."""
    from src.heldout.oneshot import folders_with_output
    done = folders_with_output(pathlib.Path(root) / "experiments" / "final_heldout_protocol")
    return (bool(done), HELD_OUT_DONE_NOTE.format(folders=", ".join(f"{f}/" for f in done)) if done else HELD_OUT_NOTE)


NO_RESEARCH = "the research model is not loaded (start the server with --research-model; it needs experiments/xgboost/model.joblib)"


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _clean(v):
    """JSON-safe: NaN / inf become None."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


class Services:
    """Everything the handlers need; built once at startup. Holds the scoring bundle and logs every file open."""

    def __init__(self, bundle=None, synthetic_demo=False, sample=None, root: pathlib.Path = ROOT, research=None):
        self.root = pathlib.Path(root)
        self.bundle = bundle
        self.research = research
        self.synthetic_demo = bool(synthetic_demo)
        self.sample = sample
        self.opened: list = []
        self._lock = threading.Lock()

    def read_text(self, key: str) -> str:
        rel = READABLE[key]
        S.refuse_sealed_path(rel)
        with self._lock:
            self.opened.append(rel)
        return (self.root / rel).read_text(encoding="utf-8")

    def read_json(self, key: str):
        return json.loads(self.read_text(key))

    def read_csv(self, key: str) -> list:
        return list(csv.DictReader(self.read_text(key).splitlines()))

    # ------------------------------------------------------------------ payloads
    def evidence(self) -> dict:
        table = self.read_json("evidence_table")
        cells = []
        for r in self.read_csv("evidence_cells"):
            f = lambda k: (float(r[k]) if r.get(k) not in (None, "") else None)  # noqa: E731
            cells.append({"cell": r["cell"], "model": r["model"], "basis": r["basis"], "runs": int(float(r["n_runs"])), "status": r["status"],
                          "pr_auc": f("pr_auc_mean"), "pr_auc_sd": f("pr_auc_sd"), "roc_auc": f("roc_auc_mean"), "pr_auc_cold": f("pr_auc_has_edge_cold_mean")})
        b_rows = []
        for r in self.read_csv("table_with_B"):
            f = lambda k: (float(r[k]) if r.get(k) not in (None, "") else None)  # noqa: E731
            b_rows.append({"cell": r["cell"], "model": r["model"], "runs": int(float(r["runs"])), "pr_auc": f("pr_auc"), "pr_auc_sd": f("pr_auc_sd"), "roc_auc": f("roc_auc"),
                           "pr_auc_cold": f("pr_auc_cold")})
        b, un, im = self.read_json("decision_B"), self.read_json("decision_unweighted"), self.read_json("decision_imbalance")
        return _clean({
            "banner": table["standing_statement"], "f_banner": table["f_banner"], "held_out": held_out_note(self.root)[1],
            "phase12_cells": cells, "comparison_b_rows": b_rows,
            "comparison_b_verdicts": {"H1_direct_B_vs_A": b["H1_direct_B_vs_A"]["verdict"], "H2_E_vs_B_sub": b["H2_E_vs_B_sub"]["verdict"],
                                      "H2_secondary_C_vs_B_sub": b["H2_secondary_C_vs_B_sub"]["verdict"], "ablations": b["ablations"]["rows"]},
            "imbalance": {"xgboost_weighting_verdict": im["verdict"], "xgboost_pr_auc_by_weight": im["pr_auc"],
                          "unweighted_followup": {"weighting_effect": {k: v["verdict"] for k, v in un["rule1_weighting_effect"].items()},
                                                  "H1_direct": un["rule2_H1_direct_unweighted"]["verdict"], "H2": un["rule3_H2_unweighted_E"]["verdict"],
                                                  "tabular_above_every_neural_seed": un["rule4_ordering"]["tabular_above_every_neural_seed"]}},
            "note_B": "Comparison B is the XGBoost + 40 backward-looking graph features comparison; rows labelled A are the same recipe without them.",
            "note_phase12_B_row": "The B row in the Phase 12 table above is the historical record made before comparison B was run (MISSING at that time); comparison B was run afterwards and is shown in the next table."})

    def communities(self) -> dict:
        c = self.read_json("cards")
        cards = []
        for card in c["cards"]:
            out = {k: card[k] for k in CARD_KEYS if k in card}
            out["timeline"] = [{k: t[k] for k in TIMELINE_KEYS if k in t} for t in card.get("timeline", [])]
            cards.append(out)
        return _clean({"standing_statement": c["standing_statement"], "not_established": c["not_established"],
                       "scope": "aggregate-only cards for exploratory Phase 9 candidate communities (window 7 days, seed 42, validation days 122-151); no member identities exist in this output",
                       "n_cards": len(cards), "cards": cards})

    def status(self) -> dict:
        fh = self.root / "experiments" / "final_heldout_protocol"
        performed, note = held_out_note(self.root)
        return {"validation_period_only": True, "held_out_evaluation_performed": performed, "held_out_authorization_present": (fh / "AUTHORIZATION.txt").exists(),
                "held_out_note": note, "phase12": "AC1-AC11 pass; AC12 BLOCKED (partially verified); Phase 12 is not formally complete",
                "h5": "not established (frozen benchmark gate G4 failed on 2 of 3 seeds)", "communities": "candidates only; not confirmed to be fraudulent or coordinated",
                "scoring_bundle_loaded": self.bundle is not None, "synthetic_demo": self.synthetic_demo,
                "research_model_loaded": self.research is not None}

    # ------------------------------------------------------------------ research model (separate mode)
    def research_info(self) -> dict:
        if self.research is None:
            raise ApiError(404, NO_RESEARCH)
        return {"mode": "research_model", "synthetic_demo": False, **self.research.info()}

    def research_score(self, payload: dict) -> dict:
        import pandas as pd
        from src.inference.research_model import ResearchInputError
        if self.research is None:
            raise ApiError(404, NO_RESEARCH)
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list) or not payload["rows"]:
            raise ApiError(400, "body must be a JSON object with a non-empty list 'rows' of objects in the processed-feature schema (see GET /api/research/model)")
        rows = payload["rows"]
        if len(rows) > MAX_ROWS:
            raise ApiError(400, f"at most {MAX_ROWS} rows per request")
        if not all(isinstance(r, dict) for r in rows):
            raise ApiError(400, "every element of 'rows' must be an object")
        labels_present = any("isFraud" in r for r in rows)
        try:
            res = self.research.score(pd.DataFrame(rows), explain=True, top_k=5)
        except ResearchInputError as e:
            raise ApiError(400, str(e)) from None
        out = []
        for r in res.itertuples(index=False):
            tid = r.TransactionID
            tid = None if tid is None or (isinstance(tid, float) and not math.isfinite(tid)) else int(tid)
            item = {"row": int(r.row), "TransactionID": tid, "score": float(r.score), "alert": bool(r.alert)}
            if bool(r.alert) and r.top_features:
                item["top_features"] = r.top_features
            out.append(item)
        return _clean({"mode": "research_model", "synthetic_demo": False, "model_sha256": self.research.model_sha256, "threshold": self.research.threshold,
                       "labels_in_request_ignored": labels_present, "n": len(out), "n_alerts": sum(o["alert"] for o in out), "results": out,
                       "statement": self.research.info()["statement"]})

    def model_info(self) -> dict:
        if self.bundle is None:
            raise ApiError(404, "no scoring bundle is loaded (start the server with --demo or --bundle DIR)")
        b = self.bundle
        return {"synthetic_demo": self.synthetic_demo, "uses_graph_features": b.use_graph, "threshold": b.threshold, "n_model_columns": len(b.columns),
                "required_columns": b.schema.required(False), "meta": {k: v for k, v in b.meta.items() if k != "threshold_selection"}, "note": SCORE_NOTE}

    def score(self, payload: dict) -> dict:
        import pandas as pd
        if self.bundle is None:
            raise ApiError(404, "no scoring bundle is loaded (start the server with --demo or --bundle DIR)")
        if not isinstance(payload, dict) or not isinstance(payload.get("new"), list) or not payload["new"]:
            raise ApiError(400, "body must be a JSON object with a non-empty list 'new' (and optionally a list 'history') of row objects")
        hist = payload.get("history") or []
        if not isinstance(hist, list):
            raise ApiError(400, "'history' must be a list of row objects")
        if len(hist) + len(payload["new"]) > MAX_ROWS:
            raise ApiError(400, f"at most {MAX_ROWS} rows (history + new) per request")
        for name, rows in (("history", hist), ("new", payload["new"])):
            if not all(isinstance(r, dict) for r in rows):
                raise ApiError(400, f"every element of '{name}' must be an object")
        need = self.bundle.schema.required(False)
        try:
            new = pd.DataFrame(payload["new"])
            old = pd.DataFrame(hist) if hist else None
            res = S.score(self.bundle, old, new, explain=True, top_k=5)
        except S.ScoringError as e:
            raise ApiError(400, str(e)) from None
        except (ValueError, TypeError, KeyError) as e:
            raise ApiError(400, f"invalid rows: {type(e).__name__}: {e}. Required columns: {need}") from None
        out = []
        for r in res.itertuples(index=False):
            item = {"TransactionID": int(r.TransactionID), "score": float(r.score), "alert": bool(r.alert)}
            if bool(r.alert) and r.top_features:
                item["top_features"] = json.loads(r.top_features)
            out.append(item)
        return _clean({"synthetic_demo": self.synthetic_demo, "threshold": self.bundle.threshold, "n": len(out), "n_alerts": sum(o["alert"] for o in out), "results": out, "statement": SCORE_NOTE})


def make_handler(services: Services):
    html = None

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "FraudStudyDashboard/1"
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):          # quiet; opened files are tracked in Services.opened
            return

        # ------------------------------------------------------------------ responses
        def _send(self, status: int, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj):
            self._send(status, (json.dumps(obj, indent=1, allow_nan=False) + "\n").encode("utf-8"), "application/json; charset=utf-8")

        def _error(self, status: int, message: str):
            self._json(status, {"error": message, "status": status})

        # ------------------------------------------------------------------ routing
        def _route(self, method: str):
            nonlocal html
            path = self.path.split("?", 1)[0]
            try:
                low = path.lower().replace("latest", "")
                if ".." in path or "\\" in path or "test" in low:
                    raise ApiError(404, "not found")
                if method == "GET":
                    if path in ("/", "/index.html"):
                        if html is None:
                            html = services.read_text("dashboard").encode("utf-8")
                        return self._send(200, html, "text/html; charset=utf-8")
                    routes = {"/health": lambda: {"status": "ok"}, "/api/status": services.status, "/api/model": services.model_info,
                              "/api/evidence": services.evidence, "/api/communities": services.communities, "/api/sample": self._sample,
                              "/api/research/model": services.research_info}
                    if path in routes:
                        return self._json(200, routes[path]())
                    if path in ("/api/score", "/api/research/score"):
                        raise ApiError(405, "use POST for " + path)
                    raise ApiError(404, "not found")
                if method == "POST":
                    if path == "/api/research/score":
                        return self._json(200, services.research_score(self._body()))
                    if path != "/api/score":
                        raise ApiError(404 if path not in ("/", "/health", "/api/status", "/api/model", "/api/evidence", "/api/communities", "/api/sample",
                                                           "/api/research/model") else 405, "not found or method not allowed")
                    return self._json(200, services.score(self._body()))
                raise ApiError(405, "method not allowed")
            except ApiError as e:
                return self._error(e.status, e.message)
            except Exception:                                        # noqa: BLE001 - never leak a trace to the client
                traceback.print_exc()
                return self._error(500, "internal error")

        def _sample(self):
            if services.sample is None:
                raise ApiError(404, "no sample rows are available (they exist only for the synthetic demo bundle)")
            return {"synthetic_demo": True, **services.sample}

        def _body(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ApiError(400, "invalid Content-Length") from None
            if n < 0:
                raise ApiError(400, "invalid Content-Length")
            if n > MAX_BODY:
                remaining = min(n, 8 * MAX_BODY)                       # drain so the client can read the answer, then refuse
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                raise ApiError(413, f"body larger than {MAX_BODY} bytes")
            raw = self.rfile.read(n) if n else b""
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ApiError(400, "body is not valid JSON") from None

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_PUT(self):
            self._route("PUT")

        def do_DELETE(self):
            self._route("DELETE")

        def do_PATCH(self):
            self._route("PATCH")

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(services: Services, host: str = "127.0.0.1", port: int = 8765, allow_remote: bool = False) -> Server:
    if host not in LOCAL_HOSTS and not allow_remote:
        raise ValueError(f"refusing to bind to non-local address '{host}': this server has no authentication; pass allow_remote only if you understand the risk")
    return Server((host, port), make_handler(services))
