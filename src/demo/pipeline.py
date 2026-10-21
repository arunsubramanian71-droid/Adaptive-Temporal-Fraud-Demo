"""End-to-end software demonstration on the SYNTHETIC fixture of src/demo/synthetic.py.

Every step calls real project code: backward-looking graph features (comparison B), the XGBoost TRAIN/VAL protocol, the shared
metric implementation, history-based scoring, native tree contributions, and the Phase 9 label-free candidate-community modules.
It demonstrates MECHANICS. It reproduces no research experiment, uses no IEEE-CIS data and no TEST period, and its numbers are not
results. Splits here are labelled demo-train / demo-valid / demo-holdout; "demo-holdout" is synthetic and unrelated to the sealed
TEST period.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd

from src.demo import _compat  # noqa: F401  (must precede the coordination imports: torch-free environments)
from src.coordination import frozen as F
from src.coordination.data import GraphView
from src.coordination.graphs import aggregate, louvain_candidates, window_rows
from src.coordination.tracking import Tracker, lifecycle_table
from src.demo.synthetic import FEATURE_COLUMNS, make_transactions
from src.inference import scorer as S
from src.models.metrics import compute_metrics

TRAIN_DAYS, VALID_DAYS, HOLDOUT_DAYS = (1, 24), (25, 32), (33, 40)
BANNER = ("SYNTHETIC DEMONSTRATION of software mechanics. Not IEEE-CIS data, not a research result, no TEST period used. "
          "Communities are candidates only.")


def _slice(df, days):
    return df[(df["day"] >= days[0]) & (df["day"] <= days[1])].reset_index(drop=True)


def _graph_view(df: pd.DataFrame) -> tuple:
    """Label-free GraphView of the synthetic stream (card, netproxy, device ids; -1 = no edge)."""
    card = pd.factorize(df["card_key"])[0].astype(np.int64)
    net = pd.factorize(df["net_key"])[0].astype(np.int64)
    dev = pd.factorize(df["device_key"])[0].astype(np.int64)
    view = GraphView(df["day"].to_numpy(dtype=np.int64), card, net, dev, int(net.max()) + 1, int(dev.max()) + 1)
    return view, card


def run_demo(out_dir, seed: int = 0, verbose: bool = True) -> dict:
    t0 = time.time()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    say(f"[demo] {BANNER}")

    df = make_transactions(seed=seed)
    tr, va, ho = _slice(df, TRAIN_DAYS), _slice(df, VALID_DAYS), _slice(df, HOLDOUT_DAYS)
    hist = pd.concat([tr, va], ignore_index=True)
    schema = S.Schema(features=tuple(FEATURE_COLUMNS))
    say(f"[1] synthetic stream: {len(df):,} transactions, {df['isFraud'].mean():.3%} fraud; "
        f"demo-train days {TRAIN_DAYS}, demo-valid {VALID_DAYS}, demo-holdout {HOLDOUT_DAYS}")

    # --- comparison A style (tabular only) versus comparison B style (tabular + backward-looking graph features)
    bundles, metrics = {}, {}
    for name, use_graph in (("tabular_only", False), ("tabular_plus_graph_features", True)):
        b = S.train_bundle(tr, va, schema, use_graph=use_graph, seed=42)
        S.save_bundle(b, out / "bundles" / name)
        sc = S.score(b, hist, ho)
        m = compute_metrics(ho["isFraud"].to_numpy(), sc["score"].to_numpy(), b.threshold)
        bundles[name], metrics[name] = b, {"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "threshold": m["threshold"], "n_alerts": m["n_alerts"],
                                           "precision": m["precision"], "recall": m["recall"], "best_iteration": b.meta["best_iteration"]}
        say(f"[2] {name:<28} PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  alerts {m['n_alerts']}  (demo-holdout, synthetic)")

    # --- scoring with explanations, from stored history only
    b = bundles["tabular_plus_graph_features"]
    scored = S.score(b, hist, ho, explain=True, top_k=3)
    scored.to_csv(out / "holdout_scores.csv", index=False)
    top = scored[scored["alert"]].sort_values("score", ascending=False).head(3)
    explain_rows = [{"TransactionID": int(r.TransactionID), "score": round(float(r.score), 4), "top_features": json.loads(r.top_features)} for r in top.itertuples()]
    say(f"[3] scored {len(scored)} holdout transactions from {len(hist)} history rows; explained the top {len(explain_rows)} alerts (native tree contributions)")

    # --- candidate-community discovery with the Phase 9 modules (label-free); planted ids are used ONLY to grade this demo
    view, card = _graph_view(df)
    planted_cards = set(card[df["is_planted"].to_numpy()].tolist())
    tracker, daily = Tracker(), []
    for t in range(VALID_DAYS[0], HOLDOUT_DAYS[1] + 1):
        cands = louvain_candidates(aggregate(window_rows(view, t, F.PRIMARY_W)), gamma=0.5, seed=42, m_min=F.M_MIN)
        tracker.step(t, [c.cards for c in cands])
        daily.append({"day": t, "n_candidates": len(cands), "sizes": sorted((len(c.cards) for c in cands), reverse=True)[:5],
                      "best_overlap_with_planted": max((len(set(c.cards.tolist()) & planted_cards) for c in cands), default=0)})
    life = lifecycle_table(tracker, HOLDOUT_DAYS[1])
    pd.DataFrame(daily).to_csv(out / "candidate_communities_by_day.csv", index=False)
    peak = max(daily, key=lambda r: r["best_overlap_with_planted"])
    say(f"[4] candidate communities (Phase 9 modules, window {F.PRIMARY_W} days): {sum(r['n_candidates'] for r in daily)} community-days over "
        f"{len(daily)} days; best overlap with the {len(planted_cards)} planted demo cards on day {peak['day']}: {peak['best_overlap_with_planted']}")

    # --- invariants shown to the audience
    inv = {"scores_in_unit_interval": bool(scored["score"].between(0, 1).all()), "one_score_per_new_row": bool(len(scored) == len(ho)),
           "label_column_not_needed_to_score": bool(np.array_equal(S.score(b, hist.drop(columns=["isFraud"]), ho.drop(columns=["isFraud"]))["score"].to_numpy(), S.score(b, hist, ho)["score"].to_numpy())),
           "first_half_unchanged_when_second_half_removed": None}
    half = ho[ho["day"] <= (HOLDOUT_DAYS[0] + 3)].reset_index(drop=True)
    inv["first_half_unchanged_when_second_half_removed"] = bool(np.array_equal(
        S.score(b, hist, half)["score"].to_numpy(), scored["score"].to_numpy()[: len(half)]))
    say(f"[5] invariants: {inv}")

    summary = {"banner": BANNER, "seed": seed, "n_transactions": int(len(df)), "splits": {"train": TRAIN_DAYS, "valid": VALID_DAYS, "holdout": HOLDOUT_DAYS},
               "metrics_demo_holdout": metrics, "explained_alerts": explain_rows, "candidate_days": daily, "n_tracks": len(life),
               "planted_cards": len(planted_cards), "invariants": inv, "seconds": round(time.time() - t0, 2)}
    deterministic = {k: v for k, v in summary.items() if k != "seconds"}
    (out / "demo_summary.json").write_text(json.dumps(deterministic, indent=1, default=float), encoding="utf-8")
    summary["output_dir"] = str(out)
    say(f"[done] {summary['seconds']}s; outputs in {out}")
    return summary
