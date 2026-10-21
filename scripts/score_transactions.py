"""Train or use a scoring bundle from files (production-style command line; see src/inference/scorer.py).

  train:  .venv/Scripts/python.exe -B scripts/score_transactions.py train --train train.csv --valid valid.csv --features f0,f1,f2 --out-dir bundle/
  score:  .venv/Scripts/python.exe -B scripts/score_transactions.py score --bundle bundle/ --history history.csv --new new.csv --out scores.csv [--explain]

Expected columns: TransactionID, TransactionDT (seconds), TransactionAmt, the --features columns, and the four entity key columns
account_key, card_key, net_key, device_key (empty = no edge); training files also need isFraud (0/1). TEST-named files are refused.
Exit codes: 0 success, 2 invalid input or refused path.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.inference import scorer as S  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fit a bundle (early stopping and threshold on --valid)")
    t.add_argument("--train", required=True)
    t.add_argument("--valid", required=True)
    t.add_argument("--features", required=True, help="comma-separated numeric tabular columns")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--no-graph", action="store_true", help="tabular features only (comparison-A style)")
    t.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("score", help="score new transactions using stored history")
    s.add_argument("--bundle", required=True)
    s.add_argument("--history", default=None, help="earlier transactions (optional)")
    s.add_argument("--new", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--explain", action="store_true", help="add the top model contributions for alerted rows")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "train":
            schema = S.Schema(features=tuple(f for f in a.features.split(",") if f))
            b = S.train_bundle(S.read_table(a.train), S.read_table(a.valid), schema, use_graph=not a.no_graph, seed=a.seed)
            m = S.save_bundle(b, a.out_dir)
            print(f"bundle written to {a.out_dir} (threshold {b.threshold:.4f}, best iteration {b.meta['best_iteration']}, sha256 {m['bundle_sha256'][:16]}...)")
        else:
            b = S.load_bundle(a.bundle)
            hist = S.read_table(a.history) if a.history else None
            out = S.score(b, hist, S.read_table(a.new), explain=a.explain)
            S.refuse_sealed_path(a.out)
            out.to_csv(a.out, index=False)
            print(f"scored {len(out)} transactions, {int(out['alert'].sum())} alerts at threshold {b.threshold:.4f}; wrote {a.out}")
        return 0
    except (S.ScoringError, S.SealedDataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
