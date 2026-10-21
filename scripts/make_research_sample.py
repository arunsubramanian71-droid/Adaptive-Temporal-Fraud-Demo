"""Write a few VALIDATION rows (label column removed) as a JSON file for the dashboard's Research model tab or POST /api/research/score.

  .venv/Scripts/python.exe -B scripts/make_research_sample.py [--n 25] [--out PATH]

Reads data/features/val_features.parquet only (validation days 122-151; never a held-out file) with the label column projected away at read time,
and picks rows deterministically (the first N in file order plus, if available, N rows the recorded research model alerted on, so the demo shows
explanations). The default output is outside the repository (your temporary folder): the rows are transaction-level data and must not be committed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.heldout.evaluator import read_features  # noqa: E402

VAL = "data/features/val_features.parquet"
REC = "experiments/xgboost/predictions_val.parquet"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default=str(pathlib.Path(tempfile.gettempdir()) / "fraud_research_rows.json"))
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out).resolve()
    if out.is_relative_to(ROOT):
        print("error: write the rows outside the repository (they are transaction-level data and must not be committed)", file=sys.stderr)
        return 2
    if not (ROOT / VAL).exists():
        print(f"error: {VAL} does not exist; build it with scripts/build_processed_data.py on the IEEE-CIS files", file=sys.stderr)
        return 2
    df = read_features(ROOT / VAL)
    pick = list(df.index[: a.n])
    if (ROOT / REC).exists():                                  # add rows the recorded model scored highest, so alerts and explanations appear
        rec = pd.read_parquet(ROOT / REC, columns=["TransactionID", "y_score"]).sort_values("y_score", ascending=False).head(a.n)
        pick += list(df.index[df["TransactionID"].isin(rec["TransactionID"])])
    rows = df.loc[list(dict.fromkeys(pick))]
    rows = rows.replace({np.nan: None})
    out.write_text(json.dumps({"rows": rows.to_dict(orient="records"), "source": VAL + " (validation days 122-151, label removed)"}, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
                   encoding="utf-8")
    print(f"wrote {len(rows)} validation rows (no label) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
