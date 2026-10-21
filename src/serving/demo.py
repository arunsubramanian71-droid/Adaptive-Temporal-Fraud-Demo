"""Services for the synthetic demonstration: trains the scoring bundle in memory on the synthetic fixture (src/demo/synthetic.py).

Everything built here is SYNTHETIC. It reproduces no research result, uses no IEEE-CIS data and no held-out period; every API response made with it
carries ``synthetic_demo: true``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.demo.synthetic import FEATURE_COLUMNS, make_transactions
from src.inference import scorer as S
from src.serving.app import Services

TRAIN_DAYS, VALID_DAYS, SAMPLE_DAY = (1, 24), (25, 32), 33


def _records(df: pd.DataFrame, columns) -> list:
    out = []
    for row in df[list(columns)].itertuples(index=False):
        rec = {}
        for c, v in zip(columns, row):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[c] = None
            elif isinstance(v, (np.integer,)):
                rec[c] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[c] = float(v)
            else:
                rec[c] = v
        out.append(rec)
    return out


def build_demo_services(seed: int = 0, n_history: int = 300, n_new: int = 30) -> Services:
    df = make_transactions(seed=seed)
    tr = df[(df["day"] >= TRAIN_DAYS[0]) & (df["day"] <= TRAIN_DAYS[1])].reset_index(drop=True)
    va = df[(df["day"] >= VALID_DAYS[0]) & (df["day"] <= VALID_DAYS[1])].reset_index(drop=True)
    schema = S.Schema(features=tuple(FEATURE_COLUMNS))
    bundle = S.train_bundle(tr, va, schema, use_graph=True, seed=42)
    cols = schema.required(False)               # no label, no planted flag: the sample rows carry only what scoring needs
    history = pd.concat([tr, va], ignore_index=True).tail(n_history)
    new = df[df["day"] == SAMPLE_DAY].head(n_new)
    sample = {"description": "synthetic rows: the last stored transactions as history and the first transactions of a later synthetic day as new rows",
              "history": _records(history, cols), "new": _records(new, cols)}
    return Services(bundle=bundle, synthetic_demo=True, sample=sample)
