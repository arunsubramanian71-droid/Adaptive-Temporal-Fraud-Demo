"""Small deterministic SYNTHETIC transaction fixture for the software demonstration and tests.

This is NOT the IEEE-CIS data and NOT the Phase 9 research benchmark. It exists only to exercise the real project code (graph
features, tabular models, scoring, explanations, candidate-community discovery) end to end on a laptop in seconds. Nothing computed
from it is a research result and none of it may be reported as one.

Planted structure (so the mechanics can be seen working):
  * tabular signal: fraud odds rise with ``f0`` and with the amount;
  * graph signal: from day 12 a set of "compromised" devices carries elevated fraud, so a device's prior activity is informative;
  * a coordinated burst: ``n_planted_cards`` cards share a few devices and netproxies on days 25-32 (``is_planted`` marks them;
    it is for demo evaluation only and is never given to a model or to discovery).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400
ENTITY_COLUMNS = {"pa": "account_key", "card": "card_key", "net": "net_key", "dev": "device_key"}
FEATURE_COLUMNS = [f"f{i}" for i in range(8)]


def make_transactions(seed: int = 0, n_days: int = 40, per_day: int = 160, n_cards: int = 500, n_devices: int = 90, n_nets: int = 40,
                      n_planted_cards: int = 14, burst_days=(25, 32)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    home_dev = rng.integers(0, n_devices, n_cards)
    home_net = rng.integers(0, n_nets, n_cards)
    has_dev = rng.random(n_cards) < 0.55
    has_net = rng.random(n_cards) < 0.35
    has_acc = rng.random(n_cards) < 0.85
    bad_dev = set(rng.choice(n_devices, size=max(2, n_devices // 9), replace=False).tolist())
    planted = rng.choice(n_cards, size=n_planted_cards, replace=False)
    burst_dev = rng.choice(n_devices, size=3, replace=False)
    burst_net = rng.choice(n_nets, size=2, replace=False)
    tid = 3_000_000
    for d in range(1, n_days + 1):
        n = per_day
        cards = (rng.zipf(1.6, n) - 1) % n_cards
        for k in range(n):
            c = int(cards[k])
            dev = int(home_dev[c]) if (has_dev[c] and rng.random() < 0.8) else (int(rng.integers(0, n_devices)) if rng.random() < 0.3 else -1)
            net = int(home_net[c]) if (has_net[c] and rng.random() < 0.8) else -1
            rows.append((tid, (d - 1) * DAY + int(rng.integers(0, DAY)), c, dev, net, c if has_acc[c] else -1, False))
            tid += 1
        if burst_days[0] <= d <= burst_days[1]:
            for c in planted:
                for _ in range(int(rng.integers(2, 5))):
                    dev = int(rng.choice(burst_dev))
                    net = int(rng.choice(burst_net))
                    rows.append((tid, (d - 1) * DAY + int(rng.integers(0, DAY)), int(c), dev, net, int(c), True))
                    tid += 1
    df = pd.DataFrame(rows, columns=["TransactionID", "TransactionDT", "card", "dev", "net", "acc", "is_planted"])
    n = len(df)
    for i, f in enumerate(FEATURE_COLUMNS):
        df[f] = rng.normal(size=n)
    df["TransactionAmt"] = np.round(rng.lognormal(3.4, 1.0, n), 2)
    day = df["TransactionDT"] // DAY + 1
    logit = (-4.3 + 1.3 * df["f0"] + 0.6 * (np.log1p(df["TransactionAmt"]) - 3.5)
             + 2.4 * (df["dev"].isin(bad_dev) & (day >= 12)) + 1.5 * df["is_planted"])
    df["isFraud"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int8)
    for col, src in (("card_key", "card"), ("device_key", "dev"), ("net_key", "net"), ("account_key", "acc")):
        df[col] = np.where(df[src] >= 0, df[src].astype(str), None)
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    df["day"] = (df["TransactionDT"] // DAY + 1).astype(int)
    return df[["TransactionID", "TransactionDT", "day", "TransactionAmt", *FEATURE_COLUMNS, "card_key", "account_key", "net_key", "device_key",
               "is_planted", "isFraud"]]
