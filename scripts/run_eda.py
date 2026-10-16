"""Generate the Phase 2 EDA/report from the built processed + feature data.

Requires scripts/build_processed_data.py to have run first. Writes:
  reports/phase2_eda_report.md   (human-readable)
  reports/phase2_eda_stats.json  (machine-readable, for reproducibility)
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.splits import compute_day

pd.set_option("display.width", 200)


def df_to_md(df: pd.DataFrame) -> str:
    """Manual markdown table formatter. pandas.to_markdown() requires the
    optional `tabulate` package, which is not installed and not worth adding
    just for report formatting."""
    df = df.reset_index()
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def section_row_counts(merged: pd.DataFrame, splits: dict, cfg: Config) -> tuple[str, dict]:
    lines = ["## Row counts\n"]
    lines.append(f"- Merged labeled training frame: **{len(merged):,}** rows, {merged.shape[1]} columns")
    lines.append(f"- With identity (`has_identity`): {int(merged['has_identity'].sum()):,} "
                f"({100*merged['has_identity'].mean():.2f}%)")
    stats = {"merged_rows": int(len(merged)), "merged_cols": int(merged.shape[1]),
              "has_identity_count": int(merged["has_identity"].sum())}
    lines.append("\n| split | rows | share |")
    lines.append("|---|---|---|")
    split_stats = {}
    for name, df in splits.items():
        share = 100 * len(df) / len(merged)
        lines.append(f"| {name} | {len(df):,} | {share:.2f}% |")
        split_stats[name] = {"rows": int(len(df))}
    stats["splits"] = split_stats
    return "\n".join(lines), stats


def section_target(merged: pd.DataFrame, splits: dict) -> tuple[str, dict]:
    lines = ["## Target (isFraud)\n"]
    vc = merged["isFraud"].value_counts().sort_index()
    lines.append(f"- Overall: {int(vc[0]):,} legitimate / {int(vc[1]):,} fraud "
                f"({100*merged['isFraud'].mean():.4f}% fraud, imbalance ratio {vc[0]/vc[1]:.2f}:1)")
    lines.append("\n| split | rows | fraud | fraud rate |")
    lines.append("|---|---|---|---|")
    stats = {"overall_fraud_rate": float(merged["isFraud"].mean())}
    split_stats = {}
    for name, df in splits.items():
        rate = df["isFraud"].mean()
        lines.append(f"| {name} | {len(df):,} | {int(df['isFraud'].sum()):,} | {100*rate:.3f}% |")
        split_stats[name] = {"fraud": int(df["isFraud"].sum()), "rate": float(rate)}
    stats["splits"] = split_stats
    return "\n".join(lines), stats


def section_missingness(merged: pd.DataFrame) -> tuple[str, dict]:
    miss = merged.isna().mean().sort_values(ascending=False)
    lines = ["## Missingness\n", "Top 20 columns by missing fraction:\n"]
    lines.append("| column | missing % |")
    lines.append("|---|---|")
    for col, frac in miss.head(20).items():
        lines.append(f"| {col} | {100*frac:.2f}% |")
    buckets = pd.cut(miss, bins=[0, 1e-9, 0.05, 0.25, 0.5, 0.75, 0.9, 1.0001],
                      labels=["0%", "0-5%", "5-25%", "25-50%", "50-75%", "75-90%", ">=90%"], include_lowest=True)
    counts = buckets.value_counts().reindex(["0%", "0-5%", "5-25%", "25-50%", "50-75%", "75-90%", ">=90%"])
    lines.append("\nColumns by missingness bucket:\n")
    lines.append("| bucket | n columns |")
    lines.append("|---|---|")
    for b, c in counts.items():
        lines.append(f"| {b} | {int(c)} |")
    stats = {"top20_missing": {k: float(v) for k, v in miss.head(20).items()},
              "bucket_counts": {str(k): int(v) for k, v in counts.items()}}
    return "\n".join(lines), stats


def section_temporal(merged: pd.DataFrame, splits: dict, cfg: Config) -> tuple[str, dict]:
    day = compute_day(merged, cfg)
    lines = ["## Temporal distribution\n"]
    lines.append(f"- Day range: {int(day.min())}-{int(day.max())} ({int(day.max()-day.min())+1} days)")
    lines.append("\n| split | day range | TransactionDT range |")
    lines.append("|---|---|---|")
    stats = {"day_min": int(day.min()), "day_max": int(day.max()), "splits": {}}
    for name, df in splits.items():
        d = compute_day(df, cfg)
        lines.append(f"| {name} | {int(d.min())}-{int(d.max())} | "
                    f"{int(df['TransactionDT'].min()):,}-{int(df['TransactionDT'].max()):,} |")
        stats["splits"][name] = {"day_min": int(d.min()), "day_max": int(d.max())}
    # 14-day window fraud rate, matching Phase 1's drift preview
    w = (day - 1) // 14
    win = pd.DataFrame({"w": w, "y": merged["isFraud"]}).groupby("w").agg(n=("y", "size"), fraud_pct=("y", lambda s: round(s.mean()*100, 3)))
    lines.append("\n14-day window fraud rate (full labeled train period):\n")
    lines.append(df_to_md(win))
    stats["window_14day_fraud_pct"] = win["fraud_pct"].tolist()
    return "\n".join(lines), stats


def section_cardinality(merged: pd.DataFrame) -> tuple[str, dict]:
    # pandas 3.0 uses a PyArrow-backed "str" extension dtype by default, which
    # is not the classic numpy `object` dtype — use the dtype-kind-agnostic
    # pandas helpers instead of comparing dtype == object directly.
    cat_cols = [c for c in merged.columns
                if pd.api.types.is_string_dtype(merged[c]) or pd.api.types.is_bool_dtype(merged[c])]
    nuniq = merged[cat_cols].nunique().sort_values(ascending=False)
    lines = ["## Categorical cardinalities\n", "| column | n unique |", "|---|---|"]
    for col, n in nuniq.items():
        lines.append(f"| {col} | {n:,} |")
    return "\n".join(lines), {str(k): int(v) for k, v in nuniq.items()}


def section_data_quality(cfg: Config) -> tuple[str, dict]:
    log_path = cfg.processed_dir / "build_log.json"
    if not log_path.exists():
        return "", {}
    with open(log_path, "r", encoding="utf-8") as fh:
        log = json.load(fh)
    lines = ["## Data quality interventions\n",
              "Negative D-column values clipped to 0 (Phase 1 risk R6 — a handful of rows "
              "reference a 'days since' event that appears to be in the future):\n"]
    clip = log.get("d_columns_clipped_negative_count", {})
    cols = sorted({c for split_counts in clip.values() for c, n in split_counts.items() if n})
    if cols:
        lines.append("| column | " + " | ".join(clip.keys()) + " |")
        lines.append("|---|" + "|".join(["---"] * len(clip)) + "|")
        for c in cols:
            lines.append("| " + c + " | " + " | ".join(str(clip[s].get(c, 0)) for s in clip) + " |")
    else:
        lines.append("(none)")
    return "\n".join(lines), log


def section_numerical(merged: pd.DataFrame) -> tuple[str, dict]:
    num_cols = ["TransactionAmt", "C1", "C2", "C13", "D1", "D15", "V1", "V258"]
    num_cols = [c for c in num_cols if c in merged.columns]
    desc = merged[num_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    lines = ["## Numerical distributions (representative columns)\n", df_to_md(desc.round(3))]
    return "\n".join(lines), json.loads(desc.round(6).to_json(orient="index"))


def main() -> None:
    cfg = load_config()
    merged = pd.read_parquet(cfg.processed_dir / "train_merged.parquet")
    splits = {name: pd.read_parquet(cfg.features_dir / f"{name}_features.parquet") for name in ("train", "val", "test")}

    report_sections = []
    all_stats = {}

    for title, fn in [
        ("row_counts", lambda: section_row_counts(merged, splits, cfg)),
        ("target", lambda: section_target(merged, splits)),
        ("missingness", lambda: section_missingness(merged)),
        ("temporal", lambda: section_temporal(merged, splits, cfg)),
        ("cardinality", lambda: section_cardinality(merged)),
        ("numerical", lambda: section_numerical(merged)),
        ("data_quality", lambda: section_data_quality(cfg)),
    ]:
        md, stats = fn()
        if not md:
            continue
        report_sections.append(md)
        all_stats[title] = stats

    header = (
        "# Phase 2 EDA Report\n\n"
        "Generated from `data/processed/train_merged.parquet` and `data/features/*_features.parquet` "
        "by `scripts/run_eda.py`. Source of the chronological split and column semantics: "
        "`reports/phase1_reconnaissance.md`.\n\n"
    )
    report_md = header + "\n\n".join(report_sections) + "\n"

    report_path = cfg.reports_dir / "phase2_eda_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    stats_path = cfg.reports_dir / "phase2_eda_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=1)

    print(f"Wrote {report_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
