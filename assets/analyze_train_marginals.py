#!/usr/bin/env python3
"""Marginal diagnostics for the assembled training sample.

Answers "does a feature actually separate failing disks at all?" — the
input for `docs/dataset.md` (a feature that is nearly constant cannot
support a meaningful linear weight). Read-only: it only prints stats.

    .venv/bin/python assets/analyze_train_marginals.py [assets/train.csv]
"""
import sys

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "pct_used", "spare_deficit", "media_err", "temp", "poh",
    "crit_warn", "realloc", "smart_fail",
]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/train.csv"
    df = pd.read_csv(path, usecols=FEATURE_NAMES + ["label"])
    print(f"{path}: rows {len(df)} base rate {df['label'].mean():.4f}\n")

    for name in FEATURE_NAMES:
        col = df[name]
        groups = pd.qcut(col, 10, duplicates="drop")
        print(f"== {name}  (min {col.min():.3f} max {col.max():.3f})")
        if groups.nunique() <= 1:
            print(f"   near-constant: rate {df['label'].mean():.4f} "
                  f"(n {len(df)}) — линейный вес не интерпретируем\n")
            continue
        stats = df.groupby(groups, observed=True)["label"].agg(["mean", "size"])
        for interval, row in stats.iterrows():
            print(f"   {str(interval):>28}  rate {row['mean']:.4f}  "
                  f"n {int(row['size'])}")
        print()

    # Collinearity between features (should be none after the exclusions).
    corr = df[FEATURE_NAMES].corr().to_numpy(copy=True)
    print("strongest feature pairs (|corr|):")
    tri = [(abs(corr[i, j]), FEATURE_NAMES[i], FEATURE_NAMES[j])
           for i in range(len(FEATURE_NAMES))
           for j in range(i + 1, len(FEATURE_NAMES))]
    for value, a, b in sorted((t for t in tri if not np.isnan(t[0])),
                              reverse=True)[:3]:
        print(f"  {a} ~ {b}: {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
