#!/usr/bin/env python3
"""Sample real feature rows for the Rust integration tests.

Writes `tests/fixtures/features_real.json`: equal numbers of POSITIVE
(pre-failure) and NEGATIVE (healthy) snapshots taken **only from the test
split** of `assets/train.csv`, using exactly the same grouped split as
`train_risk_model.py` (GroupShuffleSplit on `model#disk_id`, test_size=0.2,
random_state=42).  Rows held out from training are the honest way to assert
that the embedded ONNX model still separates them after `cargo build`.

Usage: .venv/bin/python assets/make_features_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "assets/risk/train.csv"
OUT = ROOT / "tests/fixtures/features_real.json"

FEATURE_NAMES = [
    "pct_used", "spare_deficit", "media_err", "temp", "poh",
    "crit_warn", "realloc", "smart_fail",
]
N_PER_CLASS = 40
SEED = 7


def main() -> int:
    df = pd.read_csv(TRAIN_CSV)
    groups = (df["model"].astype(str) + "#" + df["disk_id"].astype(str)).to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te = next(gss.split(df[FEATURE_NAMES].to_numpy(), df["label"].to_numpy(),
                           groups))
    test = df.iloc[te]

    rng = np.random.default_rng(SEED)
    rows = []
    for label in (0, 1):
        pool = test[test["label"] == label]
        idx = rng.choice(len(pool), size=N_PER_CLASS, replace=False)
        for r in pool.iloc[idx].itertuples(index=False):
            rows.append({
                "label": int(label),
                "model": str(r.model),
                "features": [float(getattr(r, f)) for f in FEATURE_NAMES],
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "alibaba_dsn21_30d test split (GroupShuffleSplit seed 42)",
        "feature_names": FEATURE_NAMES,
        "samples": rows,
    }, indent=1) + "\n")

    pos = [r["features"] for r in rows if r["label"] == 1]
    neg = [r["features"] for r in rows if r["label"] == 0]
    print(f"written {OUT}: {len(rows)} samples "
          f"({len(pos)} positive / {len(neg)} negative), test rows {len(test)}")
    for name, i in (("pct_used", 0), ("media_err", 2), ("crit_warn", 6)):
        print(f"  {name}: pos mean {np.mean([f[i] for f in pos]):.4f} "
              f"vs neg mean {np.mean([f[i] for f in neg]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
