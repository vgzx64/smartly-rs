#!/usr/bin/env python3
"""Learn the 'normal' SMART profile from healthy records and save an anomaly
alarm threshold for the smartly-rs 8-feature contract.

This is the unsupervised direction (the opposite of the supervised 30-day
risk model): there are no 30-day failure labels in ``notebooks/SMART``, so
instead we ask ``what does a healthy SSD look like?``.  Any later device whose
feature vector drifts far from that normal cluster is alarmed.

Inputs:
  --data         parsed SMART records CSV (default notebooks/SMART/parsed_smart_records.csv)
  --out-dir      artifact directory (default: assets/normality/)

Outputs (into --out-dir):
  normal_model.joblib   fitted IsolationForest
  anomaly_config.json   feature names, method, contamination, alarm threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ASSETS = Path("assets/normality")
FEATURE_NAMES = [
    "pct_used", "spare_deficit", "media_err", "temp",
    "poh", "crit_warn", "realloc", "smart_fail",
]
N_FEATURES = len(FEATURE_NAMES)


def build_smart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised mapping of parsed SMART records to the 8-feature contract.

    Mirrors src/features.rs (log10 scalings, clip 0..1) but is tolerant of
    missing source columns.
    """
    X = pd.DataFrame(index=df.index)
    if "poh" in df.columns:
        X["poh"] = np.log10(1.0 + df["poh"].astype(float).fillna(0.0)) / np.log10(1.0 + 87_600.0)
    else:
        X["poh"] = 0.0
    if "realloc" in df.columns:
        X["realloc"] = np.log10(1.0 + df["realloc"].astype(float).fillna(0.0)) / 6.0
    else:
        X["realloc"] = 0.0
    X["pct_used"] = df.get("pct_used", np.nan).astype(float).fillna(0.0)
    X["spare_deficit"] = df.get("spare_deficit", np.nan).astype(float).fillna(0.0)
    X["media_err"] = df.get("media_err", np.nan).astype(float).fillna(0.0)
    X["temp"] = df.get("temp", np.nan).astype(float).fillna(0.0)
    X["crit_warn"] = 0.0
    X["smart_fail"] = df.get("smart_fail", 0).astype(float)
    return X[FEATURE_NAMES].clip(0.0, 1.0)


def classify_condition(df: pd.DataFrame) -> pd.Series:
    """Heuristic 'condition' label for each record.

    A record reports a *bad condition* when it carries any canonical SMART
    failure indicator stored in the corpus:

      * self_test == FAILED
      * media_err   > 0           (media/data-integrity errors)
      * crit_warn   > 0           (critical-warning bitmask set)
      * pct_used    >= 0.90        (near end of rated life)

    Everything else is *normal*.

    Notes on edge cases
      * NaN media_err / realloc / pct_used are treated as 0 (not reported).
      * 'realloc' itself is NOT a bad-condition filter: a single reallocated
        sector is common across healthy drives and would otherwise erase the
        training set.
    """
    bad = pd.Series(False, index=df.index)
    bad = bad | (df.get("self_test","") == "FAILED")
    bad = bad | (df.get("media_err",0).fillna(0) > 0)
    bad = bad | (df.get("crit_warn",0).fillna(0) > 0)
    bad = bad | (df.get("pct_used",0).fillna(0) >= 0.90)
    return ~bad


def train_normal_model(df: pd.DataFrame, contamination: float = 0.02) -> tuple:
    """Fit the normal profile and derive the alarm threshold."""
    X = build_smart_features(df)
    healthy = X[classify_condition(df)].copy()

    if len(healthy) < 100:
        raise SystemExit(f"too few healthy records ({len(healthy)}) for a stable normal model")

    model = IsolationForest(
        n_estimators=300,
        max_samples=256,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(healthy)

    # Reference distribution of the HEALTHY population.
    if hasattr(model, "score_samples"):
        ref_score = model.score_samples(healthy)
    else:
        ref_score = model.decision_function(healthy)

    low, hi = np.percentile(ref_score, [1.0, 50.0])
    threshold = float(np.percentile(ref_score, 1.0))
    return model, healthy, ref_score, threshold, float(low), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description="train the normal-SSD anomaly model")
    ap.add_argument("--data", type=Path, default=Path("notebooks/SMART/parsed_smart_records.csv"))
    ap.add_argument("--out-dir", type=Path, default=ASSETS, help="artifact directory")
    ap.add_argument("--contamination", type=float, default=0.02,
                    help="expected anomaly rate (internal predict routing)")
    args = ap.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8", encoding_errors="surrogateescape")
    model, healthy_mask, ref_score, threshold, low, hi = train_normal_model(
        df, contamination=args.contamination
    )

    n_total = len(df)
    n_healthy = len(healthy_mask)
    n_bad = n_total - n_healthy
    print(f"records: {n_total}  healthy={n_healthy}  bad={n_bad}")
    print(f"alarm threshold: decision_function < {threshold:.4f} "
          f"(approx. 1% alarm rate on the healthy population)")
    print(f"healthy reference range: [{low:.4f}, {hi:.4f}] "
          f"(0-50%ile: [{ref_score.min():.4f}, {ref_score.max():.4f}])")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "normal_model.joblib")
    config_path = args.out_dir / "anomaly_config.json"
    with config_path.open("w") as fh:
        json.dump({
            "method": "isolation_forest",
            "feature_names": FEATURE_NAMES,
            "contamination": args.contamination,
            "alarm_threshold": threshold,
            "healthy_count": len(healthy_mask),
            "n_features": N_FEATURES,
            "score_reference": {"min": float(ref_score.min()), "max": float(ref_score.max())},
        }, fh, indent=2)
    print(f"written: {args.out_dir / 'normal_model.joblib'}")
    print(f"written: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
