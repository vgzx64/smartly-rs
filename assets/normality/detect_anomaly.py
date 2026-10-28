#!/usr/bin/env python3
"""Alarm on significant deviation from the learned 'normal' SMART profile.

Uses the artifacts produced by ``train_normal_model.py``:

  assets/normal_model.joblib     IsolationForest trained on healthy records
  assets/anomaly_config.json     method, feature ordering, alarm threshold

Input sources (pick one):
  --csv         parsed SMART records CSV (same layout as parsed_smart_records.csv)
  --features-json  JSON object or array of objects carrying the 8 features
  --device         single JSON object on stdin (robust to whitespace/order)

For every input device the script prints:

  <path|index>  score=<x>  status=<NORMAL|ALARM>  (threshold=<t>)

and finally a summary line with the number of alarms.

A status of ALARM means the device's SMART-derived 8-feature vector scores
significantly (>1 percentile) lower than the healthy training population;
it does NOT by itself diagnose a cause.  Wire the printed score into the
Rust binary's report hook to correlate with the existing disclaimer that
'Normal/ALARM only reflects deviation from a training corpus'.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load

ASSETS = Path("assets/normality")
FEATURE_NAMES = [
    "pct_used", "spare_deficit", "media_err", "temp",
    "poh", "crit_warn", "realloc", "smart_fail",
]
N_FEATURES = len(FEATURE_NAMES)


def load_model(out_dir: Path):
    config_path = out_dir / "anomaly_config.json"
    model_path = out_dir / "normal_model.joblib"
    if not config_path.exists():
        raise SystemExit(f"missing anomaly config: {config_path}")
    if not model_path.exists():
        raise SystemExit(f"missing model: {model_path}")
    with config_path.open() as fh:
        config = json.load(fh)
    model = joblib_load(model_path)
    return config, model


def to_feature_frame(obj):
    """Coerce a parsed-record row or a features JSON object into an 8-column frame."""
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, pd.DataFrame):
        obj = pd.DataFrame(obj)
    missing = [c for c in FEATURE_NAMES if c not in obj.columns]
    if missing:
        raise SystemExit(f"missing feature columns: {missing}; expected {FEATURE_NAMES}")
    X = obj[FEATURE_NAMES].astype(float).fillna(0.0)
    X = X.clip(0.0, 1.0)
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description="alarm on deviation from learned normal SSD profile")
    ap.add_argument("--config", type=Path, default=ASSETS / "anomaly_config.json",
                    help="anomaly_config.json output by train_normal_model.py")
    ap.add_argument("--model", type=Path, default=ASSETS / "normal_model.joblib",
                    help="normal_model.joblib output by train_normal_model.py")
    ap.add_argument("--csv", type=Path, help="parsed SMART records CSV (same layout as parsed_smart_records.csv)")
    ap.add_argument("--features-json", type=Path, help="JSON object(s) with the 8 features")
    ap.add_argument("--device", help="a single device JSON on stdin (or type directly)")
    args = ap.parse_args()

    config, model = load_model(args.config.parent)
    threshold = float(config["alarm_threshold"])
    if model.n_features_in_ != N_FEATURES:
        raise SystemExit(f"model expects {model.n_features_in_} features, got {N_FEATURES}")

    if args.csv:
        raw = pd.read_csv(args.csv, encoding="utf-8", encoding_errors="surrogateescape")
        X = to_feature_frame(raw)
        sources = raw.get("path", pd.Series([f"row_{i}" for i in range(len(raw))]))
    elif args.features_json:
        with open(args.features_json) as fh:
            obj = json.load(fh)
        X = to_feature_frame(obj)
        sources = pd.Series([f"item_{i}" for i in range(len(X))])
    elif args.device:
        with open(args.device) as fh:
            obj = json.load(fh)
        X = to_feature_frame(obj)
        sources = pd.Series(["device"])
    else:
        obj = sys.stdin.read()
        if not obj.strip():
            raise SystemExit("no input: pass --csv, --features-json, --device, or pipe JSON on stdin")
        obj = json.loads(obj)
        X = to_feature_frame(obj)
        sources = pd.Series(["device"])

    if hasattr(model, "score_samples"):
        scores = model.score_samples(X)
    else:
        scores = model.decision_function(X)

    def sanitize(s):
        return str(s).encode("utf-8", "replace").decode("utf-8")

    alarms = 0
    print(f"alarm threshold: {threshold:.4f}   (sample distribution: "
          f"min={scores.min():.4f}, max={scores.max():.4f}, median={np.median(scores):.4f})")
    print(f"{'source':<28}{'score':>10}  status")
    for src, s in zip(sources, scores):
        status = "ALARM" if s < threshold else "NORMAL"
        if status == "ALARM":
            alarms += 1
        print(f"{sanitize(src):<28}{float(s):>10.4f}  {status}")

    print(f"\nsummary: {alarms} alarm(s) of {len(sources)} checked")
    return 1 if alarms else 0


if __name__ == "__main__":
    raise SystemExit(main())
