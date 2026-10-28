#!/usr/bin/env python3
"""Train the 30-day SSD data-loss risk model on `notebooks/SMART` feature data.

DATA REQUIREMENT (the important part)
  `notebooks/SMART/parsed_smart_records.csv` contains no failure annotations -
it catalogues device capabilities only. To train a 30-day risk model you must
  supply a label source:

  1. Proxy labels (--proxy): a weak, demonstrable label derived from the data
     itself. This is NOT ground truth for a 30-day failure and must not ship
     as a real model.
  2. Real labels (--labels <csv>): a CSV with a 'path' column matching the
     report path, plus a binary 'label' column (0/1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

ASSETS = Path("assets")
FEATURE_NAMES = [
    "pct_used", "spare_deficit", "media_err", "temp",
    "poh", "crit_warn", "realloc", "smart_fail",
]
N_FEATURES = len(FEATURE_NAMES)


def _ints_only(value):
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.astype(np.int64)
        return value
    if isinstance(value, (list, tuple)):
        return [int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
    return value


def _install_onnx_bool_shim() -> None:
    """Workaround for skl2onnx 1.20 / onnx >= 1.23 incompatibility."""
    import onnx.helper as oh

    if getattr(oh.make_attribute, "_smartly_bool_shim", False):
        return
    original = oh.make_attribute

    def make_attribute(key, value, *args, **kwargs):
        return original(key, _ints_only(value), *args, **kwargs)

    make_attribute._smartly_bool_shim = True
    oh.make_attribute = make_attribute
    import skl2onnx.common._container as container

    container.make_attribute = make_attribute


def build_smart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map parsed SMART records to the 8-feature risk contract."""
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
    X["crit_warn"] = 0.0  # 0% coverage in this dataset (no NVMe critical-warning bitmask)
    X["smart_fail"] = df.get("smart_fail", 0).astype(float)
    return X[FEATURE_NAMES].clip(0.0, 1.0)


def proxy_labels(df: pd.DataFrame) -> pd.Series:
    """Demonstration-only proxy label.

    A device is 'risky' if it fails its self-test OR carries media/ECC errors
    OR has a large reallocated-sector count. This is a heuristic that would
    need ground truth to be replaced before any real deployment.
    """
    risk = pd.Series(False, index=df.index)
    risk = risk | (df.get("self_test", "") == "FAILED")
    if "media_and_data_integrity_errors" in df.columns:
        risk = risk | pd.to_numeric(df["media_and_data_integrity_errors"], errors="coerce").fillna(0) > 0
    if "attr_005" in df.columns:
        risk = risk | pd.to_numeric(df["attr_005"], errors="coerce").fillna(0) > 0
    return risk.astype(int)


def derive_cutoffs(y_true: np.ndarray, proba: np.ndarray, targets=(0.54, 0.71, 0.86)) -> dict:
    """Smallest score whose precision reaches each target."""
    cutoffs = {}
    order = np.argsort(-proba)
    yp = y_true[order]
    for name, target in zip(("low", "medium", "high"), targets):
        for i in range(len(y_true)):
            p = float(yp[: i + 1].mean())
            if p >= target:
                cutoffs[name] = round(float(proba[order[i]]), 4)
                break
        else:
            cutoffs[name] = 1.0
    return cutoffs


def export_onnx(clf, path: Path) -> None:
    _install_onnx_bool_shim()
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onnx = convert_sklearn(
        clf, "risk_model", [("float_input", FloatTensorType([None, N_FEATURES]))], target_opset=15
    )
    path.write_bytes(onnx.SerializeToString())


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="train smartly-rs risk model on notebooks/SMART features")
    ap.add_argument("--data", type=Path, default=Path("notebooks/SMART/parsed_smart_records.csv"))
    ap.add_argument("--labels", type=Path, help="real labels CSV (columns: path, label)")
    ap.add_argument("--proxy", action="store_true", help="use demonstration-only proxy labels")
    ap.add_argument("--out-dir", type=Path, default=ASSETS, help="outputs dir")
    args = ap.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8", encoding_errors="surrogateescape")
    X = build_smart_features(df)

    if args.proxy:
        y = proxy_labels(df).values
        print("PROXY labels: demonstration only (no 30-day ground truth)")
    elif args.labels:
        lab = pd.read_csv(args.labels)
        if not {"path", "label"} <= set(lab.columns):
            raise SystemExit("labels CSV must contain 'path' and 'label' columns")
        merged = df.merge(lab, on="path", how="left", validate="one_to_one")
        missing = int(merged["label"].isna().sum())
        if missing:
            raise SystemExit(f"{missing} rows lack a label")
        y = merged["label"].astype(int).values
    else:
        raise SystemExit("provide --labels <csv> (real) or --proxy (demonstration)")

    y = np.asarray(y).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic": LogisticRegression(max_iter=2000),
        "logistic_balanced": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "hgb": HistGradientBoostingClassifier(random_state=42),
    }
    results, fitted = {}, {}
    for name, clf in candidates.items():
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        ap = average_precision_score(yte, p)
        results[name] = {"pr_auc": float(ap)}
        fitted[name] = clf
        print(f"  {name:>15} PR-AUC {ap:.4f}")

    best = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"exported model: {best}")

    p_best = fitted[best].predict_proba(Xte)[:, 1]
    cutoffs = derive_cutoffs(yte, p_best)
    for k, v in cutoffs.items():
        print(f"  level[{k}] >= {v}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(fitted[best], args.out_dir / "risk_model.onnx")

    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        fitted[best], Xte, yte, scoring="average_precision", n_repeats=5, random_state=42, n_jobs=-1
    )
    importance = sorted(
        ({"feature": n, "importance": round(float(max(0.0, m)), 4)} for n, m in zip(FEATURE_NAMES, perm.importances_mean)),
        key=lambda d: -d["importance"],
    )
    write_json(args.out_dir / "risk_coefficients.json", {
        "data": "notebooks/SMART",
        "exported_model": best,
        "global_importance": importance,
        "risk_level_cutoffs": cutoffs,
    })
    write_json(args.out_dir / "train_metrics.json", {
        "candidates": results,
        "exported": best,
        "test_positive_rate": float(yte.mean()),
        "test_rows": int(len(yte)),
        "risk_level_cutoffs": cutoffs,
        "global_importance": importance,
    })
    print("written: risk_model.onnx, risk_coefficients.json, train_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
