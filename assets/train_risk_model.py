#!/usr/bin/env python3
"""Train the SSD failure-risk model and export it to ONNX.

The model predicts the probability that an SSD will suffer a
data-loss-causing failure within the next 30 days, given 8 normalized
SMART-derived features.

Two modes:

* ``--data assets/train.csv`` (the shipped model): real Alibaba DSN'21
  SATA-SSD samples; candidates are plain/balanced logistic regression and
  a histogram gradient-boosting ensemble, the winner by PR-AUC is
  exported, and the risk-level cut-offs plus permutation importance are
  derived on held-out disks. See ``docs/dataset.md``.
* no ``--data``: a synthetic generator based on an expert logistic
  formula, kept as a smoke test / demo. It also writes the same JSON keys
  so ``cargo test`` stays green after switching modes.

Outputs (into assets/):
  risk_model.onnx            - ONNX graph, embedded by the Rust binary
  risk_coefficients.json     - weights/bias, risk-level cut-offs,
                               permutation importance
  train_metrics.json         - candidates, per-model AP, cut-offs (real mode)

Usage:
  .venv/bin/python assets/train_risk_model.py --data assets/train.csv
  .venv/bin/python assets/train_risk_model.py            # synthetic demo
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = [
    "pct_used",         # percentage_used / 100
    "spare_deficit",    # how close available_spare is to its threshold
    "media_err",        # log10(1 + media_errors) / 5
    "temp",             # (temp_c - 30) / 60
    "poh",              # log10(1 + power_on_hours) / log10(1 + 87600)
    "crit_warn",        # popcount(critical_warning bits) / 5
    "realloc",          # log10(1 + reallocated sectors) / 6
    "smart_fail",       # 1 if SMART self-test reported FAILED
]
N_FEATURES = len(FEATURE_NAMES)

# Expert ground-truth weights (log-odds) approximating real-world SSD
# failure predictors: wear, spare deficit and media errors dominate.
TRUE_W = np.array([
    4.0,   # pct_used
    3.0,   # spare_deficit
    2.0,   # media_err
    1.2,   # temp
    1.5,   # poh
    2.5,   # crit_warn
    1.2,   # realloc
    3.0,   # smart_fail
], dtype=np.float64)
TRUE_B = -4.5

P_E_CYCLES = 1500.0  # assumed TLC P/E endurance for the wear_cycles feature


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ints_only(value):
    """Coerce Python/numpy booleans inside int-list attributes to ints."""
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.astype(np.int64).tolist()
        return value
    if isinstance(value, (list, tuple)):
        return [int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
    return value


def _install_onnx_bool_shim() -> None:
    """Work around an skl2onnx 1.20 / onnx>=1.23 incompatibility.

    sklearn >= 1.9 exposes HistGradientBoosting missing-value routing as a
    uint8 array; skl2onnx forwards it (mixed uint8 + bool) as the
    `nodes_missing_value_tracks_true` int-list attribute. onnx 1.23 uses
    strict protobuf, which rejects Python bools there:

        TypeError: Field onnx.AttributeProto.ints: Expected an int,
                   got a boolean

    Coercing bools to ints before the attribute is built keeps the emitted
    graph semantically identical (the ONNX spec wants 0/1 there anyway).
    """
    import onnx.helper as oh

    if getattr(oh.make_attribute, "_smartly_bool_shim", False):
        return
    original = oh.make_attribute

    def make_attribute(key, value, *args, **kwargs):
        return original(key, _ints_only(value), *args, **kwargs)

    make_attribute._smartly_bool_shim = True
    oh.make_attribute = make_attribute
    # `make_node` resolves `make_attribute` through onnx.helper's globals,
    # but some converters import it directly -> patch that binding too.
    import skl2onnx.common._container as container

    container.make_attribute = make_attribute


def export_model(model, onnx_path: str) -> None:
    """Convert a fitted sklearn classifier to ONNX with an `input`
    tensor [None, 10] and a float output named `probability`."""
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    _install_onnx_bool_shim()
    initial = [("input", FloatTensorType([None, N_FEATURES]))]
    onnx_model = convert_sklearn(
        model, initial_types=initial, target_opset=15,
        options={id(model): {"zipmap": False}},
    )
    # Rename the float probability output (output[0] is the int64 label).
    # Renaming must touch both the graph output AND the producing node's
    # output value, otherwise the model is invalid for ONNX Runtime.
    for o in onnx_model.graph.output:
        if o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and o.name != "probability":
            old = o.name
            for n in onnx_model.graph.node:
                if old in n.output:
                    n.output[list(n.output).index(old)] = "probability"
            o.name = "probability"
    names = [o.name for o in onnx_model.graph.output]
    assert names.count("probability") == 1, f"bad outputs: {names}"
    onnx.checker.check_model(onnx_model)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"written: {onnx_path}")


def write_coefficients(path: str, weights, bias, extra: dict | None = None) -> None:
    coefs = {
        "feature_names": FEATURE_NAMES,
        "weights": [float(v) for v in weights],
        "bias": float(bias),
    }
    if extra:
        coefs.update(extra)
    with open(path, "w") as f:
        json.dump(coefs, f, indent=2)
    print(f"written: {path}")


def gen_features(rng: np.random.Generator, n: int) -> np.ndarray:
    x = np.zeros((n, N_FEATURES), dtype=np.float64)

    # Wear: most drives are lightly used, a few are near end of life.
    pct_used = np.clip(rng.beta(2.0, 7.0, n) * 1.05, 0.0, 1.0)
    x[:, 0] = pct_used

    # Spare deficit tracks wear with scatter (and can stay 0 early on).
    x[:, 1] = np.clip(pct_used * 0.8 + rng.normal(0.0, 0.12, n), 0.0, 1.0)

    # Media errors: zero for most drives, heavy tail for sick ones.
    u = rng.random(n)
    media = np.zeros(n)
    media[u < 0.86] = 0.0
    m = (u >= 0.86) & (u < 0.96)
    media[m] = rng.poisson(3.0, m.sum())
    m = (u >= 0.96) & (u < 0.995)
    media[m] = rng.poisson(200.0, m.sum())
    media[u >= 0.995] = rng.poisson(20000.0, (u >= 0.995).sum())
    x[:, 2] = np.clip(np.log10(1.0 + media) / 5.0, 0.0, 1.0)



    # Temperature under load: normal around 40-50C, occasional hot drives.
    temp_c = np.clip(42.0 + rng.normal(0.0, 9.0, n) + pct_used * 6.0, 25.0, 95.0)
    x[:, 3] = np.clip((temp_c - 30.0) / 60.0, 0.0, 1.0)

    # Power-on hours: log-normal, ~1-6 years typical.
    poh = np.clip(10.0 ** rng.uniform(2.7, 4.9, n), 0.0, 87600.0 * 3.0)
    x[:, 4] = np.clip(np.log10(1.0 + poh) / np.log10(1.0 + 87600.0), 0.0, 1.0)

    # Critical warning bits: ~92% of healthy drives report none.
    bits = np.zeros(n)
    r = rng.random(n)
    bits[(r >= 0.92) & (r < 0.97)] = 1.0
    bits[(r >= 0.97) & (r < 0.99)] = 3.0
    bits[r >= 0.99] = 5.0
    x[:, 5] = bits / 5.0

    # Reallocated sectors.
    u = rng.random(n)
    realloc = np.zeros(n)
    m = (u >= 0.90) & (u < 0.97)
    realloc[m] = rng.poisson(8.0, m.sum())
    tail = u >= 0.97
    realloc[tail] = 10.0 ** rng.uniform(2.0, 4.5, tail.sum())
    x[:, 6] = np.clip(np.log10(1.0 + realloc) / 6.0, 0.0, 1.0)

    # SMART self-test FAILED flag: rare, mostly on already-sick drives.
    fail_prob = sigmoid(3.5 * pct_used + 2.0 * x[:, 2] - 5.5)
    x[:, 7] = (rng.random(n) < fail_prob).astype(np.float64)

    # Sensor noise: SMART values are quantized/jittery.
    x += rng.normal(0.0, 0.015, x.shape)
    return np.clip(x, 0.0, 1.0)


def train_synthetic() -> int:
    assets_dir = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(42)

    x = gen_features(rng, 8000)
    logit = x @ TRUE_W + TRUE_B + rng.normal(0.0, 0.6, x.shape[0])
    y = (rng.random(x.shape[0]) < sigmoid(logit)).astype(np.int64)

    # Hold out a quarter of the rows: the cut-offs and importance below are
    # then measured on data the model has not memorised.
    split = int(len(x) * 0.75)
    x_tr, x_te = x[:split], x[split:]
    y_tr, y_te = y[:split], y[split:]

    model = LogisticRegression(max_iter=5000, C=1.0)
    model.fit(x_tr, y_tr)
    acc = model.score(x_te, y_te)
    p_te = model.predict_proba(x_te)[:, 1]
    print(f"holdout accuracy: {acc:.3f} (baseline {1 - y_te.mean():.3f}), "
          f"positives {y_te.mean():.3f}")
    thresholds = derive_risk_thresholds(y_te, p_te, min_pred=10)
    for name, info in thresholds.items():
        print(f"  level[{name:>8}] >= {info['threshold']:.2f}  "
              f"precision {info['precision']:.3f}  recall {info['recall']:.3f}")

    from sklearn.inspection import permutation_importance
    perm = permutation_importance(model, x_te, y_te,
                                  scoring="average_precision",
                                  n_repeats=5, random_state=42, n_jobs=-1)
    global_importance = sorted(
        ({"feature": name, "importance": float(max(0.0, mean))}
         for name, mean in zip(FEATURE_NAMES, perm.importances_mean)),
        key=lambda d: -d["importance"])

    onnx_path = os.path.join(assets_dir, "risk_model.onnx")
    export_model(model, onnx_path)
    write_coefficients(
        os.path.join(assets_dir, "risk_coefficients.json"),
        model.coef_[0], model.intercept_[0],
        {"true_weights": [float(v) for v in TRUE_W], "true_bias": float(TRUE_B),
         "pe_cycles_assumed": P_E_CYCLES, "data": "synthetic",
         "exported_model": "logistic", "approx_explain": False,
         "risk_thresholds": {k: v["threshold"] for k, v in thresholds.items()},
         "global_importance": global_importance},
    )

    print("learned coefficients:")
    for name, w in zip(FEATURE_NAMES, model.coef_[0]):
        print(f"  {name:>15}: {w:+.3f}")
    print(f"  {'bias':>15}: {float(model.intercept_[0]):+.3f}")
    return 0


def derive_risk_thresholds(y_true, p, targets=(0.50, 0.70, 0.85),
                           min_pred: int = 50) -> dict:
    """Score cut-offs for the risk levels, derived from the test split.

    The score is shown to a human as "chance of a data-loss failure within
    30 days", so each level is anchored at the *lowest* score whose
    predicted positives still reach a target precision — that keeps the
    recall as high as possible for the promised precision. Cut-offs are
    rounded up to 0.01 and re-checked against the actual selection, so the
    reported precision always refers to the real set of flagged disks.
    """
    order = np.argsort(-p)
    ys = y_true[order]
    ps = p[order]
    rounded = np.ceil(ps * 100.0) / 100.0
    cum = np.cumsum(ys)
    k = np.arange(1, len(cum) + 1)
    precision = cum / k
    out = {}
    for name, target in zip(("medium", "high", "critical"), targets):
        # Largest top-k whose cumulative precision still meets the target:
        # the cut with the highest recall for that precision.
        idx = np.where((precision >= target) & (k >= min_pred))[0]
        chosen = None
        if idx.size:
            # Walk cut-offs upward until the flags really meet the target
            # (rounding and score ties can move the boundary).
            for thr in np.unique(rounded[idx[-1]:]):
                sel = p >= thr
                if sel.sum() < min_pred:
                    break
                prec = float(y_true[sel].mean())
                if chosen is None or prec >= target:
                    chosen = (float(thr), prec, int(sel.sum()))
                if prec >= target:
                    break
        if chosen is None:
            chosen = (1.0, 1.0, 0)
        thr, prec, n = chosen
        sel = p >= thr
        out[name] = {
            "threshold": thr,
            "precision": float(y_true[sel].mean()) if n else 1.0,
            "recall": float(y_true[sel].sum() / max(y_true.sum(), 1)),
            "lift": float(y_true[sel].mean() / y_true.mean()) if n else 0.0,
            "predicted_positives": n,
        }
        assert out[name]["precision"] >= target - 1e-9, \
            f"{name}: precision {out[name]['precision']:.4f} < target {target}"
    return out


def train_real(csv_path: str) -> int:
    """Train on assets/train.csv (Alibaba DSN'21, 30-day horizon).

    Split is grouped by disk_id (no disk leaks across train/test).
    Candidates: plain logistic, balanced logistic, histogram GBM.
    Winner by test PR-AUC is exported to ONNX; the plain logistic's
    coefficients are always written for explainability/fallback.
    """
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss, \
        roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    assets_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    # disk_id is unique per model only -> composite group key
    groups = (df["model"].astype(str) + "#" + df["disk_id"].astype(str)).to_numpy()
    print(f"loaded {csv_path}: {len(df)} rows, positives {y.mean():.4f}, "
          f"{df['disk_id'].nunique()} disks")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    x_tr, x_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    print(f"split: train {len(tr)} (pos {y_tr.mean():.4f}) / "
          f"test {len(te)} (pos {y_te.mean():.4f})")

    candidates = {
        "logistic": LogisticRegression(max_iter=5000, C=1.0),
        "logistic_balanced": LogisticRegression(max_iter=5000, C=1.0,
                                                 class_weight="balanced"),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                               random_state=42),
    }
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, m in candidates.items():
        m.fit(x_tr, y_tr)
        p = m.predict_proba(x_te)[:, 1]
        results[name] = {
            "pr_auc": float(average_precision_score(y_te, p)),
            "roc_auc": float(roc_auc_score(y_te, p)),
            "brier": float(brier_score_loss(y_te, p)),
            "base_rate": float(y_te.mean()),
        }
        fitted[name] = m
        print(f"{name:>18}: PR-AUC {results[name]['pr_auc']:.4f}  "
              f"ROC-AUC {results[name]['roc_auc']:.4f}  "
              f"Brier {results[name]['brier']:.4f}")

    # Winner by PR-AUC; prefer plain logistic when it is within 10%
    # (interpretability + calibrated fallback in Rust).
    best = max(results, key=lambda k: results[k]["pr_auc"])
    if best != "logistic" and results["logistic"]["pr_auc"] >= \
            results[best]["pr_auc"] * 0.90:
        best = "logistic"
    print(f"exporting: {best}")

    # Winner's scores on the held-out disks: drives both the risk-level
    # thresholds and the per-model breakdown below.
    p_best = fitted[best].predict_proba(x_te)[:, 1]
    thresholds = derive_risk_thresholds(y_te, p_best)
    for name, info in thresholds.items():
        print(f"  level[{name:>8}] >= {info['threshold']:.2f}  "
              f"precision {info['precision']:.3f}  recall {info['recall']:.3f}  "
              f"lift {info['lift']:.1f}  ({info['predicted_positives']} rows)")

    export_model(fitted[best], os.path.join(assets_dir, "risk_model.onnx"))

    # Global (dataset-level) importance: which features the *exported* model
    # actually uses. This is what the UI should show — the plain logistic
    # coefficients are only a surrogate (several features are near-constant
    # here, which makes their linear weights meaningless).
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(fitted[best], x_te, y_te,
                                  scoring="average_precision",
                                  n_repeats=5, random_state=42, n_jobs=-1)
    global_importance = sorted(
        ({"feature": name, "importance": float(max(0.0, mean))}
         for name, mean in zip(FEATURE_NAMES, perm.importances_mean)),
        key=lambda d: -d["importance"])
    print("global importance (permutation, dPR-AUC):")
    for item in global_importance:
        print(f"  {item['feature']:>15}: {item['importance']:.4f}")

    # Explainability: plain logistic coefficients (approximation when the
    # exported model is HGB; contributions in the UI are then indicative).
    # The risk-level cut-offs are data-derived, so the Rust binary reads
    # them from this same file instead of hard-coding constants.
    logit = fitted["logistic"]
    write_coefficients(
        os.path.join(assets_dir, "risk_coefficients.json"),
        logit.coef_[0], logit.intercept_[0],
        {"data": "alibaba_dsn21_30d", "exported_model": best,
         "metrics": results, "approx_explain": best != "logistic",
         "risk_thresholds": {k: v["threshold"] for k, v in thresholds.items()},
         "global_importance": global_importance},
    )

    # Per-model AP of the winner on the test split.
    te_models = df["model"].to_numpy()[te]
    per_model = {}
    for m in sorted(set(te_models)):
        mask = te_models == m
        if mask.sum() and len(set(y_te[mask])) == 2:
            per_model[str(m)] = float(average_precision_score(y_te[mask],
                                                              p_best[mask]))
            print(f"  AP[{m}] = {per_model[str(m)]:.4f} ({int(mask.sum())} rows)")

    import json as _json
    quantiles = {q: float(np.quantile(p_best, q))
                 for q in (0.5, 0.9, 0.99, 0.999, 1.0)}
    with open(os.path.join(assets_dir, "train_metrics.json"), "w") as f:
        _json.dump({"candidates": results, "exported": best,
                    "per_model_ap": per_model,
                    "risk_thresholds": thresholds,
                    "global_importance": global_importance,
                    "test_rows": int(len(te)),
                    "test_positive_rate": float(y_te.mean()),
                    "score_quantiles": quantiles,
                    "rows": len(df), "positive_rate": float(y.mean())},
                   f, indent=2)
    print("written: train_metrics.json")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="train smartly-rs risk model")
    ap.add_argument("--data", metavar="CSV",
                    help="real dataset train.csv (default: synthetic)")
    args = ap.parse_args()
    if args.data:
        return train_real(args.data)
    return train_synthetic()


if __name__ == "__main__":
    sys.exit(main())

