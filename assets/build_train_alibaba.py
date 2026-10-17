#!/usr/bin/env python
"""Build train.csv for smartly-rs v2 from the Alibaba SSD dataset.

Single streaming pass over both zips (no extraction):
  * positive: for every failed disk one random daily snapshot from the
    30 days before its failure_time (label=1, prediction horizon 30d);
  * negative: ~500 random disks per day that never appear in the label
    file (label=0), global ratio ~1:20;
  * features: 8-dim contract of src/features.rs + wear_unknown flag.

Split by disk_id happens later in train_risk_model.py.

Usage: .venv/bin/python assets/build_train_alibaba.py
"""

from __future__ import annotations

import random
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIPS = [ROOT / "downloads/smartlog2018ssd.zip",
        ROOT / "downloads/smartlog2019ssd.zip"]
LABELS = ROOT / "downloads/ssd_failure_label.csv.zip"
OUT = ROOT / "assets/train.csv"

NEG_PER_DAY = 500
HORIZON = 30  # days
SEED = 20261001

USECOLS = ["disk_id", "ds", "model", "n_5", "r_5", "r_9", "n_173", "r_173",
           "r_187", "r_194", "r_195", "r_197", "r_198",
           "n_232", "r_232", "n_233", "r_233", "r_241", "r_242"]
USESET = set(USECOLS)

FEATS = ["pct_used", "spare_deficit", "media_err", "temp", "poh",
         "crit_warn", "realloc", "smart_fail"]

DATA_MIN = date(2018, 1, 1)
DATA_MAX = date(2019, 12, 31)


def clamp01(s: pd.Series) -> pd.Series:
    return s.clip(0.0, 1.0)


def available_days() -> set[date]:
    days: set[date] = set()
    for zpath in ZIPS:
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if n.endswith(".csv"):
                    d = n[:8]
                    days.add(date(int(d[:4]), int(d[4:6]), int(d[6:8])))
    return days


def pick_pos_days(labels: pd.DataFrame, days: set[date]) -> dict[date, set[str]]:
    # disk_id is unique per MODEL, not globally -> composite "model#id" keys
    rng = random.Random(SEED)
    targets: dict[date, set[str]] = {}
    dropped = 0
    for row in labels.itertuples(index=False):
        fday = row.failure_time.date()
        lo = max(fday - timedelta(days=HORIZON), DATA_MIN)
        hi = fday - timedelta(days=1)
        candidates = sorted(d for d in days if lo <= d <= hi)
        if not candidates:
            dropped += 1
            continue
        day = rng.choice(candidates)
        targets.setdefault(day, set()).add(f"{row.model}#{int(row.disk_id)}")
    print(f"positives: {sum(len(v) for v in targets.values())} samples, "
          f"{dropped} failed disks skipped (no snapshot in window)")
    return targets


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised mapping to the 10-feature contract (see README)."""
    # vendor files contain non-numeric junk in some cells -> force numeric
    num = {c: pd.to_numeric(df[c], errors="coerce")
           for c in ["n_5", "r_5", "r_9", "n_173", "r_194",
                     "r_197", "r_198", "n_232", "n_233"]}
    n233, n173 = num["n_233"], num["n_173"]
    wear_known = n233.notna() | n173.notna()
    life = n233.where(n233.notna(), n173)
    pct_used = (100.0 - life).clip(0.0, 100.0).fillna(0.0) / 100.0

    # spare deficit, threshold assumed 10 (as NVMe default); fallback=pct_used
    thr = 10.0
    spare = num["n_232"]
    deficit = (1.0 - (spare - thr) / (100.0 - thr)).clip(0.0, 1.0)
    deficit = deficit.where(spare.notna(), pct_used)

    media = np.log10(1.0 + num["r_198"].fillna(0) + num["r_197"].fillna(0)) / 5.0
    temp = ((num["r_194"].fillna(30.0) - 30.0) / 60.0).clip(0.0, 1.0)
    poh = np.log10(1.0 + num["r_9"].fillna(0.0)) / np.log10(1.0 + 87_600.0)
    crit = ((num["r_198"].fillna(0) > 0).astype(int)
            + (num["r_197"].fillna(0) > 0).astype(int)
            + (num["n_5"].fillna(100) < 100).astype(int)) / 5.0
    realloc = np.log10(1.0 + num["r_5"].fillna(0.0)) / 6.0
    smart_fail = 0.0  # no SMART self-test verdict in this dataset

    out = pd.DataFrame({
        "pct_used": clamp01(pct_used), "spare_deficit": clamp01(deficit),
        "media_err": clamp01(media), "temp": clamp01(temp),
        "poh": clamp01(poh),
        "crit_warn": clamp01(crit),
        "realloc": clamp01(realloc), "smart_fail": smart_fail,
        "wear_unknown": (~wear_known).astype("int8"),
    })
    return out



def main() -> int:
    rng = np.random.default_rng(SEED)
    with zipfile.ZipFile(LABELS) as z, z.open("ssd_failure_label.csv") as f:
        labels = pd.read_csv(f)
    labels["failure_time"] = pd.to_datetime(labels["failure_time"])
    failed_keys = {f"{m}#{int(d)}" for m, d in
                   zip(labels["model"], labels["disk_id"])}
    print(f"label rows: {len(labels)}, unique failed disks: {len(failed_keys)}")

    days = available_days()
    print(f"available snapshots: {len(days)}")
    targets = pick_pos_days(labels, days)

    parts: list[pd.DataFrame] = []
    for zpath in ZIPS:
        with zipfile.ZipFile(zpath) as z:
            members = sorted(n for n in z.namelist() if n.endswith(".csv"))
            for i, member in enumerate(members):
                day = date(int(member[:4]), int(member[4:6]), int(member[6:8]))
                with z.open(member) as fh:
                    df = pd.read_csv(fh, usecols=lambda c: c in USESET,
                                      dtype={"disk_id": "int64"})
                df = df.drop_duplicates(subset=["model", "disk_id"])
                df["key"] = df["model"].astype(str) + "#" + df["disk_id"].astype(str)
                pos_ids = targets.get(day, set())
                pos = df[df["key"].isin(pos_ids)] if pos_ids else df.iloc[0:0]
                healthy = df[~df["key"].isin(failed_keys)]
                k = min(NEG_PER_DAY, len(healthy))
                neg = healthy.iloc[rng.choice(len(healthy), size=k, replace=False)]
                chunk = pd.concat([pos, neg], ignore_index=True)
                if len(chunk):
                    feat = features(chunk)
                    out = pd.concat(
                        [chunk[["disk_id", "ds", "model"]].reset_index(drop=True),
                         feat], axis=1)
                    out["label"] = [1] * len(pos) + [0] * len(neg)
                    parts.append(out)
                if (i + 1) % 100 == 0:
                    print(f"{zpath.name}: {i + 1}/{len(members)} files",
                          flush=True)
    train = pd.concat(parts, ignore_index=True)
    train = train[["disk_id", "ds", "model", "label"] + FEATS + ["wear_unknown"]]
    train.to_csv(OUT, index=False)
    pos = int(train["label"].sum())
    print(f"written {OUT}: {len(train)} rows, {pos} positives "
          f"({pos / len(train):.3%}), {train['disk_id'].nunique()} disks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
