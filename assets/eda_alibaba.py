#!/usr/bin/env python
"""EDA for the Alibaba SSD SMART dataset (DSN'21 / ssd_smart_logs).

Streams the zips directly (no extraction), samples one day per month,
computes per-model attribute coverage, key distributions, checks
monotonicity of host-write counters (r_241/r_242 units sanity check)
and summarises failure labels. Writes docs/dataset.md.

Usage:
  .venv/bin/python assets/eda_alibaba.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIPS = {2018: ROOT / "downloads/smartlog2018ssd.zip",
        2019: ROOT / "downloads/smartlog2019ssd.zip"}
LABELS = ROOT / "downloads/ssd_failure_label.csv.zip"

KEY_RAW = ["r_5", "r_9", "r_173", "r_187", "r_192", "r_194", "r_195",
           "r_196", "r_197", "r_198", "r_232", "r_233", "r_241", "r_242"]
KEY_NORM = ["n_5", "n_173", "n_232", "n_233"]
COLS = ["disk_id", "ds", "model"] + KEY_RAW + KEY_NORM

# 15th of each month (or nearest available snapshot), both years -> ~24 days
SAMPLE_DAYS: list[str] = []
MONO_DAYS: list[str] = []  # 7 consecutive available snapshots in 2019


def member_index() -> dict[int, list[str]]:
    """zip member names by year -> sorted list."""
    out: dict[int, list[str]] = {}
    for year, zpath in ZIPS.items():
        with zipfile.ZipFile(zpath) as z:
            out[year] = sorted(n for n in z.namelist() if n.endswith(".csv"))
    return out


def pick_sample_days() -> None:
    """For each month pick the snapshot closest to the 15th; for the
    monotonicity check pick 7 consecutive available days of 2019-06."""
    by_year = member_index()
    for year, members in by_year.items():
        for month in range(1, 13):
            prefix = f"{year}{month:02d}"
            month_members = [m for m in members if m.startswith(prefix)]
            if not month_members:
                continue
            # closest to day 15
            best = min(month_members, key=lambda m: abs(int(m[6:8]) - 15))
            SAMPLE_DAYS.append(best)
    june = [m for m in by_year[2019] if m.startswith("201906")]
    MONO_DAYS.extend(june[:7])



def zip_for(member: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(ZIPS[int(member[:4])])


def read_day(member: str) -> pd.DataFrame:
    with zip_for(member) as z, z.open(member) as f:
        return pd.read_csv(f, usecols=lambda c: c in set(COLS),
                           dtype={"disk_id": "int64", "ds": "int64",
                                  "model": "string"})


def main() -> int:
    lines: list[str] = []
    add = lines.append
    pick_sample_days()
    with zipfile.ZipFile(LABELS) as z, z.open("ssd_failure_label.csv") as f:
        lab = pd.read_csv(f)
    lab["failure_time"] = pd.to_datetime(lab["failure_time"])
    add("# Датасет Alibaba ssd_smart_logs — EDA\n")
    add("Источник: Tianchi dataId=95044, Xu et al., DSN 2021. "
        "Каждый файл `yyyymmdd.csv` — ежедневный SMART-срез всех SSD.\n")
    add("## Метки отказов\n")
    add("| model | отказов |")
    add("|---|---|")
    for m, n in lab["model"].value_counts().items():
        add(f"| {m} | {n} |")
    add(f"| **всего** | **{len(lab)}** |")
    add("")
    add(f"Период меток: {lab['failure_time'].min().date()} … "
        f"{lab['failure_time'].max().date()}\n")

    failed_ids = set(lab["disk_id"].unique())
    add(f"Уникальных дисков с отказом: {len(failed_ids)}\n")

    # ---- coverage over sampled days ----
    add("## Покрытие атрибутов (доля непустых значений, среднее по 24 суткам)\n")
    cov_rows: list[dict] = []
    dist_frames: list[pd.DataFrame] = []
    for day in SAMPLE_DAYS:
        try:
            df = read_day(day)
        except KeyError:
            continue
        grp = {}
        for m, g in df.groupby("model"):
            grp[m] = {"n_rows": len(g)}
            for c in KEY_RAW + KEY_NORM:
                grp[m][c] = float(g[c].notna().mean())
        cov_rows.append({"day": day, **{f"{m}|{k}": v for m, d in grp.items()
                                        for k, v in d.items()}})
        dist_frames.append(df[["model", "r_9", "r_194", "r_197", "r_198",
                               "r_241", "r_233"]])

    cov = pd.DataFrame(cov_rows)
    models = sorted({c.split("|")[0] for c in cov.columns if "|" in c})
    attrs = KEY_RAW + KEY_NORM
    add("| attr | " + " | ".join(models) + " |")
    add("|---" * (len(models) + 1) + "|")
    for a in attrs:
        cells = []
        for m in models:
            col = f"{m}|{a}"
            if col in cov.columns:
                v = cov[col].mean()
                cells.append(f"{v:.2f}" if pd.notna(v) else "—")
            else:
                cells.append("—")
        add(f"| `{a}` | " + " | ".join(cells) + " |")
    add("")

    # ---- distributions ----
    dist = pd.concat(dist_frames, ignore_index=True)
    add("## Распределения (суммарно по сэмплам)\n")
    add("| метрика | p10 | p50 | p90 |")
    add("|---|---|---|---|")
    for c, name in [("r_9", "часы работы (poh)"),
                    ("r_194", "температура (raw)"),
                    ("r_241", "host writes (raw)"),
                    ("r_233", "износ/остаток ресурса")]:
        s = dist[c].dropna()
        if len(s):
            add(f"| {name} | {s.quantile(.1):.0f} | {s.quantile(.5):.0f} | "
                f"{s.quantile(.9):.0f} |")
    add("")

    # ---- monotonicity of r_241/r_242 (units sanity check) ----
    add("## Проверка монотонности host-write счётчиков (7 суток)\n")
    frames = []
    for day in MONO_DAYS:
        try:
            d = read_day(day)[["disk_id", "model", "r_241", "r_242"]]
        except KeyError:
            continue
        d["day"] = day
        frames.append(d)
    if frames:
        mono = pd.concat(frames).sort_values(["disk_id", "day"])
        for c in ("r_241", "r_242"):
            g = mono.groupby("disk_id")[c]
            diff = g.diff().dropna()
            valid = diff.dropna()
            frac = float((valid >= 0).mean()) if len(valid) else float("nan")
            add(f"- `{c}`: доля непадающих приростов = {frac:.3f} "
                f"(строк: {len(valid)}) — "
                + ("счётчик монотонный, единицы годны для вычисления wear"
                   if frac > 0.95 else
                   "НЕ монотонный — unit-нормализация рискованна, используем fallback"))
    add("")

    Path(ROOT / "docs/dataset.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"written docs/dataset.md ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
