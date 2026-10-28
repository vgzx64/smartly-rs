#!/usr/bin/env python3
"""Analyze `notebooks/SMART/parsed_smart_records.csv` for the 8-feature risk
model contract (src/features.rs).

Outputs a text summary and a machine-readable JSON summary of coverage.
"""

import csv
import json
from collections import Counter
from pathlib import Path

CSV_PATH = Path("notebooks/SMART/parsed_smart_records.csv")
KEY_ATTRS = [1, 5, 9, 12, 192, 194, 197, 198, 231, 232, 233]
FEATURES = [
    "pct_used", "spare_deficit", "media_err", "temp", "poh",
    "crit_warn", "realloc", "smart_fail",
]


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8", errors="surrogateescape") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    total = len(rows)
    ssd = sum(1 for r in rows if r["sub_collection"] == "ssd")
    nvme = total - ssd

    # Attribute coverage (SATA attr columns are keyed attr_XXX).
    attr_coverage = {}
    for aid in KEY_ATTRS:
        col = f"attr_{aid:03d}"
        nonempty = sum(1 for r in rows if r.get(col) not in (None, ""))
        attr_coverage[str(aid)] = {"nonempty": nonempty, "pct": round(100 * nonempty / total, 2)}

    # Feature coverage: features are present when not empty and numeric.
    feat_stats = {}
    for f in FEATURES:
        nonempty = sum(1 for r in rows if r.get(f) not in (None, ""))
        feat_stats[f] = {"nonempty": nonempty, "pct": round(100 * nonempty / total, 2)}

    # smart_fail (derived) + self_test FAILED.
    failed = sum(1 for r in rows if r.get("self_test") == "FAILED")
    smart_fail = sum(1 for r in rows if r.get("smart_fail") == "1")

    # Numeric sanity for key numeric columns.
    numeric_check = {}
    for col in ["pct_used", "spare_deficit", "media_err", "temp", "poh", "realloc", "crit_warn", "smart_fail"]:
        vals = [r.get(col) for r in rows if r.get(col) not in (None, "")]
        numeric_check[col] = {
            "count": len(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }

    summary = {
        "total_rows": total,
        "ssd_rows": ssd,
        "nvme_rows": nvme,
        "attribute_coverage": attr_coverage,
        "feature_coverage": feat_stats,
        "self_test_failed": failed,
        "smart_fail_derived": smart_fail,
        "numeric_check": numeric_check,
    }

    text = []
    text.append("notebooks/SMART coverage analysis (8-feature risk-model contract)")
    text.append(f"rows: {total}  (SSD={ssd}, NVMe={nvme})")
    text.append("")
    text.append("Attribute coverage (raw SMART attributes present in reports):")
    for aid, info in sorted(attr_coverage.items(), key=lambda kv: int(kv[0])):
        text.append(f"  attr {aid:>3}: {info['nonempty']:>7} ({info['pct']}%)")
    text.append("")
    text.append("Derived risk-feature coverage:")
    for f, info in feat_stats.items():
        text.append(f"  {f:>13}: {info['nonempty']:>7} ({info['pct']}%)")
    text.append(f"")
    text.append(f"self_test = FAILED: {failed}")
    text.append(f"smart_fail (derived =1): {smart_fail}")
    text.append("")
    text.append("Numeric sanity slices:")
    for col, info in numeric_check.items():
        text.append(f"  {col:>13}: n={info['count']}, min={info['min']}, max={info['max']}")
    text.append("")
    text.append("Read the JSON summary for machine use.")

    out = Path("notebooks/SMART/coverage_summary.txt")
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(out)
    print("\n".join(text))
    with Path("notebooks/SMART/coverage_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
