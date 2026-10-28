#!/usr/bin/env python3
"""Parse the `notebooks/SMART` smartctl text reports into a structured, ML-ready CSV.

Each input file is one `smartctl -a` report for a single SSD (SATA under `SSD/`,
NVMe under `NVMe/`, text reports from the linuxhw/SMART repository). The script
emits one row per device with:

  * device metadata (model, firmware, capacity, form factor, trim, ATA security,
    sanitize flags, self-test verdict)
  * SATA-style reports: every SMART attribute id -> raw value, flat and wide
  * derived risk features used by the smartly-rs risk model, filled where the
    report contains the source data (poh, realloc, media_err, temp, pct_used,
    spare_deficit, crit_warn, smart_fail)

Labels are intentionally NOT produced: the dataset contains no failure
annotations, so this is a feature-extraction stage only.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# SATA / ATA text report (the `smartctl -A` attribute table)
#   "  5 Reallocated_Sector_Ct   -O--CK   100   100   050    -    0"
# --------------------------------------------------------------------------
ATTR_LINE_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z0-9_]+)\s+([^\s]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s*$")

META_PATTERNS = {
    "device_model": re.compile(r"^Device Model:\s*(.+)$", re.M),
    "model_number": re.compile(r"^Model Number:\s*(.+)$", re.M),
    "serial": re.compile(r"^Serial Number:\s*(.+)$", re.M),
    "firmware": re.compile(r"^Firmware Version:\s*(.+)$", re.M),
    "capacity": re.compile(r"^User Capacity:\s*(.+)$", re.M),
    "sector_size": re.compile(r"^Sector Size:\s*(.+)$", re.M),
    "rotation_rate": re.compile(r"^Rotation Rate:\s*(.+)$", re.M),
    "form_factor": re.compile(r"^Form Factor:\s*(.+)$", re.M),
    "trim": re.compile(r"^TRIM Command:\s*(.+)$", re.M),
    "ata_security": re.compile(r"^ATA Security is:\s*(.+)$", re.M),
}

SELF_TEST_RE = re.compile(r"SMART overall-health self-assessment test result:\s*(\w+)", re.I)
SMART_CAPABLE_RE = re.compile(r"SMART support is:\s*(Available.*)$", re.I)
SANITIZE_RE = re.compile(r"SANITIZE DEVICE", re.I)

# --------------------------------------------------------------------------
# NVMe text report (smartctl renders NVMe SMART logs as keyword=value lines)
# --------------------------------------------------------------------------
NVME_PATTERNS = {
    "percentage_used": re.compile(r"^Percentage Used:\s+([0-9.]+)\%", re.M),
    "available_spare": re.compile(r"^Available Spare:\s+([0-9.]+)\%", re.M),
    "spare_threshold": re.compile(r"^Available Spare Threshold:\s+([0-9.]+)\%", re.M),
    "temperature": re.compile(r"^Temperature:\s+([0-9.-]+)\s*Celsius", re.M),
    "critical_warning": re.compile(r"^Critical Warning:\s+(0x[0-9a-fA-F]+)", re.M),
    "media_and_data_integrity_errors": re.compile(r"^Media and Data Integrity Errors:\s+([0-9]+)", re.M),
    "error_information_log_entries": re.compile(r"^Error Information Log Entries:\s+([0-9]+)", re.M),
    "power_on_hours": re.compile(r"^Power On Hours:\s+([0-9]+)", re.M),
}

NVME_META = {
    "model_number": re.compile(r"^Model Number:\s*(.+)$", re.M),
    "serial": re.compile(r"^Serial Number:\s*(.+)$", re.M),
    "firmware": re.compile(r"^Firmware Version:\s*(.+)$", re.M),
    "capacity": re.compile(r"^User Capacity:\s*(.+)$", re.M),
    "form_factor": re.compile(r"^Form Factor:\s*(.+)$", re.M),
}

# Model's 8 feature contract: in-order FEM for describe().
FEATURE_NAMES = [
    "pct_used", "spare_deficit", "media_err", "temp",
    "poh", "crit_warn", "realloc", "smart_fail",
]


def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_num(v) -> float | None:
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _popcount(bits: int) -> int:
    return bin(bits).count("1")


def parse_sata_attrs(text: str) -> dict:
    """{attribute_id: raw_value} for every attribute row in a SATA report."""
    attrs = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit():
            attrs[int(parts[0])] = _to_int(parts[7])
    return attrs


def parse_sata(text: str) -> dict:
    rec = {}
    for key, pattern in META_PATTERNS.items():
        m = pattern.search(text)
        rec[key] = m.group(1).strip() if m else ""
    rec["self_test"] = "PASSED"
    m = SELF_TEST_RE.search(text)
    if m:
        rec["self_test"] = m.group(1).upper()
    rec["smart_capable"] = bool(SMART_CAPABLE_RE.search(text))
    rec["sanitize_observed"] = bool(SANITIZE_RE.search(text))
    rec.update(parse_sata_attrs(text))
    return rec


def parse_nvme(text: str) -> dict:
    rec = {}
    for key, pattern in NVME_PATTERNS.items():
        m = pattern.search(text)
        rec[key] = m.group(1) if m else ""
    for key, pattern in NVME_META.items():
        m = pattern.search(text)
        rec[key] = m.group(1).strip() if m else ""
    m = SELF_TEST_RE.search(text)
    rec["self_test"] = m.group(1).upper() if m else ""
    rec["smart_capable"] = bool(SMART_CAPABLE_RE.search(text))
    rec["sanitize_observed"] = bool(SANITIZE_RE.search(text))
    return rec


def derive_features(rec: dict, fmt: str) -> dict:
    """Fill the model's 8-feature contract where the report permits."""
    f = {}
    if fmt == "sata":
        f["poh"] = rec.get(9)
        f["realloc"] = rec.get(5)
        r197 = rec.get(197, 0) or 0
        r198 = rec.get(198, 0) or 0
        if r197 or r198:
            f["media_err"] = math.log10(1.0 + r197 + r198) / 5.0
        else:
            f["media_err"] = None
        t = rec.get(194)
        f["temp"] = ((float(t) - 30.0) / 60.0) if t is not None else None
        life = rec.get(233)
        if life is None:
            life = rec.get(231)
        f["pct_used"] = ((100.0 - float(life)) / 100.0) if life is not None else None
        f["spare_deficit"] = None
        f["crit_warn"] = None
        f["smart_fail"] = 1 if rec.get("self_test") == "FAILED" else 0
    else:
        f["poh"] = _to_num(rec.get("power_on_hours"))
        f["realloc"] = None
        mdie = _to_num(rec.get("media_and_data_integrity_errors"))
        f["media_err"] = (mdie / 100.0) if mdie is not None else None
        t = _to_num(rec.get("temperature"))
        f["temp"] = ((t - 30.0) / 60.0) if t is not None else None
        pu_v = _to_num(rec.get("percentage_used"))
        f["pct_used"] = (pu_v / 100.0) if pu_v is not None else None
        as_v = _to_num(rec.get("available_spare"))
        st_v = _to_num(rec.get("spare_threshold"))
        if as_v is not None and st_v is not None:
            span = max(1.0, 100.0 - st_v)
            f["spare_deficit"] = max(0.0, min(1.0, 1.0 - (as_v - st_v) / span))
        else:
            f["spare_deficit"] = None
        cw = _to_num(rec.get("critical_warning"))
        f["crit_warn"] = (_popcount(int(cw)) / 5.0) if cw is not None else None
        f["smart_fail"] = 1 if rec.get("self_test") == "FAILED" else 0
    return f


def rel_parts(path: Path, ssd_dir: Path):
    rel = path.relative_to(ssd_dir)
    parts = list(rel.parts)
    return (parts[0] if parts else None, str(rel))


def find_reports(ssd_dir: Path):
    roots = [ssd_dir / "SSD", ssd_dir / "NVMe"]
    files = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.name.lower() not in {"readme", "readme.md"}:
                files.append(p)
    return files


def build_fieldnames(ssd_dir: Path, sata_paths: list, nvme_paths: list) -> list:
    """Union of every column that will be written, with a stable order."""
    names = [
        "path", "manufacturer", "format", "sub_collection",
        "device_model", "model_number", "serial", "firmware", "capacity",
        "sector_size", "rotation_rate", "form_factor", "trim",
        "ata_security", "self_test", "smart_capable", "sanitize_observed",
    ]
    for p in sata_paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        for k in parse_sata_attrs(text):
            names.append(f"attr_{k:03d}")
    for p in nvme_paths:
        for k in NVME_PATTERNS:
            names.append(k)
        for k in NVME_META:
            names.append(k)
    names += FEATURE_NAMES
    # dedupe preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def discover_pass(files) -> tuple[list, list, set]:
    """Classify sata/nvme and collect attribute ids. Cheap relative parse."""
    sata_files = []
    nvme_files = []
    all_attr_ids = set()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "Device Model:" in text:
            sata_files.append((path, text))
            all_attr_ids.update(parse_sata_attrs(text).keys())
        else:
            nvme_files.append((path, text))
    return sata_files, nvme_files, all_attr_ids


def row_for(ssd_dir: Path, sata_files: list, nvme_files: list, fieldnames: list) -> list:
    """Yield complete CSV rows for both sub-collections."""
    # SATA rows
    for path, text in sata_files:
        rec = parse_sata(text)
        feats = derive_features(rec, "sata")
        attr_by_id = parse_sata_attrs(text)
        row = {
            "path": str(path.relative_to(ssd_dir)),
            "manufacturer": rel_parts(path, ssd_dir)[0],
            "format": "sata",
            "sub_collection": "ssd",
        }
        for key, val in rec.items():
            if key in ("device_model", "model_number", "serial", "firmware",
                       "capacity", "sector_size", "rotation_rate", "form_factor",
                       "trim", "ata_security", "self_test", "smart_capable",
                       "sanitize_observed"):
                row[key] = val
        for k, v in attr_by_id.items():
            row[f"attr_{k:03d}"] = v
        for k, v in feats.items():
            row[k] = v
        yield row
    # NVMe rows
    for path, text in nvme_files:
        rec = parse_nvme(text)
        feats = derive_features(rec, "nvme")
        row = {
            "path": str(path.relative_to(ssd_dir)),
            "manufacturer": rel_parts(path, ssd_dir)[0],
            "format": "nvme",
            "sub_collection": "nvme",
        }
        for key in list(NVME_PATTERNS) + list(NVME_META):
            if key in rec:
                row[key] = rec[key]
        for key in ("self_test", "smart_capable", "sanitize_observed"):
            row[key] = rec.get(key, "")
        for k, v in feats.items():
            row[k] = v
        yield row


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract SMART features from notebooks/SMART text reports")
    ap.add_argument("ssd_dir", type=Path, help="Path to the SMART dataset root (contains SSD/ and NVMe/)")
    ap.add_argument("--output", type=Path, default=Path("notebooks/SMART/parsed_smart_records.csv"))
    ap.add_argument("--max-files", type=int, default=None, help="Cap on the number of files parsed")
    args = ap.parse_args()

    ssd_dir = args.ssd_dir.resolve()
    for sub in ("SSD", "NVMe"):
        if not (ssd_dir / sub).exists():
            raise SystemExit(f"no {sub}/ subdir under {ssd_dir}")

    files = find_reports(ssd_dir)
    if args.max_files:
        files = files[: args.max_files]
    print(f"parsing {len(files)} reports under {ssd_dir}", file=sys.stderr)

    sata_files, nvme_files, all_attr_ids = discover_pass(files)
    sata_paths = [p for p, _ in sata_files]
    nvme_paths = [p for p, _ in nvme_files]
    print(f"  sata={len(sata_files)} nvme={len(nvme_files)} attr_ids={len(all_attr_ids)}", file=sys.stderr)

    fieldnames = build_fieldnames(ssd_dir, sata_paths, nvme_paths)
    # canonical attribute columns first
    attr_cols = sorted((n for n in fieldnames if n.startswith("attr_")), key=lambda x: int(x[5:]))
    rest = [n for n in fieldnames if not n.startswith("attr_")]
    fieldnames = rest + attr_cols

    rows_written = 0
    bad = 0
    with args.output.open("w", newline="", encoding="utf-8", errors="surrogateescape") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in row_for(ssd_dir, sata_files, nvme_files, fieldnames):
            w.writerow({k: row.get(k, "") for k in fieldnames})
            rows_written += 1
            if rows_written % 20000 == 0:
                print(f"  wrote {rows_written}", file=sys.stderr)

    print(f"wrote {args.output} ({rows_written} rows, {len(fieldnames)} columns)", file=sys.stderr)
    if bad:
        print(f"{bad} unreadable files", file=sys.stderr)


if __name__ == "__main__":
    main()
