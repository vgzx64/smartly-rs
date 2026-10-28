//! Unified SMART health model: parses `smartctl -j -a` JSON for both
//! NVMe and SATA (ATA) devices into a single [`Health`] structure.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Normalized health snapshot of a storage device.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Health {
    /// Device path, e.g. `/dev/nvme0n1`.
    pub device: String,
    /// `nvme`, `sata`, `usb`, ... (filled by collector from lsblk).
    pub transport: String,
    pub model: String,
    pub serial: String,
    pub capacity_bytes: Option<u64>,
    /// `Some(true)` for SSD/flash, `Some(false)` for rotational drives.
    pub is_ssd: Option<bool>,
    pub temperature_c: Option<f64>,
    /// Endurance used, percent (NVMe `percentage_used`, SATA SSD life left).
    pub percentage_used: Option<f64>,
    /// NVMe available spare, percent.
    pub available_spare: Option<f64>,
    /// NVMe available spare threshold, percent.
    pub spare_threshold: Option<f64>,
    /// NVMe media errors (uncorrectable ECC errors over lifetime).
    pub media_errors: Option<u64>,
    /// ATA attr 5: reallocated sectors.
    pub realloc_sectors: Option<u64>,
    /// ATA attr 197: pending (unstable) sectors.
    pub pending_sectors: Option<u64>,
    /// ATA attr 198: offline uncorrectable errors.
    pub uncorr_errors: Option<u64>,
    pub power_on_hours: Option<u64>,
    pub power_cycles: Option<u64>,
    pub unsafe_shutdowns: Option<u64>,
    /// NVMe critical_warning bitmask (0 = no warnings).
    pub critical_warning: Option<u8>,
    /// Estimated lifetime host writes in bytes (NVMe data_units_written * 512000).
    pub bytes_written: Option<u64>,
    /// SMART self-test overall status.
    pub smart_passed: Option<bool>,
    /// NVMe critical_warning bit3: controller reports read-only mode.
    pub read_only_mode: bool,
    /// NVMe sanitize capabilities (from `nvme id-ctrl`, if queried).
    pub sanitize_supported: Option<bool>,
    /// Sanitize crypto erase supported (NVMe sanicap CES bit / ATA Sanitize).
    pub crypto_erase_supported: Option<bool>,
    /// Device (or its partition) is inside an active LUKS/dm-crypt mapping.
    pub encryption_active: bool,
    /// Where the data came from: `smartctl`, `nvme-cli`, `demo`, `fixture`.
    pub source: String,
    /// Non-fatal problems (missing tools, permission denied, ...).
    pub notes: Vec<String>,
}

fn loose_u64(v: &Value) -> Option<u64> {
    match v {
        Value::Number(n) => n.as_u64().or_else(|| n.as_f64().map(|f| f as u64)),
        Value::String(s) => s.trim().parse::<u64>().ok(),
        _ => None,
    }
}

fn loose_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn get_u64(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(loose_u64)
}

fn get_f64(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(loose_f64)
}

fn find_attr(entries: &[Value], id: u64) -> Option<&Value> {
    entries
        .iter()
        .find(|e| e.get("id").and_then(loose_u64) == Some(id))
}

impl Health {
    /// Parse `smartctl --json --all <device>` output.
    pub fn from_smartctl(device: &str, v: &Value) -> Health {
        let mut h = Health {
            device: device.to_string(),
            source: "smartctl".into(),
            ..Default::default()
        };

        h.model = v
            .get("model_name")
            .and_then(Value::as_str)
            .or_else(|| v.get("model").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        h.serial = v
            .get("serial_number")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        h.capacity_bytes = v
            .get("user_capacity")
            .and_then(|c| c.get("bytes"))
            .and_then(loose_u64)
            .or_else(|| v.get("physical_size").and_then(loose_u64));
        h.smart_passed = v
            .get("smart_status")
            .and_then(|s| s.get("passed"))
            .and_then(Value::as_bool);
        h.temperature_c = match v.get("temperature") {
            Some(Value::Object(_)) => v
                .get("temperature")
                .and_then(|t| t.get("current"))
                .and_then(loose_f64),
            Some(other) => loose_f64(other),
            None => None,
        };
        if h.temperature_c.is_none() {
            h.temperature_c = v
                .get("nvme_smart_health_information_log")
                .and_then(|l| l.get("temperature"))
                .and_then(loose_f64);
        }

        if let Some(log) = v.get("nvme_smart_health_information_log") {
            // --- NVMe ---
            h.percentage_used = get_f64(log, "percentage_used");
            h.available_spare = get_f64(log, "available_spare");
            h.spare_threshold = get_f64(log, "available_spare_threshold");
            h.media_errors = get_u64(log, "media_errors");
            h.power_on_hours = get_u64(log, "power_on_hours");
            h.power_cycles = get_u64(log, "power_cycles");
            h.unsafe_shutdowns = get_u64(log, "unsafe_shutdowns");
            h.critical_warning = get_u64(log, "critical_warning").map(|b| b as u8);
            h.bytes_written = get_u64(log, "data_units_written")
                // 1 NVMe data unit = 512 bytes * 1000.
                .map(|units| units.saturating_mul(512_000));
            if let Some(cw) = h.critical_warning {
                h.read_only_mode = cw & 0b100 != 0;
            }
            h.is_ssd = Some(true);
        } else if let Some(ata) = v.get("ata_smart_data") {
            // --- SATA / ATA ---
            let empty: Vec<Value> = Vec::new();
            let entries = ata
                .get("table")
                .and_then(|t| t.get("entries"))
                .and_then(Value::as_array)
                .map(|a| a.as_slice())
                .unwrap_or(&empty);

            h.realloc_sectors = attr_raw(entries, 5);
            h.pending_sectors = attr_raw(entries, 197);
            h.uncorr_errors = attr_raw(entries, 198);
            h.power_on_hours = v
                .get("power_on_time")
                .and_then(|p| p.get("hours"))
                .and_then(loose_u64)
                .or_else(|| attr_raw(entries, 9));
            h.power_cycles = attr_raw(entries, 12);

            // SSD life left attributes: 233 (SSD_Life_Left) or 231 (Drive life).
            let life = [233u64, 231]
                .iter()
                .find_map(|id| find_attr(entries, *id))
                .and_then(|e| {
                    let raw = e
                        .get("raw")
                        .and_then(|r| r.get("value"))
                        .and_then(loose_u64);
                    let norm = e.get("value").and_then(loose_u64);
                    raw.filter(|r| *r <= 100).or(norm)
                });
            if let Some(left) = life {
                h.percentage_used = Some((100 - left.min(100)) as f64);
            }

            // Host writes: only trust attr 241 when the unit is in the name.
            if let Some(e) = find_attr(entries, 241) {
                let name = e.get("name").and_then(Value::as_str).unwrap_or("");
                if name.contains("32MiB") {
                    h.bytes_written =
                        attr_raw(entries, 241).map(|w| w.saturating_mul(32 * 1024 * 1024));
                }
            }

            // smartctl JSON may carry a top-level rotation_rate (rpm):
            // >0  -> rotational HDD, 0 -> SSD, absent -> leave lsblk's hint.
            if let Some(rr) = v.get("rotation_rate").and_then(loose_u64) {
                h.is_ssd = if rr > 0 { Some(false) } else { Some(true) };
            }
        }

        h
    }
}

fn attr_raw(entries: &[Value], id: u64) -> Option<u64> {
    find_attr(entries, id)
        .and_then(|e| e.get("raw"))
        .and_then(|raw| {
            raw.get("value")
                .and_then(loose_u64)
                .or_else(|| loose_u64(raw))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NVME: &str = include_str!("../tests/fixtures/nvme_smart.json");
    const SATA: &str = include_str!("../tests/fixtures/sata_smart.json");

    #[test]
    fn parses_nvme_smartctl() {
        let v: Value = serde_json::from_str(NVME).unwrap();
        let h = Health::from_smartctl("/dev/nvme0n1", &v);
        assert_eq!(h.percentage_used, Some(3.0));
        assert_eq!(h.available_spare, Some(100.0));
        assert_eq!(h.spare_threshold, Some(10.0));
        assert_eq!(h.media_errors, Some(0));
        assert_eq!(h.power_on_hours, Some(8760));
        assert_eq!(h.unsafe_shutdowns, Some(150));
        assert_eq!(h.critical_warning, Some(0));
        assert_eq!(h.bytes_written, Some(987_654_321 * 512_000));
        assert_eq!(h.temperature_c, Some(39.0));
        assert_eq!(h.smart_passed, Some(true));
        assert!(!h.read_only_mode);
    }

    #[test]
    fn parses_sata_smartctl() {
        let v: Value = serde_json::from_str(SATA).unwrap();
        let h = Health::from_smartctl("/dev/sda", &v);
        assert_eq!(h.percentage_used, Some(3.0)); // 100 - 97 life left
        assert_eq!(h.realloc_sectors, Some(0));
        assert_eq!(h.pending_sectors, Some(0));
        assert_eq!(h.uncorr_errors, Some(0));
        assert_eq!(h.power_on_hours, Some(4000));
        assert_eq!(h.capacity_bytes, Some(500_107_862_016));
        assert_eq!(h.temperature_c, Some(33.0));
        // 30000 * 32 MiB
        assert_eq!(h.bytes_written, Some(30_000 * 32 * 1024 * 1024));
        assert_eq!(h.smart_passed, Some(true));
    }
}
