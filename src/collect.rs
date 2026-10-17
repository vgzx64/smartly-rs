//! Collect diagnostic info from Linux tools: `lsblk`, `smartctl`,
//! `nvme-cli`. Read-only; this module never writes to a device.

use std::process::Command;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::model::Health;

#[derive(Debug, Clone, Deserialize)]
pub struct BlockDev {
    pub name: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tran: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub rota: Option<bool>,
    #[serde(default)]
    pub serial: Option<String>,
    #[serde(rename = "type", default)]
    pub dev_type: Option<String>,
    #[serde(default)]
    pub children: Vec<BlockDev>,
}

#[derive(Debug, Deserialize)]
struct LsblkOut {
    blockdevices: Vec<BlockDev>,
}

fn run(cmd: &str, args: &[&str]) -> Result<(String, String, i32)> {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .with_context(|| format!("не удалось запустить `{cmd}`"))?;
    let code = out.status.code().unwrap_or(-1);
    Ok((
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        code,
    ))
}

fn tool_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// List block devices via `lsblk --json`.
pub fn list_block_devices() -> Result<Vec<BlockDev>> {
    let (stdout, stderr, code) = run(
        "lsblk",
        &[
            "-J",
            "-b",
            "-o",
            "NAME,PATH,TYPE,MODEL,TRAN,SIZE,ROTA,SERIAL",
        ],
    )?;
    if code != 0 {
        anyhow::bail!("lsblk завершился с кодом {code}: {stderr}");
    }
    let parsed: LsblkOut = serde_json::from_str(&stdout).context("разбор lsblk -J")?;
    Ok(parsed.blockdevices)
}

/// Find a top-level disk device (or any node) by name, e.g. `nvme0n1`.
fn find_device(devs: &[BlockDev], name: &str) -> Option<BlockDev> {
    for d in devs {
        let matches = d.name == name
            || d.path.as_deref() == Some(name)
            || d.path.as_deref() == Some(&format!("/dev/{name}"));
        if matches {
            return Some(d.clone());
        }
        if let Some(c) = find_device(&d.children, name) {
            return Some(c);
        }
    }
    None
}

fn any_has_crypt(devs: &[BlockDev]) -> bool {
    devs.iter()
        .any(|d| d.dev_type.as_deref() == Some("crypt") || any_has_crypt(&d.children))
}

/// Run `smartctl --json --all <dev>` and parse it.
pub fn smartctl_json(dev: &str) -> Result<Value> {
    let (stdout, stderr, _code) = run("smartctl", &["-j", "-a", dev])?;
    if stdout.trim().is_empty() {
        anyhow::bail!(
            "smartctl не вернул данных для {dev}. stderr: {} \
             (нужен пакет smartmontools и, как правило, права root)",
            stderr.trim()
        );
    }
    serde_json::from_str(&stdout).with_context(|| format!("разбор smartctl -j для {dev}"))
}

/// Sanitize capabilities from `nvme id-ctrl <dev> --json` (best effort).
/// NVMe sanicap bits: 0 = crypto erase, 1 = block erase, 2 = overwrite.
pub fn nvme_id_ctrl_sanitize(dev: &str) -> Option<(bool, bool)> {
    let (stdout, _, code) = run("nvme", &["id-ctrl", dev, "-o", "json"]).ok()?;
    if code != 0 {
        return None;
    }
    let v: Value = serde_json::from_str(&stdout).ok()?;
    let san = v.get("san")?;
    let san = match san {
        Value::String(s) => u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()?,
        Value::Number(n) => n.as_u64()?,
        _ => return None,
    };
    let any_sanitize = san & 0b111 != 0;
    let crypto = san & 0b001 != 0;
    Some((any_sanitize, crypto))
}

/// Collect a full [`Health`] snapshot for a device (read-only).
pub fn probe(device: &str) -> Result<Health> {
    let mut notes: Vec<String> = Vec::new();

    // -- lsblk: identity, transport, rotational flag, encryption ---------
    let devs = list_block_devices().unwrap_or_default();
    let short = device.rsplit('/').next().unwrap_or(device);
    let blk = find_device(&devs, short);
    let mut h = match &blk {
        Some(b) => Health {
            device: b.path.clone().unwrap_or_else(|| format!("/dev/{}", b.name)),
            transport: b.tran.clone().unwrap_or_else(|| "unknown".into()),
            model: b.model.clone().unwrap_or_default(),
            serial: b.serial.clone().unwrap_or_default(),
            capacity_bytes: b.size,
            is_ssd: b.rota.map(|rota| !rota),
            encryption_active: find_device(&devs, short)
                .map(|d| any_has_crypt(&d.children) || d.dev_type.as_deref() == Some("crypt"))
                .unwrap_or(false),
            source: "lsblk".into(),
            ..Default::default()
        },
        None => {
            notes.push(format!("lsblk: устройство {device} не найдено"));
            Health {
                device: device.to_string(),
                source: "нет данных".into(),
                ..Default::default()
            }
        }
    };

    // -- smartctl: the actual SMART data --------------------------------
    if !tool_exists("smartctl") {
        notes.push(
            "smartmontools (smartctl) не установлен — установите: \
             sudo apt install smartmontools"
                .into(),
        );
    } else {
        match smartctl_json(device) {
            Ok(v) => {
                let parsed = Health::from_smartctl(device, &v);
                if !parsed.model.is_empty() {
                    h.model = parsed.model;
                }
                if !parsed.serial.is_empty() {
                    h.serial = parsed.serial;
                }
                if parsed.capacity_bytes.is_some() {
                    h.capacity_bytes = parsed.capacity_bytes;
                }
                h.temperature_c = parsed.temperature_c;
                h.percentage_used = parsed.percentage_used;
                h.available_spare = parsed.available_spare;
                h.spare_threshold = parsed.spare_threshold;
                h.media_errors = parsed.media_errors;
                h.realloc_sectors = parsed.realloc_sectors;
                h.pending_sectors = parsed.pending_sectors;
                h.uncorr_errors = parsed.uncorr_errors;
                h.power_on_hours = parsed.power_on_hours;
                h.power_cycles = parsed.power_cycles;
                h.unsafe_shutdowns = parsed.unsafe_shutdowns;
                h.critical_warning = parsed.critical_warning;
                h.bytes_written = parsed.bytes_written;
                h.smart_passed = parsed.smart_passed;
                h.read_only_mode = parsed.read_only_mode;
                h.is_ssd = parsed.is_ssd.or(h.is_ssd);
                h.source = "smartctl".into();
            }
            Err(e) => notes.push(format!("smartctl: {e:#}")),
        }
    }

    // -- nvme-cli: sanitize capabilities --------------------------------
    if h.transport == "nvme" {
        if tool_exists("nvme") {
            match nvme_id_ctrl_sanitize(device) {
                Some((any, crypto)) => {
                    h.sanitize_supported = Some(any);
                    h.crypto_erase_supported = Some(crypto);
                }
                None => notes.push(format!("nvme id-ctrl: не удалось прочитать {device}")),
            }
        } else {
            notes.push("nvme-cli не установлен (опционально): sudo apt install nvme-cli".into());
        }
    }

    if h.source == "lsblk" {
        h.source = "lsblk (без SMART)".into();
    }
    h.notes = notes;
    Ok(h)
}
