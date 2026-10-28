//! Collect diagnostic info from Linux tools: `lsblk`, `smartctl`,
//! `nvme-cli`. Read-only; this module never writes to a device.

use std::io::Read;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use wait_timeout::ChildExt;

use crate::model::Health;

/// Hard cap on any single external-tool call. A non-responsive SATA/USB
/// bridge can make `smartctl` block forever; this keeps the program
/// responsive and reporting partial results instead of hanging silently.
const TOOL_TIMEOUT: Duration = Duration::from_secs(20);

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
    let mut child = Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run `{cmd}`"))?;
    // We always pipe stdout/stderr, so take() always yields Some.
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    // Read pipes on separate threads so a full pipe buffer on one side
    // cannot deadlock the child.
    let out_handle = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = out.read_to_end(&mut v);
        v
    });
    let err_handle = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = err.read_to_end(&mut v);
        v
    });
    let (stdout, stderr, code) = match child.wait_timeout(TOOL_TIMEOUT) {
        Ok(Some(status)) => {
            let code = status.code().unwrap_or(-1);
            (
                String::from_utf8_lossy(&out_handle.join().unwrap_or_default()).into_owned(),
                String::from_utf8_lossy(&err_handle.join().unwrap_or_default()).into_owned(),
                code,
            )
        }
        Ok(None) => {
            // Timed out: kill the child and report failure instead of
            // letting smartctl block the whole program forever.
            let _ = child.kill();
            // Drain pipes so the reader threads finish.
            let _ = out_handle.join();
            let stderr =
                String::from_utf8_lossy(&err_handle.join().unwrap_or_default()).into_owned();
            anyhow::bail!(
                "`{cmd}` timed out after {}s (stuck bridge/device): {}",
                TOOL_TIMEOUT.as_secs(),
                stderr.trim()
            );
        }
        Err(e) => {
            let _ = child.kill();
            anyhow::bail!("waiting for `{cmd}`: {e}");
        }
    };
    Ok((stdout, stderr, code))
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
        anyhow::bail!("lsblk exited with code {code}: {stderr}");
    }
    let parsed: LsblkOut = serde_json::from_str(&stdout).context("parsing lsblk -J")?;
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

/// Infer SSD vs HDD from lsblk's ROTA flag + transport.
///
/// The ROTA flag reported over a USB-SATA bridge is unreliable: many
/// bridges report `rota=1` for an SSD (or nothing at all). Treating a
/// USB device as "rotational -> HDD" would mislabel every USB-attached SSD,
/// so on `transport == "usb"` we deliberately return `None` ("unknown")
/// instead of `Some(false)` ("HDD"). The caller lets smartctl refine the
/// answer when it can read rotation rate / ATA data.
pub(crate) fn infer_is_ssd(rota: Option<bool>, tran: Option<&str>) -> Option<bool> {
    match (rota, tran) {
        (Some(false), _) => Some(true), // non-rotational = SSD (trusted everywhere)
        (Some(true), Some("usb")) => None, // USB bridge flag: untrusted -> unknown, not HDD
        (Some(true), _) => Some(false), // internal rotational drive = HDD
        (None, _) => None,              // no signal at all
    }
}

/// Check whether `device` (e.g. `/dev/sda` or `sda`) holds the root
/// filesystem ("/" in `/proc/mounts`). Safety guard so we don't print
/// destructive erase commands for the system drive without `--force`.
pub fn is_root_device(device: &str) -> Result<bool> {
    let want = device.strip_prefix("/dev/").unwrap_or(device);
    let content = std::fs::read_to_string("/proc/mounts")
        .with_context(|| "failed to read /proc/mounts")?;
    for line in content.lines() {
        let mut parts = line.split_whitespace();
        let dev = parts.next().unwrap_or("");
        let mnt = parts.next().unwrap_or(""); // mount point is field right after device
        if mnt != "/" {
            continue;
        }
        let have = dev.strip_prefix("/dev/").unwrap_or(dev);
        let rest = &have[want.len()..];
        if rest.is_empty()
            || rest
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit() || c == 'p')
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Run `smartctl --json --all <dev>` and parse it.
/// On failure, retry with `-d <driver>` (ATA passthrough over USB-SATA
/// bridges). Plain `smartctl -j -a /dev/sdX` often cannot reach a drive
/// behind a bridge; USB bridges in particular need a driver hint, and
/// different bridges need different ones (`sat`, `usbjmicron`, `scsi`, `ata`).
pub fn smartctl_json(dev: &str, tran: Option<&str>) -> Result<Value> {
    let mut last_err: Option<anyhow::Error> = None;
    // USB-SATA bridges are inconsistent: try several `-d` driver types so an
    // SSD behind a bridge is actually reachable. Non-USB devices keep the
    // original behaviour (plain, then `-d sat`).
    let drivers = match tran {
        Some("usb") => &["sat", "usbjmicron", "scsi", "ata"][..],
        _ => &["sat"][..],
    };
    let attempts: Vec<Vec<&str>> = std::iter::once(vec!["-j", "-a", dev])
        .chain(drivers.iter().map(|d| vec!["-j", "-a", "-d", d, dev]))
        .collect();
    for args in attempts {
        let (stdout, stderr, _code) = run("smartctl", &args)?;
        if stdout.trim().is_empty() {
            last_err = Some(anyhow::anyhow!(
                "smartctl returned no data for {dev}. stderr: {} \
                 (need smartmontools package, typically root)",
                stderr.trim()
            ));
            continue;
        }
        return serde_json::from_str(&stdout)
            .with_context(|| format!("parsing smartctl -j for {dev}"));
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("smartctl returned no data for {dev}")))
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
    let devs = match list_block_devices() {
        Ok(d) => d,
        Err(e) => {
            notes.push(format!("lsblk: {e:#}"));
            Vec::new()
        }
    };
    let short = device.rsplit('/').next().unwrap_or(device);
    let blk = find_device(&devs, short);
    let mut h = match &blk {
        Some(b) => Health {
            device: b.path.clone().unwrap_or_else(|| format!("/dev/{}", b.name)),
            transport: b.tran.clone().unwrap_or_else(|| "unknown".into()),
            model: b.model.clone().unwrap_or_default(),
            serial: b.serial.clone().unwrap_or_default(),
            capacity_bytes: b.size,
            is_ssd: infer_is_ssd(b.rota, b.tran.as_deref()),
            encryption_active: find_device(&devs, short)
                .map(|d| any_has_crypt(&d.children) || d.dev_type.as_deref() == Some("crypt"))
                .unwrap_or(false),
            source: "lsblk".into(),
            ..Default::default()
        },
        None => {
            notes.push(format!("lsblk: device {device} not found"));
            Health {
                device: device.to_string(),
                source: "no data".into(),
                ..Default::default()
            }
        }
    };

    // -- smartctl: the actual SMART data --------------------------------
    if !tool_exists("smartctl") {
        notes.push(
            "smartmontools (smartctl) not installed — install: \
             sudo apt install smartmontools"
                .into(),
        );
    } else {
        match smartctl_json(device, Some(&h.transport)) {
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
                None => notes.push(format!("nvme id-ctrl: failed to read {device}")),
            }
        } else {
            notes.push("nvme-cli not installed (optional): sudo apt install nvme-cli".into());
        }
    }

    if h.source == "lsblk" {
        h.source = "lsblk (without SMART)".into();
    }
    if h.is_ssd.is_none() {
        notes.push(
            "media type not determined (SSD/HDD) — recommendations given without drive type".into(),
        );
    }
    h.notes = notes;
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_ssd_from_non_rotational_flag() {
        assert_eq!(infer_is_ssd(Some(false), Some("sata")), Some(true));
        assert_eq!(infer_is_ssd(Some(false), Some("nvme")), Some(true));
        assert_eq!(infer_is_ssd(Some(false), None), Some(true));
    }

    #[test]
    fn infers_hdd_from_rotational_flag() {
        assert_eq!(infer_is_ssd(Some(true), Some("sata")), Some(false));
        assert_eq!(infer_is_ssd(Some(true), Some("nvme")), Some(false));
    }

    #[test]
    fn usb_rotational_flag_is_unknown_not_hdd() {
        // USB-SATA bridges report ROTA unreliably: never label a USB
        // device HDD purely from that flag.
        assert_eq!(infer_is_ssd(Some(true), Some("usb")), None);
    }

    #[test]
    fn no_signal_is_unknown() {
        assert_eq!(infer_is_ssd(None, Some("sata")), None);
        assert_eq!(infer_is_ssd(None, None), None);
    }
}
