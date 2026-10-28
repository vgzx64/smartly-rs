//! Expert rules on top of the ML risk score: drive state, data-recovery
//! feasibility and the "erase without recovery" verdict.

use serde::Serialize;

use crate::infer::RiskResult;
use crate::model::Health;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DriveState {
    Ok,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Feasibility {
    High,
    Medium,
    Low,
}

/// Verdict: can the data be destroyed so that recovery is impossible?
/// (`No` is part of the public JSON schema for completeness.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum EraseVerdict {
    /// Yes: erase command (crypto/block sanitize / secure erase / key wipe).
    Yes,
    /// Partially: logical data becomes unreachable, physical remnants may survive.
    Partial,
    /// No reliable method with the tools/medium at hand.
    No,
}

#[derive(Debug, Clone, Serialize)]
pub struct Assessment {
    pub state: DriveState,
    pub state_reasons: Vec<String>,
    pub risk_level: RiskLevel,
    pub recovery: RecoveryAdvice,
    pub erase: EraseAdvice,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecoveryAdvice {
    pub feasibility: Feasibility,
    pub explanation: String,
    pub steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EraseAdvice {
    pub verdict: EraseVerdict,
    pub summary: String,
    /// Commands/methods (printed, never executed by smartly-rs).
    pub methods: Vec<String>,
    pub warnings: Vec<String>,
}

/// Map a probability to a risk level using the score cut-offs derived on
/// the model's held-out test split (`assets/train_metrics.json`).
pub fn risk_level(p: f64) -> RiskLevel {
    let t = crate::infer::risk_thresholds();
    if p < t.medium {
        RiskLevel::Low
    } else if p < t.high {
        RiskLevel::Medium
    } else if p < t.critical {
        RiskLevel::High
    } else {
        RiskLevel::Critical
    }
}

fn bump(s: DriveState, cur: DriveState) -> DriveState {
    if (s as usize) > (cur as usize) {
        s
    } else {
        cur
    }
}

fn classify(h: &Health) -> (DriveState, Vec<String>) {
    let mut reasons: Vec<String> = Vec::new();
    let mut state = DriveState::Ok;

    if h.smart_passed == Some(false) {
        reasons.push("SMART self-test reported FAILED".into());
        state = bump(DriveState::Critical, state);
    }
    if let (Some(spare), Some(thr)) = (h.available_spare, h.spare_threshold) {
        if spare <= thr {
            reasons.push(format!(
                "available_spare {spare:.0}% <= threshold {thr:.0}% — spare block resource exhausted"
            ));
            state = bump(DriveState::Critical, state);
        } else if spare <= thr + 10.0 {
            reasons.push(format!(
                "available_spare {spare:.0}% close to threshold {thr:.0}%%"
            ));
            state = bump(DriveState::Warning, state);
        }
    }
    if h.read_only_mode {
        reasons.push("controller switched to read-only mode (critical_warning bit3)".into());
        state = bump(DriveState::Critical, state);
    }
    if let Some(cw) = h.critical_warning {
        if cw & 0b011 != 0 {
            reasons.push(format!(
                "NVMe critical_warning = 0x{cw:02x} (overheat / spare below threshold)"
            ));
            state = bump(DriveState::Warning, state);
        }
        if cw & 0b100 != 0 {
            reasons.push("NVMe critical_warning: reliability degrade".into());
            state = bump(DriveState::Critical, state);
        }
    }
    if let Some(pu) = h.percentage_used {
        if pu >= 95.0 {
            reasons.push(format!("wear {pu:.0}% — resource almost exhausted"));
            state = bump(DriveState::Critical, state);
        } else if pu >= 70.0 {
            reasons.push(format!("wear {pu:.0}% — resource significantly consumed"));
            state = bump(DriveState::Warning, state);
        }
    }
    if let Some(m) = h.media_errors {
        if m >= 100 {
            reasons.push(format!(
                "media_errors = {m} — uncorrectable read errors accumulating"
            ));
            state = bump(DriveState::Critical, state);
        } else if m > 0 {
            reasons.push(format!("media_errors = {m}"));
            state = bump(DriveState::Warning, state);
        }
    }
    if let Some(u) = h.uncorr_errors
        && u > 0
    {
        reasons.push(format!("uncorrectable read errors (attr 198) = {u}"));
        state = bump(DriveState::Warning, state);
    }
    if let Some(p) = h.pending_sectors
        && p > 0
    {
        reasons.push(format!("pending sectors (attr 197) = {p}"));
        state = bump(DriveState::Warning, state);
    }
    if let Some(r) = h.realloc_sectors
        && r >= 100
    {
        reasons.push(format!("reallocated sectors = {r}"));
        state = bump(DriveState::Warning, state);
    }
    if let Some(t) = h.temperature_c {
        if t >= 75.0 {
            reasons.push(format!("temperature {t:.0}C — critically high"));
            state = bump(DriveState::Critical, state);
        } else if t >= 60.0 {
            reasons.push(format!("temperature {t:.0}C above normal"));
            state = bump(DriveState::Warning, state);
        }
    }
    if h.is_ssd == Some(false) {
        reasons.push("device is not SSD (rotating disk) — SSD SMART thresholds not applicable".into());
    }
    if reasons.is_empty() {
        reasons.push("no SMART deviations detected".into());
    }
    (state, reasons)
}

pub fn assess(h: &Health, risk: &RiskResult) -> Assessment {
    let (state, state_reasons) = classify(h);
    let risk_level = risk_level(risk.probability);

    let recovery = match state {
        DriveState::Ok => RecoveryAdvice {
            feasibility: Feasibility::High,
            explanation:
                "Disk is healthy. Files lost on a healthy SSD are recoverable \
                 with standard software, BUT: after TRIM the controller has already zeroed freed blocks, so \
                 deleted files on SSD are typically unrecoverable without a backup."
                    .into(),
            steps: vec![
                "Back up important data immediately (rsync/restic/borg).".into(),
                "Filesystem integrity check: PhotoRec, R-Studio, UFS Explorer — from another disk."
                    .into(),
                "TRIM + backups are the best \"protection against recovery\" after file deletion.".into(),
                "Regularly save smartctl -a reports to track wear trends.".into(),
            ],
        },
        DriveState::Warning => RecoveryAdvice {
            feasibility: Feasibility::Medium,
            explanation:
                "Signs of degradation. Existing data is still readable, but failure can \
                 occur at any moment: recovery chances drop with every hour of disk operation."
                    .into(),
            steps: vec![
                "Priority #1: full disk image: sudo ddrescue /dev/sdX image.img mapfile".into(),
                "Do not write new data to this disk and do not run fsck on it.".into(),
                "Restore files from the image, as soon as possible.".into(),
                "Plan disk replacement; save SMART report as proof of degradation."
                    .into(),
            ],
        },
        DriveState::Critical => RecoveryAdvice {
            feasibility: Feasibility::Low,
            explanation:
                "Disk is critically failing (spare exhausted, read-only mode or mass \
                 media errors). Read may abort; repeated attempts accelerate degradation. \
                 On physical controller failure only a lab (PC-3000, DeepSpar) can help."
                    .into(),
            steps: vec![
                "IMMEDIATELY: ddrescue (multiple passes, few retries), do not power the disk on again without need."
                    .into(),
                "Connect directly (no USB bridges), disable automatic fsck/mounting."
                    .into(),
                "Do NOT run badblocks/repartition — this destroys data.".into(),
                "If the disk is not physically detected — only a data recovery lab."
                    .into(),
            ],
        },
    };

    Assessment {
        state,
        state_reasons,
        risk_level,
        recovery,
        erase: erase_advice(h),
    }
}

fn erase_advice(h: &Health) -> EraseAdvice {
    let mut warnings = vec![
        "smartly-rs does NOT perform deletion — commands are for reference only; before \
         erasing make a backup and double-check the device name."
            .into(),
        "Repeated writes (shred/wipe) on SSD are unreliable due to wear-leveling and over-provisioning: \
         some old pages may remain in spare areas."
            .into(),
    ];

    if h.encryption_active {
        warnings.push(
            "Disk/partition encrypted (LUKS/dm-crypt): destroying the key makes data \
             instantly and guaranteed unreadable."
                .into(),
        );
    }

    if h.is_ssd == Some(false) {
        return EraseAdvice {
            verdict: EraseVerdict::Yes,
            summary: "This is a rotating HDD: data can be destroyed by multiple overwrites; \
                 afterwards physical reading is impossible."
                .into(),
            methods: vec![
                "sudo nwipe /dev/sdX             (reliable wiping with journal)".into(),
                "sudo shred -uzn 3 /dev/sdX      (3 passes + deletion)".into(),
                "sudo dd if=/dev/urandom of=/dev/sdX bs=4M status=progress".into(),
            ],
            warnings,
        };
    }

    let transport_nvme = h.transport == "nvme";
    let usb_bridge = h.transport == "usb";

    let (methods, verdict, summary) = if transport_nvme {
        if h.sanitize_supported == Some(false) && h.crypto_erase_supported == Some(false) {
            (
                vec![
                    "nvme format /dev/nvmeXn1 --ses=1   (user-data erase: controller erases data itself)"
                        .into(),
                    "blkdiscard /dev/nvmeXn1            (fast discard — logical, not physical)"
                        .into(),
                ],
                EraseVerdict::Yes,
                "Sanitize unavailable, but mandatory Format NVM (ses=1) destroys data \
                 data; afterwards reading old data is impossible.",
            )
        } else {
            (
                vec![
                    "nvme format /dev/nvmeXn1 --ses=2    (crypto-erase: keys are destroyed)".into(),
                    "nvme sanitize /dev/nvmeXn1 -a 2     (crypto erase) or -a 1 (block erase)"
                        .into(),
                ],
                EraseVerdict::Yes,
                "NVMe supports Sanitize/Format NVM (including crypto-erase): controller \
                 immediately and irrevocably destroys data including spare areas — recovery impossible.",
            )
        }
    } else if usb_bridge {
        warnings.push(
            "USB bridge may not pass through ATA Sanitize/TRIM — check support via hdparm -I"
                .into(),
        );
        (
            vec![
                "hdparm --user-master u --security-set-pass p /dev/sdX && \
                 hdparm --user-master u --security-erase p /dev/sdX"
                    .into(),
                "sudo blkdiscard /dev/sdX           (logical only; does not guarantee OP erasure)"
                    .into(),
            ],
            EraseVerdict::Partial,
            "SSD in USB enclosure: ATA Secure Erase unavailable through some bridges, discard — only \
             logical. Complete destruction of physical remnants not guaranteed.",
        )
    } else {
        (
            vec![
                "hdparm --user-master u --security-set-pass p /dev/sdX && \
                 hdparm --user-master u --security-erase p /dev/sdX (ATA Secure Erase)"
                    .into(),
                "sudo sg_sanitize /dev/sdX -c       (ATA Sanitize, if supported)".into(),
                "sudo blkdiscard -f /dev/sdX         (fast TRIM zeroing of entire device)".into(),
            ],
            EraseVerdict::Yes,
            "Native SATA SSD: ATA Secure Erase/Sanitize erases data and spare areas — \
             recovery impossible.",
        )
    };

    EraseAdvice {
        verdict,
        summary: summary.to_string(),
        methods,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::RiskResult;

    fn risk(p: f64) -> RiskResult {
        RiskResult {
            probability: p,
            engine: "test".into(),
            error: None,
            proxy_probability: p,
            approx_explain: false,
            contributions: vec![],
        }
    }

    fn healthy_nvme() -> Health {
        Health {
            transport: "nvme".into(),
            is_ssd: Some(true),
            percentage_used: Some(3.0),
            available_spare: Some(100.0),
            spare_threshold: Some(10.0),
            media_errors: Some(0),
            temperature_c: Some(39.0),
            critical_warning: Some(0),
            smart_passed: Some(true),
            ..Default::default()
        }
    }

    #[test]
    fn healthy_nvme_is_ok_with_high_recovery() {
        let a = assess(&healthy_nvme(), &risk(0.1));
        assert_eq!(a.state, DriveState::Ok);
        assert_eq!(a.recovery.feasibility, Feasibility::High);
        assert_eq!(a.erase.verdict, EraseVerdict::Yes); // nvme sanitize
    }

    #[test]
    fn critical_nvme_is_low_recovery() {
        let h = Health {
            percentage_used: Some(100.0),
            available_spare: Some(5.0),
            media_errors: Some(25_000),
            temperature_c: Some(68.0),
            critical_warning: Some(0b111),
            smart_passed: Some(false),
            ..healthy_nvme()
        };
        let a = assess(&h, &risk(0.97));
        assert_eq!(a.state, DriveState::Critical);
        assert_eq!(a.recovery.feasibility, Feasibility::Low);
        assert_eq!(a.risk_level, RiskLevel::Critical);
        assert!(a.state_reasons.len() >= 3);
    }

    #[test]
    fn usb_bridge_ssd_erase_is_partial() {
        let h = Health {
            transport: "usb".into(),
            ..healthy_nvme()
        };
        let a = assess(&h, &risk(0.1));
        assert_eq!(a.erase.verdict, EraseVerdict::Partial);
    }

    #[test]
    fn hdd_erase_uses_overwrite() {
        let h = Health {
            transport: "sata".into(),
            is_ssd: Some(false),
            smart_passed: None,
            ..healthy_nvme()
        };
        let a = assess(&h, &risk(0.1));
        assert_eq!(a.erase.verdict, EraseVerdict::Yes);
        assert!(a.erase.methods[0].contains("nwipe"));
    }

    #[test]
    fn risk_levels_thresholds() {
        let t = crate::infer::risk_thresholds();
        assert!(
            t.medium < t.high && t.high < t.critical,
            "cut-offs must be ordered: {t:?}"
        );
        assert_eq!(risk_level(0.0), RiskLevel::Low);
        assert_eq!(risk_level(t.medium - 0.001), RiskLevel::Low);
        assert_eq!(risk_level(t.medium), RiskLevel::Medium);
        assert_eq!(risk_level(t.high - 0.001), RiskLevel::Medium);
        assert_eq!(risk_level(t.high), RiskLevel::High);
        assert_eq!(risk_level(t.critical - 0.001), RiskLevel::High);
        assert_eq!(risk_level(t.critical), RiskLevel::Critical);
        assert_eq!(risk_level(1.0), RiskLevel::Critical);
    }
}
