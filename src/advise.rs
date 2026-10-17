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
        reasons.push("SMART-самотест сообщил FAILED".into());
        state = bump(DriveState::Critical, state);
    }
    if let (Some(spare), Some(thr)) = (h.available_spare, h.spare_threshold) {
        if spare <= thr {
            reasons.push(format!(
                "available_spare {spare:.0}% ≤ порога {thr:.0}% — ресурс резервных блоков исчерпан"
            ));
            state = bump(DriveState::Critical, state);
        } else if spare <= thr + 10.0 {
            reasons.push(format!(
                "available_spare {spare:.0}% близок к порогу {thr:.0}%"
            ));
            state = bump(DriveState::Warning, state);
        }
    }
    if h.read_only_mode {
        reasons.push("контроллер перешёл в режим read-only (critical_warning bit3)".into());
        state = bump(DriveState::Critical, state);
    }
    if let Some(cw) = h.critical_warning {
        if cw & 0b011 != 0 {
            reasons.push(format!(
                "NVMe critical_warning = 0x{cw:02x} (перегрев / запас ниже порога)"
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
            reasons.push(format!("износ {pu:.0}% — ресурс почти исчерпан"));
            state = bump(DriveState::Critical, state);
        } else if pu >= 70.0 {
            reasons.push(format!("износ {pu:.0}% — ресурс заметно израсходован"));
            state = bump(DriveState::Warning, state);
        }
    }
    if let Some(m) = h.media_errors {
        if m >= 100 {
            reasons.push(format!(
                "media_errors = {m} — накапливаются неисправимые ошибки чтения"
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
        reasons.push(format!("некорректируемые ошибки чтения (attr 198) = {u}"));
        state = bump(DriveState::Warning, state);
    }
    if let Some(p) = h.pending_sectors
        && p > 0
    {
        reasons.push(format!("pending-сектора (attr 197) = {p}"));
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
            reasons.push(format!("температура {t:.0}°C — критически высоко"));
            state = bump(DriveState::Critical, state);
        } else if t >= 60.0 {
            reasons.push(format!("температура {t:.0}°C выше нормы"));
            state = bump(DriveState::Warning, state);
        }
    }
    if h.is_ssd == Some(false) {
        reasons.push("устройство не SSD (вращающийся диск) — SMART-пороги SSD неприменимы".into());
    }
    if reasons.is_empty() {
        reasons.push("отклонений SMART не обнаружено".into());
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
                "Диск работоспособен. Файлы, потерянные на исправном SSD, восстанавливаются \
                 обычным ПО, НО: после TRIM контроллер уже обнуллил свободные блоки, поэтому \
                 «удалённые» файлы на SSD обычно невосстановимы без бэкапа."
                    .into(),
            steps: vec![
                "Немедленно сделайте бэкап важных данных (rsync/restic/borg).".into(),
                "Восстановление целостности ФС: PhotoRec, R-Studio, UFS Explorer — с другого диска."
                    .into(),
                "TRIM + бэкапы — лучшая «защита от восстановления» после удаления файлов.".into(),
                "Регулярно сохраняйте smartctl -a отчёт для отслеживания динамики износа.".into(),
            ],
        },
        DriveState::Warning => RecoveryAdvice {
            feasibility: Feasibility::Medium,
            explanation:
                "Есть признаки деградации. Существующие данные ещё читаются, но отказ может \
                 наступить в любой момент: шансы на восстановление падают с каждым часом работы диска."
                    .into(),
            steps: vec![
                "Приоритет №1: полный образ диска: sudo ddrescue /dev/sdX image.img mapfile".into(),
                "Не записывайте новые данные на этот диск и не запускайте на нём fsck.".into(),
                "Восстанавливайте файлы уже с образа, по возможности сразу.".into(),
                "Запланируйте замену диска; сохраните SMART-отчёт как доказательство деградации."
                    .into(),
            ],
        },
        DriveState::Critical => RecoveryAdvice {
            feasibility: Feasibility::Low,
            explanation:
                "Диск критически неисправен (исчерпан spare-ресурс, read-only режим или массовые \
                 media errors). Чтение может обрываться; многократные попытки ускоряют деградацию. \
                 При физическом отказе контроллера поможет только лаборатория (PC-3000, DeepSpar)."
                    .into(),
            steps: vec![
                "НЕМЕДЛЕННО: ddrescue (несколько проходов, малое число повторов), не включайте диск повторно без нужды."
                    .into(),
                "Подключайте напрямую (без USB-мостов), отключите автоматический fsck/монтирование."
                    .into(),
                "НЕ делайте badblocks/переразметку — это уничтожает данные.".into(),
                "Если диск не определяется физически — только лаборатория восстановления данных."
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
        "smartly-rs НЕ выполняет удаление — команды приведены только для справки; перед \
         любым стиранием сделайте резервную копию и перепроверьте имя устройства."
            .into(),
        "Повторная запись (shred/wipe) на SSD ненадёжна из-за wear-leveling и over-provisioning: \
         часть старых страниц может остаться в резервных зонах."
            .into(),
    ];

    if h.encryption_active {
        warnings.push(
            "Диск/раздел под шифрованием (LUKS/dm-crypt): уничтожение ключа делает данные \
             нечитаемыми мгновенно и гарантированно."
                .into(),
        );
    }

    if h.is_ssd == Some(false) {
        return EraseAdvice {
            verdict: EraseVerdict::Yes,
            summary: "Это вращающийся HDD: данные можно уничтожить многократной перезаписью; \
                 после неё чтение физически невозможно."
                .into(),
            methods: vec![
                "sudo nwipe /dev/sdX             (надёжное затирание с журналом)".into(),
                "sudo shred -uzn 3 /dev/sdX      (3 прохода + удаление)".into(),
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
                    "nvme format /dev/nvmeXn1 --ses=1   (user-data erase: контроллер затирает данные сам)"
                        .into(),
                    "blkdiscard /dev/nvmeXn1            (быстрый discard — логически, не физически)"
                        .into(),
                ],
                EraseVerdict::Yes,
                "Sanitize недоступен, но обязательная команда Format NVM (ses=1) уничтожает \
                 данные; после неё чтение старых данных невозможно.",
            )
        } else {
            (
                vec![
                    "nvme format /dev/nvmeXn1 --ses=2    (crypto-erase: ключи уничтожаются)".into(),
                    "nvme sanitize /dev/nvmeXn1 -a 2     (crypto erase) или -a 1 (block erase)"
                        .into(),
                ],
                EraseVerdict::Yes,
                "NVMe поддерживает Sanitize/Format NVM (в т.ч. crypto-erase): контроллер \
                 необратимо уничтожает данные, включая резервные зоны — восстановление невозможно.",
            )
        }
    } else if usb_bridge {
        warnings.push(
            "USB-мост может не пропускать ATA Sanitize/TRIM — проверьте поддержку через hdparm -I"
                .into(),
        );
        (
            vec![
                "hdparm --user-master u --security-set-pass p /dev/sdX && \
                 hdparm --user-master u --security-erase p /dev/sdX"
                    .into(),
                "sudo blkdiscard /dev/sdX           (только логически; не гарантирует стирание OP)"
                    .into(),
            ],
            EraseVerdict::Partial,
            "SSD в USB-корпусе: ATA Secure Erase недоступен через часть мостов, discard — только \
             логический. Полное уничтожение физических остатков не гарантируется.",
        )
    } else {
        (
            vec![
                "hdparm --user-master u --security-set-pass p /dev/sdX && \
                 hdparm --user-master u --security-erase p /dev/sdX (ATA Secure Erase)"
                    .into(),
                "sudo sg_sanitize /dev/sdX -c       (ATA Sanitize, если поддерживается)".into(),
                "sudo blkdiscard -f /dev/sdX         (быстрый TRIM-обнул всего устройства)".into(),
            ],
            EraseVerdict::Yes,
            "Нативный SATA SSD: ATA Secure Erase/Sanitize стирает данные и резервные зоны — \
             восстановление невозможно.",
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
