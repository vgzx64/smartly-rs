//! Human-readable (and JSON) report rendering.

use std::io::IsTerminal;

use serde::Serialize;

use crate::advise::{Assessment, DriveState, EraseVerdict, Feasibility, RiskLevel};
use crate::features;
use crate::infer::RiskResult;
use crate::model::Health;

#[derive(Serialize)]
pub struct Report {
    pub health: Health,
    pub features: Vec<(String, f32)>,
    pub risk: RiskResult,
    pub assessment: Assessment,
}

impl Report {
    pub fn new(health: Health, feats: &[f32; 8], risk: RiskResult, assessment: Assessment) -> Self {
        Report {
            health,
            features: features::FEATURE_NAMES
                .iter()
                .map(|n| n.to_string())
                .zip(feats.iter().copied())
                .collect(),
            risk,
            assessment,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("report serializes")
    }
}

fn color(on: bool, code: &str, s: &str) -> String {
    if on {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

fn fmt_bytes(b: Option<u64>) -> String {
    match b {
        Some(b) => {
            const T: f64 = 1_000_000_000_000.0;
            if b as f64 >= T {
                format!("{:.1} TB", b as f64 / T)
            } else {
                format!("{:.1} GB", b as f64 / 1_000_000_000.0)
            }
        }
        None => "n/a".into(),
    }
}

pub fn print_text(r: &Report, explain: bool) {
    let c = std::io::stdout().is_terminal();
    let h = &r.health;
    let a = &r.assessment;

    println!("{}", color(c, "1", "=== 1. СОСТОЯНИЕ SSD ==="));
    println!("Устройство : {} ({})", h.device, h.source);
    println!(
        "Модель     : {}  s/n {}",
        if h.model.is_empty() { "n/a" } else { &h.model },
        if h.serial.is_empty() {
            "n/a"
        } else {
            &h.serial
        }
    );
    println!(
        "Транспорт : {} | тип: {} | объём: {}",
        h.transport,
        match h.is_ssd {
            Some(true) => "SSD (NAND)",
            Some(false) => "HDD (вращающийся)",
            None => "неизвестно",
        },
        fmt_bytes(h.capacity_bytes)
    );
    let temp = h
        .temperature_c
        .map(|t| format!("{t:.0}°C"))
        .unwrap_or_else(|| "n/a".into());
    let wear = h
        .percentage_used
        .map(|p| format!("{p:.0}%"))
        .unwrap_or_else(|| "n/a".into());
    let spare = match (h.available_spare, h.spare_threshold) {
        (Some(s), Some(t)) => format!("{s:.0}% (порог {t:.0}%)"),
        (None, _) if h.transport == "sata" => "n/a (нет данных SATA)".into(),
        _ => "n/a".into(),
    };
    println!("Температура: {temp} | износ (percentage_used): {wear} | запас: {spare}");
    println!(
        "Наработка : {} ч | включений: {} | небезопасных выключений: {}",
        h.power_on_hours
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".into()),
        h.power_cycles
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".into()),
        h.unsafe_shutdowns
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".into())
    );
    let media = h
        .media_errors
        .map(|v| v.to_string())
        .or_else(|| h.uncorr_errors.map(|v| v.to_string()))
        .unwrap_or_else(|| "n/a".into());
    let realloc = h
        .realloc_sectors
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".into());
    println!("Ошибки    : media/uncorr = {media} | reallocated = {realloc}");
    println!(
        "SMART     : {}",
        match h.smart_passed {
            Some(true) => color(c, "32", "PASSED"),
            Some(false) => color(c, "31", "FAILED"),
            None => "нет данных".into(),
        }
    );
    let (state_str, color_code) = match a.state {
        DriveState::Ok => ("OK", "32"),
        DriveState::Warning => ("WARNING", "33"),
        DriveState::Critical => ("CRITICAL", "31"),
    };
    println!("Состояние : {}", color(c, color_code, state_str));
    for r in &a.state_reasons {
        println!("  - {r}");
    }
    for n in &h.notes {
        println!("  {} {n}", color(c, "33", "примечание:"));
    }

    println!();
    println!("{}", color(c, "1", "=== 2. РИСК ПОТЕРИ ДАННЫХ (ML) ==="));
    let pct = r.risk.probability * 100.0;
    let (level_str, level_code) = match a.risk_level {
        RiskLevel::Low => ("низкий", "32"),
        RiskLevel::Medium => ("средний", "33"),
        RiskLevel::High => ("высокий", "31"),
        RiskLevel::Critical => ("критический", "31;1"),
    };
    println!(
        "Вероятность отказа с потерей данных: {}% — {} риск",
        color(c, "1", &format!("{pct:.1}")),
        color(c, level_code, level_str)
    );
    println!(
        "Движок    : {} (ONNX-модель на датасете Alibaba DSN'21, вшита в бинарник)",
        r.risk.engine
    );
    if let Some(e) = &r.risk.error {
        println!(
            "  {}/ ONNX недоступен, применена формула: {e}",
            color(c, "33", "предупреждение")
        );
    }
    if explain {
        let t = crate::infer::risk_thresholds();
        println!(
            "Уровни    : низкий <{:.2} ≤ средний <{:.2} ≤ высокий <{:.2} ≤ критический \
             (пороги выведены на тестовой выборке модели)",
            t.medium, t.high, t.critical
        );
    }
    if r.risk.approx_explain {
        println!(
            "Вклады    : условные — линейное приближение даёт {:.1}%, бустинг {:.1}% \
             (смотрите направление вкладов, а не величину)",
            r.risk.proxy_probability * 100.0,
            pct
        );
    }
    println!("Ключевые факторы риска:");
    let shown = if explain { 10 } else { 5 };
    for contrib in r.risk.contributions.iter().take(shown) {
        println!(
            "  {:>15} = {:.3} (w {:+.2}, вклад в log-odds {:+.2})",
            contrib.feature, contrib.value, contrib.weight, contrib.impact
        );
    }
    if explain {
        let feats: Vec<f32> = r.features.iter().map(|(_, v)| *v).collect();
        let mut fake = [0.0f32; 8];
        for (i, v) in feats.iter().enumerate().take(8) {
            fake[i] = *v;
        }
        println!("Все признаки: {}", features::describe(&fake));
        let gi = crate::infer::global_importance();
        if !gi.is_empty() {
            let top: Vec<String> = gi
                .iter()
                .filter(|g| g.importance > 0.0)
                .take(5)
                .map(|g| format!("{} {:.3}", g.feature, g.importance))
                .collect();
            if !top.is_empty() {
                println!(
                    "Значимость для модели (permutation, падение PR-AUC): {}",
                    top.join(", ")
                );
            }
        }
    }

    println!();
    println!(
        "{}",
        color(c, "1", "=== 3. ВОЗМОЖНОСТЬ ВОССТАНОВЛЕНИЯ ДАННЫХ ===")
    );
    let (feas_str, feas_code) = match a.recovery.feasibility {
        Feasibility::High => ("ВЫСОКАЯ", "32"),
        Feasibility::Medium => ("СРЕДНЯЯ", "33"),
        Feasibility::Low => ("НИЗКАЯ", "31"),
    };
    println!("Оценка: {}", color(c, feas_code, feas_str));
    println!("{}", a.recovery.explanation);

    println!();
    println!(
        "{}",
        color(c, "1", "=== 4. РЕКОМЕНДАЦИИ ПО ВОССТАНОВЛЕНИЮ ===")
    );
    for (i, step) in a.recovery.steps.iter().enumerate() {
        println!("  {}. {step}", i + 1);
    }

    println!();
    println!(
        "{}",
        color(
            c,
            "1",
            "=== 5. БЕЗВОЗВРАТНОЕ УДАЛЕНИЕ ДАННЫХ (отдельный блок) ==="
        )
    );
    let (verdict_str, verdict_code) = match a.erase.verdict {
        EraseVerdict::Yes => (
            "ДА — данные можно уничтожить без возможности восстановления",
            "32",
        ),
        EraseVerdict::Partial => (
            "ЧАСТИЧНО — логически недоступны, физические остатки возможны",
            "33",
        ),
        EraseVerdict::No => ("НЕТ — надёжного способа нет", "31"),
    };
    println!(
        "Возможно ли удалить данные без восстановления: {}",
        color(c, verdict_code, verdict_str)
    );
    println!("{}", a.erase.summary);
    println!("Методы (smartly-rs ничего НЕ выполняет автоматически):");
    for m in &a.erase.methods {
        println!("  - {m}");
    }
    println!("Предупреждения:");
    for w in &a.erase.warnings {
        println!("  ! {w}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advise::assess;
    use crate::features::extract;
    use crate::infer::predict;

    #[test]
    fn json_report_round_trips() {
        let health = Health {
            device: "/dev/nvme0n1".into(),
            transport: "nvme".into(),
            model: "TEST SSD".into(),
            is_ssd: Some(true),
            percentage_used: Some(3.0),
            available_spare: Some(100.0),
            spare_threshold: Some(10.0),
            smart_passed: Some(true),
            source: "demo".into(),
            ..Default::default()
        };
        let feats = extract(&health);
        let risk = predict(&feats);
        let assessment = assess(&health, &risk);
        let report = Report::new(health, &feats, risk, assessment);
        let json = report.to_json();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["health"]["model"], "TEST SSD");
        assert!(v["risk"]["probability"].as_f64().unwrap() < 1.0);
        assert!(v["risk"]["proxy_probability"].is_number());
        assert!(v["risk"]["approx_explain"].is_boolean());
        assert!(v["assessment"]["erase"]["verdict"].is_string());
        assert!(v["features"].as_array().unwrap().len() == 8);
    }
}
