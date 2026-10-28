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
    /// Honest provenance warning: the ML model was trained on data-center
    /// SATA SSDs, so the probability is an extrapolation on other devices.
    pub model_caveat: String,
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
            model_caveat: "ML model trained on data-center SATA SSDs (Alibaba DSN'21); probability for other device types is an extrapolation, not an absolute assessment."
                .to_string(),
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

    println!("{}", color(c, "1", "== 1. SSD HEALTH =="));
    println!("Device    : {} ({})", h.device, h.source);
    println!(
        "Model     : {}  s/n {}",
        if h.model.is_empty() { "n/a" } else { &h.model },
        if h.serial.is_empty() {
            "n/a"
        } else {
            &h.serial
        }
    );
    println!(
        "Transport : {} | type: {} | size: {}",
        h.transport,
        match h.is_ssd {
            Some(true) => "SSD (NAND)",
            Some(false) => "HDD (rotating)",
            None => "unknown",
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
        (Some(s), Some(t)) => format!("{s:.0}% (threshold {t:.0}%)"),
        (None, _) if h.transport == "sata" => "n/a (no SATA data)".into(),
        _ => "n/a".into(),
    };
    println!("Temperature : {temp} | wear (percentage_used): {wear} | spare: {spare}");
    println!(
        "Power-on hours : {} | power cycles : {} | unsafe shutdowns : {}",
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
    println!("Errors     : media/uncorr = {media} | reallocated = {realloc}");
    println!(
        "SMART     : {}",
        match h.smart_passed {
            Some(true) => color(c, "32", "PASSED"),
            Some(false) => color(c, "31", "FAILED"),
            None => "no data".into(),
        }
    );
    let (state_str, color_code) = match a.state {
        DriveState::Ok => ("OK", "32"),
        DriveState::Warning => ("WARNING", "33"),
        DriveState::Critical => ("CRITICAL", "31"),
    };
    println!("State     : {}", color(c, color_code, state_str));
    for r in &a.state_reasons {
        println!("  - {r}");
    }
    for n in &h.notes {
        println!("  {} {n}", color(c, "33", "note:"));
    }

    println!();
    println!("{}", color(c, "1", "== 2. DATA LOSS RISK (ML) =="));
    if let Some(e) = &r.risk.error {
        println!(
            "Data loss probability: n/a — {}",
            color(c, "33", e)
        );
    } else {
        let pct = r.risk.probability * 100.0;
        let (level_str, level_code) = match a.risk_level {
            RiskLevel::Low => ("low", "32"),
            RiskLevel::Medium => ("medium", "33"),
            RiskLevel::High => ("high", "31"),
            RiskLevel::Critical => ("critical", "31;1"),
        };
        println!(
            "Data loss probability: {}% — {} risk",
            color(c, "1", &format!("{pct:.1}")),
            color(c, level_code, level_str)
        );
        println!(
            "Engine     : {} (ONNX model on Alibaba DSN'21 dataset, embedded in binary)",
            r.risk.engine
        );
    }
    println!("  {} {}", color(c, "33", "note:"), r.model_caveat);
    if explain {
        let t = crate::infer::risk_thresholds();
        println!(
            "Levels    : low <{:.2} <= medium <{:.2} <= high <{:.2} <= critical \
             (cut-offs from held-out model test split)",
            t.medium, t.high, t.critical
        );
    }
    if !r.risk.error.is_some() && r.risk.approx_explain {
        println!(
            "Contributions : conditional — linear approximation {:.1}%, boosting {:.1}% \
             (check directions, not magnitudes)",
            r.risk.proxy_probability * 100.0,
            r.risk.probability * 100.0,
        );
    }
    println!("Key risk factors (model relies on):");
    if r.risk.error.is_none() {
        let gi = crate::infer::global_importance();
        let shown = if explain { gi.len() } else { 5 };
        for g in gi.iter().take(shown).filter(|g| g.importance > 0.0) {
            println!(
                "  {:>15} — importance {:.4} (PR-AUC drop)",
                g.feature, g.importance
            );
        }
        if explain {
            println!();
            println!("Feature contributions for this device (linear approximation):");
            for contrib in r.risk.contributions.iter().take(8) {
                println!(
                    "  {:>15} = {:.3} (w {:+.2}, log-odds {:+.2})",
                    contrib.feature, contrib.value, contrib.weight, contrib.impact
                );
            }
            let feats: Vec<f32> = r.features.iter().map(|(_, v)| *v).collect();
            let mut fake = [0.0f32; 8];
            for (i, v) in feats.iter().enumerate().take(8) {
                fake[i] = *v;
            }
            println!("All features: {}", features::describe(&fake));
        }
    }

    println!();
    println!(
        "{}",
        color(c, "1", "== 3. DATA RECOVERY FEASIBILITY ==")
    );
    let (feas_str, feas_code) = match a.recovery.feasibility {
        Feasibility::High => ("HIGH", "32"),
        Feasibility::Medium => ("MEDIUM", "33"),
        Feasibility::Low => ("LOW", "31"),
    };
    println!("Assessment: {}", color(c, feas_code, feas_str));
    println!("{}", a.recovery.explanation);

    println!();
    println!(
        "{}",
        color(c, "1", "== 4. RECOVERY RECOMMENDATIONS ==")
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
            "== 5. UNRECOVERABLE DATA ERASURE (separate block) =="
        )
    );
    let (verdict_str, verdict_code) = match a.erase.verdict {
        EraseVerdict::Yes => (
            "YES — data can be destroyed with no recovery possible",
            "32",
        ),
        EraseVerdict::Partial => (
            "PARTIALLY — logically inaccessible, physical remnants may remain",
            "33",
        ),
        EraseVerdict::No => ("NO — no reliable method available", "31"),
    };
    println!(
        "Erase without recovery: {}",
        color(c, verdict_code, verdict_str)
    );
    println!("{}", a.erase.summary);
    println!("Methods (smartly-rs never executes anything automatically):");
    for m in &a.erase.methods {
        println!("  - {m}");
    }
    println!("Warnings:");
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
