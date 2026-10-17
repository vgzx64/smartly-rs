//! Extract the 8 normalized model features from a [`Health`] snapshot.
//!
//! Feature order must match `assets/train_risk_model.py` exactly.

use crate::model::Health;

/// Feature names, same order as in the ONNX model input.
pub const FEATURE_NAMES: [&str; 8] = [
    "pct_used",
    "spare_deficit",
    "media_err",
    "temp",
    "poh",
    "crit_warn",
    "realloc",
    "smart_fail",
];

pub fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn popcount(mut b: u8) -> u32 {
    let mut c = 0;
    while b != 0 {
        c += (b & 1) as u32;
        b >>= 1;
    }
    c
}

/// Build the feature vector `[f32; 8]` for the risk model.
pub fn extract(h: &Health) -> [f32; 8] {
    let pct_used = clamp01(h.percentage_used.unwrap_or(0.0) / 100.0);

    // How close the spare blocks are to their threshold:
    // 0 = spare at 100%, 1 = spare at/below threshold.
    // For SATA (no spare metric) approximate with consumed life.
    let spare_deficit = match (h.available_spare, h.spare_threshold) {
        (Some(spare), Some(thr)) => {
            let span = (100.0 - thr).max(1.0);
            clamp01(1.0 - (spare - thr) / span)
        }
        _ => pct_used,
    };

    let media_err = match h.media_errors {
        Some(m) => clamp01((1.0 + m as f64).log10() / 5.0),
        None => {
            // SATA: combine uncorrectable + pending + reallocated as errors.
            let e = h.uncorr_errors.unwrap_or(0) + h.pending_sectors.unwrap_or(0);
            clamp01((1.0 + e as f64).log10() / 5.0)
        }
    };

    let temp = clamp01((h.temperature_c.unwrap_or(30.0) - 30.0) / 60.0);

    let poh =
        clamp01((1.0 + h.power_on_hours.unwrap_or(0) as f64).log10() / (1.0f64 + 87_600.0).log10());

    let crit_warn = match h.critical_warning {
        Some(cw) => clamp01(popcount(cw) as f64 / 5.0),
        None => 0.0,
    };

    let realloc = clamp01((1.0 + h.realloc_sectors.unwrap_or(0) as f64).log10() / 6.0);

    let smart_fail = if h.smart_passed == Some(false) {
        1.0
    } else {
        0.0
    };

    [
        pct_used as f32,
        spare_deficit as f32,
        media_err as f32,
        temp as f32,
        poh as f32,
        crit_warn as f32,
        realloc as f32,
        smart_fail as f32,
    ]
}

/// Pretty-print features as `name=value` pairs.
pub fn describe(feats: &[f32; 8]) -> String {
    FEATURE_NAMES
        .iter()
        .zip(feats.iter())
        .map(|(n, v)| format!("{n}={v:.3}"))
        .collect::<Vec<_>>()
        .join("  ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_nvme_features_saturate() {
        let h = Health {
            percentage_used: Some(100.0),
            available_spare: Some(5.0),
            spare_threshold: Some(10.0),
            media_errors: Some(25_000),
            temperature_c: Some(68.0),
            power_on_hours: Some(60_000),
            unsafe_shutdowns: Some(1500),
            critical_warning: Some(0b111),
            bytes_written: Some(1_000_000_000_000),
            capacity_bytes: Some(500_000_000_000),
            smart_passed: Some(false),
            ..Default::default()
        };
        let f = extract(&h);
        assert_eq!(f[0], 1.0); // pct_used
        assert_eq!(f[1], 1.0); // spare below threshold -> saturated
        assert!(f[2] > 0.85); // media errors
        assert!(f[7] == 1.0); // smart fail
        assert!(f.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn sata_without_spare_falls_back_to_life_left() {
        let h = Health {
            percentage_used: Some(40.0),
            ..Default::default()
        };
        let f = extract(&h);
        assert!((f[1] - 0.4).abs() < 1e-6);
        assert!((f[0] - 0.4).abs() < 1e-6);
    }
}
