//! Built-in demo scenarios so the whole pipeline can be exercised
//! without root access or a spare SSD.

use anyhow::{Result, bail};

use crate::model::Health;

/// Build a synthetic [`Health`] snapshot for one of the demo scenarios.
pub fn scenario(name: &str) -> Result<Health> {
    let base = Health {
        device: "/dev/nvme0n1 (demo)".into(),
        transport: "nvme".into(),
        model: "DEMO SSD 512GB".into(),
        serial: "DEMO-0001".into(),
        capacity_bytes: Some(512_110_190_592),
        is_ssd: Some(true),
        power_cycles: Some(2048),
        sanitize_supported: Some(true),
        crypto_erase_supported: Some(true),
        encryption_active: false,
        source: "demo".into(),
        ..Default::default()
    };

    match name {
        "healthy" => Ok(Health {
            temperature_c: Some(39.0),
            percentage_used: Some(3.0),
            available_spare: Some(100.0),
            spare_threshold: Some(10.0),
            media_errors: Some(0),
            power_on_hours: Some(8760),
            unsafe_shutdowns: Some(150),
            critical_warning: Some(0),
            // 3% wear of 1500 P/E cycles.
            bytes_written: Some(512_110_190_592u64 * 1500 * 3 / 100),
            smart_passed: Some(true),
            ..base
        }),
        "worn" => Ok(Health {
            temperature_c: Some(46.0),
            percentage_used: Some(45.0),
            available_spare: Some(85.0),
            spare_threshold: Some(10.0),
            media_errors: Some(20),
            power_on_hours: Some(12_000),
            unsafe_shutdowns: Some(100),
            critical_warning: Some(0),
            bytes_written: Some((512_110_190_592u64 * 1500) * 45 / 100),
            smart_passed: Some(true),
            ..base
        }),
        "critical" => Ok(Health {
            temperature_c: Some(68.0),
            percentage_used: Some(100.0),
            available_spare: Some(5.0),
            spare_threshold: Some(10.0),
            media_errors: Some(25_000),
            power_on_hours: Some(60_000),
            unsafe_shutdowns: Some(1500),
            critical_warning: Some(0b0111),
            bytes_written: Some((512_110_190_592u64 * 1500) * 99 / 100),
            smart_passed: Some(false),
            ..base
        }),
        other => bail!("unknown demo scenario `{other}`; available: healthy, worn, critical"),
    }
}
