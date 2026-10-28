//! smartly-rs: SSD health diagnostics, data-loss risk scoring (ONNX ML),
//! recovery advice and "erase without recovery" verdict.
//!
//! Read-only: smartly-rs never writes to or erases any device.

mod advise;
mod collect;
mod demo;
mod features;
mod infer;
mod model;
mod report;

use anyhow::{Context, Result};
use clap::Parser;

use crate::advise::assess;
use crate::infer::{RiskResult, predict};
use crate::report::Report;

#[derive(Parser, Debug)]
#[command(
    name = "smartly-rs",
    version,
    about = "SSD health diagnostics: state, data-loss risk, recovery, erase without recovery"
)]
struct Cli {
    /// Device path, e.g. /dev/nvme0n1 or /dev/sda (requires smartctl; often sudo).
    #[arg(long)]
    disk: Option<String>,
    /// Demo mode without touching the disk: healthy | worn | critical.
    #[arg(long)]
    demo: Option<String>,
    /// List all block devices (lsblk).
    #[arg(long)]
    list: bool,
    /// JSON report instead of text report.
    #[arg(long)]
    json: bool,
    /// Full model interpretation (all 8 features).
    #[arg(long)]
    explain: bool,
    /// Confirm device is not system, before printing erase commands.
    #[arg(long)]
    force: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list {
        for d in collect::list_block_devices()? {
            let mark = match collect::infer_is_ssd(d.rota, d.tran.as_deref()) {
                Some(true) => "SSD",
                Some(false) => "HDD",
                None => " ? ",
            };
            let tran = d.tran.as_deref().unwrap_or("-");
            let size = d
                .size
                .map(|b| format!("{:.1}G", b as f64 / 1e9))
                .unwrap_or_else(|| "?".into());
            let model = d.model.clone().unwrap_or_default();
            println!("{:10} {mark} {tran:5} {size:>8}  {model}", d.name);
        }
        return Ok(());
    }

    let health = if let Some(name) = &cli.demo {
        demo::scenario(name)?
    } else if let Some(dev) = &cli.disk {
        if !cli.force && collect::is_root_device(dev)? {
            anyhow::bail!(
                "device {dev} contains the root filesystem (/). \
                 I will not print erase commands for a system drive. \
                 If this is really the target, rerun with --force."
            );
        }
        collect::probe(dev).context("collecting diagnostics")?
    } else {
        anyhow::bail!("provide --disk /dev/... or --demo healthy|worn|critical (or --list)");
    };

    let feats = features::extract(&health);
    let risk = if health.is_ssd == Some(false) {
        // The risk model was trained exclusively on SSDs (Alibaba DSN'21).
        // An HDD is out of distribution — a probability would be meaningless.
        RiskResult {
            probability: 0.0,
            engine: "n/a".into(),
            error: Some("risk model trained only on SSDs; for HDDs probability not applicable".into()),
            proxy_probability: 0.0,
            approx_explain: true,
            contributions: vec![],
        }
    } else {
        let mut risk = predict(&feats);
        // Safety override: the ML model was trained on data where SMART self-test
        // was always passed, so `smart_fail` carries zero model weight. A drive
        // that self-reports FAILED must not show a low risk number.
        if health.smart_passed == Some(false) && risk.probability < 0.50 {
            risk.probability = 0.50;
        }
        risk
    };
    let assessment = assess(&health, &risk);
    let report = Report::new(health, &feats, risk, assessment);

    if cli.json {
        println!("{}", report.to_json());
    } else {
        crate::report::print_text(&report, cli.explain);
    }
    Ok(())
}
