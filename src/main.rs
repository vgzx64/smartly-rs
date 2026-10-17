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
use crate::infer::predict;
use crate::report::Report;

#[derive(Parser, Debug)]
#[command(
    name = "smartly-rs",
    version,
    about = "Диагностика SSD: состояние, риск потери данных, восстановление, удаление"
)]
struct Cli {
    /// Устройство, например /dev/nvme0n1 или /dev/sda (нужен smartctl; часто sudo).
    #[arg(long)]
    disk: Option<String>,
    /// Демо-режим без обращения к диску: healthy | worn | critical.
    #[arg(long)]
    demo: Option<String>,
    /// Показать все блочные устройства (lsblk).
    #[arg(long)]
    list: bool,
    /// JSON-отчёт вместо текстового.
    #[arg(long)]
    json: bool,
    /// Полная интерпретация модели (все 8 признаков).
    #[arg(long)]
    explain: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list {
        for d in collect::list_block_devices()? {
            let mark = match d.rota {
                Some(false) => "SSD",
                Some(true) => "HDD",
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
        collect::probe(dev).context("сбор диагностики")?
    } else {
        anyhow::bail!("укажите --disk /dev/... или --demo healthy|worn|critical (или --list)");
    };

    let feats = features::extract(&health);
    let risk = predict(&feats);
    let assessment = assess(&health, &risk);
    let report = Report::new(health, &feats, risk, assessment);

    if cli.json {
        println!("{}", report.to_json());
    } else {
        crate::report::print_text(&report, cli.explain);
    }
    Ok(())
}
