//! End-to-end Pass 1 smoke test against the deployed GLM-OCR Modal worker.
//!
//! Usage:
//!     cargo run --example pass1_smoke -- <pdf_path> [page1,page2,...]
//!
//! Example:
//!     cargo run --example pass1_smoke -- \
//!         data/corpus/coilcraft_xal7030.pdf 0,1

use extractor_client::{modal::ModalExtractor, Extractor};
use ir::Content;
use std::time::Instant;

const PASS1_URL: &str = "https://ferrite-systems--parselab-glm-ocr-serve.modal.run";
const PASS2_URL: &str = "https://ferrite-systems--parselab-infinity-parser2-serve.modal.run";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let pdf_path = args
        .next()
        .ok_or("usage: pass1_smoke <pdf_path> [comma-separated-pages]")?;
    let pages: Option<Vec<u32>> = args
        .next()
        .map(|s| s.split(',').map(|p| p.trim().parse()).collect())
        .transpose()?;

    let pdf = std::fs::read(&pdf_path)?;
    println!(
        "loaded {} ({} bytes); pages={:?}",
        pdf_path,
        pdf.len(),
        pages
    );

    let client = ModalExtractor::new(PASS1_URL, PASS2_URL);

    let started = Instant::now();
    let result = client.pass1(&pdf, pages.as_deref()).await?;
    let wall_secs = started.elapsed().as_secs_f64();

    let extraction = &result.extraction;
    println!("\n--- extraction ---");
    println!("uuid:      {}", extraction.uuid.0);
    println!("model:     {}", extraction.model.0);
    println!("prompt:    {:?}", extraction.prompt.0);
    println!("params:    {:?}", extraction.params);
    println!("created:   {}", extraction.created_at);
    println!("wall_secs: {wall_secs:.2}");

    // Dump per-page markdown to /tmp so we can read it without re-running.
    for p in &result.pages {
        if let Content::Markdown(m) = &p.content {
            let path = format!("/tmp/pass1_smoke_page_{}.md", p.page);
            std::fs::write(&path, &m.markdown)?;
            println!("wrote {path}");
        }
    }

    println!("\n--- pages ({}) ---", result.pages.len());
    let mut total_in_tok = 0u64;
    let mut total_out_tok = 0u64;
    let mut total_gpu_secs = 0f64;
    let mut errors = 0;
    for p in &result.pages {
        let kind = match &p.content {
            Content::Markdown(m) => format!("markdown ({} chars)", m.markdown.len()),
            Content::Structured(s) => format!("structured ({} blocks)", s.blocks.len()),
            Content::Error(e) => {
                errors += 1;
                format!("ERROR kind={} message={:?}", e.kind, e.message)
            }
        };
        let metrics = p
            .metrics
            .map(|m| {
                format!(
                    "{:.2}s in={} out={}",
                    m.elapsed_secs,
                    m.input_tokens.unwrap_or(0),
                    m.output_tokens.unwrap_or(0)
                )
            })
            .unwrap_or_else(|| "no-metrics".into());
        println!(
            "  page {:>3}  uuid={}  {}  [{}]",
            p.page, p.page_uuid.0, kind, metrics
        );
        if let Some(m) = p.metrics {
            total_in_tok += m.input_tokens.unwrap_or(0) as u64;
            total_out_tok += m.output_tokens.unwrap_or(0) as u64;
            total_gpu_secs += m.elapsed_secs as f64;
        }
    }

    println!("\n--- totals ---");
    println!("ok:            {}", result.pages.len() - errors);
    println!("errors:        {errors}");
    println!("gpu_secs:      {total_gpu_secs:.2}");
    println!("wall_secs:     {wall_secs:.2}");
    println!("input_tokens:  {total_in_tok}");
    println!("output_tokens: {total_out_tok}");

    Ok(())
}
