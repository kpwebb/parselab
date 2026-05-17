//! Ad-hoc smoke test (FER-120) — send a single PNG + prompt to one of
//! the deployed VLMs and print the response.
//!
//! Usage:
//!     cargo run --example ad_hoc_smoke -- <png_path> <model> "<prompt>"
//!
//! Where <model> is one of: glm-ocr | infinity | qwen36

use extractor_client::ad_hoc::{AdHocClient, AdHocModel};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let png_path = args
        .next()
        .ok_or("usage: ad_hoc_smoke <png_path> <model> <prompt>")?;
    let model_arg = args.next().ok_or("missing model name")?;
    let prompt = args.next().ok_or("missing prompt")?;

    let model = match model_arg.as_str() {
        "glm-ocr" | "glm" => AdHocModel::GlmOcr,
        "infinity" | "inf2" => AdHocModel::InfinityParser2Pro,
        "qwen36" | "qwen" => AdHocModel::Qwen36MoE,
        other => {
            return Err(format!(
                "unknown model {other}; expected glm-ocr | infinity | qwen36"
            )
            .into());
        }
    };

    let png = std::fs::read(&png_path)?;
    println!(
        "loaded {} ({} bytes); model={}",
        png_path,
        png.len(),
        model.label(),
    );

    let client = AdHocClient::new();
    let started = Instant::now();
    let resp = client.extract(model, &png, &prompt).await?;
    let wall = started.elapsed().as_secs_f64();

    println!("\n--- response ---");
    println!("model:         {}", resp.model.label());
    println!("elapsed_secs:  {:.2}", resp.elapsed_secs);
    println!("wall_secs:     {wall:.2}");
    println!("finish_reason: {:?}", resp.finish_reason);
    if let Some(u) = resp.usage {
        println!("input_tokens:  {}", u.prompt_tokens);
        println!("output_tokens: {}", u.completion_tokens);
    }
    println!("\n{}", resp.content);
    Ok(())
}
