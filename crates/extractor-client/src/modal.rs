//! Async client for the deployed Modal SGLang workers.
//!
//! Both passes hit a stock SGLang OpenAI-compatible server; the workers
//! do nothing but serve the model. All page rendering, prompt assembly,
//! concurrency control, and IR construction live here.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine;
use futures::future::join_all;
use ir::{
    Content, ErrorContent, Extraction, ExtractionId, MarkdownContent, ModelId, PageExtraction,
    PageId, PageMeta, PageMetrics, PromptText, StructuredPage,
};
use tokio::sync::Semaphore;

use crate::openai::{ChatRequest, ChatResponse};
use crate::pass2::parse_layout_json;
use crate::render::{render_pages, PageImage, DEFAULT_RENDER_DPI};
use crate::{Error, ExtractionResult, Extractor};

/// Which IR `Content` variant to construct from a successful response.
/// Pass 1 → `Markdown` (raw OCR text). Pass 2 → `Structured` after
/// parsing the model's layout JSON via [`parse_layout_json`]; on parse
/// failure we still emit a `Content::Error` so sibling pages survive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassFormat {
    Markdown,
    StructuredJson,
}

/// Default request timeout. Generous enough to cover 35B-MoE Pass 2
/// requests under load.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Pass 1 (GLM-OCR) — model id sent in the OpenAI request.
const PASS1_MODEL_ID: &str = "zai-org/GLM-OCR";
/// Recommended doc-parsing prompt from the GLM-OCR model card.
const PASS1_PROMPT: &str = "Text Recognition:";
/// Token budget per page. ~95% of legitimate pages are <1000 tokens;
/// dense pages should route to Pass 2.
const PASS1_MAX_TOKENS: u32 = 1024;
/// Default in-flight request cap — empirical probe (2026-05-03) showed
/// throughput plateaus near concurrency=16 on L40S.
const PASS1_DEFAULT_CONCURRENCY: usize = 16;

/// Pass 2 (Infinity-Parser2-Pro) — model id sent in the OpenAI request.
const PASS2_MODEL_ID: &str = "infly/Infinity-Parser2-Pro";
/// FER-117 production prompt. Replaces the earlier one-line
/// `"Extract layout with bboxes as JSON"` which produced degenerate
/// output (duplicate `"bbox"` keys, broken JSON) on dense pages — see
/// FER-117 ticket for the failure analysis. Verified 48/48 (100%) clean
/// parse rate on the FER-80 corpus vs the prior prompt's 31/48 (65%).
/// The explicit JSON schema in the prompt body anchors the model's
/// structured-output decoder; the `Skip page furniture` clause cuts
/// noise from headers/footers/page numbers.
const PASS2_PROMPT: &str = "Extract document layout as a JSON array, \
with as few blocks as possible. One table = one block (full HTML in text). \
Group adjacent paragraphs into one text block. Skip page furniture. \
Each element: {\"bbox\":[x1,y1,x2,y2], \"category\":\"...\", \"text\":\"...\"}. \
Return ONLY the JSON array.";
/// Higher token budget — Pass 2 emits structured layout JSON which can
/// run several thousand tokens for dense pages.
const PASS2_MAX_TOKENS: u32 = 4096;
/// Default in-flight cap — Pass 2 saturates the H100 at concurrency=4.
const PASS2_DEFAULT_CONCURRENCY: usize = 4;

const TEMPERATURE: f32 = 0.0;

pub struct ModalExtractor {
    pub pass1_url: String,
    pub pass2_url: String,
    pub pass1_concurrency: usize,
    pub pass2_concurrency: usize,
    client: reqwest::Client,
}

impl ModalExtractor {
    pub fn new(pass1_url: impl Into<String>, pass2_url: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("reqwest client builder failed with default config");
        Self {
            pass1_url: pass1_url.into(),
            pass2_url: pass2_url.into(),
            pass1_concurrency: PASS1_DEFAULT_CONCURRENCY,
            pass2_concurrency: PASS2_DEFAULT_CONCURRENCY,
            client,
        }
    }

    /// Override concurrency caps post-construction (builder-style).
    pub fn with_concurrency(mut self, pass1: usize, pass2: usize) -> Self {
        self.pass1_concurrency = pass1.max(1);
        self.pass2_concurrency = pass2.max(1);
        self
    }

    async fn run_pass(
        &self,
        url: &str,
        model: &'static str,
        prompt: &'static str,
        max_tokens: u32,
        concurrency: usize,
        format: PassFormat,
        pdf: &[u8],
        pages: Vec<u32>,
    ) -> Result<ExtractionResult, Error> {
        // Render pages off the async runtime.
        let pdf_owned = pdf.to_vec();
        let page_images = render_pages(pdf_owned, pages, DEFAULT_RENDER_DPI).await?;

        let extraction = Extraction {
            uuid: ExtractionId::new(),
            model: ModelId(model.to_string()),
            prompt: PromptText(prompt.to_string()),
            params: BTreeMap::from([
                ("dpi".into(), serde_json::json!(DEFAULT_RENDER_DPI)),
                ("max_tokens".into(), serde_json::json!(max_tokens)),
                ("temperature".into(), serde_json::json!(TEMPERATURE)),
            ]),
            created_at: chrono::Utc::now(),
        };
        let extraction_uuid = extraction.uuid;

        let semaphore = Arc::new(Semaphore::new(concurrency));
        let url = url.to_string();
        let model_str = model.to_string();
        let prompt_str = prompt.to_string();

        let futures = page_images.into_iter().map(|img| {
            let sem = semaphore.clone();
            let client = self.client.clone();
            let url = url.clone();
            let model = model_str.clone();
            let prompt = prompt_str.clone();
            async move {
                let _permit = sem.acquire().await.expect("semaphore closed");
                send_one_request(
                    &client, &url, &model, &prompt, max_tokens, &img, extraction_uuid, format,
                )
                .await
            }
        });
        let pages: Vec<PageExtraction> = join_all(futures).await;
        Ok(ExtractionResult { extraction, pages })
    }
}

#[async_trait]
impl Extractor for ModalExtractor {
    async fn pass1(
        &self,
        pdf: &[u8],
        pages: Option<&[u32]>,
    ) -> Result<ExtractionResult, Error> {
        let pages_owned: Vec<u32> = match pages {
            Some(p) => p.to_vec(),
            None => {
                // None = all pages; need a count without async.
                let bytes = pdf.to_vec();
                let count =
                    tokio::task::spawn_blocking(move || crate::render::page_count_blocking(&bytes))
                        .await
                        .map_err(|e| Error::Render(format!("page-count task panicked: {e}")))??;
                (0..count).collect()
            }
        };
        self.run_pass(
            &self.pass1_url,
            PASS1_MODEL_ID,
            PASS1_PROMPT,
            PASS1_MAX_TOKENS,
            self.pass1_concurrency,
            PassFormat::Markdown,
            pdf,
            pages_owned,
        )
        .await
    }

    async fn pass2(
        &self,
        pdf: &[u8],
        pages: &[u32],
    ) -> Result<ExtractionResult, Error> {
        if pages.is_empty() {
            return Err(Error::EmptyPagesForPass2);
        }
        self.run_pass(
            &self.pass2_url,
            PASS2_MODEL_ID,
            PASS2_PROMPT,
            PASS2_MAX_TOKENS,
            self.pass2_concurrency,
            PassFormat::StructuredJson,
            pdf,
            pages.to_vec(),
        )
        .await
    }
}

/// One pre-rendered page in a bulk job. `key` is an arbitrary client-side
/// identifier (e.g. `"part_id/page_idx"`) that the caller uses to
/// demultiplex results back to source documents. The `bulk_extract`
/// method below pairs with this and is meant for cross-document workflows
/// where amortizing per-PDF render gaps + tail-latency-per-batch matters
/// (FER-80 corpus run, future FER-105 orchestrator).
#[derive(Debug, Clone)]
pub struct ExtractTask {
    pub key: String,
    pub page_image: PageImage,
}

/// Outcome of one bulk task. `content` is `Ok(markdown)` on success or
/// `Err(message)` on transport / decode failure. `key` is echoed from
/// the input task; `page` is from the underlying `PageImage`.
#[derive(Debug, Clone)]
pub struct ExtractTaskResult {
    pub key: String,
    pub page: u32,
    pub content: Result<String, String>,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub elapsed_secs: f32,
}

impl ModalExtractor {
    /// Send pre-rendered pages concurrently against `url`, returning
    /// results in arbitrary completion order. Caller is responsible for
    /// rendering, batching by `(model, prompt, max_tokens)`, and
    /// demultiplexing results by `key`.
    ///
    /// Compared to `pass1` / `pass2` (which assume one-PDF-per-call),
    /// this lets a corpus-runner pre-render every page upfront and feed
    /// SGLang a sustained stream of requests — eliminating the
    /// per-document render gaps and tail-latency-amplifies-per-PDF
    /// effects that the simple per-PDF dispatch suffers from.
    pub async fn bulk_extract(
        &self,
        url: &str,
        model: &str,
        prompt: &str,
        max_tokens: u32,
        concurrency: usize,
        tasks: Vec<ExtractTask>,
    ) -> Vec<ExtractTaskResult> {
        let semaphore = Arc::new(Semaphore::new(concurrency.max(1)));
        let url = url.to_string();
        let model = model.to_string();
        let prompt = prompt.to_string();
        let futures = tasks.into_iter().map(|task| {
            let sem = semaphore.clone();
            let client = self.client.clone();
            let url = url.clone();
            let model = model.clone();
            let prompt = prompt.clone();
            async move {
                let _permit = sem.acquire().await.expect("semaphore closed");
                send_one_bulk_request(&client, &url, &model, &prompt, max_tokens, task).await
            }
        });
        join_all(futures).await
    }
}

async fn send_one_bulk_request(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    task: ExtractTask,
) -> ExtractTaskResult {
    let started = Instant::now();
    let png_b64 =
        base64::engine::general_purpose::STANDARD.encode(&task.page_image.png_bytes);
    let request = ChatRequest::single_page(model, prompt, &png_b64, max_tokens, TEMPERATURE);

    let result: Result<(String, Option<u32>, Option<u32>), String> = async {
        let resp = client
            .post(format!("{}/v1/chat/completions", url.trim_end_matches('/')))
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("transport: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("http {}: {}", status.as_u16(), body));
        }
        let body: ChatResponse = resp
            .json()
            .await
            .map_err(|e| format!("decode response: {e}"))?;
        let content = body
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| "response had no message content".to_string())?;
        let usage = body.usage;
        Ok((
            content,
            usage.map(|u| u.prompt_tokens),
            usage.map(|u| u.completion_tokens),
        ))
    }
    .await;

    let elapsed = started.elapsed().as_secs_f32();
    let (content_result, input_tokens, output_tokens) = match result {
        Ok((content, in_tok, out_tok)) => (Ok(content), in_tok, out_tok),
        Err(message) => (Err(message), None, None),
    };
    ExtractTaskResult {
        key: task.key,
        page: task.page_image.page,
        content: content_result,
        input_tokens,
        output_tokens,
        elapsed_secs: elapsed,
    }
}

/// Send one chat-completion request for one rendered page. Always returns
/// a `PageExtraction` — failures become `Content::Error` records so the
/// caller's `Vec<PageExtraction>` is dense (one entry per requested page,
/// in the same order as the input).
async fn send_one_request(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    img: &PageImage,
    extraction_uuid: ExtractionId,
    format: PassFormat,
) -> PageExtraction {
    let started = Instant::now();
    let png_b64 = base64::engine::general_purpose::STANDARD.encode(&img.png_bytes);
    let request = ChatRequest::single_page(model, prompt, &png_b64, max_tokens, TEMPERATURE);

    let result: Result<(String, Option<u32>, Option<u32>), String> = async {
        let resp = client
            .post(format!("{}/v1/chat/completions", url.trim_end_matches('/')))
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("transport: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("http {}: {}", status.as_u16(), body));
        }
        let body: ChatResponse = resp
            .json()
            .await
            .map_err(|e| format!("decode response: {e}"))?;
        let content = body
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| "response had no message content".to_string())?;
        let usage = body.usage;
        Ok((
            content,
            usage.map(|u| u.prompt_tokens),
            usage.map(|u| u.completion_tokens),
        ))
    }
    .await;

    let elapsed = started.elapsed().as_secs_f32();
    match result {
        Ok((raw, input_tokens, output_tokens)) => {
            let content = build_content(format, raw, img);
            PageExtraction {
                page_uuid: PageId::new(),
                page: img.page,
                extraction_uuid,
                content,
                metrics: Some(PageMetrics {
                    elapsed_secs: elapsed,
                    input_tokens,
                    output_tokens,
                }),
            }
        }
        Err(message) => PageExtraction {
            page_uuid: PageId::new(),
            page: img.page,
            extraction_uuid,
            content: Content::Error(ErrorContent {
                kind: "request_error".into(),
                message,
            }),
            metrics: Some(PageMetrics {
                elapsed_secs: elapsed,
                input_tokens: None,
                output_tokens: None,
            }),
        },
    }
}

/// Convert a raw model response into the appropriate `Content` variant.
///
/// Pass 1 stores the response verbatim as markdown. Pass 2 routes
/// through [`parse_layout_json`]; parse failures degrade to
/// `Content::Error { kind = "parse_error" }` per the FER-82 per-page
/// error pattern, so a single bad page doesn't fail the extraction.
fn build_content(format: PassFormat, raw: String, img: &PageImage) -> Content {
    match format {
        PassFormat::Markdown => Content::Markdown(MarkdownContent { markdown: raw }),
        PassFormat::StructuredJson => {
            match parse_layout_json(&raw, (img.width_pts, img.height_pts)) {
                Ok(blocks) => Content::Structured(StructuredPage {
                    blocks,
                    page_meta: Some(PageMeta {
                        width_pts: img.width_pts,
                        height_pts: img.height_pts,
                        rotation_deg: img.rotation_deg,
                        dpi: Some(img.dpi as u16),
                    }),
                }),
                Err(err) => Content::Error(ErrorContent {
                    kind: "parse_error".into(),
                    message: err.to_string(),
                }),
            }
        }
    }
}
