//! Ad-hoc image+prompt requests to the deployed VLM workers (FER-120).
//!
//! The ToC builder (FER-103) and Pass 1 / Pass 2 (FER-83) ship fixed
//! prompts against fixed workers — that's the production path. This
//! module adds a thin user-driven wrapper: send an arbitrary image +
//! arbitrary prompt to one of the three deployed VLMs and get the
//! response back as a string. No IR materialisation — the desktop app
//! (FER-121) holds session-only state for the result.

use std::time::{Duration, Instant};

use base64::Engine as _;

use crate::openai::{ChatRequest, ChatResponse};
use crate::Error;

/// Catalog of the deployed VLMs available to ad-hoc prompts. URLs and
/// model IDs mirror the production constants in `modal::*` and
/// `toc::QWEN_35B_*` — keep them in sync if those move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdHocModel {
    GlmOcr,
    InfinityParser2Flash,
    Qwen36MoE,
}

impl AdHocModel {
    /// Stable iteration order for UI pickers.
    pub const ALL: &'static [Self] =
        &[Self::GlmOcr, Self::InfinityParser2Flash, Self::Qwen36MoE];

    pub fn label(self) -> &'static str {
        match self {
            Self::GlmOcr => "GLM-OCR",
            Self::InfinityParser2Flash => "Infinity-Parser2-Flash",
            Self::Qwen36MoE => "Qwen3.6 35B-A3B",
        }
    }

    pub fn url(self) -> &'static str {
        match self {
            Self::GlmOcr => "https://ferrite-systems--parselab-glm-ocr-serve.modal.run",
            Self::InfinityParser2Flash => {
                "https://ferrite-systems--parselab-inf2-flash-serve.modal.run"
            }
            Self::Qwen36MoE => {
                "https://ferrite-systems--parselab-qwen36-35b-a3b-serve.modal.run"
            }
        }
    }

    /// Modal app name that hosts this model. Matches the deployed
    /// `parselab-*` app names; used by `discovery::discover_deployed_workers`
    /// to filter the model picker to live endpoints.
    pub fn modal_app_name(self) -> &'static str {
        match self {
            Self::GlmOcr => "parselab-glm-ocr",
            Self::InfinityParser2Flash => "parselab-inf2-flash",
            Self::Qwen36MoE => "parselab-qwen36-35b-a3b",
        }
    }

    pub fn model_id(self) -> &'static str {
        match self {
            Self::GlmOcr => "zai-org/GLM-OCR",
            Self::InfinityParser2Flash => "infly/Infinity-Parser2-Flash",
            Self::Qwen36MoE => "Qwen/Qwen3.6-35B-A3B",
        }
    }

    /// Qwen3.6 emits `<think>...</think>` blocks by default. For ad-hoc
    /// UI use the user wants the answer, not the reasoning trace —
    /// disable thinking. Other workers ignore the chat-template kwarg.
    fn disable_thinking(self) -> bool {
        matches!(self, Self::Qwen36MoE)
    }
}

/// Generous request timeout — covers Qwen3.6 cold-start (~3 min when
/// `min_containers=0`) and long responses on saturated GPUs. The UI
/// surfaces elapsed time so a hung request is visible to the user.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
const TEMPERATURE: f32 = 0.0;
/// Output budget — large enough for a few thousand words of free-form
/// response, well beyond what most ad-hoc prompts need.
const MAX_TOKENS: u32 = 4096;
/// Output budget for text-in / structured-JSON-out post-processing
/// (FER-124). The component summary tends to run a few thousand tokens
/// when the source markdown spans 10+ pages.
const TEXT_MAX_TOKENS: u32 = 8192;

#[derive(Debug, Clone)]
pub struct AdHocResponse {
    pub model: AdHocModel,
    pub content: String,
    pub elapsed_secs: f32,
    pub usage: Option<AdHocUsage>,
    /// `"stop"` (EOS), `"length"` (hit `MAX_TOKENS`), `"content_filter"`,
    /// or `None`. UI uses this to flag truncated responses.
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct AdHocUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

pub struct AdHocClient {
    client: reqwest::Client,
}

impl AdHocClient {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("reqwest client builder failed with default config");
        Self { client }
    }

    pub async fn extract(
        &self,
        model: AdHocModel,
        png_bytes: &[u8],
        prompt: &str,
    ) -> Result<AdHocResponse, Error> {
        let png_b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
        let mut request = ChatRequest::single_page(
            model.model_id(),
            prompt,
            &png_b64,
            MAX_TOKENS,
            TEMPERATURE,
        );
        if model.disable_thinking() {
            request = request.without_thinking();
        }
        self.send_request(model, request).await
    }

    /// Text-only ad-hoc dispatch (FER-124). For tasks where the input
    /// is already text — typically Pass 1 markdown the caller wants
    /// post-processed via Qwen3.6 — skip the image and send the prompt
    /// alone. The output budget is sized for structured-JSON responses
    /// over a multi-page input; bump if you need more.
    pub async fn extract_text(
        &self,
        model: AdHocModel,
        prompt: &str,
    ) -> Result<AdHocResponse, Error> {
        let mut request =
            ChatRequest::text_only(model.model_id(), prompt, TEXT_MAX_TOKENS, TEMPERATURE);
        if model.disable_thinking() {
            request = request.without_thinking();
        }
        self.send_request(model, request).await
    }

    async fn send_request(
        &self,
        model: AdHocModel,
        request: ChatRequest,
    ) -> Result<AdHocResponse, Error> {
        let started = Instant::now();
        let endpoint =
            format!("{}/v1/chat/completions", model.url().trim_end_matches('/'));
        let resp = self.client.post(&endpoint).json(&request).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::HttpStatus {
                status: status.as_u16(),
                body,
            });
        }
        let body: ChatResponse = resp.json().await?;
        let choice = body
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::WorkerResponse("response had no choices".into()))?;
        let content = choice
            .message
            .body()
            .map(str::to_string)
            .ok_or_else(|| Error::WorkerResponse("response had no message content".into()))?;
        let usage = body.usage.map(|u| AdHocUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
        });
        Ok(AdHocResponse {
            model,
            content,
            elapsed_secs: started.elapsed().as_secs_f32(),
            usage,
            finish_reason: choice.finish_reason,
        })
    }
}

impl Default for AdHocClient {
    fn default() -> Self {
        Self::new()
    }
}
