//! Minimal OpenAI-compatible chat completions request/response types.
//!
//! SGLang serves the `/v1/chat/completions` endpoint with the standard
//! OpenAI shape; we only need the subset that supports a single
//! image+text user message, so building the types ourselves is cheaper
//! than depending on a full OpenAI SDK.

use serde::{Deserialize, Serialize};

/// Chat completions request body. Only the fields we actually send.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Penalty on repeated tokens (OpenAI-spec range -2.0..=2.0). Higher
    /// values discourage the model from repeating already-emitted
    /// tokens. Used by FER-117 to escape the duplicate-`"bbox"`-key
    /// failure mode in Pass 2 layout extraction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// One-shot penalty on tokens that have appeared at all (OpenAI
    /// spec). Less granular than `frequency_penalty`; included for
    /// completeness when probing repetition issues.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Qwen3-family extra: pass `{"enable_thinking": false}` to suppress
    /// `<think>...</think>` blocks for cleaner / cheaper outputs. Other
    /// SGLang servers ignore this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,
}

/// One message in a chat. We only ever send a single user message with
/// `image_url` + `text` parts.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: Vec<ContentPart>,
}

/// One content part in a message — either text or a base64-encoded image.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ImageUrl,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

/// Chat completions response. Subset of the OpenAI shape we read.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
    /// `"stop"` (EOS reached), `"length"` (hit `max_tokens`),
    /// `"content_filter"`, or `null`. Used to detect truncation.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: Option<u32>,
}

impl ChatRequest {
    /// Convenience: build a single-page request with one image + one text
    /// prompt. Image is base64-encoded PNG bytes (data URI).
    pub fn single_page(
        model: impl Into<String>,
        prompt: impl Into<String>,
        png_base64: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Self {
        Self {
            model: model.into(),
            messages: vec![Message {
                role: "user".into(),
                content: vec![
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: format!("data:image/png;base64,{png_base64}"),
                        },
                    },
                    ContentPart::Text {
                        text: prompt.into(),
                    },
                ],
            }],
            max_tokens,
            temperature,
            frequency_penalty: None,
            presence_penalty: None,
            chat_template_kwargs: None,
        }
    }

    /// Text-only request — for post-processing tasks where the input is
    /// already-extracted text (FER-113 evaluation).
    pub fn text_only(
        model: impl Into<String>,
        prompt: impl Into<String>,
        max_tokens: u32,
        temperature: f32,
    ) -> Self {
        Self {
            model: model.into(),
            messages: vec![Message {
                role: "user".into(),
                content: vec![ContentPart::Text {
                    text: prompt.into(),
                }],
            }],
            max_tokens,
            temperature,
            frequency_penalty: None,
            presence_penalty: None,
            chat_template_kwargs: None,
        }
    }

    /// Builder helper: turn off Qwen3 thinking mode for this request.
    pub fn without_thinking(mut self) -> Self {
        self.chat_template_kwargs =
            Some(serde_json::json!({ "enable_thinking": false }));
        self
    }

    /// Builder helper: set the OpenAI-spec `frequency_penalty`. Range
    /// is -2.0..=2.0 per the spec; in practice 0.3..=1.5 is the useful
    /// band for combating repetition. FER-117 uses this to escape
    /// duplicate-key failures in Pass 2 layout JSON.
    pub fn with_frequency_penalty(mut self, p: f32) -> Self {
        self.frequency_penalty = Some(p);
        self
    }
}
