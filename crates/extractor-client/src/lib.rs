//! Async client for the deployed Modal SGLang workers (Pass 1 GLM-OCR,
//! Pass 2 Infinity-Parser2-Pro).
//!
//! The Modal workers are stock SGLang OpenAI-compatible servers — no
//! custom Python. This crate handles the rest of the pipeline: PDF
//! rendering (via pdfium), prompt assembly, concurrent dispatch, and
//! IR construction. Tracking issue: FER-83.
//!
//! # Layout
//!
//! * [`Extractor`] — the trait both implementations satisfy
//! * [`ExtractionResult`] — IR-ready output (one [`ir::Extraction`] +
//!   its [`ir::PageExtraction`]s, sharing one `extraction_uuid`)
//! * [`mock::MockExtractor`] — in-memory impl for tests
//! * [`modal::ModalExtractor`] — HTTP impl talking to the deployed SGLang
//!   workers via OpenAI's `/v1/chat/completions` endpoint
//! * [`render`] — PDF page rendering (pdfium-render wrapper)
//! * [`openai`] — minimal OpenAI request/response types

pub mod ad_hoc;
mod error;
pub mod mock;
pub mod modal;
pub mod openai;
pub mod pass2;
pub mod render;
pub mod toc;

use async_trait::async_trait;
use ir::{Extraction, PageExtraction};

pub use error::Error;

/// IR-ready output from an extractor call. The [`Extraction`] and
/// [`PageExtraction`]s share a common `extraction_uuid`; insert both into
/// the IR `Doc`'s `extractions` and `extracted_pages` lists respectively.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractionResult {
    pub extraction: Extraction,
    pub pages: Vec<PageExtraction>,
}

/// The two-pass extraction interface. `pass1` is cheap full-doc OCR
/// (GLM-OCR); `pass2` is on-demand structured extraction with bboxes
/// (Infinity-Parser2-Pro). Both return IR-ready records so downstream
/// consumers (orchestrator, cache, harness) get uniform
/// [`ExtractionResult`]s regardless of which pass ran.
#[async_trait]
pub trait Extractor: Send + Sync {
    async fn pass1(
        &self,
        pdf: &[u8],
        pages: Option<&[u32]>,
    ) -> Result<ExtractionResult, Error>;

    async fn pass2(
        &self,
        pdf: &[u8],
        pages: &[u32],
    ) -> Result<ExtractionResult, Error>;
}
