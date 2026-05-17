//! Document IR — per-page extraction store + extraction-anchored ToC.
//!
//! Tracking issue: FER-101. This is the contract between the two-pass
//! extraction pipeline (cheap GLM-OCR / Pass 1, on-demand Infinity-Parser2-Pro
//! / Pass 2) and downstream interpreters (FER-89 / FER-90 / FER-91). IR is
//! **data**, not behavior — interpreters live in the `component-model` crate.
//!
//! # Shape
//!
//! A [`Doc`] owns three lists that accumulate over the document's lifetime:
//!
//! * [`Extraction`] records — one per LLM run (Pass 1, Pass 2, or ToC build).
//!   Carry the model id, full prompt, and a timestamp. Re-running with a new
//!   prompt appends a new record; old ones are never mutated.
//! * [`PageExtraction`] records — one per `(page, extraction)` pair. Holds
//!   either Markdown (Pass 1) or Structured content (Pass 2). Multiple
//!   extractions per page are normal; consumers query [`Doc::page_extractions`]
//!   and pick the format they need.
//! * [`TocEntry`] records — hierarchical, each entry carries page refs back
//!   to the [`PageExtraction`] records it was derived from, plus the
//!   [`ExtractionId`] of the run that built the ToC entry itself.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use uuid::Uuid;

mod kdl_serde;
pub use kdl_serde::IrKdlError;

/// Palette for the `kind` strings produced by Pass 2 (Infinity-Parser2-Pro).
/// Returned as 24-bit `0xRRGGBB` so callers can wrap in their UI toolkit's
/// color type without the IR crate depending on gpui (or anything else).
///
/// Vocabulary observed in the FER-80 corpus to date: `title`, `text`,
/// `table`, `table_footnote`, `figure`, `figure_caption`, `list`,
/// `list_item`, `header`, `footer`, `logo`. Unknown kinds fall back to a
/// magenta-ish color so they're visible at a glance.
pub fn block_kind_color_rgb(kind: &str) -> u32 {
    match kind.to_ascii_lowercase().as_str() {
        "table" => 0xdc1e1e,
        "table_footnote" => 0xc85050,
        "text" => 0x1e78dc,
        "title" => 0x1eb43c,
        "figure" => 0xf08c1e,
        "figure_caption" => 0xdcaa3c,
        "footer" => 0x828282,
        "header" => 0x1ec8c8,
        "logo" => 0xffd700,
        "list" | "list_item" => 0xc83cc8,
        _ => 0xb43cc8,
    }
}

/// On-disk IR schema version. Independent of the wire envelope's
/// `schema_version` (FER-82) — they evolve separately and both currently
/// sit at 1.
pub const SCHEMA_VERSION: u32 = 1;

pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// SHA-256 of the source PDF bytes, lowercase hex. The cache key for the
/// extraction store.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(pub String);

/// Stable model identifier, conventionally `"name@version"` (e.g.
/// `"glm-ocr@v1"`, `"infinity-parser2-pro@v1"`). Prompt revisions usually
/// bump the version suffix so cache keys diverge.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

/// Full prompt text used for an [`Extraction`] run. Stored verbatim so the
/// IR captures enough provenance to reproduce the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PromptText(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExtractionId(pub Uuid);

impl ExtractionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ExtractionId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PageId(pub Uuid);

impl PageId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for PageId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TocEntryId(pub Uuid);

impl TocEntryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TocEntryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Format discriminator for [`Content`]. Variants line up 1:1 so a mismatched
/// `(format_type, content)` payload is impossible to construct or deserialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatType {
    Markdown,
    StructuredJson,
    Error,
}

/// Per-page extraction payload. Internally tagged via `format_type` so the
/// wire format (FER-82) matches the in-memory shape and so that constructing
/// a mismatched `(format_type, content)` pair is impossible at the type
/// level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "format_type", content = "content", rename_all = "snake_case")]
pub enum Content {
    Markdown(MarkdownContent),
    #[serde(rename = "structured_json")]
    Structured(StructuredPage),
    Error(ErrorContent),
}

impl Content {
    pub fn format_type(&self) -> FormatType {
        match self {
            Content::Markdown(_) => FormatType::Markdown,
            Content::Structured(_) => FormatType::StructuredJson,
            Content::Error(_) => FormatType::Error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkdownContent {
    pub markdown: String,
}

/// Per-page failure payload. `kind` is a short stable token (e.g. `"oom"`,
/// `"render_failed"`, `"model_error"`); `message` is free-form for humans.
/// Sibling pages in the same extraction can succeed alongside an error page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorContent {
    pub kind: String,
    pub message: String,
}

/// Per-page operational metrics — modeled on OpenAI's `usage` shape but
/// emitted per page since each page is its own GPU call. Optional: workers
/// fill what they can measure, consumers tolerate missing fields.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PageMetrics {
    pub elapsed_secs: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
}

/// Pass 2 (Infinity-Parser2-Pro) output for one page. The exact block
/// vocabulary is seeded from the FER-80 spike — this stub captures the
/// minimum the IR needs to round-trip the wire format. Refinements land
/// alongside the spike findings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StructuredPage {
    pub blocks: Vec<StructuredBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_meta: Option<PageMeta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredBlock {
    /// Block kind reported by the model — `"table"`, `"figure"`, `"heading"`,
    /// `"text"`, etc. Vocabulary is empirical until FER-80 lands.
    pub kind: String,
    pub bbox: BBox,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Bounding box in PDF points, page-relative (origin at the page's
/// upper-left corner). The page is implicit — it's whichever
/// [`PageExtraction`] this BBox lives in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PageMeta {
    pub width_pts: f32,
    pub height_pts: f32,
    /// 0, 90, 180, 270.
    pub rotation_deg: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpi: Option<u16>,
}

/// One LLM run. Whether it's Pass 1 (GLM-OCR), Pass 2 (Infinity-Parser2-
/// Pro), or a ToC build, the shape is the same: model id, full prompt,
/// tuning params, and timestamp. Associated outputs (Markdown / Structured
/// pages, ToC entries) are linked back via `extraction_uuid` / `built_by`
/// fields.
///
/// `params` captures worker-specific tuning knobs that affect output (e.g.
/// `dpi`, `max_new_tokens`). A re-run with different params is captured as a
/// distinct `Extraction` rather than overwriting the prior one — `BTreeMap`
/// gives stable serialization order so identical params hash identically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Extraction {
    pub uuid: ExtractionId,
    pub model: ModelId,
    pub prompt: PromptText,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, serde_json::Value>,
    pub created_at: Timestamp,
}

/// One page's worth of output from a single [`Extraction`]. Multiple
/// `PageExtraction`s for the same `page` are normal — they're produced by
/// different `Extraction` runs (different models, different prompt versions,
/// or re-runs).
///
/// `page_uuid` is minted client-side when materializing the wire envelope
/// into the IR — workers never generate or echo it. Keeps page identity a
/// local concern so the wire format stays stateless.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageExtraction {
    pub page_uuid: PageId,
    pub page: u32,
    pub extraction_uuid: ExtractionId,
    #[serde(flatten)]
    pub content: Content,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<PageMetrics>,
}

impl PageExtraction {
    pub fn format_type(&self) -> FormatType {
        self.content.format_type()
    }
}

/// Reference from a [`TocEntry`] back to a specific [`PageExtraction`].
/// `page` and `model` are denormalized for fast filtering without a join
/// through `extractions` / `extracted_pages`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageRef {
    pub page: u32,
    pub model: ModelId,
    pub page_uuid: PageId,
    pub extraction_uuid: ExtractionId,
}

/// One node in the document's hierarchical Table of Contents. Multi-page
/// sections carry multiple `page_refs` (one per source page); we don't model
/// page spans because the underlying extractions are per-page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TocEntry {
    pub id: TocEntryId,
    pub title: String,
    pub level: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<TocEntryId>,
    pub page_refs: Vec<PageRef>,
    /// The [`Extraction`] run that produced this ToC entry — i.e. the LLM
    /// (or heuristic builder) that read the page markdown and emitted this
    /// node.
    pub built_by: ExtractionId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Doc {
    pub schema_version: u32,
    pub content_hash: ContentHash,
    pub extractions: Vec<Extraction>,
    pub extracted_pages: Vec<PageExtraction>,
    pub toc: Vec<TocEntry>,
}

impl Doc {
    /// All [`PageExtraction`] records on the given page, in declaration order.
    pub fn page_extractions(&self, page: u32) -> impl Iterator<Item = &PageExtraction> {
        self.extracted_pages.iter().filter(move |p| p.page == page)
    }

    /// Look up a [`PageExtraction`] by its `page_uuid`.
    pub fn page_extraction(&self, page_uuid: PageId) -> Option<&PageExtraction> {
        self.extracted_pages.iter().find(|p| p.page_uuid == page_uuid)
    }

    /// Look up an [`Extraction`] by its uuid.
    pub fn extraction(&self, uuid: ExtractionId) -> Option<&Extraction> {
        self.extractions.iter().find(|e| e.uuid == uuid)
    }

    /// Look up a [`TocEntry`] by its id.
    pub fn toc_entry(&self, id: TocEntryId) -> Option<&TocEntry> {
        self.toc.iter().find(|t| t.id == id)
    }

    /// Validate the FER-101 invariants. Returns every violation found; an
    /// empty result means the document is internally consistent.
    pub fn validate(&self) -> Result<(), Vec<ValidationError>> {
        let mut errs = Vec::new();
        let extraction_ids: HashSet<_> = self.extractions.iter().map(|e| e.uuid).collect();
        let page_uuids: HashSet<_> = self.extracted_pages.iter().map(|p| p.page_uuid).collect();
        let toc_ids: HashSet<_> = self.toc.iter().map(|t| t.id).collect();

        for p in &self.extracted_pages {
            if !extraction_ids.contains(&p.extraction_uuid) {
                errs.push(ValidationError::DanglingExtractionRef {
                    via: format!("PageExtraction(page_uuid={})", p.page_uuid.0),
                    extraction_uuid: p.extraction_uuid,
                });
            }
        }

        for t in &self.toc {
            if !extraction_ids.contains(&t.built_by) {
                errs.push(ValidationError::DanglingExtractionRef {
                    via: format!("TocEntry(id={}).built_by", t.id.0),
                    extraction_uuid: t.built_by,
                });
            }
            if let Some(parent) = t.parent {
                if !toc_ids.contains(&parent) {
                    errs.push(ValidationError::DanglingTocParent { entry: t.id, parent });
                }
            }
            for r in &t.page_refs {
                if !extraction_ids.contains(&r.extraction_uuid) {
                    errs.push(ValidationError::DanglingExtractionRef {
                        via: format!("TocEntry(id={}).page_refs.extraction_uuid", t.id.0),
                        extraction_uuid: r.extraction_uuid,
                    });
                }
                if !page_uuids.contains(&r.page_uuid) {
                    errs.push(ValidationError::DanglingPageRef {
                        entry: t.id,
                        page_uuid: r.page_uuid,
                    });
                } else if let Some(pe) = self.page_extraction(r.page_uuid) {
                    if pe.page != r.page {
                        errs.push(ValidationError::PageRefPageMismatch {
                            entry: t.id,
                            page_uuid: r.page_uuid,
                            ref_page: r.page,
                            actual_page: pe.page,
                        });
                    }
                }
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    DanglingExtractionRef {
        via: String,
        extraction_uuid: ExtractionId,
    },
    DanglingPageRef {
        entry: TocEntryId,
        page_uuid: PageId,
    },
    DanglingTocParent {
        entry: TocEntryId,
        parent: TocEntryId,
    },
    PageRefPageMismatch {
        entry: TocEntryId,
        page_uuid: PageId,
        ref_page: u32,
        actual_page: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_ts() -> Timestamp {
        chrono::Utc.with_ymd_and_hms(2026, 4, 29, 12, 0, 0).unwrap()
    }

    fn build_doc() -> Doc {
        let extr_pass1 = Extraction {
            uuid: ExtractionId::new(),
            model: ModelId("glm-ocr@v1".into()),
            prompt: PromptText("extract markdown from this page".into()),
            params: BTreeMap::from([
                ("dpi".into(), serde_json::json!(200)),
                ("max_new_tokens".into(), serde_json::json!(8192)),
            ]),
            created_at: fixed_ts(),
        };
        let extr_pass2 = Extraction {
            uuid: ExtractionId::new(),
            model: ModelId("infinity-parser2-pro@v1".into()),
            prompt: PromptText("extract structured blocks with bboxes".into()),
            params: BTreeMap::new(),
            created_at: fixed_ts(),
        };
        let extr_toc = Extraction {
            uuid: ExtractionId::new(),
            model: ModelId("toc-builder@v1".into()),
            prompt: PromptText("build a hierarchical ToC from per-page markdown".into()),
            params: BTreeMap::new(),
            created_at: fixed_ts(),
        };

        let pe1 = PageExtraction {
            page_uuid: PageId::new(),
            page: 0,
            extraction_uuid: extr_pass1.uuid,
            content: Content::Markdown(MarkdownContent {
                markdown: "# Datasheet Title".into(),
            }),
            metrics: Some(PageMetrics {
                elapsed_secs: 1.4,
                input_tokens: Some(1024),
                output_tokens: Some(217),
            }),
        };
        let pe2 = PageExtraction {
            page_uuid: PageId::new(),
            page: 1,
            extraction_uuid: extr_pass1.uuid,
            content: Content::Markdown(MarkdownContent {
                markdown: "## Electrical Characteristics\n\n| Param | Min | Typ | Max |".into(),
            }),
            metrics: None,
        };
        let pe3 = PageExtraction {
            page_uuid: PageId::new(),
            page: 1,
            extraction_uuid: extr_pass2.uuid,
            content: Content::Structured(StructuredPage {
                blocks: vec![StructuredBlock {
                    kind: "table".into(),
                    bbox: BBox {
                        x: 72.0,
                        y: 120.0,
                        w: 450.0,
                        h: 200.0,
                    },
                    text: Some("Param | Min | Typ | Max | …".into()),
                    confidence: Some(0.92),
                }],
                page_meta: Some(PageMeta {
                    width_pts: 612.0,
                    height_pts: 792.0,
                    rotation_deg: 0,
                    dpi: Some(300),
                }),
            }),
            metrics: None,
        };

        let toc_root = TocEntry {
            id: TocEntryId::new(),
            title: "Electrical Characteristics".into(),
            level: 1,
            parent: None,
            page_refs: vec![PageRef {
                page: 1,
                model: ModelId("glm-ocr@v1".into()),
                page_uuid: pe2.page_uuid,
                extraction_uuid: pe2.extraction_uuid,
            }],
            built_by: extr_toc.uuid,
        };

        Doc {
            schema_version: SCHEMA_VERSION,
            content_hash: ContentHash("0123456789abcdef".repeat(4)),
            extractions: vec![extr_pass1, extr_pass2, extr_toc],
            extracted_pages: vec![pe1, pe2, pe3],
            toc: vec![toc_root],
        }
    }

    #[test]
    fn round_trip_doc() {
        let doc = build_doc();
        let json = serde_json::to_string(&doc).expect("serialize");
        let parsed: Doc = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(doc, parsed);
    }

    #[test]
    fn round_trip_markdown_page_extraction() {
        let pe = PageExtraction {
            page_uuid: PageId::new(),
            page: 3,
            extraction_uuid: ExtractionId::new(),
            content: Content::Markdown(MarkdownContent {
                markdown: "hello".into(),
            }),
            metrics: None,
        };
        let json = serde_json::to_value(&pe).unwrap();
        assert_eq!(json["format_type"], "markdown");
        assert_eq!(json["content"]["markdown"], "hello");
        assert!(json.get("metrics").is_none(), "absent metrics shouldn't serialize");
        let parsed: PageExtraction = serde_json::from_value(json).unwrap();
        assert_eq!(pe, parsed);
    }

    #[test]
    fn round_trip_structured_page_extraction() {
        let pe = PageExtraction {
            page_uuid: PageId::new(),
            page: 7,
            extraction_uuid: ExtractionId::new(),
            content: Content::Structured(StructuredPage::default()),
            metrics: None,
        };
        let json = serde_json::to_value(&pe).unwrap();
        assert_eq!(json["format_type"], "structured_json");
        let parsed: PageExtraction = serde_json::from_value(json).unwrap();
        assert_eq!(pe, parsed);
    }

    #[test]
    fn round_trip_error_page_extraction() {
        let pe = PageExtraction {
            page_uuid: PageId::new(),
            page: 4,
            extraction_uuid: ExtractionId::new(),
            content: Content::Error(ErrorContent {
                kind: "render_failed".into(),
                message: "pymupdf raised on encrypted page".into(),
            }),
            metrics: Some(PageMetrics {
                elapsed_secs: 0.25,
                input_tokens: None,
                output_tokens: None,
            }),
        };
        let json = serde_json::to_value(&pe).unwrap();
        assert_eq!(json["format_type"], "error");
        assert_eq!(json["content"]["kind"], "render_failed");
        assert_eq!(json["metrics"]["elapsed_secs"], 0.25);
        let parsed: PageExtraction = serde_json::from_value(json).unwrap();
        assert_eq!(pe, parsed);
    }

    #[test]
    fn round_trip_extraction_params() {
        let extr = Extraction {
            uuid: ExtractionId::new(),
            model: ModelId("glm-ocr@v1".into()),
            prompt: PromptText("Text Recognition:".into()),
            params: BTreeMap::from([
                ("dpi".into(), serde_json::json!(200)),
                ("max_new_tokens".into(), serde_json::json!(8192)),
            ]),
            created_at: fixed_ts(),
        };
        let json = serde_json::to_value(&extr).unwrap();
        assert_eq!(json["params"]["dpi"], 200);
        assert_eq!(json["params"]["max_new_tokens"], 8192);
        let parsed: Extraction = serde_json::from_value(json).unwrap();
        assert_eq!(extr, parsed);
    }

    #[test]
    fn empty_params_omitted_from_serialization() {
        let extr = Extraction {
            uuid: ExtractionId::new(),
            model: ModelId("toc-builder@v1".into()),
            prompt: PromptText("...".into()),
            params: BTreeMap::new(),
            created_at: fixed_ts(),
        };
        let json = serde_json::to_value(&extr).unwrap();
        assert!(json.get("params").is_none(), "empty params shouldn't serialize");
        let parsed: Extraction = serde_json::from_value(json).unwrap();
        assert_eq!(extr, parsed);
    }

    #[test]
    fn validate_passes_on_well_formed_doc() {
        let doc = build_doc();
        if let Err(errs) = doc.validate() {
            panic!("validation errors: {errs:?}");
        }
    }

    #[test]
    fn page_extractions_returns_all_for_page() {
        let doc = build_doc();
        let pages: Vec<_> = doc.page_extractions(1).collect();
        assert_eq!(pages.len(), 2);
        let formats: Vec<_> = pages.iter().map(|p| p.format_type()).collect();
        assert!(formats.contains(&FormatType::Markdown));
        assert!(formats.contains(&FormatType::StructuredJson));
    }

    #[test]
    fn extraction_lookup_by_uuid() {
        let doc = build_doc();
        let id = doc.extractions[0].uuid;
        assert_eq!(
            doc.extraction(id).map(|e| &e.model),
            Some(&doc.extractions[0].model)
        );
        assert!(doc.extraction(ExtractionId::new()).is_none());
    }

    #[test]
    fn deserialize_rejects_format_type_content_mismatch() {
        // markdown format with structured-shaped content — must fail.
        let json = r#"{
            "page_uuid": "00000000-0000-0000-0000-000000000000",
            "page": 0,
            "extraction_uuid": "00000000-0000-0000-0000-000000000000",
            "format_type": "markdown",
            "content": { "blocks": [] }
        }"#;
        let res: Result<PageExtraction, _> = serde_json::from_str(json);
        assert!(res.is_err(), "expected serde error, got {res:?}");
    }

    #[test]
    fn validate_catches_dangling_extraction_ref() {
        let mut doc = build_doc();
        doc.extracted_pages[0].extraction_uuid = ExtractionId::new();
        let errs = doc.validate().expect_err("validation should fail");
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::DanglingExtractionRef { .. })));
    }

    #[test]
    fn validate_catches_page_mismatch() {
        let mut doc = build_doc();
        doc.toc[0].page_refs[0].page = 99;
        let errs = doc.validate().expect_err("validation should fail");
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::PageRefPageMismatch { .. })));
    }

    #[test]
    fn validate_catches_dangling_toc_parent() {
        let mut doc = build_doc();
        doc.toc[0].parent = Some(TocEntryId::new());
        let errs = doc.validate().expect_err("validation should fail");
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::DanglingTocParent { .. })));
    }
}
