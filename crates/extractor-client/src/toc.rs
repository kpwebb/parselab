//! FER-103 ToC builder. Consumes Pass 1 markdown `PageExtraction`s,
//! calls Qwen3.6-35B-A3B (or a heuristic fallback) over the document's
//! ToC page range, and emits a hierarchical `Vec<TocEntry>` rooted at a
//! fresh `Extraction` record.
//!
//! Production prompt landed via the FER-113 spike; the dual-source
//! variant from `extractor-eval/src/combined.rs` was authoritative for
//! FER-89/90/91 but the spike's resolution called out that ToC needs
//! only Pass 1 (Pass 2 adds nothing for this task), so the preamble
//! here is a Pass-1-only narrowing of the dual-source one. Body is
//! lifted verbatim — that's the prompt the spike validated against the
//! corpus.
//!
//! The builder is **pure**: it doesn't mutate the input `Doc`. The
//! caller is responsible for appending the new `Extraction` and
//! `TocEntry` records and persisting. That keeps the network-bound
//! call composable with cache layers and re-extraction policy.

use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::time::{Duration, Instant};

use ir::{
    Content, Doc, Extraction, ExtractionId, ModelId, PageExtraction, PageRef, PromptText,
    TocEntry, TocEntryId,
};
use serde::Deserialize;

use crate::openai::{ChatRequest, ChatResponse};
use crate::Error;

/// Modal endpoint for the production post-processor (per FER-113).
const QWEN_35B_URL: &str =
    "https://ferrite-systems--parselab-qwen36-35b-a3b-serve.modal.run";
const QWEN_35B_MODEL: &str = "Qwen/Qwen3.6-35B-A3B";

/// Generous output budget — a multi-page ToC for a dense MCU datasheet
/// can run a few thousand tokens of `rows[]`.
const MAX_TOKENS: u32 = 16384;
const TEMPERATURE: f32 = 0.0;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(900);

/// How long to keep retrying the warm-up ping while the Modal container
/// is cold-starting. 35B model load on H100 is ~60-180s; 600s gives
/// generous headroom for queue + load.
const WARM_UP_MAX_WAIT: Duration = Duration::from_secs(600);
/// Backoff between warm-up retries. Modal's edge proxy returns 408
/// almost immediately while SGLang is still loading, so we don't want
/// to hammer it — 20s is short enough that a freshly-loaded server
/// gets picked up quickly without burning quota with retries.
const WARM_UP_BACKOFF: Duration = Duration::from_secs(20);

/// Pass-1-only preamble. Narrowed from the FER-113 dual-source preamble
/// — no `_sources` provenance gate, no Pass 2 cross-referencing, since
/// the spike found Pass 2 contributes nothing to ToC quality. The
/// `issues` self-scoring layer is preserved so the model can flag
/// ambiguous / OCR-garbled rows.
const PASS1_ONLY_PREAMBLE: &str = r#"You are extracting a structured table of contents from a semiconductor datasheet's printed ToC pages. Input is the Pass 1 (GLM-OCR) markdown for the supplied page range.

Output a single JSON object with two top-level fields:

  "rows":   [ ...ToC entries (see schema below)... ]
  "issues": [ ...self-scored quality flags... ]

EVERY row entry MUST include the task-specific schema fields PLUS this provenance field:
  "_page": integer — the page number of the ToC page the entry appeared on (note the underscore!)

Field name MUST be `_page` (with leading underscore), NOT `page`. The page number for the ToC-page itself is `_page`; the page number a ToC entry POINTS TO is the task-specific `page` field defined per task.

NEVER fill missing values from prior knowledge of the part. If a value is missing, leave it `null`. Do not extrapolate from training data.

The "issues" array surfaces problems for human review. Each issue:
  "kind":        one of "missing_destination_page" | "ambiguous" | "ocr_garbled" | "pass_quality"
  "page":        integer or null (null for global issues)
  "description": string — what's wrong (one sentence)
  "evidence":    string — short quote from the input that supports the flag

Only emit issues where action is warranted — minor formatting differences are not issues.

Return ONLY the JSON object. No markdown, no explanation, no code fences.

"#;

/// Lifted from `extractor-eval/src/combined.rs::FER103_BODY`. This is
/// the prompt body the FER-113 spike validated against the corpus —
/// keep it byte-for-byte aligned so behavior matches the eval results.
const FER103_BODY: &str = r#"TASK: Extract every entry from the printed table-of-contents on this page (range), in document order.

Row schema (alongside _page):
  "level":  integer — heading depth, inferred from numbering: "5" → 1, "5.4" → 2, "5.4.2" → 3
  "number": string or null — section number as printed ("5.4")
  "title":  string — entry title with numbering AND footnote markers stripped (see below)
  "page":   integer or null — the page number the entry points to (this is the section's destination page, distinct from `_page` which is the page of the ToC entry itself)

TITLE HYGIENE:
  - Strip the leading numbering: "5.4 Thermal Information" → "Thermal Information"
  - Strip trailing footnote markers like "(1)", "(2)" or "$^{(1)}$": "Absolute Maximum Ratings(1)" → "Absolute Maximum Ratings"
  - Preserve LaTeX/math glyphs verbatim: "$V_{BAT}$ monitoring characteristics" stays as "$V_{BAT}$ monitoring characteristics" — do NOT flatten to "V BAT".
  - Preserve trademark/special chars: "Cortex®-M4", "I²S", "Embedded Trace Macrocell™" stay as-is.

SKIP:
  - Section banners ("Table of Contents", "Contents", "Index")
  - Revision-history bullets that may bleed into the same OCR page (e.g. "Changes from Revision X")
  - Prose paragraphs

INPUTS:
"#;

/// Result of one ToC build. The builder is pure — caller appends
/// `extraction` to `doc.extractions` and `entries` to `doc.toc`. Issues
/// are advisory; surface them in the UI / logs.
#[derive(Debug, Clone)]
pub struct TocBuildResult {
    pub extraction: Extraction,
    pub entries: Vec<TocEntry>,
    pub issues: Vec<TocIssue>,
}

/// Self-scored quality flag from the model. Mirrors the FER-113 issue
/// shape; empty for the heuristic builder.
#[derive(Debug, Clone, Deserialize)]
pub struct TocIssue {
    pub kind: String,
    #[serde(default)]
    pub page: Option<u32>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub evidence: Option<String>,
}

pub struct TocBuilder {
    backend: Backend,
}

enum Backend {
    Llm { url: String, model: String },
    Heuristic,
}

impl TocBuilder {
    /// Production builder — calls the deployed Qwen3.6-35B-A3B
    /// post-processor on Modal.
    pub fn new_qwen_35b() -> Self {
        Self {
            backend: Backend::Llm {
                url: QWEN_35B_URL.into(),
                model: QWEN_35B_MODEL.into(),
            },
        }
    }

    /// Configurable builder — point at a different OpenAI-compatible
    /// endpoint or model. Used by tests with a mocked HTTP server and by
    /// non-default deployments.
    pub fn with_endpoint(url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            backend: Backend::Llm {
                url: url.into(),
                model: model.into(),
            },
        }
    }

    /// Heuristic markdown-heading parser. No network, no LLM —
    /// emits a (typically shallower) ToC by scanning the Pass 1 page
    /// markdown for `^#+ ` lines. Useful as a fallback for tests and
    /// for cheap re-runs where round-tripping the LLM is undesirable.
    pub fn heuristic() -> Self {
        Self {
            backend: Backend::Heuristic,
        }
    }

    /// Build a ToC from the Pass 1 markdown `PageExtraction`s in `doc`
    /// covering the inclusive `pages` range.
    pub async fn build(
        &self,
        doc: &Doc,
        pages: RangeInclusive<u32>,
    ) -> Result<TocBuildResult, Error> {
        match &self.backend {
            Backend::Llm { url, model } => build_via_llm(doc, pages, url, model).await,
            Backend::Heuristic => Ok(build_via_heuristic(doc, pages)),
        }
    }
}

async fn build_via_llm(
    doc: &Doc,
    pages: RangeInclusive<u32>,
    url: &str,
    model: &str,
) -> Result<TocBuildResult, Error> {
    let pass1_pages = collect_pass1_pages(doc, pages.clone());
    if pass1_pages.is_empty() {
        return Err(Error::TocBuild(format!(
            "no Pass 1 markdown PageExtractions in range {}..={}",
            pages.start(),
            pages.end()
        )));
    }

    let prompt = build_prompt(&pass1_pages);

    let extraction = Extraction {
        uuid: ExtractionId::new(),
        model: ModelId(model.to_string()),
        prompt: PromptText(prompt.clone()),
        params: BTreeMap::from([
            ("max_tokens".into(), serde_json::json!(MAX_TOKENS)),
            ("temperature".into(), serde_json::json!(TEMPERATURE)),
            ("page_start".into(), serde_json::json!(*pages.start())),
            ("page_end_inclusive".into(), serde_json::json!(*pages.end())),
        ]),
        created_at: chrono::Utc::now(),
    };

    let response_text = call_llm(url, model, &prompt).await?;
    let parsed = parse_response(&response_text)?;

    let entries = build_entries(&parsed.rows, doc, extraction.uuid);

    Ok(TocBuildResult {
        extraction,
        entries,
        issues: parsed.issues,
    })
}

fn build_via_heuristic(doc: &Doc, pages: RangeInclusive<u32>) -> TocBuildResult {
    let pass1_pages = collect_pass1_pages(doc, pages.clone());

    let extraction = Extraction {
        uuid: ExtractionId::new(),
        model: ModelId("toc-heuristic@v1".into()),
        prompt: PromptText("(heuristic markdown-heading parser, no LLM call)".into()),
        params: BTreeMap::from([
            ("page_start".into(), serde_json::json!(*pages.start())),
            ("page_end_inclusive".into(), serde_json::json!(*pages.end())),
        ]),
        created_at: chrono::Utc::now(),
    };

    let mut rows: Vec<ParsedRow> = Vec::new();
    for pe in &pass1_pages {
        let Content::Markdown(md) = &pe.content else {
            continue;
        };
        for line in md.markdown.lines() {
            // ATX headings only — `# foo`, `## foo` etc. Trim trailing
            // whitespace (and any closing `#` markers, which markdown
            // permits) before treating the rest as the title.
            let stripped = line.trim_start();
            if !stripped.starts_with('#') {
                continue;
            }
            let level = stripped.bytes().take_while(|&b| b == b'#').count();
            if level == 0 || level > 6 {
                continue;
            }
            let after_hashes = &stripped[level..];
            // ATX requires a space (or end-of-line) after the hashes;
            // `#foo` is just text per CommonMark.
            if !after_hashes.starts_with(' ') && !after_hashes.is_empty() {
                continue;
            }
            let title = after_hashes
                .trim_start()
                .trim_end_matches(|c: char| c == '#' || c.is_whitespace())
                .to_string();
            if title.is_empty() {
                continue;
            }
            rows.push(ParsedRow {
                level: level as u8,
                number: None,
                title,
                // Heuristic: every heading "points to" the page it
                // appears on. The LLM path can produce a different
                // destination page if the printed ToC says so.
                destination_page: Some(pe.page),
                source_page: pe.page,
            });
        }
    }

    let entries = build_entries(&rows, doc, extraction.uuid);

    TocBuildResult {
        extraction,
        entries,
        issues: Vec::new(),
    }
}

/// Collect Pass 1 markdown `PageExtraction`s in the requested range,
/// preserving doc order. If a page has multiple Pass 1 extractions
/// (re-runs with different prompts), the most recent one wins —
/// `Doc.extracted_pages` is append-only so the last entry per page is
/// the latest. Pages with only Pass 2 / Error content are skipped.
fn collect_pass1_pages<'a>(doc: &'a Doc, pages: RangeInclusive<u32>) -> Vec<&'a PageExtraction> {
    let mut by_page: BTreeMap<u32, &PageExtraction> = BTreeMap::new();
    for pe in &doc.extracted_pages {
        if !pages.contains(&pe.page) {
            continue;
        }
        if !matches!(pe.content, Content::Markdown(_)) {
            continue;
        }
        by_page.insert(pe.page, pe);
    }
    by_page.into_values().collect()
}

fn build_prompt(pass1_pages: &[&PageExtraction]) -> String {
    let mut buf = String::new();
    buf.push_str(PASS1_ONLY_PREAMBLE);
    buf.push_str(FER103_BODY);
    for pe in pass1_pages {
        buf.push_str(&format!("=== PAGE {} ===\n\n", pe.page));
        if let Content::Markdown(md) = &pe.content {
            buf.push_str(&md.markdown);
            if !md.markdown.ends_with('\n') {
                buf.push('\n');
            }
        }
        buf.push('\n');
    }
    buf
}

async fn call_llm(url: &str, model: &str, prompt: &str) -> Result<String, Error> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?;

    // Modal scales the 35B container to zero after `scaledown_window`
    // seconds of idle (`modal/qwen36_35b_a3b/app.py`). The first call
    // after a cold-start triggers SGLang to load the model on H100,
    // which takes ~60-180s; during that window the edge proxy returns
    // 408 Request Timeout. Sending the heavy ToC prompt straight at a
    // cold endpoint either fails outright or burns prompt-token cost
    // on a request the proxy abandons. The cheap warm-up ping below
    // absorbs the cold-start retry loop on a 1-token request before
    // committing the real prompt.
    warm_up_endpoint(&client, url, model).await?;

    let request =
        ChatRequest::text_only(model, prompt, MAX_TOKENS, TEMPERATURE).without_thinking();

    let resp = client
        .post(format!("{}/v1/chat/completions", url.trim_end_matches('/')))
        .json(&request)
        .send()
        .await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::HttpStatus {
            status: status.as_u16(),
            body,
        });
    }
    let body: ChatResponse = resp.json().await?;
    let content = body
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .ok_or_else(|| Error::TocBuild("response had no message content".into()))?;
    Ok(content)
}

/// Cheap 1-token completion that probes whether the endpoint is ready
/// to serve the real request. On a warm server it returns in <1s and
/// is essentially free; on a cold-start it loops on 408 / timeout
/// until the model finishes loading or [`WARM_UP_MAX_WAIT`] elapses.
///
/// Stays silent when the endpoint is already warm (the common case);
/// surfaces a one-line stderr notice on the *first* cold-start tick so
/// a human running `build-toc` understands why the call is hanging,
/// then a closing line when it succeeds. Test runs only see this if
/// they actually hit the network, so unit tests stay quiet.
async fn warm_up_endpoint(
    client: &reqwest::Client,
    url: &str,
    model: &str,
) -> Result<(), Error> {
    let endpoint = format!("{}/v1/chat/completions", url.trim_end_matches('/'));
    let request = ChatRequest::text_only(model, "ping", 1, TEMPERATURE).without_thinking();
    let started = Instant::now();
    let mut announced = false;

    loop {
        match client.post(&endpoint).json(&request).send().await {
            Ok(r) if r.status().is_success() => {
                // Drain the body so the connection can be reused for
                // the real request that follows.
                let _ = r.bytes().await;
                if announced {
                    eprintln!(
                        "  endpoint warm ({:.0}s).",
                        started.elapsed().as_secs_f32()
                    );
                }
                return Ok(());
            }
            Ok(r) if r.status().as_u16() == 408 => {
                // 408 = Modal proxy timed out waiting for SGLang to
                // bind. Container is still loading; sleep and retry.
            }
            Ok(r) => {
                let status = r.status().as_u16();
                let body = r.text().await.unwrap_or_default();
                return Err(Error::HttpStatus { status, body });
            }
            Err(e) if e.is_timeout() || e.is_connect() => {
                // Transport-level cold-start signal — same handling.
            }
            Err(e) => return Err(Error::Transport(e)),
        }

        if !announced {
            eprintln!(
                "  endpoint cold-starting (35B model load on H100 takes ~60-180s); polling…"
            );
            announced = true;
        }
        if started.elapsed() >= WARM_UP_MAX_WAIT {
            return Err(Error::TocBuild(format!(
                "warm-up timed out after {:.0}s",
                started.elapsed().as_secs_f32()
            )));
        }
        tokio::time::sleep(WARM_UP_BACKOFF).await;
    }
}

#[derive(Debug, Deserialize)]
struct LlmResponse {
    rows: Vec<LlmRow>,
    #[serde(default)]
    issues: Vec<TocIssue>,
}

#[derive(Debug, Deserialize)]
struct LlmRow {
    #[serde(default)]
    level: Option<u8>,
    #[serde(default)]
    number: Option<String>,
    #[serde(default)]
    title: Option<String>,
    /// Destination page the entry links to ("Section 5 starts on p. 23").
    #[serde(default)]
    page: Option<u32>,
    /// Source page — which ToC page emitted this row. Note the
    /// underscore: required so the field doesn't collide with `page`.
    #[serde(default, rename = "_page")]
    source_page: Option<u32>,
}

#[derive(Debug)]
struct ParsedResponse {
    rows: Vec<ParsedRow>,
    issues: Vec<TocIssue>,
}

#[derive(Debug, Clone)]
struct ParsedRow {
    level: u8,
    number: Option<String>,
    title: String,
    destination_page: Option<u32>,
    /// Page where the row was found in the printed ToC. Used as a
    /// fallback for `page_refs` if the model omits the destination.
    source_page: u32,
}

fn parse_response(raw: &str) -> Result<ParsedResponse, Error> {
    // Strip code fences if the model ignored "no code fences". Cheap
    // belt-and-suspenders — the spike never hit this in practice but
    // it's a one-line guard.
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|s| s.trim_start_matches('\n'))
        .map(|s| s.strip_suffix("```").unwrap_or(s))
        .unwrap_or(trimmed);

    let resp: LlmResponse = serde_json::from_str(stripped)
        .map_err(|e| Error::TocBuild(format!("decode: {e}")))?;

    let mut rows = Vec::with_capacity(resp.rows.len());
    for row in resp.rows {
        let Some(title) = row.title.filter(|t| !t.is_empty()) else {
            continue;
        };
        let level = row.level.unwrap_or(1).max(1);
        let Some(source_page) = row.source_page else {
            // _page is required for provenance — without it we can't
            // link the row back to its source PageExtraction. Skip.
            continue;
        };
        rows.push(ParsedRow {
            level,
            number: row.number,
            title,
            destination_page: row.page,
            source_page,
        });
    }
    Ok(ParsedResponse {
        rows,
        issues: resp.issues,
    })
}

/// Walk parsed rows in document order, computing parent IDs from level
/// transitions, and produce `TocEntry` records. Page refs resolve via
/// `doc.extracted_pages`: the entry's destination page (or, falling
/// back, its source page) is looked up in the doc's Pass 1 extractions.
fn build_entries(rows: &[ParsedRow], doc: &Doc, built_by: ExtractionId) -> Vec<TocEntry> {
    // Stack of (level, id) representing the current ancestor chain.
    // For each new row at level L, pop ancestors with level >= L; the
    // remaining top is the parent.
    let mut stack: Vec<(u8, TocEntryId)> = Vec::new();
    let mut entries: Vec<TocEntry> = Vec::with_capacity(rows.len());

    for row in rows {
        while stack.last().map(|(l, _)| *l >= row.level).unwrap_or(false) {
            stack.pop();
        }
        let parent = stack.last().map(|(_, id)| *id);
        let id = TocEntryId::new();

        // Prefer the destination page when the model gave us one;
        // otherwise fall back to the source page so the entry still
        // links somewhere meaningful.
        let target_page = row.destination_page.unwrap_or(row.source_page);
        let page_refs = page_refs_for(doc, target_page);

        // Reconstitute "5.4 Foo" if the model split number from title.
        // The body says to strip numbering from the title itself, so we
        // honor that and skip the prefix unless the user wants it.
        let title = row.title.clone();

        entries.push(TocEntry {
            id,
            title,
            level: row.level,
            parent,
            page_refs,
            built_by,
        });
        let _ = row.number; // currently unused; reserved for richer rendering
        stack.push((row.level, id));
    }

    entries
}

/// All Pass 1 PageExtractions for `page`, materialised as `PageRef`s.
/// Multiple Pass 1 records on the same page (re-runs) all get refs —
/// downstream consumers can pick the latest.
fn page_refs_for(doc: &Doc, page: u32) -> Vec<PageRef> {
    doc.extracted_pages
        .iter()
        .filter(|p| p.page == page && matches!(p.content, Content::Markdown(_)))
        .filter_map(|pe| {
            let extraction = doc.extraction(pe.extraction_uuid)?;
            Some(PageRef {
                page: pe.page,
                model: extraction.model.clone(),
                page_uuid: pe.page_uuid,
                extraction_uuid: pe.extraction_uuid,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::{ContentHash, MarkdownContent, PageId, PageMetrics, SCHEMA_VERSION};

    fn fixture_doc() -> Doc {
        let pass1_uuid = ExtractionId::new();
        let extraction = Extraction {
            uuid: pass1_uuid,
            model: ModelId("zai-org/GLM-OCR".into()),
            prompt: PromptText("Text Recognition:".into()),
            params: BTreeMap::new(),
            created_at: chrono::Utc::now(),
        };

        let page_md = |page: u32, md: &str| PageExtraction {
            page_uuid: PageId::new(),
            page,
            extraction_uuid: pass1_uuid,
            content: Content::Markdown(MarkdownContent {
                markdown: md.into(),
            }),
            metrics: Some(PageMetrics {
                elapsed_secs: 0.0,
                input_tokens: None,
                output_tokens: None,
            }),
        };

        Doc {
            schema_version: SCHEMA_VERSION,
            content_hash: ContentHash("sha256:test".into()),
            extractions: vec![extraction],
            extracted_pages: vec![
                page_md(0, "# Datasheet\n\nIntroduction prose."),
                page_md(
                    1,
                    "# Table of Contents\n\n## Overview\n\n### Features\n\n## Electrical Characteristics\n\n### Absolute Maximum Ratings",
                ),
                page_md(2, "Continued ToC content..."),
            ],
            toc: Vec::new(),
        }
    }

    #[tokio::test]
    async fn heuristic_emits_hierarchical_toc_and_validates() {
        let doc = fixture_doc();
        let builder = TocBuilder::heuristic();
        let result = builder.build(&doc, 0..=2).await.unwrap();
        assert!(!result.entries.is_empty(), "should emit entries");

        // Hierarchy should reflect the ## / ### nesting on page 1.
        let titles: Vec<&str> = result.entries.iter().map(|e| e.title.as_str()).collect();
        assert!(titles.iter().any(|t| *t == "Overview"));
        assert!(titles.iter().any(|t| *t == "Features"));
        assert!(titles.iter().any(|t| *t == "Electrical Characteristics"));

        // Features (level 3) should parent to Overview (level 2).
        let overview = result
            .entries
            .iter()
            .find(|e| e.title == "Overview")
            .expect("Overview present");
        let features = result
            .entries
            .iter()
            .find(|e| e.title == "Features")
            .expect("Features present");
        assert_eq!(features.parent, Some(overview.id));
        assert_eq!(features.level, 3);

        // Each entry's built_by must match the Extraction record.
        for entry in &result.entries {
            assert_eq!(entry.built_by, result.extraction.uuid);
        }

        // After appending the result to the doc, FER-101 invariants
        // should hold (parent IDs resolve, page refs resolve).
        let mut doc = doc;
        doc.extractions.push(result.extraction.clone());
        doc.toc.extend(result.entries.clone());
        doc.validate().expect("FER-101 invariants must hold");
    }

    #[test]
    fn parse_response_strips_code_fence() {
        let raw = "```json\n{\"rows\": [{\"level\": 1, \"title\": \"Foo\", \"_page\": 0}], \"issues\": []}\n```";
        let parsed = parse_response(raw).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.rows[0].title, "Foo");
        assert_eq!(parsed.rows[0].source_page, 0);
    }

    #[test]
    fn parse_response_skips_rows_without_source_page() {
        let raw = r#"{"rows": [
            {"level": 1, "title": "WithPage", "_page": 0},
            {"level": 1, "title": "MissingPage"}
        ], "issues": []}"#;
        let parsed = parse_response(raw).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.rows[0].title, "WithPage");
    }

    #[test]
    fn build_entries_links_destination_page_and_validates() {
        // Simulates the LLM path: rows whose destination_page differs
        // from the source_page (entry was *printed* on the ToC page but
        // *links to* a content page elsewhere in the doc).
        let doc = fixture_doc();
        let extraction_uuid = ExtractionId::new();
        let rows = vec![
            ParsedRow {
                level: 1,
                number: Some("1".into()),
                title: "Overview".into(),
                destination_page: Some(0),
                source_page: 1,
            },
            ParsedRow {
                level: 2,
                number: Some("1.1".into()),
                title: "Features".into(),
                destination_page: Some(2),
                source_page: 1,
            },
            ParsedRow {
                level: 1,
                number: Some("2".into()),
                title: "Electrical".into(),
                destination_page: Some(2),
                source_page: 1,
            },
        ];

        let entries = build_entries(&rows, &doc, extraction_uuid);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].parent, None);
        assert_eq!(entries[1].parent, Some(entries[0].id));
        assert_eq!(entries[2].parent, None);

        // page_refs should resolve to the destination page's Pass 1
        // PageExtraction (page 0 for Overview, page 2 for Features).
        assert_eq!(entries[0].page_refs.first().map(|r| r.page), Some(0));
        assert_eq!(entries[1].page_refs.first().map(|r| r.page), Some(2));

        // After splicing into the doc, FER-101 invariants hold.
        let mut doc = doc;
        doc.extractions.push(Extraction {
            uuid: extraction_uuid,
            model: ModelId(QWEN_35B_MODEL.into()),
            prompt: PromptText("test".into()),
            params: BTreeMap::new(),
            created_at: chrono::Utc::now(),
        });
        doc.toc.extend(entries);
        doc.validate().expect("FER-101 invariants must hold");
    }
}
