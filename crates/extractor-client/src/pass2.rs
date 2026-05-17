//! Pass 2 (Infinity-Parser2-Pro) layout-JSON parser.
//!
//! FER-112. The Pass 2 worker is a stock SGLang server that returns the
//! model's raw text in the OpenAI chat-completions response. The model
//! emits a JSON blob — typically wrapped in markdown code fences —
//! describing per-block layout: `{bbox, category, text}` per element,
//! with bboxes in **image-pixel** coordinates (xyxy). This module
//! normalizes those into IR [`StructuredBlock`]s with bboxes in **PDF
//! points** (xywh), matching the [`ir::BBox`] contract.
//!
//! Ported from `_parse_layout_json` in the pre-2026-05-03 Python worker
//! (git history at `64867c4^:modal/infinity_parser2/app.py`).
//!
//! # Output shapes the parser tolerates
//!
//! Empirically (across the FER-80 corpus run), the model emits one of:
//!
//! 1. Top-level array of element objects: `[{bbox, category, text}, ...]`
//!    — the common case.
//! 2. Top-level dict wrapping the array under one of `layout_elements`,
//!    `elements`, or `regions` — defensive; not seen in the corpus but
//!    documented in the model card.
//! 3. Bbox-only arrays (`{bbox}` per element with no `category`/`text`)
//!    — degenerate model output we still parse so the caller can detect
//!    it (every element has `text: None`).
//!
//! Anything else (or syntactically broken JSON, which the model emits
//! when it loses its place mid-page) returns [`LayoutParseError`].
//! Callers typically convert that into `Content::Error { kind:
//! "parse_error" }` so sibling pages survive (FER-82 per-page error
//! pattern).

use ir::{BBox, StructuredBlock};
use serde_json::Value;

/// Failure modes for [`parse_layout_json`].
///
/// Distinguishes "JSON didn't decode" (model emitted broken syntax —
/// usually a degenerate page) from "JSON decoded but didn't match any
/// shape we recognize" (schema drift signal).
#[derive(Debug, thiserror::Error)]
pub enum LayoutParseError {
    /// `serde_json` couldn't decode the cleaned (post-fence-strip) text.
    /// Most common with degenerate Pass 2 outputs that emit duplicate
    /// keys or unterminated objects.
    #[error("layout json decode failed: {source} (head: {head:?})")]
    JsonDecode {
        #[source]
        source: serde_json::Error,
        /// First 120 chars of the cleaned text — enough to identify the
        /// failure mode at a glance without bloating logs.
        head: String,
    },

    /// JSON decoded but the top-level value was neither an array nor a
    /// dict carrying any recognized array key
    /// (`layout_elements`/`elements`/`regions`).
    #[error("unexpected layout shape (top level was {kind}, no element list found)")]
    UnexpectedShape { kind: &'static str },
}

/// Width and height of the model's normalized output coordinate
/// system. Empirically (verified 2026-05-04 against the production
/// Pass 2 worker), Infinity-Parser2-Pro emits bboxes in a
/// `[0, 1000] × [0, 1000]` grid regardless of the actual image
/// resolution it was given — a common Qwen-VL family convention for
/// grounding outputs. The earlier code (and the Python parser it
/// ported from) assumed image-pixel coords; that worked for the old
/// worker only because it happened to render at a DPI that produced
/// images close to 1000 px on a side. Don't change this without
/// re-empirically verifying against the model's output.
const MODEL_COORD_MAX: f32 = 1000.0;

/// Parse a Pass 2 chat-completion response body into a list of
/// [`StructuredBlock`]s with bboxes rescaled into PDF points.
///
/// `pdf_dim` is the source PDF page size in points — the multiplier
/// applied to the model's normalized [0, 1000] coords. The image
/// dimensions sent to the model do not appear here because the model
/// emits in a fixed normalized space (see [`MODEL_COORD_MAX`]).
///
/// Elements without a 4-element `bbox` field are silently dropped. The
/// `category` field becomes `StructuredBlock::kind` (defaulting to
/// `"unknown"` if absent); `text` and `confidence` are optional and
/// preserved when present.
pub fn parse_layout_json(
    raw: &str,
    pdf_dim: (f32, f32),
) -> Result<Vec<StructuredBlock>, LayoutParseError> {
    let cleaned = strip_fences(raw);
    let head: String = cleaned.chars().take(120).collect();

    let value: Value = serde_json::from_str(&cleaned)
        .map_err(|source| LayoutParseError::JsonDecode { source, head })?;

    let elements = normalize_elements(&value)?;

    let (pdf_w, pdf_h) = pdf_dim;
    let sx = pdf_w / MODEL_COORD_MAX;
    let sy = pdf_h / MODEL_COORD_MAX;

    let mut blocks = Vec::with_capacity(elements.len());
    for el in elements {
        let Some(bbox_xyxy) = el.get("bbox").and_then(|v| v.as_array()) else {
            continue;
        };
        if bbox_xyxy.len() != 4 {
            continue;
        }
        let coords: Option<Vec<f32>> = bbox_xyxy
            .iter()
            .map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        let Some(c) = coords else {
            continue;
        };
        let (x1, y1, x2, y2) = (c[0], c[1], c[2], c[3]);

        let kind = el
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let text = el
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let confidence = el
            .get("confidence")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32);

        blocks.push(StructuredBlock {
            kind,
            bbox: BBox {
                x: x1 * sx,
                y: y1 * sy,
                w: (x2 - x1) * sx,
                h: (y2 - y1) * sy,
            },
            text,
            confidence,
        });
    }
    Ok(blocks)
}

/// Strip surrounding ```json ... ``` (or bare ``` ... ```) markdown
/// fences if present. The model wraps inconsistently; we tolerate both.
/// Returns the inner content trimmed of whitespace.
fn strip_fences(raw: &str) -> String {
    let trimmed = raw.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }
    // Drop the opening fence line (```json or ```).
    let after_open = match trimmed.find('\n') {
        Some(nl) => &trimmed[nl + 1..],
        None => return trimmed.to_string(),
    };
    // The fence line might be `json` on its own (`)json` or no language).
    // We've already consumed the whole opening line via the newline split.
    // Strip the trailing fence if present.
    let body = match after_open.rfind("```") {
        Some(idx) => &after_open[..idx],
        None => after_open,
    };
    body.trim().to_string()
}

/// Normalize the parsed JSON value into the iterable element list.
/// Top-level array → use directly. Top-level dict → look for one of the
/// known keys. Anything else → schema-drift error.
fn normalize_elements(value: &Value) -> Result<Vec<&Value>, LayoutParseError> {
    if let Some(arr) = value.as_array() {
        return Ok(arr.iter().collect());
    }
    if let Some(obj) = value.as_object() {
        for key in ["layout_elements", "elements", "regions"] {
            if let Some(arr) = obj.get(key).and_then(|v| v.as_array()) {
                return Ok(arr.iter().collect());
            }
        }
        return Err(LayoutParseError::UnexpectedShape { kind: "object" });
    }
    Err(LayoutParseError::UnexpectedShape {
        kind: match value {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            _ => "other",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// US Letter page dimensions in PDF points.
    const US_LETTER_PTS: (f32, f32) = (612.0, 792.0);

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn parses_top_level_array_with_fences() {
        // Bbox values are in the model's [0, 1000] normalized space.
        let raw = r#"```json
[{"bbox": [100, 200, 500, 400], "category": "table", "text": "hi"}]
```"#;
        let blocks = parse_layout_json(raw, US_LETTER_PTS).expect("parse");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].kind, "table");
        assert_eq!(blocks[0].text.as_deref(), Some("hi"));
        // x = 100/1000 × 612 = 61.2 pts; w = (500-100)/1000 × 612 = 244.8 pts
        assert!(approx_eq(blocks[0].bbox.x, 61.2, 0.01));
        assert!(approx_eq(blocks[0].bbox.w, 244.8, 0.01));
    }

    #[test]
    fn parses_top_level_array_without_fences() {
        let raw = r#"[{"bbox": [0, 0, 100, 100], "category": "title", "text": "T"}]"#;
        let blocks = parse_layout_json(raw, (200.0, 200.0)).expect("parse");
        assert_eq!(blocks.len(), 1);
        // 100/1000 × 200 = 20 pts
        assert!(approx_eq(blocks[0].bbox.w, 20.0, 0.001));
        assert!(approx_eq(blocks[0].bbox.h, 20.0, 0.001));
    }

    #[test]
    fn parses_dict_wrapped_layout_elements() {
        let raw = r#"{"layout_elements": [{"bbox": [1, 2, 3, 4], "category": "x"}]}"#;
        let blocks = parse_layout_json(raw, (10.0, 10.0)).expect("parse");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].kind, "x");
        assert!(blocks[0].text.is_none());
    }

    #[test]
    fn parses_dict_wrapped_elements_alias() {
        let raw = r#"{"elements": [{"bbox": [0, 0, 1, 1]}]}"#;
        let blocks = parse_layout_json(raw, (10.0, 10.0)).expect("parse");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].kind, "unknown");
    }

    #[test]
    fn parses_dict_wrapped_regions_alias() {
        let raw = r#"{"regions": [{"bbox": [0, 0, 1, 1]}]}"#;
        let blocks = parse_layout_json(raw, (10.0, 10.0)).expect("parse");
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn bbox_only_elements_parse_with_text_none() {
        // The degenerate Pass 2 case: layout array but no text content.
        let raw = r#"[{"bbox": [10, 20, 30, 40]}, {"bbox": [50, 60, 70, 80]}]"#;
        let blocks = parse_layout_json(raw, (100.0, 100.0)).expect("parse");
        assert_eq!(blocks.len(), 2);
        assert!(blocks.iter().all(|b| b.text.is_none()));
        assert!(blocks.iter().all(|b| b.kind == "unknown"));
    }

    #[test]
    fn elements_without_valid_bbox_are_skipped() {
        let raw = r#"[
            {"bbox": [1, 2, 3, 4], "category": "ok"},
            {"category": "no_bbox"},
            {"bbox": [1, 2, 3], "category": "wrong_len"},
            {"bbox": "not_an_array"},
            {"bbox": [1, 2, 3, 4], "category": "ok2"}
        ]"#;
        let blocks = parse_layout_json(raw, (10.0, 10.0)).expect("parse");
        // Only the two with valid 4-elem array bboxes survive.
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].kind, "ok");
        assert_eq!(blocks[1].kind, "ok2");
    }

    #[test]
    fn confidence_field_is_preserved_when_present() {
        let raw = r#"[{"bbox": [0, 0, 1, 1], "category": "table", "confidence": 0.92}]"#;
        let blocks = parse_layout_json(raw, (10.0, 10.0)).expect("parse");
        assert_eq!(blocks[0].confidence, Some(0.92));
    }

    #[test]
    fn malformed_json_returns_decode_error_with_head_snippet() {
        // Real degenerate output from tlv757p p2: duplicate `bbox` keys
        // and an unterminated final object — model lost its place.
        let raw = r#"[{"bbox": [85, 34, 249, 80], "bbox": "broken}]"#;
        let err = parse_layout_json(raw, (10.0, 10.0)).expect_err("should fail");
        match err {
            LayoutParseError::JsonDecode { head, .. } => {
                assert!(head.contains("85, 34"), "head should carry the start: {head:?}");
            }
            other => panic!("expected JsonDecode, got {other:?}"),
        }
    }

    #[test]
    fn dict_with_no_known_array_key_returns_unexpected_shape() {
        let raw = r#"{"foo": [1, 2, 3]}"#;
        let err = parse_layout_json(raw, (10.0, 10.0)).expect_err("should fail");
        assert!(matches!(err, LayoutParseError::UnexpectedShape { kind: "object" }));
    }

    #[test]
    fn top_level_string_returns_unexpected_shape() {
        let raw = r#""I am a string""#;
        let err = parse_layout_json(raw, (10.0, 10.0)).expect_err("should fail");
        assert!(matches!(err, LayoutParseError::UnexpectedShape { kind: "string" }));
    }

    #[test]
    fn empty_array_yields_zero_blocks() {
        let raw = "[]";
        let blocks = parse_layout_json(raw, (10.0, 10.0)).expect("parse");
        assert!(blocks.is_empty());
    }

    #[test]
    fn rescale_preserves_relative_position_and_size() {
        // A bbox at half-resolution in the [0, 1000] grid should land at
        // 50% of the PDF page width, 50% of the height.
        let raw = r#"[{"bbox": [250, 250, 750, 1000], "category": "table"}]"#;
        let pdf = US_LETTER_PTS;
        let blocks = parse_layout_json(raw, pdf).expect("parse");
        let b = &blocks[0];
        // x = 250/1000 × 612 = 153 pts
        // y = 250/1000 × 792 = 198 pts
        // w = (750-250)/1000 × 612 = 306 pts (50% of width)
        // h = (1000-250)/1000 × 792 = 594 pts (75% of height)
        assert!(approx_eq(b.bbox.x, 153.0, 0.01));
        assert!(approx_eq(b.bbox.y, 198.0, 0.01));
        assert!(approx_eq(b.bbox.w, 306.0, 0.01));
        assert!(approx_eq(b.bbox.h, 594.0, 0.01));
    }

    #[test]
    fn full_page_bbox_maps_to_full_pdf_bounds() {
        // A bbox spanning the full [0, 1000] grid should map to the
        // full PDF page bounds.
        let raw = r#"[{"bbox": [0, 0, 1000, 1000], "category": "x"}]"#;
        let blocks = parse_layout_json(raw, US_LETTER_PTS).expect("parse");
        let b = &blocks[0];
        assert!(approx_eq(b.bbox.x, 0.0, 0.01));
        assert!(approx_eq(b.bbox.y, 0.0, 0.01));
        assert!(approx_eq(b.bbox.w, 612.0, 0.01));
        assert!(approx_eq(b.bbox.h, 792.0, 0.01));
    }

    #[test]
    fn fence_with_language_tag_and_no_trailing_fence() {
        // Model sometimes truncates the closing fence when hitting
        // max_tokens. Tolerate it.
        let raw = "```json\n[{\"bbox\": [0, 0, 1, 1], \"category\": \"x\"}]";
        let blocks = parse_layout_json(raw, (10.0, 10.0)).expect("parse");
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn html_table_text_round_trips_intact() {
        // Pass 2 emits tables as HTML strings inside the `text` field —
        // FER-89's interpreter wants them preserved character-for-character.
        let raw = r#"[{"bbox": [0, 0, 100, 100], "category": "table",
            "text": "<table><tr><th>Min</th><th>Max</th></tr><tr><td>40</td><td>—</td></tr></table>"}]"#;
        let blocks = parse_layout_json(raw, (1000.0, 1000.0)).expect("parse");
        let text = blocks[0].text.as_deref().unwrap();
        assert!(text.contains("<table>"));
        assert!(text.contains("<th>Min</th>"));
        assert!(text.contains("<td>40</td>"));
    }
}
