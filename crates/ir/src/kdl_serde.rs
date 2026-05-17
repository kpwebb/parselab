//! KDL on-disk format for [`Doc`] — FER-95 phase 1.
//!
//! Per CLAUDE.md, all on-disk Ferrite data is KDL. This module owns the
//! `Doc` ↔ KDL translation, hand-rolled against `kdl-rs`'s structural
//! node model rather than a serde adapter so the disk shape stays
//! readable and stable as the IR evolves. Wire formats (Modal workers
//! → Rust client) stay JSON; KDL is for what lands on disk.
//!
//! Schema sketch:
//!
//! ```kdl
//! doc schema_version=1 content_hash="..." {
//!     extractions {
//!         extraction uuid="..." model="glm-ocr@v1" created_at="..." {
//!             prompt "..."
//!             params {
//!                 dpi 200
//!                 max_new_tokens 8192
//!             }
//!         }
//!     }
//!     extracted_pages {
//!         page page=0 page_uuid="..." extraction_uuid="..." {
//!             markdown "..."
//!             metrics elapsed_secs=1.4 input_tokens=1024 output_tokens=217
//!         }
//!         page page=14 page_uuid="..." extraction_uuid="..." {
//!             structured {
//!                 page_meta width_pts=612 height_pts=792 rotation_deg=0 dpi=300
//!                 block kind="table" x=72 y=120 w=450 h=200 confidence=0.92 {
//!                     text "..."
//!                 }
//!             }
//!         }
//!     }
//!     toc {
//!         entry id="..." title="..." level=1 built_by="..." {
//!             page_ref page=14 model="..." page_uuid="..." extraction_uuid="..."
//!         }
//!     }
//! }
//! ```

use std::collections::BTreeMap;

use chrono::SecondsFormat;
use kdl::{KdlDocument, KdlEntry, KdlNode, KdlValue};
use thiserror::Error;
use uuid::Uuid;

use crate::{
    BBox, Content, ContentHash, Doc, ErrorContent, Extraction, ExtractionId, MarkdownContent,
    ModelId, PageExtraction, PageId, PageMeta, PageMetrics, PageRef, PromptText, StructuredBlock,
    StructuredPage, Timestamp, TocEntry, TocEntryId,
};

#[derive(Debug, Error)]
pub enum IrKdlError {
    #[error("kdl parse: {0}")]
    Parse(#[from] kdl::KdlError),
    #[error("missing required node `{0}`")]
    MissingNode(&'static str),
    #[error("missing required property `{prop}` on `{node}`")]
    MissingProp {
        node: &'static str,
        prop: &'static str,
    },
    #[error("missing required argument on `{0}`")]
    MissingArg(&'static str),
    #[error("invalid value for `{field}`: {message}")]
    InvalidValue { field: String, message: String },
    #[error("unknown content variant `{0}` (expected `markdown`, `structured`, or `error`)")]
    UnknownContent(String),
    #[error("invalid uuid `{value}`: {source}")]
    InvalidUuid {
        value: String,
        #[source]
        source: uuid::Error,
    },
    #[error("invalid timestamp `{value}`: {source}")]
    InvalidTimestamp {
        value: String,
        #[source]
        source: chrono::ParseError,
    },
    #[error("invalid json for params field `{field}`: {source}")]
    InvalidJsonParam {
        field: String,
        #[source]
        source: serde_json::Error,
    },
}

// =============================================================================
// Public API
// =============================================================================

impl Doc {
    /// Serialize this `Doc` to a KDL string. Auto-formatted; round-trip
    /// safe with [`Doc::from_kdl_str`].
    pub fn to_kdl_string(&self) -> String {
        let mut document = KdlDocument::new();
        document.nodes_mut().push(doc_to_kdl_node(self));
        document.autoformat();
        document.to_string()
    }

    pub fn from_kdl_str(s: &str) -> Result<Self, IrKdlError> {
        let document = KdlDocument::parse_v2(s)?;
        Self::from_kdl_document(&document)
    }

    pub fn from_kdl_document(document: &KdlDocument) -> Result<Self, IrKdlError> {
        let root = document.get("doc").ok_or(IrKdlError::MissingNode("doc"))?;
        doc_from_kdl_node(root)
    }
}

// =============================================================================
// Encoders
// =============================================================================

fn doc_to_kdl_node(doc: &Doc) -> KdlNode {
    let mut node = KdlNode::new("doc");
    push_prop_int(&mut node, "schema_version", doc.schema_version as i128);
    push_prop_string(&mut node, "content_hash", &doc.content_hash.0);

    let mut children = KdlDocument::new();

    let mut extractions_node = KdlNode::new("extractions");
    let mut extractions_doc = KdlDocument::new();
    for e in &doc.extractions {
        extractions_doc.nodes_mut().push(extraction_to_kdl_node(e));
    }
    extractions_node.set_children(extractions_doc);
    children.nodes_mut().push(extractions_node);

    let mut pages_node = KdlNode::new("extracted_pages");
    let mut pages_doc = KdlDocument::new();
    for pe in &doc.extracted_pages {
        pages_doc.nodes_mut().push(page_extraction_to_kdl_node(pe));
    }
    pages_node.set_children(pages_doc);
    children.nodes_mut().push(pages_node);

    let mut toc_node = KdlNode::new("toc");
    let mut toc_doc = KdlDocument::new();
    for entry in &doc.toc {
        toc_doc.nodes_mut().push(toc_entry_to_kdl_node(entry));
    }
    toc_node.set_children(toc_doc);
    children.nodes_mut().push(toc_node);

    node.set_children(children);
    node
}

fn extraction_to_kdl_node(extraction: &Extraction) -> KdlNode {
    let mut node = KdlNode::new("extraction");
    push_prop_string(&mut node, "uuid", &extraction.uuid.0.to_string());
    push_prop_string(&mut node, "model", &extraction.model.0);
    push_prop_string(
        &mut node,
        "created_at",
        &timestamp_to_string(extraction.created_at),
    );

    let mut children = KdlDocument::new();

    let mut prompt_node = KdlNode::new("prompt");
    prompt_node.push(KdlEntry::new(KdlValue::String(extraction.prompt.0.clone())));
    children.nodes_mut().push(prompt_node);

    if !extraction.params.is_empty() {
        let mut params_node = KdlNode::new("params");
        let mut params_doc = KdlDocument::new();
        for (k, v) in &extraction.params {
            params_doc.nodes_mut().push(json_param_to_kdl_node(k, v));
        }
        params_node.set_children(params_doc);
        children.nodes_mut().push(params_node);
    }

    node.set_children(children);
    node
}

fn page_extraction_to_kdl_node(pe: &PageExtraction) -> KdlNode {
    let mut node = KdlNode::new("page");
    push_prop_int(&mut node, "page", pe.page as i128);
    push_prop_string(&mut node, "page_uuid", &pe.page_uuid.0.to_string());
    push_prop_string(
        &mut node,
        "extraction_uuid",
        &pe.extraction_uuid.0.to_string(),
    );

    let mut children = KdlDocument::new();
    children.nodes_mut().push(content_to_kdl_node(&pe.content));
    if let Some(metrics) = &pe.metrics {
        children.nodes_mut().push(metrics_to_kdl_node(metrics));
    }
    node.set_children(children);
    node
}

fn content_to_kdl_node(content: &Content) -> KdlNode {
    match content {
        Content::Markdown(MarkdownContent { markdown }) => {
            let mut node = KdlNode::new("markdown");
            node.push(KdlEntry::new(KdlValue::String(markdown.clone())));
            node
        }
        Content::Structured(sp) => structured_page_to_kdl_node(sp),
        Content::Error(ErrorContent { kind, message }) => {
            let mut node = KdlNode::new("error");
            push_prop_string(&mut node, "kind", kind);
            push_prop_string(&mut node, "message", message);
            node
        }
    }
}

fn structured_page_to_kdl_node(sp: &StructuredPage) -> KdlNode {
    let mut node = KdlNode::new("structured");
    let mut children = KdlDocument::new();

    if let Some(meta) = &sp.page_meta {
        let mut meta_node = KdlNode::new("page_meta");
        push_prop_float(&mut meta_node, "width_pts", f32_as_clean_f64(meta.width_pts));
        push_prop_float(&mut meta_node, "height_pts", f32_as_clean_f64(meta.height_pts));
        push_prop_int(&mut meta_node, "rotation_deg", meta.rotation_deg as i128);
        if let Some(dpi) = meta.dpi {
            push_prop_int(&mut meta_node, "dpi", dpi as i128);
        }
        children.nodes_mut().push(meta_node);
    }

    for block in &sp.blocks {
        children.nodes_mut().push(block_to_kdl_node(block));
    }

    node.set_children(children);
    node
}

fn block_to_kdl_node(b: &StructuredBlock) -> KdlNode {
    let mut node = KdlNode::new("block");
    push_prop_string(&mut node, "kind", &b.kind);
    push_prop_float(&mut node, "x", f32_as_clean_f64(b.bbox.x));
    push_prop_float(&mut node, "y", f32_as_clean_f64(b.bbox.y));
    push_prop_float(&mut node, "w", f32_as_clean_f64(b.bbox.w));
    push_prop_float(&mut node, "h", f32_as_clean_f64(b.bbox.h));
    if let Some(c) = b.confidence {
        push_prop_float(&mut node, "confidence", f32_as_clean_f64(c));
    }
    if let Some(t) = &b.text {
        let mut children = KdlDocument::new();
        let mut text_node = KdlNode::new("text");
        text_node.push(KdlEntry::new(KdlValue::String(t.clone())));
        children.nodes_mut().push(text_node);
        node.set_children(children);
    }
    node
}

fn metrics_to_kdl_node(m: &PageMetrics) -> KdlNode {
    let mut node = KdlNode::new("metrics");
    push_prop_float(&mut node, "elapsed_secs", f32_as_clean_f64(m.elapsed_secs));
    if let Some(t) = m.input_tokens {
        push_prop_int(&mut node, "input_tokens", t as i128);
    }
    if let Some(t) = m.output_tokens {
        push_prop_int(&mut node, "output_tokens", t as i128);
    }
    node
}

fn toc_entry_to_kdl_node(entry: &TocEntry) -> KdlNode {
    let mut node = KdlNode::new("entry");
    push_prop_string(&mut node, "id", &entry.id.0.to_string());
    push_prop_string(&mut node, "title", &entry.title);
    push_prop_int(&mut node, "level", entry.level as i128);
    if let Some(parent) = entry.parent {
        push_prop_string(&mut node, "parent", &parent.0.to_string());
    }
    push_prop_string(&mut node, "built_by", &entry.built_by.0.to_string());

    let mut children = KdlDocument::new();
    for r in &entry.page_refs {
        children.nodes_mut().push(page_ref_to_kdl_node(r));
    }
    node.set_children(children);
    node
}

fn page_ref_to_kdl_node(r: &PageRef) -> KdlNode {
    let mut node = KdlNode::new("page_ref");
    push_prop_int(&mut node, "page", r.page as i128);
    push_prop_string(&mut node, "model", &r.model.0);
    push_prop_string(&mut node, "page_uuid", &r.page_uuid.0.to_string());
    push_prop_string(&mut node, "extraction_uuid", &r.extraction_uuid.0.to_string());
    node
}

/// Encode a single param as a child node `key value` (e.g. `dpi 200`).
/// Primitive JSON types get native KDL atoms; arrays/objects fall back
/// to a `(json)`-typed string so any complex value round-trips.
fn json_param_to_kdl_node(key: &str, value: &serde_json::Value) -> KdlNode {
    let mut node = KdlNode::new(key);
    match value {
        serde_json::Value::Bool(b) => {
            node.push(KdlEntry::new(KdlValue::Bool(*b)));
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                node.push(KdlEntry::new(KdlValue::Integer(i as i128)));
            } else if let Some(u) = n.as_u64() {
                node.push(KdlEntry::new(KdlValue::Integer(u as i128)));
            } else if let Some(f) = n.as_f64() {
                node.push(KdlEntry::new(KdlValue::Float(f)));
            } else {
                push_json_typed_arg(&mut node, value);
            }
        }
        serde_json::Value::String(s) => {
            node.push(KdlEntry::new(KdlValue::String(s.clone())));
        }
        serde_json::Value::Null => {
            node.push(KdlEntry::new(KdlValue::Null));
        }
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            push_json_typed_arg(&mut node, value);
        }
    }
    node
}

fn push_json_typed_arg(node: &mut KdlNode, value: &serde_json::Value) {
    let json = serde_json::to_string(value).unwrap_or_default();
    let mut entry = KdlEntry::new(KdlValue::String(json));
    entry.set_ty("json");
    node.push(entry);
}

// =============================================================================
// Decoders
// =============================================================================

fn doc_from_kdl_node(node: &KdlNode) -> Result<Doc, IrKdlError> {
    let schema_version = read_prop_int(node, "doc", "schema_version")? as u32;
    let content_hash = ContentHash(read_prop_string(node, "doc", "content_hash")?);

    let children = node.children();
    let extractions = children
        .and_then(|c| c.get("extractions"))
        .map(extractions_from_kdl_node)
        .transpose()?
        .unwrap_or_default();
    let extracted_pages = children
        .and_then(|c| c.get("extracted_pages"))
        .map(extracted_pages_from_kdl_node)
        .transpose()?
        .unwrap_or_default();
    let toc = children
        .and_then(|c| c.get("toc"))
        .map(toc_from_kdl_node)
        .transpose()?
        .unwrap_or_default();

    Ok(Doc {
        schema_version,
        content_hash,
        extractions,
        extracted_pages,
        toc,
    })
}

fn extractions_from_kdl_node(node: &KdlNode) -> Result<Vec<Extraction>, IrKdlError> {
    let Some(children) = node.children() else {
        return Ok(Vec::new());
    };
    children
        .nodes()
        .iter()
        .filter(|n| n.name().value() == "extraction")
        .map(extraction_from_kdl_node)
        .collect()
}

fn extraction_from_kdl_node(node: &KdlNode) -> Result<Extraction, IrKdlError> {
    let uuid = ExtractionId(read_prop_uuid(node, "extraction", "uuid")?);
    let model = ModelId(read_prop_string(node, "extraction", "model")?);
    let created_at = read_prop_timestamp(node, "extraction", "created_at")?;

    let mut prompt = PromptText(String::new());
    let mut params = BTreeMap::new();
    if let Some(children) = node.children() {
        for child in children.nodes() {
            match child.name().value() {
                "prompt" => {
                    let s = read_first_arg_string(child, "prompt")?;
                    prompt = PromptText(s);
                }
                "params" => {
                    if let Some(params_children) = child.children() {
                        for p in params_children.nodes() {
                            let key = p.name().value().to_string();
                            let value = json_param_from_kdl_node(p)?;
                            params.insert(key, value);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(Extraction {
        uuid,
        model,
        prompt,
        params,
        created_at,
    })
}

fn extracted_pages_from_kdl_node(
    node: &KdlNode,
) -> Result<Vec<PageExtraction>, IrKdlError> {
    let Some(children) = node.children() else {
        return Ok(Vec::new());
    };
    children
        .nodes()
        .iter()
        .filter(|n| n.name().value() == "page")
        .map(page_extraction_from_kdl_node)
        .collect()
}

fn page_extraction_from_kdl_node(node: &KdlNode) -> Result<PageExtraction, IrKdlError> {
    let page = read_prop_int(node, "page", "page")? as u32;
    let page_uuid = PageId(read_prop_uuid(node, "page", "page_uuid")?);
    let extraction_uuid = ExtractionId(read_prop_uuid(node, "page", "extraction_uuid")?);

    let mut content: Option<Content> = None;
    let mut metrics: Option<PageMetrics> = None;
    if let Some(children) = node.children() {
        for child in children.nodes() {
            match child.name().value() {
                "markdown" => {
                    let s = read_first_arg_string(child, "markdown")?;
                    content = Some(Content::Markdown(MarkdownContent { markdown: s }));
                }
                "structured" => {
                    content = Some(Content::Structured(structured_page_from_kdl_node(child)?));
                }
                "error" => {
                    let kind = read_prop_string(child, "error", "kind")?;
                    let message = read_prop_string(child, "error", "message")?;
                    content = Some(Content::Error(ErrorContent { kind, message }));
                }
                "metrics" => {
                    metrics = Some(metrics_from_kdl_node(child)?);
                }
                other => {
                    return Err(IrKdlError::UnknownContent(other.to_string()));
                }
            }
        }
    }
    let content = content.ok_or(IrKdlError::MissingNode("page content"))?;
    Ok(PageExtraction {
        page_uuid,
        page,
        extraction_uuid,
        content,
        metrics,
    })
}

fn structured_page_from_kdl_node(node: &KdlNode) -> Result<StructuredPage, IrKdlError> {
    let mut blocks = Vec::new();
    let mut page_meta: Option<PageMeta> = None;
    if let Some(children) = node.children() {
        for child in children.nodes() {
            match child.name().value() {
                "page_meta" => {
                    page_meta = Some(page_meta_from_kdl_node(child)?);
                }
                "block" => {
                    blocks.push(block_from_kdl_node(child)?);
                }
                _ => {}
            }
        }
    }
    Ok(StructuredPage { blocks, page_meta })
}

fn page_meta_from_kdl_node(node: &KdlNode) -> Result<PageMeta, IrKdlError> {
    let width_pts = read_prop_float(node, "page_meta", "width_pts")? as f32;
    let height_pts = read_prop_float(node, "page_meta", "height_pts")? as f32;
    let rotation_deg = read_prop_int(node, "page_meta", "rotation_deg")? as u16;
    let dpi = read_optional_prop_int(node, "dpi")?.map(|v| v as u16);
    Ok(PageMeta {
        width_pts,
        height_pts,
        rotation_deg,
        dpi,
    })
}

fn block_from_kdl_node(node: &KdlNode) -> Result<StructuredBlock, IrKdlError> {
    let kind = read_prop_string(node, "block", "kind")?;
    let bbox = BBox {
        x: read_prop_float(node, "block", "x")? as f32,
        y: read_prop_float(node, "block", "y")? as f32,
        w: read_prop_float(node, "block", "w")? as f32,
        h: read_prop_float(node, "block", "h")? as f32,
    };
    let confidence = read_optional_prop_float(node, "confidence")?.map(|v| v as f32);
    let mut text = None;
    if let Some(children) = node.children() {
        for child in children.nodes() {
            if child.name().value() == "text" {
                text = Some(read_first_arg_string(child, "text")?);
            }
        }
    }
    Ok(StructuredBlock {
        kind,
        bbox,
        text,
        confidence,
    })
}

fn metrics_from_kdl_node(node: &KdlNode) -> Result<PageMetrics, IrKdlError> {
    let elapsed_secs = read_prop_float(node, "metrics", "elapsed_secs")? as f32;
    let input_tokens = read_optional_prop_int(node, "input_tokens")?.map(|v| v as u32);
    let output_tokens = read_optional_prop_int(node, "output_tokens")?.map(|v| v as u32);
    Ok(PageMetrics {
        elapsed_secs,
        input_tokens,
        output_tokens,
    })
}

fn toc_from_kdl_node(node: &KdlNode) -> Result<Vec<TocEntry>, IrKdlError> {
    let Some(children) = node.children() else {
        return Ok(Vec::new());
    };
    children
        .nodes()
        .iter()
        .filter(|n| n.name().value() == "entry")
        .map(toc_entry_from_kdl_node)
        .collect()
}

fn toc_entry_from_kdl_node(node: &KdlNode) -> Result<TocEntry, IrKdlError> {
    let id = TocEntryId(read_prop_uuid(node, "entry", "id")?);
    let title = read_prop_string(node, "entry", "title")?;
    let level = read_prop_int(node, "entry", "level")? as u8;
    let parent = read_optional_prop_string(node, "parent")?
        .map(|s| {
            Uuid::parse_str(&s)
                .map(TocEntryId)
                .map_err(|source| IrKdlError::InvalidUuid { value: s, source })
        })
        .transpose()?;
    let built_by = ExtractionId(read_prop_uuid(node, "entry", "built_by")?);

    let mut page_refs = Vec::new();
    if let Some(children) = node.children() {
        for child in children.nodes() {
            if child.name().value() == "page_ref" {
                page_refs.push(page_ref_from_kdl_node(child)?);
            }
        }
    }

    Ok(TocEntry {
        id,
        title,
        level,
        parent,
        page_refs,
        built_by,
    })
}

fn page_ref_from_kdl_node(node: &KdlNode) -> Result<PageRef, IrKdlError> {
    Ok(PageRef {
        page: read_prop_int(node, "page_ref", "page")? as u32,
        model: ModelId(read_prop_string(node, "page_ref", "model")?),
        page_uuid: PageId(read_prop_uuid(node, "page_ref", "page_uuid")?),
        extraction_uuid: ExtractionId(read_prop_uuid(node, "page_ref", "extraction_uuid")?),
    })
}

fn json_param_from_kdl_node(node: &KdlNode) -> Result<serde_json::Value, IrKdlError> {
    let entry = node
        .entries()
        .first()
        .ok_or_else(|| IrKdlError::MissingArg("params child"))?;
    let ty = entry.ty().map(|t| t.value());
    match (ty, entry.value()) {
        (Some("json"), KdlValue::String(s)) => serde_json::from_str(s).map_err(|source| {
            IrKdlError::InvalidJsonParam {
                field: node.name().value().to_string(),
                source,
            }
        }),
        (_, KdlValue::Bool(b)) => Ok(serde_json::Value::Bool(*b)),
        (_, KdlValue::Integer(i)) => Ok(serde_json::Value::Number((*i as i64).into())),
        (_, KdlValue::Float(f)) => Ok(serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)),
        (_, KdlValue::String(s)) => Ok(serde_json::Value::String(s.clone())),
        (_, KdlValue::Null) => Ok(serde_json::Value::Null),
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn push_prop_string(node: &mut KdlNode, name: &str, value: &str) {
    node.push(KdlEntry::new_prop(name, KdlValue::String(value.to_string())));
}

fn push_prop_int(node: &mut KdlNode, name: &str, value: i128) {
    node.push(KdlEntry::new_prop(name, KdlValue::Integer(value)));
}

fn push_prop_float(node: &mut KdlNode, name: &str, value: f64) {
    node.push(KdlEntry::new_prop(name, KdlValue::Float(value)));
}

/// f32 → KDL float that stringifies cleanly. Direct `f as f64` preserves
/// the f32 bit pattern exactly but smears it across a long decimal
/// (e.g. `0.92f32 as f64 → 0.9200000166893005`). Round-tripping through
/// f32's shortest-round-trip text picks the closest f64, which prints
/// as `0.92`. Casting back to f32 on read still yields the original
/// value, so round-trip equality is preserved.
fn f32_as_clean_f64(v: f32) -> f64 {
    format!("{v}").parse::<f64>().unwrap_or(v as f64)
}

fn read_prop_string(
    node: &KdlNode,
    node_name: &'static str,
    prop: &'static str,
) -> Result<String, IrKdlError> {
    match node.get(prop) {
        Some(KdlValue::String(s)) => Ok(s.clone()),
        Some(other) => Err(IrKdlError::InvalidValue {
            field: format!("{node_name}.{prop}"),
            message: format!("expected string, got {other:?}"),
        }),
        None => Err(IrKdlError::MissingProp {
            node: node_name,
            prop,
        }),
    }
}

fn read_prop_int(
    node: &KdlNode,
    node_name: &'static str,
    prop: &'static str,
) -> Result<i128, IrKdlError> {
    match node.get(prop) {
        Some(KdlValue::Integer(i)) => Ok(*i),
        Some(other) => Err(IrKdlError::InvalidValue {
            field: format!("{node_name}.{prop}"),
            message: format!("expected integer, got {other:?}"),
        }),
        None => Err(IrKdlError::MissingProp {
            node: node_name,
            prop,
        }),
    }
}

fn read_prop_float(
    node: &KdlNode,
    node_name: &'static str,
    prop: &'static str,
) -> Result<f64, IrKdlError> {
    match node.get(prop) {
        Some(KdlValue::Float(f)) => Ok(*f),
        Some(KdlValue::Integer(i)) => Ok(*i as f64),
        Some(other) => Err(IrKdlError::InvalidValue {
            field: format!("{node_name}.{prop}"),
            message: format!("expected number, got {other:?}"),
        }),
        None => Err(IrKdlError::MissingProp {
            node: node_name,
            prop,
        }),
    }
}

fn read_optional_prop_int(node: &KdlNode, prop: &str) -> Result<Option<i128>, IrKdlError> {
    match node.get(prop) {
        Some(KdlValue::Integer(i)) => Ok(Some(*i)),
        Some(other) => Err(IrKdlError::InvalidValue {
            field: prop.to_string(),
            message: format!("expected integer, got {other:?}"),
        }),
        None => Ok(None),
    }
}

fn read_optional_prop_float(node: &KdlNode, prop: &str) -> Result<Option<f64>, IrKdlError> {
    match node.get(prop) {
        Some(KdlValue::Float(f)) => Ok(Some(*f)),
        Some(KdlValue::Integer(i)) => Ok(Some(*i as f64)),
        Some(other) => Err(IrKdlError::InvalidValue {
            field: prop.to_string(),
            message: format!("expected number, got {other:?}"),
        }),
        None => Ok(None),
    }
}

fn read_optional_prop_string(node: &KdlNode, prop: &str) -> Result<Option<String>, IrKdlError> {
    match node.get(prop) {
        Some(KdlValue::String(s)) => Ok(Some(s.clone())),
        Some(other) => Err(IrKdlError::InvalidValue {
            field: prop.to_string(),
            message: format!("expected string, got {other:?}"),
        }),
        None => Ok(None),
    }
}

fn read_prop_uuid(
    node: &KdlNode,
    node_name: &'static str,
    prop: &'static str,
) -> Result<Uuid, IrKdlError> {
    let s = read_prop_string(node, node_name, prop)?;
    Uuid::parse_str(&s).map_err(|source| IrKdlError::InvalidUuid { value: s, source })
}

fn read_prop_timestamp(
    node: &KdlNode,
    node_name: &'static str,
    prop: &'static str,
) -> Result<Timestamp, IrKdlError> {
    let s = read_prop_string(node, node_name, prop)?;
    chrono::DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|source| IrKdlError::InvalidTimestamp { value: s, source })
}

fn read_first_arg_string(node: &KdlNode, name: &'static str) -> Result<String, IrKdlError> {
    let entry = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .ok_or(IrKdlError::MissingArg(name))?;
    match entry.value() {
        KdlValue::String(s) => Ok(s.clone()),
        other => Err(IrKdlError::InvalidValue {
            field: name.to_string(),
            message: format!("expected string, got {other:?}"),
        }),
    }
}

fn timestamp_to_string(ts: Timestamp) -> String {
    ts.to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SCHEMA_VERSION;

    fn fixed_ts() -> Timestamp {
        chrono::TimeZone::with_ymd_and_hms(&chrono::Utc, 2026, 4, 29, 12, 0, 0).unwrap()
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
        let pe4 = PageExtraction {
            page_uuid: PageId::new(),
            page: 4,
            extraction_uuid: extr_pass1.uuid,
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
            extracted_pages: vec![pe1, pe2, pe3, pe4],
            toc: vec![toc_root],
        }
    }

    #[test]
    fn round_trip_doc() {
        let doc = build_doc();
        let kdl = doc.to_kdl_string();
        let parsed = Doc::from_kdl_str(&kdl)
            .unwrap_or_else(|e| panic!("parse failed: {e}\n--- KDL ---\n{kdl}"));
        assert_eq!(doc, parsed, "round-trip mismatch\n--- KDL ---\n{kdl}");
    }

    #[test]
    fn empty_doc_round_trips() {
        let doc = Doc {
            schema_version: SCHEMA_VERSION,
            content_hash: ContentHash("deadbeef".into()),
            extractions: Vec::new(),
            extracted_pages: Vec::new(),
            toc: Vec::new(),
        };
        let kdl = doc.to_kdl_string();
        let parsed = Doc::from_kdl_str(&kdl).unwrap();
        assert_eq!(doc, parsed);
    }

    #[test]
    fn complex_param_uses_json_typed_string() {
        let extr = Extraction {
            uuid: ExtractionId::new(),
            model: ModelId("test@v1".into()),
            prompt: PromptText("...".into()),
            params: BTreeMap::from([(
                "stop_tokens".into(),
                serde_json::json!(["<|end|>", "###"]),
            )]),
            created_at: fixed_ts(),
        };
        let doc = Doc {
            schema_version: SCHEMA_VERSION,
            content_hash: ContentHash("abc".into()),
            extractions: vec![extr.clone()],
            extracted_pages: Vec::new(),
            toc: Vec::new(),
        };
        let kdl = doc.to_kdl_string();
        assert!(
            kdl.contains("(json)"),
            "complex param should use (json) type annotation:\n{kdl}"
        );
        let parsed = Doc::from_kdl_str(&kdl).unwrap();
        assert_eq!(doc, parsed);
    }
}
