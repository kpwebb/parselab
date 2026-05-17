use thiserror::Error;

/// Errors surfaced by the extractor-client trait. Distinguishes transport
/// failures from worker failures from envelope-shape issues so callers
/// (orchestrator, cache, harness) can react appropriately — e.g. retry on
/// transport, surface to user on malformed PDF, escalate on schema drift.
#[derive(Debug, Error)]
pub enum Error {
    /// HTTP/transport failure (network unreachable, TLS error, timeout).
    /// Idempotent retries are safe per FER-83.
    #[error("transport: {0}")]
    Transport(#[from] reqwest::Error),

    /// Worker returned a non-2xx status. Body captured for diagnostics —
    /// FastAPI typically wraps exceptions in `{"detail": "..."}`.
    #[error("worker http {status}: {body}")]
    HttpStatus { status: u16, body: String },

    /// 2xx response but the payload didn't deserialize as the FER-82
    /// envelope. Indicates worker / wire schema drift.
    #[error("malformed envelope: {0}")]
    MalformedEnvelope(#[from] serde_json::Error),

    /// `pages` argument was empty for Pass 2, which always requires an
    /// explicit page set.
    #[error("pass2 requires at least one page")]
    EmptyPagesForPass2,

    /// PDF rendering failed (load, page get, bitmap → PNG, or pdfium
    /// binding lookup). The string carries the detail.
    #[error("render: {0}")]
    Render(String),

    /// Pass 2 layout-JSON parse failure (FER-112). The model returned
    /// 2xx but the body wasn't decodable as the expected layout-element
    /// shape. Per-page parse failures usually surface as
    /// `Content::Error { kind = "parse_error" }` so the rest of the
    /// extraction is preserved; this variant is for cases where the
    /// failure is fatal to the call.
    #[error("layout parse: {0}")]
    LayoutParse(String),

    /// FER-103 ToC builder failure — model emitted JSON the parser
    /// couldn't decode, or the response had no extractable rows. The
    /// caller can fall back to the heuristic builder.
    #[error("toc build: {0}")]
    TocBuild(String),

    /// Worker returned 200 OK but the response payload was missing a
    /// required field (e.g. `choices[0].message.content`). Distinguished
    /// from `HttpStatus` so callers can tell "worker rejected the
    /// request" from "worker accepted but emitted nothing".
    #[error("worker response: {0}")]
    WorkerResponse(String),
}
