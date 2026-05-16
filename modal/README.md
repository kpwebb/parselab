# Ferrite Modal workers

Python apps that run the two-pass PDF extraction pipeline on Modal serverless
GPUs. The Rust workspace's `extractor-client` (FER-83) calls these.

```
modal/
├── pyproject.toml          # uv-managed local deps (driver only; the worker
│                           # image installs torch/transformers itself)
├── glm_ocr/app.py            # Pass 1 — GLM-OCR (FER-81)
├── infinity_parser2/app.py   # Pass 2 — Infinity-Parser2-Pro (FER-102)
├── docling/app.py            # Docling pipeline eval worker (FER-125)
├── granite_docling/app.py    # Granite-Docling-258M VLM eval worker (FER-126)
├── shared/
│   ├── pdf_utils.py          # PDF → page images via PyMuPDF
│   └── wire.py               # FER-82 envelope (pydantic, mirrors the Rust IR)
└── harness/
    ├── run_docling.py        # FER-125 corpus driver for the docling pipeline worker
    └── run_granite_docling.py  # FER-126 corpus driver for the granite-docling VLM
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for dependency management
- A Modal account; install the CLI and authenticate:

  ```sh
  uv tool install modal
  modal token new
  ```

## Setup

```sh
cd modal
uv sync
```

This installs the local driver deps (`modal`, `pymupdf`, `pillow`,
`pydantic`). The GPU image installs its own torch/transformers/accelerate at
container build time — nothing extra to do locally.

## Deploy GLM-OCR (Pass 1)

```sh
cd modal
modal deploy glm_ocr/app.py
```

First deploy creates the `ferrite-hf-cache` volume and provisions an L40S
container; first invocation downloads the ~1.8GB weights into the volume.
Subsequent cold starts skip the download.

The deployed app exposes:

- `GlmOcr.extract(pdf_bytes, pages=None, prompt="Text Recognition:", dpi=200, max_new_tokens=8192)`
  — returns the FER-82 envelope as a JSON-serializable dict
- `GlmOcr.health()` — liveness probe; returns `{model, loaded}`

## Smoke test

```sh
cd modal
modal run glm_ocr/app.py --pdf-path ../data/corpus/<part>/datasheet.pdf
```

Limit to specific pages (zero-indexed):

```sh
modal run glm_ocr/app.py --pdf-path datasheet.pdf --pages 0,3,12
```

The driver prints the full envelope JSON to stdout. Per-page wall-clock
elapsed and total timings are printed by the worker (visible in `modal logs`).

## Cost monitoring

Modal logs GPU seconds per invocation in the dashboard. The worker also prints
per-page wall-clock elapsed and total time to stdout — pipe `modal logs
ferrite-glm-ocr` through your favorite parser to aggregate.

## Deploy Infinity-Parser2-Pro (Pass 2)

```sh
cd modal
modal deploy infinity_parser2/app.py
```

Pass 2 runs on a single H100. First cold start downloads ~70GB of weights
into the shared `ferrite-hf-cache` volume (one-time); subsequent starts mount
and skip the download.

The deployed app exposes:

- `InfinityParser2.extract(pdf_bytes, pages, prompt="Extract layout with bboxes as JSON", dpi=200, max_new_tokens=32768)`
  — returns the FER-82 envelope; per-page entries are `format_type:
  structured_json` with blocks (bbox + category + text). `pages` is required.
- `InfinityParser2.health()` — liveness probe.

## Smoke test (Pass 2)

Pass 2 is always page-targeted, so explicit zero-indexed pages are required:

```sh
cd modal
modal run infinity_parser2/app.py --pdf-path ../data/corpus/mmbt3904.pdf --pages 1
```

Bboxes in the wire output are in PDF points (xywh, page-relative), already
rescaled from the model's pixel-space xyxy output. The worker maps the
model's `category` strings 1:1 to the IR's `StructuredBlock.kind`.

## Wire format

Every worker returns the same envelope shape (FER-82). The `format_type`
field on each page result discriminates the `content` payload:

```jsonc
// Pass 1 (GLM-OCR)
{
  "page": 0,
  "format_type": "markdown",
  "content": { "markdown": "# …" }
}

// Pass 2 (Infinity-Parser2-Pro)
{
  "page": 1,
  "format_type": "structured_json",
  "content": {
    "blocks": [
      {
        "kind": "table",
        "bbox": { "x": 72.0, "y": 120.0, "w": 450.0, "h": 200.0 },
        "text": "Param | Min | Typ | Max | …"
      }
    ],
    "page_meta": { "width_pts": 612.0, "height_pts": 792.0, "rotation_deg": 0, "dpi": 200 }
  }
}
```

The full envelope wraps these:

```jsonc
{
  "schema_version": 1,
  "extraction_uuid": "<uuid>",
  "model": "glm-ocr@v1" | "infinity-parser2-pro@v1",
  "prompt": "…",
  "created_at": "2026-04-29T…Z",
  "pages": [ /* Pass 1 or Pass 2 entries */ ]
}
```

## Deploy docling (FER-125 spike)

Docling is IBM's layout-aware PDF parser — local pipeline, not an LLM
server. Unlike Pass 1 / Pass 2, the worker is an `@app.cls` exposing a
`@modal.method()` rather than an OpenAI-compatible HTTP endpoint, so
the eval harness dispatches via the Modal SDK.

```sh
cd modal
modal deploy docling/app.py
```

L40S (mirrors GLM-OCR for cost-comparable numbers), `min_containers=1`
warm, shares the `ferrite-hf-cache` volume.

The deployed class exposes:

- `DoclingExtractor.extract(pdf_bytes, pages=None)` — returns
  `{pages, raw_document, elapsed_ms, docling_version}`. `pages` is our
  normalized per-page envelope (markdown + items with bboxes);
  `raw_document` is the full `DoclingDocument.export_to_dict()` so we
  can review what docling natively produces.

### Smoke test (docling)

```sh
cd modal
modal run docling/app.py --pdf-path ../data/corpus/ao3400a.pdf
modal run docling/app.py --pdf-path ../data/corpus/stm32f411ce.pdf --pages 1,2,3
```

### Run the docling corpus harness

Drives the deployed worker across the FER-86 corpus and writes outputs
to `target/docling-runs/<utc>/`:

- `envelopes/<part_id>.json` — normalized per-page envelopes (diff against
  the Rust harness's Pass 1 / Pass 2 envelopes in `target/harness-runs/`).
- `raw/<part_id>.json` — raw `DoclingDocument.export_to_dict()` for review.
- `docling.csv` — per-page metrics.
- `summary.txt` — per-PDF rollup + totals.

```sh
cd modal
uv run python -m harness.run_docling                        # whole corpus, 10 pages each
uv run python -m harness.run_docling --pages 4              # first 4 pages each
uv run python -m harness.run_docling --parts ao3400a tps562200
uv run python -m harness.run_docling --concurrency 2
```

## Deploy granite-docling-258M (FER-126 spike)

Granite-Docling-258M is IBM's compact document VLM (Idefics3-based:
siglip2-base-patch16-512 vision encoder + Granite 165M decoder, ~258M
params total). Successor to SmolDocling. End-to-end image → DocTags;
parses to a `DoclingDocument` via `docling-core`.

Hosted as a vLLM OpenAI-compatible server (SGLang's generic multimodal
processor hits an internal bug on Idefics3 — `Modality.MULTI_IMAGES`
attribute error — and the model card recommends vLLM specifically). The
harness POSTs `{image, prompt}` to `/v1/chat/completions` and gets a
DocTags string back; the OpenAI surface is identical to Pass 1 / Pass 2.

```sh
cd modal
modal deploy granite_docling/app.py
```

L40S (matches Pass 1 for cost-comparable BENCHMARKS.md numbers, even
though the model is tiny), `min_containers=1` warm, `--revision untied`
to avoid the tied-weights serving issue called out on the model card.

The deployed app exposes:

- `POST https://ferrite-systems--ferrite-granite-docling-serve.modal.run/v1/chat/completions`
  — standard OpenAI chat-completions with image_url + text content.
  Returns DocTags in `choices[0].message.content`.

### Run the granite-docling corpus harness

Drives the deployed worker across the FER-86 corpus and writes outputs
to `target/granite-docling-runs/<utc>/`:

- `envelopes/<part_id>.json` — normalized per-page envelope (markdown +
  items + bboxes), parsed via `docling-core`.
- `raw/<part_id>/page_<N>.json` — the raw DocTags string + parsed
  `DoclingDocument` per page.
- `granite_docling.csv` — per-page metrics.
- `summary.txt` — per-PDF rollup + totals.

```sh
cd modal
uv run python -m harness.run_granite_docling                       # whole corpus, 10 pages each
uv run python -m harness.run_granite_docling --pages 4
uv run python -m harness.run_granite_docling --parts ao3400a tps562200
uv run python -m harness.run_granite_docling --concurrency 16
```

## What's next

- FER-80: spike harness that drives both workers across the corpus and
  produces a comparison report
