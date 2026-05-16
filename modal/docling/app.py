"""Docling Modal worker — IBM's layout-aware PDF parser as a callable Modal class.

Unlike the SGLang-backed Pass 1 / Pass 2 workers (which expose
OpenAI-compatible chat-completions over HTTP), docling is a Python
pipeline: it ingests a PDF, runs layout + table-structure models
locally, and emits a `DoclingDocument`. We expose it as an `@app.cls`
with a `@modal.method()` so the Python eval driver can dispatch
directly via the Modal SDK without HTTP/JSON glue. L40S to mirror the
GLM-OCR Pass 1 worker so per-page cost numbers fold cleanly into
BENCHMARKS.md.

Tracking issue: FER-125.

Deploy:

    cd modal
    modal deploy docling/app.py

Smoke test:

    cd modal
    modal run docling/app.py --pdf-path ../data/corpus/ao3400a.pdf
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import modal

app = modal.App("ferrite-docling")

image = (
    # cu13 + python 3.12 — same base ABI as the SGLang workers (FER-127).
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    # libgl1 / libglib2.0-0 — opencv runtime deps pulled in transitively
    # by docling's image preprocessing path.
    .apt_install("libnuma1", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "docling>=2.0",
        "pydantic>=2.0",
    )
)

# Reuse the shared Ferrite HF model cache. Docling pulls its layout
# (DocLayNet) + TableFormer artifacts from HF on first run; the volume
# means subsequent cold starts skip the download.
model_cache = modal.Volume.from_name("ferrite-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=120,
    # Single warm container — predictable cost/latency, mirrors GLM-OCR.
    max_containers=1,
    min_containers=1,
)
# Docling is a multi-stage pipeline (layout → tables → assembly) that
# already parallelizes internally. Keep request concurrency low so the
# GPU isn't oversubscribed; 2 in flight gives some interleave between a
# decoding-stage request and a layout-stage request without thrashing.
@modal.concurrent(max_inputs=2)
class DoclingExtractor:
    @modal.enter()
    def load(self) -> None:
        import os

        os.environ["HF_HOME"] = "/cache/hf"
        os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
        # NOTE: do *not* set `DOCLING_ARTIFACTS_PATH`. That env var tells
        # docling "models are pre-downloaded at this path, use them" —
        # not "download into this path." Pointing it at an empty dir
        # raises "is not valid. When defined, it must point to a folder
        # containing all models required by the pipeline." Leaving it
        # unset makes docling fall back to the standard HF cache, which
        # `HF_HOME=/cache/hf` already redirects to the persistent volume.
        # EasyOCR has its own cache (~/.EasyOCR by default) — point it
        # at the volume too so OCR weights persist across cold starts.
        os.environ["EASYOCR_MODULE_PATH"] = "/cache/easyocr"

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        # Cell-matching links table-structure cells back to original PDF
        # text spans — cheap, gives us better text fidelity in tables.
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA,
            num_threads=4,
        )

        self._pipeline_options = pipeline_options
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        print("docling loaded; accelerator=CUDA, ocr=on, table_structure=on")

    @modal.method()
    def extract(
        self,
        pdf_bytes: bytes,
        pages: list[int] | None = None,
    ) -> dict:
        """Convert a PDF and return per-page envelopes.

        Args:
            pdf_bytes: Raw PDF content.
            pages: Optional 1-indexed page list to restrict the run.
                Docling supports a contiguous `page_range`; if `pages` is
                non-contiguous, we run the min..max range and filter the
                output.

        Returns:
            dict with:
              - `pages`: per-page envelope `[{page_no, width, height,
                markdown, items}, ...]` (our normalized shape).
              - `raw_document`: full `DoclingDocument.export_to_dict()`
                so we can review what docling natively produces and
                compare its shape to the IR.
              - `elapsed_ms`, `docling_version`, `input_format`.
        """
        from docling.datamodel.base_models import DocumentStream

        stream = DocumentStream(name="input.pdf", stream=io.BytesIO(pdf_bytes))

        # Docling's contiguous-range filter is a `convert()` kwarg, not
        # a pipeline option. If `pages` is non-contiguous, we run
        # min..max and filter the output below via `wanted`.
        convert_kwargs: dict = {}
        if pages:
            convert_kwargs["page_range"] = (min(pages), max(pages))

        t0 = time.monotonic()
        result = self._converter.convert(stream, **convert_kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        doc = result.document
        wanted = set(pages) if pages else None
        pages_out = _build_per_page_output(doc, wanted=wanted)
        raw_document = _export_raw_document(doc)

        import docling

        return {
            "docling_version": getattr(docling, "__version__", "unknown"),
            "pages": pages_out,
            "raw_document": raw_document,
            "elapsed_ms": elapsed_ms,
            "input_format": "pdf",
        }

def _build_per_page_output(doc, wanted: set[int] | None):
    """Walk a DoclingDocument and return per-page envelopes.

    Each envelope:
      - page_no: 1-indexed
      - width, height: PDF-page coordinates (None if unavailable)
      - markdown: per-page markdown export (best-effort; falls back to
        empty if docling's API doesn't support page-filtered export in
        this version)
      - items: list of structured items, each with label, text, bbox
    """
    pages_out = []

    page_keys = sorted(doc.pages.keys()) if getattr(doc, "pages", None) else []
    if wanted is not None:
        page_keys = [p for p in page_keys if p in wanted]

    # Bin items per page in a single pass — `iterate_items` walks the
    # full doc, so doing this once is cheaper than re-walking per page.
    items_by_page: dict[int, list[dict]] = {p: [] for p in page_keys}
    for item, _level in doc.iterate_items():
        provs = getattr(item, "prov", None) or []
        for prov in provs:
            page_no = getattr(prov, "page_no", None)
            if page_no is None or page_no not in items_by_page:
                continue
            items_by_page[page_no].append(
                {
                    "label": _label_str(item),
                    "self_ref": getattr(item, "self_ref", None),
                    "text": getattr(item, "text", None),
                    "bbox": _bbox_dict(getattr(prov, "bbox", None)),
                }
            )

    for page_no in page_keys:
        page_item = doc.pages[page_no]
        size = getattr(page_item, "size", None)
        width = getattr(size, "width", None) if size else None
        height = getattr(size, "height", None) if size else None

        markdown = _export_page_markdown(doc, page_no)

        pages_out.append(
            {
                "page_no": page_no,
                "width": width,
                "height": height,
                "markdown": markdown,
                "items": items_by_page.get(page_no, []),
            }
        )

    return pages_out


def _export_raw_document(doc) -> dict:
    """Full `DoclingDocument` as a dict.

    Prefer `model_dump(mode="json")` (pydantic v2) so nested types like
    enums and bboxes serialize cleanly. Fall back to `export_to_dict()`
    if available, else an empty dict — we don't want a serialization
    quirk to fail the whole extract.
    """
    try:
        return doc.model_dump(mode="json")
    except Exception:
        pass
    try:
        return doc.export_to_dict()
    except Exception as e:
        return {"error": f"raw_document export failed: {e!r}"}


def _export_page_markdown(doc, page_no: int) -> str:
    """Best-effort per-page markdown export.

    Docling's `export_to_markdown` accepts a `page_no` filter in 2.x; if
    the running version doesn't, return empty rather than dump the whole
    document into every page (the harness can still rely on `items`).
    """
    try:
        return doc.export_to_markdown(page_no=page_no)
    except TypeError:
        return ""


def _label_str(item) -> str | None:
    label = getattr(item, "label", None)
    if label is None:
        return None
    val = getattr(label, "value", None)
    return val if val is not None else str(label)


def _bbox_dict(bbox):
    if bbox is None:
        return None
    return {
        "l": getattr(bbox, "l", None),
        "t": getattr(bbox, "t", None),
        "r": getattr(bbox, "r", None),
        "b": getattr(bbox, "b", None),
        "coord_origin": str(getattr(bbox, "coord_origin", "")),
    }


@app.local_entrypoint()
def smoke(pdf_path: str, pages: str = "") -> None:
    """Local smoke test.

    Example:
        modal run docling/app.py --pdf-path ../data/corpus/ao3400a.pdf
        modal run docling/app.py --pdf-path ../data/corpus/stm32f411ce.pdf --pages 1,2,3
    """
    page_list = (
        [int(p) for p in pages.split(",") if p.strip()] if pages else None
    )
    import json as _json

    pdf_bytes = Path(pdf_path).read_bytes()
    extractor = DoclingExtractor()
    result = extractor.extract.remote(pdf_bytes, pages=page_list)
    raw_size = len(_json.dumps(result.get("raw_document") or {}))
    print(f"docling_version: {result['docling_version']}")
    print(f"elapsed_ms:      {result['elapsed_ms']:.1f}")
    print(f"pages:           {len(result['pages'])}")
    print(f"raw_doc_bytes:   {raw_size}")
    for p in result["pages"]:
        md = (p["markdown"] or "").strip()
        snippet = md[:120].replace("\n", " ")
        w = p["width"] or 0.0
        h = p["height"] or 0.0
        print(
            f"  p{p['page_no']:>3}  {w:>5.0f}x{h:>5.0f}  "
            f"items={len(p['items']):>3}  md={snippet!r}"
        )
