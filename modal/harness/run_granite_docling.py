"""Run the deployed `ferrite-granite-docling` SGLang worker against the FER-86 corpus.

Per-page dispatch (the model takes one page image at a time):

    PDF → PyMuPDF render → PNG → base64 → /v1/chat/completions
        → DocTags string → docling_core parse → DoclingDocument

Output mirrors `harness/run_docling.py` so all three runs (FER-125
docling pipeline, this granite-docling VLM, plus existing Pass 1 / Pass
2 envelopes) are diff-able with the same tooling:

    target/granite-docling-runs/<utc>/
        envelopes/<part_id>.json       # normalized per-page envelope
        raw/<part_id>/page_<N>.json    # raw DoclingDocument per page
        granite_docling.csv            # per-page metrics
        summary.txt                    # per-PDF rollup

Tracking issue: FER-126.

Prerequisite: the worker is deployed (`modal deploy granite_docling/app.py`).

Usage (from `modal/`):

    uv run python -m harness.run_granite_docling                    # whole corpus, 10 pages each
    uv run python -m harness.run_granite_docling --pages 4
    uv run python -m harness.run_granite_docling --parts ao3400a tps562200
    uv run python -m harness.run_granite_docling --concurrency 16
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Resolve workspace root from this file's location: modal/harness/run_granite_docling.py
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = WORKSPACE_ROOT / "data" / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.toml"
RUNS_ROOT = WORKSPACE_ROOT / "target" / "granite-docling-runs"

# Mirror PASS1_PAGE_LIMIT in the Rust harness so granite-docling and
# Pass 1 cover the same first-N-pages slice.
DEFAULT_PAGE_LIMIT = 10

# Matches the worker's @modal.concurrent(max_inputs=16) and the Pass 1
# Rust harness dispatch concurrency.
DEFAULT_CONCURRENCY = 16

DEFAULT_DPI = 200
# Output token cap. The worker's `--max-model-len 8192` matches the
# model's trained `max_position_embeddings`, so we can't expand total
# context — we have to split it between input and output. Observed
# input across the FER-86 corpus: 878–1142 tokens (image patches +
# text prompt). 7000 output + 1142 input = 8142 fits, with ~50 tokens
# margin. The earlier 4096 cap truncated dense MCU pages mid-DocTags-
# stream (stm32f411ce p3–p10 returned 1 item each because the parser
# couldn't recover from a missing `</doctag>` close).
DEFAULT_MAX_TOKENS = 7000
DEFAULT_PROMPT = "Convert this page to docling."

ENDPOINT_URL = (
    "https://ferrite-systems--ferrite-granite-docling-serve.modal.run/v1/chat/completions"
)
MODEL_ID = "ibm-granite/granite-docling-258M"


@dataclass
class Part:
    part_id: str
    file: str
    cls: str
    vendor: str
    part_number: str
    pages: int


@dataclass
class PageTask:
    part: Part
    page_no: int  # 1-indexed
    png_bytes: bytes
    width_px: int
    height_px: int


@dataclass
class PageResult:
    part_id: str
    page_no: int
    width_px: int
    height_px: int
    elapsed_secs: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    doctags: str | None = None
    docling_doc: dict | None = None
    markdown: str | None = None
    items: list[dict] = field(default_factory=list)
    error: str | None = None


def load_manifest() -> dict[str, Part]:
    raw = tomllib.loads(MANIFEST_PATH.read_text())
    return {
        pid: Part(
            part_id=pid,
            file=body["file"],
            cls=body["class"],
            vendor=body["vendor"],
            part_number=body["part_number"],
            pages=int(body["pages"]),
        )
        for pid, body in raw.items()
    }


def render_pages(part: Part, page_limit: int, dpi: int) -> list[PageTask]:
    """Render the first `page_limit` pages of `part` to PNG."""
    import pymupdf  # PyMuPDF; lazy import — only the harness needs it

    pdf_path = CORPUS_DIR / part.file
    tasks: list[PageTask] = []
    with pymupdf.open(pdf_path) as doc:
        n = min(page_limit, doc.page_count)
        scale = dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)
        for i in range(n):
            pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
            tasks.append(
                PageTask(
                    part=part,
                    page_no=i + 1,
                    png_bytes=pix.tobytes("png"),
                    width_px=pix.width,
                    height_px=pix.height,
                )
            )
    return tasks


def build_chat_request(task: PageTask, prompt: str, max_tokens: int) -> dict:
    b64 = base64.b64encode(task.png_bytes).decode("ascii")
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # Granite-docling emits DocTags using element-type *special tokens*
        # (`<text>`, `<title>`, `<table>`, ...) that wrap each region's
        # `<loc_*>` coordinates and content. vLLM's OpenAI chat output
        # defaults to skip_special_tokens=true, which strips these
        # wrappers and leaves only bare-coordinate text — which docling-
        # core can't parse (every page returns 0 items). Force-keep them.
        "skip_special_tokens": False,
    }


def parse_doctags(doctags: str, png_bytes: bytes) -> tuple[dict, str, list[dict]]:
    """DocTags + source image → (raw DoclingDocument dict, markdown, items list).

    `docling_core` needs the source image so DocTags' image-pixel
    coordinates can be resolved into the document's bbox space.
    """
    from PIL import Image
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    image = Image.open(io.BytesIO(png_bytes))
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    # `load_from_doctags` is a @staticmethod that *returns* the populated
    # document — it doesn't mutate `self`. Calling it on a constructed
    # stub and ignoring the return value left us inspecting an empty doc.
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Page")

    markdown = ""
    try:
        markdown = doc.export_to_markdown()
    except Exception:
        pass

    items: list[dict] = []
    for item, _level in doc.iterate_items():
        for prov in getattr(item, "prov", None) or []:
            bbox = getattr(prov, "bbox", None)
            label = getattr(item, "label", None)
            label_str = (
                getattr(label, "value", None) if label is not None else None
            ) or (str(label) if label is not None else None)
            items.append(
                {
                    "label": label_str,
                    "self_ref": getattr(item, "self_ref", None),
                    "text": getattr(item, "text", None),
                    "bbox": (
                        {
                            "l": getattr(bbox, "l", None),
                            "t": getattr(bbox, "t", None),
                            "r": getattr(bbox, "r", None),
                            "b": getattr(bbox, "b", None),
                            "coord_origin": str(getattr(bbox, "coord_origin", "")),
                        }
                        if bbox is not None
                        else None
                    ),
                }
            )

    try:
        raw = doc.model_dump(mode="json")
    except Exception:
        raw = doc.export_to_dict()
    return raw, markdown, items


def dispatch_one(
    client: httpx.Client, task: PageTask, prompt: str, max_tokens: int
) -> PageResult:
    body = build_chat_request(task, prompt, max_tokens)
    started = time.monotonic()
    try:
        resp = client.post(ENDPOINT_URL, json=body, timeout=300.0)
        resp.raise_for_status()
        payload = resp.json()
        elapsed = time.monotonic() - started

        choice = payload["choices"][0]
        doctags = choice["message"]["content"]
        usage = payload.get("usage") or {}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")

        raw, md, items = parse_doctags(doctags, task.png_bytes)
        return PageResult(
            part_id=task.part.part_id,
            page_no=task.page_no,
            width_px=task.width_px,
            height_px=task.height_px,
            elapsed_secs=elapsed,
            input_tokens=in_tok,
            output_tokens=out_tok,
            doctags=doctags,
            docling_doc=raw,
            markdown=md,
            items=items,
        )
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - started
        return PageResult(
            part_id=task.part.part_id,
            page_no=task.page_no,
            width_px=task.width_px,
            height_px=task.height_px,
            elapsed_secs=elapsed,
            error=f"{type(e).__name__}: {e}",
        )


def write_envelope(envelopes_dir: Path, part: Part, results: list[PageResult]) -> None:
    pages = [
        {
            "page_no": r.page_no,
            "width_px": r.width_px,
            "height_px": r.height_px,
            "elapsed_secs": r.elapsed_secs,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "markdown": r.markdown,
            "items": r.items,
            "error": r.error,
        }
        for r in results
    ]
    envelope = {
        "part_id": part.part_id,
        "vendor": part.vendor,
        "part_number": part.part_number,
        "class": part.cls,
        "extractor": "granite-docling-258M",
        "model": MODEL_ID,
        "endpoint": ENDPOINT_URL,
        "pages": pages,
    }
    (envelopes_dir / f"{part.part_id}.json").write_text(json.dumps(envelope, indent=2))


def write_raw(raw_root: Path, part: Part, results: list[PageResult]) -> None:
    """One raw DoclingDocument per page (granite-docling is per-page)."""
    part_dir = raw_root / part.part_id
    part_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if r.docling_doc is None:
            continue
        out = {
            "doctags": r.doctags,
            "docling_document": r.docling_doc,
        }
        (part_dir / f"page_{r.page_no:03d}.json").write_text(
            json.dumps(out, indent=2)
        )


def write_csv_rows(
    writer: csv.writer, part: Part, results: list[PageResult]
) -> None:
    for r in results:
        writer.writerow(
            [
                part.part_id,
                part.vendor,
                part.part_number,
                part.cls,
                r.page_no,
                "granite-docling",
                f"{r.elapsed_secs:.3f}",
                r.input_tokens if r.input_tokens is not None else "",
                r.output_tokens if r.output_tokens is not None else "",
                len(r.items),
                len(r.markdown or ""),
                "error" if r.error else "ok",
            ]
        )


def write_summary(
    path: Path,
    by_part: dict[str, list[PageResult]],
    page_limit: int,
    concurrency: int,
    wall_secs: float,
) -> None:
    lines: list[str] = []
    lines.append("FER-126 granite-docling corpus run")
    lines.append("==================================================")
    lines.append("")
    lines.append(f"page_limit={page_limit}  concurrency={concurrency}")
    lines.append(f"wall_secs={wall_secs:.1f}")
    lines.append("")
    header = (
        f"{'part_id':<24} {'part_number':<22} {'pdf_p':>6} {'ext_p':>6} "
        f"{'errs':>6} {'sum_req_s':>10} {'in_tok':>10} {'out_tok':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    total_pages = 0
    total_errors = 0
    total_in = 0
    total_out = 0
    total_sum_req = 0.0

    for part_id, results in sorted(by_part.items()):
        in_tok = sum(r.input_tokens or 0 for r in results)
        out_tok = sum(r.output_tokens or 0 for r in results)
        errs = sum(1 for r in results if r.error)
        sum_req = sum(r.elapsed_secs for r in results)
        ext_p = sum(1 for r in results if not r.error)
        total_pages += ext_p
        total_errors += errs
        total_in += in_tok
        total_out += out_tok
        total_sum_req += sum_req
        # part_number/pdf_pages live in the run report as best-effort —
        # the manifest is the source of truth, but we don't need it for
        # totals. Print part_id where part_number normally goes.
        lines.append(
            f"{part_id:<24} {'':<22} {'':>6} {ext_p:>6} {errs:>6} "
            f"{sum_req:>10.1f} {in_tok:>10} {out_tok:>10}"
        )

    lines.append("")
    pages_per_sec = total_pages / wall_secs if wall_secs > 0 else 0.0
    lines.append(
        f"totals: parts={len(by_part)} extracted_pages={total_pages} "
        f"errors={total_errors} sum_req_secs={total_sum_req:.1f} "
        f"in_tok={total_in} out_tok={total_out} wall_pages_per_sec={pages_per_sec:.2f}"
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run granite-docling-258M against the FER-86 corpus"
    )
    p.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_PAGE_LIMIT,
        help="First-N-pages per PDF (default: %(default)s)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Parallel chat-completion requests (default: %(default)s)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Render DPI (default: %(default)s)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="DocTags output token cap (default: %(default)s)",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Per-page prompt sent alongside the image (default: %(default)r)",
    )
    p.add_argument(
        "--parts",
        nargs="*",
        default=[],
        help="Restrict to these part_ids (default: whole corpus)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    manifest = load_manifest()
    if args.parts:
        unknown = [p for p in args.parts if p not in manifest]
        if unknown:
            print(f"unknown part_ids: {unknown}", file=sys.stderr)
            print(f"manifest has: {sorted(manifest)}", file=sys.stderr)
            return 2
        active = [manifest[p] for p in args.parts]
    else:
        active = sorted(manifest.values(), key=lambda p: p.part_id)

    print(f"loaded manifest, running {len(active)} parts (page_limit={args.pages})")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_ROOT / run_id
    envelopes_dir = output_dir / "envelopes"
    raw_dir = output_dir / "raw"
    envelopes_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "granite_docling.csv"
    summary_path = output_dir / "summary.txt"

    # Phase 1: render every requested page upfront. Mirrors the Rust
    # extractor-harness's render-then-dispatch shape so SGLang stays
    # under sustained pressure rather than bursts-and-valleys.
    render_started = time.monotonic()
    all_tasks: list[PageTask] = []
    for part in active:
        tasks = render_pages(part, args.pages, args.dpi)
        all_tasks.extend(tasks)
        print(f"  rendered {part.part_id} ({len(tasks)}p)")
    render_secs = time.monotonic() - render_started
    print(f"\nrendered {len(all_tasks)} pages in {render_secs:.1f}s")

    # Phase 2: bulk dispatch.
    print(
        f"\ndispatching {len(all_tasks)} chat-completions, concurrency={args.concurrency}"
    )
    dispatch_started = time.monotonic()
    by_part: dict[str, list[PageResult]] = {p.part_id: [] for p in active}

    with httpx.Client(http2=False) as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {
                pool.submit(dispatch_one, client, t, args.prompt, args.max_tokens): t
                for t in all_tasks
            }
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                by_part[r.part_id].append(r)
                done += 1
                if r.error:
                    print(f"  [{done}/{len(all_tasks)}] {r.part_id} p{r.page_no}  ERROR  {r.error}")
                else:
                    print(
                        f"  [{done}/{len(all_tasks)}] {r.part_id} p{r.page_no}  "
                        f"items={len(r.items):>3}  "
                        f"in={r.input_tokens} out={r.output_tokens}  "
                        f"{r.elapsed_secs:.1f}s"
                    )

    dispatch_secs = time.monotonic() - dispatch_started

    # Phase 3: write outputs.
    for results in by_part.values():
        results.sort(key=lambda r: r.page_no)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "part_id",
                "vendor",
                "part_number",
                "class",
                "page",
                "format_type",
                "elapsed_secs",
                "input_tokens",
                "output_tokens",
                "item_count",
                "markdown_chars",
                "status",
            ]
        )
        for part in active:
            results = by_part[part.part_id]
            write_envelope(envelopes_dir, part, results)
            write_raw(raw_dir, part, results)
            write_csv_rows(writer, part, results)

    wall_secs = render_secs + dispatch_secs
    write_summary(
        summary_path,
        by_part,
        page_limit=args.pages,
        concurrency=args.concurrency,
        wall_secs=wall_secs,
    )

    print()
    print("=== run complete ===")
    print(f"wall: render {render_secs:.1f}s + dispatch {dispatch_secs:.1f}s = {wall_secs:.1f}s")
    print(f"output dir: {output_dir}")
    print(f"  envelopes: {envelopes_dir}")
    print(f"  raw:       {raw_dir}")
    print(f"  csv:       {csv_path}")
    print(f"  summary:   {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
