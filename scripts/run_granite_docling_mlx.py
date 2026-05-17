"""Run granite-docling-258M locally on Apple Silicon via MLX.

Mirrors `modal/harness/run_granite_docling.py` but drives the
MLX-converted weights from `ibm-granite/granite-docling-258M-mlx`
locally, no Modal involvement. Same envelope / raw / CSV / summary
shape so the output is directly diff-able with the vLLM run.

Prerequisites: macOS on Apple Silicon, the scripts venv installed:

    cd scripts
    uv sync

Usage:

    uv run python run_granite_docling_mlx.py                       # whole corpus
    uv run python run_granite_docling_mlx.py --pages 4
    uv run python run_granite_docling_mlx.py --parts ao3400a tps562200
    uv run python run_granite_docling_mlx.py --max-tokens 7000

Output: target/granite-docling-mlx-runs/<utc>/{envelopes,raw,csv,summary}.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Resolve workspace root from this file's location: scripts/run_granite_docling_mlx.py
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = WORKSPACE_ROOT / "data" / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.toml"
RUNS_ROOT = WORKSPACE_ROOT / "target" / "granite-docling-mlx-runs"

DEFAULT_PAGE_LIMIT = 10
DEFAULT_DPI = 200
# Match the vLLM driver's headroom budget. The MLX runtime doesn't
# enforce the same `max_model_len` rule as vLLM (it uses the model's
# own context window), so this is just an output-side cap.
DEFAULT_MAX_TOKENS = 7000
DEFAULT_PROMPT = "Convert this page to docling."

MODEL_PATH = "ibm-granite/granite-docling-258M-mlx"
# The MLX-converted repo doesn't ship `preprocessor_config.json` with
# an `image_processor_type` key, so transformers' AutoProcessor fails
# to load it. The upstream non-MLX repo has the full processor config;
# we load the processor (tokenizer + image processor) from there and
# the (quantized) weights from the MLX repo. Same architecture so the
# processor is interchangeable.
PROCESSOR_FALLBACK_PATH = "ibm-granite/granite-docling-258M"


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
    import pymupdf

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


def extract_doctags_text(generate_output) -> str:
    """mlx-vlm's `generate` return shape has shifted across versions:
    in some, a plain `str`; in others, a `(text, stats)` tuple; in
    newer ones, a `GenerationResult` with a `.text` attribute. Handle
    all three so this driver doesn't break on minor mlx-vlm bumps."""
    if isinstance(generate_output, str):
        return generate_output
    if isinstance(generate_output, tuple) and generate_output:
        first = generate_output[0]
        return first if isinstance(first, str) else str(first)
    text = getattr(generate_output, "text", None)
    if isinstance(text, str):
        return text
    return str(generate_output)


def parse_doctags(doctags: str, png_bytes: bytes) -> tuple[dict, str, list[dict]]:
    from PIL import Image
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    image = Image.open(io.BytesIO(png_bytes))
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    # `load_from_doctags` is a @staticmethod that *returns* the doc.
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


def generate_one(
    model,
    processor,
    config,
    formatted_prompt: str,
    task: PageTask,
    max_tokens: int,
) -> PageResult:
    from mlx_vlm import generate

    # mlx-vlm versions vary on whether `generate` accepts PIL.Image or
    # only file paths — temp files are the lowest-friction common
    # denominator and the cost is one fsync per page.
    started = time.monotonic()
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(task.png_bytes)
            tmp_path = tmp.name
        try:
            output = generate(
                model,
                processor,
                formatted_prompt,
                image=[tmp_path],
                max_tokens=max_tokens,
                verbose=False,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        elapsed = time.monotonic() - started
        doctags = extract_doctags_text(output)

        raw, md, items = parse_doctags(doctags, task.png_bytes)
        return PageResult(
            part_id=task.part.part_id,
            page_no=task.page_no,
            width_px=task.width_px,
            height_px=task.height_px,
            elapsed_secs=elapsed,
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
        "extractor": "granite-docling-258M-mlx",
        "model": MODEL_PATH,
        "runtime": "mlx-vlm",
        "pages": pages,
    }
    (envelopes_dir / f"{part.part_id}.json").write_text(json.dumps(envelope, indent=2))


def write_raw(raw_root: Path, part: Part, results: list[PageResult]) -> None:
    part_dir = raw_root / part.part_id
    part_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if r.docling_doc is None:
            continue
        out = {"doctags": r.doctags, "docling_document": r.docling_doc}
        (part_dir / f"page_{r.page_no:03d}.json").write_text(json.dumps(out, indent=2))


def write_csv_rows(writer, part: Part, results: list[PageResult]) -> None:
    for r in results:
        writer.writerow(
            [
                part.part_id,
                part.vendor,
                part.part_number,
                part.cls,
                r.page_no,
                "granite-docling-mlx",
                f"{r.elapsed_secs:.3f}",
                len(r.items),
                len(r.markdown or ""),
                "error" if r.error else "ok",
            ]
        )


def write_summary(
    path: Path,
    by_part: dict[str, list[PageResult]],
    page_limit: int,
    wall_secs: float,
    model_load_secs: float,
) -> None:
    lines: list[str] = []
    lines.append("Parselab granite-docling-mlx local run")
    lines.append("==================================================")
    lines.append("")
    lines.append(f"page_limit={page_limit}  runtime=mlx-vlm  model={MODEL_PATH}")
    lines.append(f"model_load_secs={model_load_secs:.1f}  wall_secs={wall_secs:.1f}")
    lines.append("")
    header = (
        f"{'part_id':<24} {'pdf_p':>6} {'ext_p':>6} {'errs':>6} {'sum_req_s':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    total_pages = 0
    total_errors = 0
    total_sum_req = 0.0
    for part_id, results in sorted(by_part.items()):
        errs = sum(1 for r in results if r.error)
        sum_req = sum(r.elapsed_secs for r in results)
        ext_p = sum(1 for r in results if not r.error)
        total_pages += ext_p
        total_errors += errs
        total_sum_req += sum_req
        lines.append(
            f"{part_id:<24} {len(results):>6} {ext_p:>6} {errs:>6} {sum_req:>10.1f}"
        )

    lines.append("")
    pages_per_sec = total_pages / wall_secs if wall_secs > 0 else 0.0
    lines.append(
        f"totals: parts={len(by_part)} extracted_pages={total_pages} "
        f"errors={total_errors} sum_req_secs={total_sum_req:.1f} "
        f"wall_pages_per_sec={pages_per_sec:.2f}"
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run granite-docling-258M (MLX) locally against the corpus"
    )
    p.add_argument("--pages", type=int, default=DEFAULT_PAGE_LIMIT)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--parts", nargs="*", default=[])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # Defer mlx-vlm import so `--help` works on non-Apple-Silicon
    # machines without crashing.
    try:
        from mlx_vlm import load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import (
            get_model_path,
            load_config,
            load_image_processor,
            load_model,
            load_processor,
        )
    except ImportError as e:
        print(
            "mlx-vlm is not installed. From the workspace root:\n"
            "    cd scripts && uv sync\n"
            f"(import error: {e})",
            file=sys.stderr,
        )
        return 2

    manifest = load_manifest()
    if args.parts:
        unknown = [p for p in args.parts if p not in manifest]
        if unknown:
            print(f"unknown part_ids: {unknown}", file=sys.stderr)
            return 2
        active = [manifest[p] for p in args.parts]
    else:
        active = sorted(manifest.values(), key=lambda p: p.part_id)

    print(f"loaded manifest, running {len(active)} parts (page_limit={args.pages})")

    # Phase 0: model load. Try the standard `load()`; fall back to a
    # split load (weights from MLX repo, processor manually constructed
    # from the upstream non-MLX repo's configs) if AutoProcessor's
    # auto-resolution can't pick an image processor for granite-docling.
    print(f"\nloading {MODEL_PATH} via mlx-vlm ...")
    load_started = time.monotonic()
    try:
        model, processor = load(MODEL_PATH)
    except ValueError as e:
        if "Unrecognized image processor" not in str(e):
            raise
        print(
            "  AutoProcessor couldn't resolve the image processor; "
            f"constructing manually from {PROCESSOR_FALLBACK_PATH}"
        )
        from transformers import (
            AutoTokenizer,
            Idefics3ImageProcessor,
            Idefics3Processor,
        )

        mlx_local = get_model_path(MODEL_PATH)
        upstream_local = get_model_path(PROCESSOR_FALLBACK_PATH)
        model = load_model(mlx_local, lazy=False)
        image_processor = Idefics3ImageProcessor.from_pretrained(upstream_local)
        tokenizer = AutoTokenizer.from_pretrained(mlx_local)
        processor = Idefics3Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

    config = load_config(MODEL_PATH)
    model_load_secs = time.monotonic() - load_started
    print(f"  loaded in {model_load_secs:.1f}s")

    formatted_prompt = apply_chat_template(processor, config, args.prompt, num_images=1)

    # Phase 1: render every requested page.
    print()
    render_started = time.monotonic()
    all_tasks: list[PageTask] = []
    for part in active:
        tasks = render_pages(part, args.pages, args.dpi)
        all_tasks.extend(tasks)
        print(f"  rendered {part.part_id} ({len(tasks)}p)")
    render_secs = time.monotonic() - render_started
    print(f"\nrendered {len(all_tasks)} pages in {render_secs:.1f}s")

    # Phase 2: sequential dispatch (MLX is single-stream — concurrency
    # would oversubscribe the GPU and not help wall time).
    print(f"\ngenerating {len(all_tasks)} pages sequentially")
    by_part: dict[str, list[PageResult]] = {p.part_id: [] for p in active}
    dispatch_started = time.monotonic()
    for i, task in enumerate(all_tasks, 1):
        r = generate_one(model, processor, config, formatted_prompt, task, args.max_tokens)
        by_part[r.part_id].append(r)
        if r.error:
            print(f"  [{i}/{len(all_tasks)}] {r.part_id} p{r.page_no}  ERROR  {r.error}")
        else:
            print(
                f"  [{i}/{len(all_tasks)}] {r.part_id} p{r.page_no}  "
                f"items={len(r.items):>3}  md={len(r.markdown or ''):>5}  "
                f"{r.elapsed_secs:.1f}s"
            )
    dispatch_secs = time.monotonic() - dispatch_started

    # Phase 3: write outputs.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_ROOT / run_id
    envelopes_dir = output_dir / "envelopes"
    raw_dir = output_dir / "raw"
    envelopes_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "granite_docling_mlx.csv"
    summary_path = output_dir / "summary.txt"

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
        wall_secs=wall_secs,
        model_load_secs=model_load_secs,
    )

    print()
    print("=== run complete ===")
    print(f"model load: {model_load_secs:.1f}s")
    print(f"wall: render {render_secs:.1f}s + dispatch {dispatch_secs:.1f}s = {wall_secs:.1f}s")
    print(f"output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
