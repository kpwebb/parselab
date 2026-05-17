"""Run the deployed `parselab-docling` Modal worker against the corpus.

Walks `data/corpus/manifest.toml`, dispatches one extract per PDF to
the Modal class, and writes:

    target/docling-runs/<utc>/
        docling.csv             # per-page metrics
        envelopes/<part_id>.json   # our normalized per-page envelope
        raw/<part_id>.json      # raw DoclingDocument.export_to_dict()
        summary.txt             # per-PDF rollup

`raw/` exists so we can review what docling natively produces — it's
the "what does docling do" view; `envelopes/` is the normalized view
that diffs cleanly against other extractors.

Prerequisite: the worker is deployed (`modal deploy docling/app.py`).

Usage (from `modal/`):

    uv run python -m harness.run_docling                    # whole corpus, 10 pages each
    uv run python -m harness.run_docling --pages 4          # first 4 pages each
    uv run python -m harness.run_docling --parts ao3400a tps562200
    uv run python -m harness.run_docling --concurrency 2

Or with `modal run` (uses Modal's app context):

    modal run harness/run_docling.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import modal

# Resolve workspace root from this file's location: modal/harness/run_docling.py
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = WORKSPACE_ROOT / "data" / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.toml"
RUNS_ROOT = WORKSPACE_ROOT / "target" / "docling-runs"

# Mirror PASS1_PAGE_LIMIT in the Rust harness so docling and Pass 1
# extract the same first-N-pages slice — comparable side-by-side.
DEFAULT_PAGE_LIMIT = 10

# Modal client dispatch concurrency. Matches the worker's
# `@modal.concurrent(max_inputs=2)` cap.
DEFAULT_CONCURRENCY = 2


@dataclass
class Part:
    part_id: str
    file: str
    cls: str
    vendor: str
    part_number: str
    pages: int


@dataclass
class PartResult:
    part: Part
    requested_pages: list[int]
    extract_response: dict | None
    error: str | None
    elapsed_secs: float


def load_manifest() -> dict[str, Part]:
    raw = tomllib.loads(MANIFEST_PATH.read_text())
    parts: dict[str, Part] = {}
    for part_id, body in raw.items():
        parts[part_id] = Part(
            part_id=part_id,
            file=body["file"],
            cls=body["class"],
            vendor=body["vendor"],
            part_number=body["part_number"],
            pages=int(body["pages"]),
        )
    return parts


def select_pages(part: Part, page_limit: int) -> list[int]:
    """1-indexed page list, capped at the PDF's actual page count."""
    n = min(page_limit, part.pages)
    return list(range(1, n + 1))


def dispatch_one(extractor, part: Part, requested_pages: list[int]) -> PartResult:
    pdf_bytes = (CORPUS_DIR / part.file).read_bytes()
    started = time.monotonic()
    try:
        response = extractor.extract.remote(pdf_bytes, pages=requested_pages)
        elapsed = time.monotonic() - started
        return PartResult(
            part=part,
            requested_pages=requested_pages,
            extract_response=response,
            error=None,
            elapsed_secs=elapsed,
        )
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - started
        return PartResult(
            part=part,
            requested_pages=requested_pages,
            extract_response=None,
            error=f"{type(e).__name__}: {e}",
            elapsed_secs=elapsed,
        )


def write_envelope(envelopes_dir: Path, part: Part, result: PartResult) -> None:
    """Normalized per-page envelope, comparable to the Rust harness output."""
    response = result.extract_response or {}
    envelope = {
        "part_id": part.part_id,
        "vendor": part.vendor,
        "part_number": part.part_number,
        "class": part.cls,
        "extractor": "docling",
        "docling_version": response.get("docling_version"),
        "requested_pages": result.requested_pages,
        "pdf_elapsed_secs": result.elapsed_secs,
        "worker_elapsed_ms": response.get("elapsed_ms"),
        "error": result.error,
        "pages": response.get("pages") or [],
    }
    (envelopes_dir / f"{part.part_id}.json").write_text(
        json.dumps(envelope, indent=2)
    )


def write_raw(raw_dir: Path, part: Part, result: PartResult) -> None:
    """Raw `DoclingDocument.export_to_dict()` — the native docling view."""
    response = result.extract_response or {}
    raw = response.get("raw_document")
    if raw is None:
        return
    (raw_dir / f"{part.part_id}.json").write_text(json.dumps(raw, indent=2))


def write_csv_rows(writer: csv.writer, part: Part, result: PartResult) -> None:
    response = result.extract_response or {}
    pages = response.get("pages") or []
    if not pages:
        # Capture errors as a single row so they're visible in the CSV.
        writer.writerow(
            [
                part.part_id,
                part.vendor,
                part.part_number,
                part.cls,
                "",
                "error",
                f"{result.elapsed_secs:.3f}",
                "0",
                "0",
                result.error or "no_pages_returned",
            ]
        )
        return
    for p in pages:
        items = p.get("items") or []
        markdown = p.get("markdown") or ""
        writer.writerow(
            [
                part.part_id,
                part.vendor,
                part.part_number,
                part.cls,
                p.get("page_no"),
                "docling",
                # Per-PDF elapsed; docling doesn't expose per-page
                # timing. Repeated across rows so a CSV reader can do
                # rough cost-per-page math = elapsed / page_count.
                f"{result.elapsed_secs:.3f}",
                str(len(items)),
                str(len(markdown)),
                "ok",
            ]
        )


def write_summary(
    path: Path,
    results: list[PartResult],
    page_limit: int,
    concurrency: int,
    wall_secs: float,
) -> None:
    lines = []
    lines.append("Parselab docling corpus run")
    lines.append("==================================================")
    lines.append("")
    lines.append(f"page_limit={page_limit}  concurrency={concurrency}")
    lines.append(f"wall_secs={wall_secs:.1f}")
    lines.append("")
    header = (
        f"{'part_id':<24} {'part_number':<22} {'pdf_p':>6} "
        f"{'ext_p':>6} {'errs':>6} {'elapsed_s':>10} {'items':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    total_pages = 0
    total_items = 0
    total_errors = 0
    total_elapsed = 0.0
    for r in results:
        response = r.extract_response or {}
        pages = response.get("pages") or []
        items = sum(len(p.get("items") or []) for p in pages)
        errors = 1 if r.error else 0
        total_pages += len(pages)
        total_items += items
        total_errors += errors
        total_elapsed += r.elapsed_secs
        lines.append(
            f"{r.part.part_id:<24} {r.part.part_number:<22} {r.part.pages:>6} "
            f"{len(pages):>6} {errors:>6} {r.elapsed_secs:>10.1f} {items:>8}"
        )

    lines.append("")
    pages_per_sec = total_pages / wall_secs if wall_secs > 0 else 0.0
    lines.append(
        f"totals: parts={len(results)} extracted_pages={total_pages} "
        f"errors={total_errors} sum_elapsed_secs={total_elapsed:.1f} "
        f"items={total_items} wall_pages_per_sec={pages_per_sec:.2f}"
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run docling against the corpus")
    parser.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_PAGE_LIMIT,
        help="First-N-pages to extract per PDF (default: %(default)s, matches Pass 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Parallel Modal dispatches (default: %(default)s)",
    )
    parser.add_argument(
        "--parts",
        nargs="*",
        default=[],
        help="Restrict to these part_ids (default: whole corpus)",
    )
    return parser.parse_args(argv)


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

    DoclingExtractor = modal.Cls.from_name("parselab-docling", "DoclingExtractor")
    extractor = DoclingExtractor()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_ROOT / run_id
    envelopes_dir = output_dir / "envelopes"
    raw_dir = output_dir / "raw"
    envelopes_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "docling.csv"
    summary_path = output_dir / "summary.txt"

    started = time.monotonic()

    pending = [(part, select_pages(part, args.pages)) for part in active]
    results: list[PartResult] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(dispatch_one, extractor, part, pages): part
            for part, pages in pending
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r.error:
                print(f"  {r.part.part_id:<24} ERROR  {r.error}")
            else:
                response = r.extract_response or {}
                pages = response.get("pages") or []
                items = sum(len(p.get("items") or []) for p in pages)
                print(
                    f"  {r.part.part_id:<24} pages={len(pages):>3}  "
                    f"items={items:>4}  elapsed={r.elapsed_secs:.1f}s"
                )

    wall_secs = time.monotonic() - started

    # Stable on disk regardless of completion order.
    results.sort(key=lambda r: r.part.part_id)

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
                "pdf_elapsed_secs",
                "item_count",
                "markdown_chars",
                "status",
            ]
        )
        for r in results:
            write_envelope(envelopes_dir, r.part, r)
            write_raw(raw_dir, r.part, r)
            write_csv_rows(writer, r.part, r)

    write_summary(
        summary_path,
        results,
        page_limit=args.pages,
        concurrency=args.concurrency,
        wall_secs=wall_secs,
    )

    print()
    print("=== run complete ===")
    print(f"output dir: {output_dir}")
    print(f"  envelopes: {envelopes_dir}")
    print(f"  raw:       {raw_dir}")
    print(f"  csv:       {csv_path}")
    print(f"  summary:   {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
