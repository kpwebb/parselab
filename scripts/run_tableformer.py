"""Run TableFormerV2 over the table regions identified by a granite-docling run.

Granite-docling gives us table-level bboxes but `bbox=None` for every
cell inside a table (DocTags is structure-only, no per-cell pixel
coords). TableFormerV2 fills that gap: a small encoder-decoder that
takes a table-region image and emits per-cell bboxes plus row/col
indices.

This script:

1. Loads a granite-docling run (latest by default) under
   `target/granite-docling-runs/<utc>/`.
2. For each per-page raw doc, finds every `tables[i]` and pulls its
   table-level bbox.
3. Re-renders the source PDF page at the same DPI granite used (200),
   so granite's bbox values can be passed straight to TableFormer
   without rescaling.
4. Calls `TFPredictor.multi_table_predict()` with `do_matching=False`
   (no PDF-text-span tokens — geometric cells only) — this is the
   simplest path for the spike. With matching turned on, cells get
   tighter bboxes + matched text, but it requires PyMuPDF
   `page.get_text("words")` rescaled into the same pixel space.
5. Writes per-cell results to
   `target/tableformer-runs/<utc>/tables/<part_id>/page_<NNN>.json`,
   plus a summary.

Tracking issue: FER-* (TableFormer hybrid spike — to file when we know
this is the path forward).

Usage (from `scripts/`):

    uv sync                                                       # one-time, pulls docling-ibm-models + opencv
    uv run python run_tableformer.py                              # latest granite run, whole corpus
    uv run python run_tableformer.py --parts ao3400a tps562200    # subset
    uv run python run_tableformer.py --granite-run target/granite-docling-runs/20260509T152846Z

The first run downloads ~250MB of TableFormer artifacts via
`huggingface_hub.snapshot_download` into the HF cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = WORKSPACE_ROOT / "data" / "corpus"
GRANITE_RUNS_ROOT = WORKSPACE_ROOT / "target" / "granite-docling-runs"
RUNS_ROOT = WORKSPACE_ROOT / "target" / "tableformer-runs"

# Granite's run_granite_docling.py uses DEFAULT_DPI = 200 to render
# pages. Bboxes in the granite raw doc are in pixel space at that DPI
# (top-left origin). We render at the same DPI here so we can pass
# bboxes through untouched.
GRANITE_RENDER_DPI = 200

# The canonical artifact bundle TFPredictor was built against. The
# standalone `docling-project/TableFormerV2` HF repo has only the
# safetensors; TFPredictor expects the wrapping `tm_config.json` and
# directory layout from this repo at this revision.
DOCLING_MODELS_REPO = "ds4sd/docling-models"
DOCLING_MODELS_REVISION = "v2.2.0"
TABLEFORMER_VARIANT = "accurate"  # vs "fast" — accurate IS V2


@dataclass
class TableTask:
    part_id: str
    page_no: int  # 1-indexed
    table_idx: int  # index within the page's tables[]
    bbox_px: tuple[float, float, float, float]  # l, t, r, b at GRANITE_RENDER_DPI
    granite_grid_text: list[list[str | None]]  # text[row][col] from granite, for quick comparison


def find_latest_granite_run() -> Path | None:
    if not GRANITE_RUNS_ROOT.exists():
        return None
    runs = sorted(p for p in GRANITE_RUNS_ROOT.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def collect_tasks_for_part(
    granite_run: Path, part_id: str
) -> list[TableTask]:
    """Walk the per-page raw docs, pull every table's prov bbox + grid text."""
    raw_dir = granite_run / "raw" / part_id
    if not raw_dir.exists():
        return []
    tasks: list[TableTask] = []
    for page_path in sorted(raw_dir.glob("page_*.json")):
        page_no = int(page_path.stem.split("_")[1])
        try:
            data = json.loads(page_path.read_text())
        except json.JSONDecodeError:
            continue
        dd = data.get("docling_document") or {}
        for t_idx, table in enumerate(dd.get("tables") or []):
            provs = table.get("prov") or []
            if not provs:
                continue
            bbox = provs[0].get("bbox") or {}
            try:
                l = float(bbox["l"])
                t = float(bbox["t"])
                r = float(bbox["r"])
                b = float(bbox["b"])
            except (KeyError, TypeError, ValueError):
                continue

            # Pull cell text from the grid for downstream comparison.
            grid = (table.get("data") or {}).get("grid") or []
            grid_text: list[list[str | None]] = []
            for row in grid:
                row_text: list[str | None] = []
                for cell in row:
                    if isinstance(cell, dict):
                        row_text.append(cell.get("text"))
                    else:
                        row_text.append(None)
                grid_text.append(row_text)

            tasks.append(
                TableTask(
                    part_id=part_id,
                    page_no=page_no,
                    table_idx=t_idx,
                    bbox_px=(l, t, r, b),
                    granite_grid_text=grid_text,
                )
            )
    return tasks


def render_and_extract_page(pdf_path: Path, page_no_1indexed: int, dpi: int):
    """Render a page AND extract its PDF word tokens at the same DPI.

    Returns (bgr_ndarray, png_bytes, tokens) where `tokens` is the
    schema TFPredictor's `do_matching=True` path expects:
        [{"id": <int>, "bbox": [l, t, r, b], "text": <str>}, ...]
    Bboxes are in image-pixel space at the requested DPI, top-left
    origin — same coordinate system as the rendered image, which is
    what the matcher requires to score intersections.
    """
    import pymupdf
    import numpy as np
    import cv2

    scale = dpi / 72.0
    with pymupdf.open(pdf_path) as doc:
        page = doc.load_page(page_no_1indexed - 1)
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), alpha=False
        )
        png_bytes = pix.tobytes("png")

        # `get_text("words")` returns tuples:
        #   (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        # in PDF points, top-left origin (PyMuPDF normalizes the y-axis
        # — same orientation as the rendered image).
        raw_words = page.get_text("words")

    tokens = []
    for idx, w in enumerate(raw_words):
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if not text:
            continue
        tokens.append(
            {
                "id": idx,
                "bbox": [x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                "text": text,
            }
        )

    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return bgr, png_bytes, tokens


def build_iocr_page(bgr_image, png_path: Path, tokens: list[dict]) -> dict:
    """Construct the dict TFPredictor.multi_table_predict expects.

    `tokens` should be PDF words rescaled into the same pixel space as
    `bgr_image`. Pass `[]` to skip matching (geometric cells only).
    """
    h, w = bgr_image.shape[:2]
    return {
        "image": bgr_image,
        "tokens": tokens,
        "width": w,
        "height": h,
        "png_image_fn": str(png_path),
    }


def setup_predictor():
    """Download the docling-models artifacts and build a TFPredictor."""
    from huggingface_hub import snapshot_download
    from docling_ibm_models.tableformer.data_management.tf_predictor import (
        TFPredictor,
    )

    print(
        f"resolving {DOCLING_MODELS_REPO} @ {DOCLING_MODELS_REVISION} "
        f"(variant={TABLEFORMER_VARIANT}) ..."
    )
    artifacts = snapshot_download(
        repo_id=DOCLING_MODELS_REPO,
        revision=DOCLING_MODELS_REVISION,
        allow_patterns=[
            f"model_artifacts/tableformer/{TABLEFORMER_VARIANT}/*",
        ],
    )
    artifacts = Path(artifacts)
    save_dir = artifacts / "model_artifacts" / "tableformer" / TABLEFORMER_VARIANT
    cfg_path = save_dir / "tm_config.json"
    print(f"  artifacts at {save_dir}")

    config = json.loads(cfg_path.read_text())
    config["model"]["save_dir"] = str(save_dir)

    # device="cpu" — docling-ibm-models force-disables MPS on macOS,
    # so this is what we actually run on regardless.
    predictor = TFPredictor(config, device="cpu", num_threads=4)
    return predictor


def _ink_bbox(text_cell_bboxes) -> dict | None:
    """Tight ink-wrapping bbox = union of matched-token bboxes.

    The cell's predicted bbox can be loose (extends past actual ink).
    `do_matching=True` records the PDF-token bboxes that contributed
    text to a given cell; their union is a tight bbox snapped to the
    rendered text. Useful for UI cell-level highlighting.
    """
    if not text_cell_bboxes:
        return None
    ls, ts, rs, bs = [], [], [], []
    for tb in text_cell_bboxes:
        if isinstance(tb, dict):
            l, t, r, b = tb.get("l"), tb.get("t"), tb.get("r"), tb.get("b")
        elif isinstance(tb, (list, tuple)) and len(tb) >= 4:
            l, t, r, b = tb[0], tb[1], tb[2], tb[3]
        else:
            continue
        if None in (l, t, r, b):
            continue
        ls.append(float(l)); ts.append(float(t))
        rs.append(float(r)); bs.append(float(b))
    if not ls:
        return None
    return {"l": min(ls), "t": min(ts), "r": max(rs), "b": max(bs)}


def _matched_text(text_cell_bboxes) -> str:
    """Concatenate matched PDF tokens in reading order (top→bottom, left→right).

    TFPredictor doesn't populate the cell's `text` field directly — it
    only attaches the matched PDF tokens to `text_cell_bboxes`, with
    each entry carrying the original `token` string from PyMuPDF. We
    sort tokens by position and join with spaces.
    """
    if not text_cell_bboxes:
        return ""
    rows: list[tuple[float, float, str]] = []
    for tb in text_cell_bboxes:
        if not isinstance(tb, dict):
            continue
        token = tb.get("token")
        if not token:
            continue
        t = tb.get("t")
        l = tb.get("l")
        if t is None or l is None:
            continue
        rows.append((float(t), float(l), str(token)))
    if not rows:
        return ""
    # Group tokens into visual lines by y-overlap, then sort within
    # each line left-to-right. Avoids "B 25" coming out as "25 B"
    # when a row's tokens land at slightly different t values.
    rows.sort(key=lambda r: (r[0], r[1]))
    lines: list[list[tuple[float, float, str]]] = []
    line_threshold_px = 4.0  # tokens within this many y-pixels = same line
    for top, left, tok in rows:
        if lines and abs(top - lines[-1][0][0]) <= line_threshold_px:
            lines[-1].append((top, left, tok))
        else:
            lines.append([(top, left, tok)])
    parts: list[str] = []
    for line in lines:
        line.sort(key=lambda r: r[1])
        parts.append(" ".join(tok for _, _, tok in line))
    return " ".join(parts)


def run_one_page(
    predictor,
    pdf_path: Path,
    part_id: str,
    page_no: int,
    page_tasks: list[TableTask],
    tmp_dir: Path,
    do_matching: bool,
) -> dict:
    """Render the page once, dispatch all of its tables in one TFPredict call."""
    bgr, png_bytes, tokens = render_and_extract_page(
        pdf_path, page_no, GRANITE_RENDER_DPI
    )

    # TFPredictor wants a path on disk; honor that even though we have
    # bytes in memory.
    png_path = tmp_dir / f"{part_id}_p{page_no:03d}.png"
    png_path.write_bytes(png_bytes)

    iocr_page = build_iocr_page(
        bgr, png_path, tokens if do_matching else []
    )

    # `multi_table_predict` takes the bboxes as a list-of-list-of-floats.
    table_bboxes = [list(t.bbox_px) for t in page_tasks]

    started = time.monotonic()
    out = predictor.multi_table_predict(
        iocr_page,
        table_bboxes,
        do_matching=do_matching,
        sort_row_col_indexes=True,
    )
    elapsed = time.monotonic() - started

    # `out` is a list, one entry per input bbox.
    page_record = {
        "part_id": part_id,
        "page_no": page_no,
        "render_dpi": GRANITE_RENDER_DPI,
        "page_width_px": iocr_page["width"],
        "page_height_px": iocr_page["height"],
        "do_matching": do_matching,
        "pdf_token_count": len(tokens),
        "elapsed_secs": elapsed,
        "tables": [],
    }
    for task, tf_result in zip(page_tasks, out):
        details = tf_result.get("predict_details") or {}
        responses = tf_result.get("tf_responses") or []
        cells_clean = []
        for c in responses:
            bbox = c.get("bbox") or {}
            tcb = c.get("text_cell_bboxes") or []
            cell_out = {
                "bbox_px": {
                    "l": bbox.get("l"),
                    "t": bbox.get("t"),
                    "r": bbox.get("r"),
                    "b": bbox.get("b"),
                },
                "row_span": c.get("row_span"),
                "col_span": c.get("col_span"),
                "start_row": c.get("start_row_offset_idx"),
                "start_col": c.get("start_col_offset_idx"),
                "end_row": c.get("end_row_offset_idx"),
                "end_col": c.get("end_col_offset_idx"),
                "column_header": c.get("column_header"),
                "row_header": c.get("row_header"),
                "row_section": c.get("row_section"),
            }
            if do_matching:
                # TFPredictor leaves cell["text"] as None and stashes
                # the matched PDF tokens (each with a "token" string)
                # in text_cell_bboxes. Derive the cell's text from
                # those tokens in reading order.
                cell_out["matched_text"] = _matched_text(tcb)
                cell_out["raw_text_field"] = c.get("text")  # keep for debugging
                cell_out["text_cell_bboxes"] = tcb
                cell_out["ink_bbox_px"] = _ink_bbox(tcb)
            cells_clean.append(cell_out)
        page_record["tables"].append(
            {
                "table_idx": task.table_idx,
                "granite_bbox_px": list(task.bbox_px),
                "tf_num_rows": details.get("num_rows"),
                "tf_num_cols": details.get("num_cols"),
                "cells": cells_clean,
                "granite_grid_text": task.granite_grid_text,
            }
        )

    png_path.unlink(missing_ok=True)
    return page_record


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TableFormerV2 over granite-docling table regions")
    p.add_argument(
        "--granite-run",
        type=Path,
        default=None,
        help="Path to a target/granite-docling-runs/<utc>/ dir (default: latest)",
    )
    p.add_argument(
        "--parts",
        nargs="*",
        default=[],
        help="Restrict to these part_ids (default: every part with raw output)",
    )
    p.add_argument(
        "--no-match",
        action="store_true",
        help="Disable do_matching=True (geometric cells only, no PDF text)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    granite_run = args.granite_run or find_latest_granite_run()
    if granite_run is None or not granite_run.exists():
        print(
            "no granite-docling run found at "
            f"{GRANITE_RUNS_ROOT}; run modal/harness/run_granite_docling.py first",
            file=sys.stderr,
        )
        return 2

    raw_root = granite_run / "raw"
    if not raw_root.exists():
        print(f"granite run missing raw/ subdir: {granite_run}", file=sys.stderr)
        return 2

    part_ids = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    if args.parts:
        unknown = [p for p in args.parts if p not in part_ids]
        if unknown:
            print(f"unknown part_ids: {unknown}", file=sys.stderr)
            return 2
        part_ids = list(args.parts)

    print(f"using granite run: {granite_run}")
    print(f"running TableFormerV2 over {len(part_ids)} parts")

    # Read manifest for PDF paths.
    import tomllib

    manifest = tomllib.loads((CORPUS_DIR / "manifest.toml").read_text())

    predictor = setup_predictor()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_ROOT / run_id
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.txt"

    summary_lines: list[str] = []
    summary_lines.append("TableFormerV2 hybrid spike")
    summary_lines.append("=" * 50)
    summary_lines.append(f"granite run: {granite_run}")
    summary_lines.append(f"render_dpi:  {GRANITE_RENDER_DPI}")
    summary_lines.append("")

    total_tables = 0
    total_cells = 0
    total_pages = 0
    total_secs = 0.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for part_id in part_ids:
            tasks = collect_tasks_for_part(granite_run, part_id)
            if not tasks:
                print(f"  {part_id}: no tables found, skipping")
                summary_lines.append(f"{part_id:<24}  tables=  0")
                continue
            part_dir = tables_dir / part_id
            part_dir.mkdir(parents=True, exist_ok=True)

            pdf_file = manifest.get(part_id, {}).get("file")
            if not pdf_file:
                print(f"  {part_id}: not in manifest, skipping")
                continue
            pdf_path = CORPUS_DIR / pdf_file

            # Group tasks by page so we can do one PDF render + one
            # multi_table_predict per page.
            tasks_by_page: dict[int, list[TableTask]] = {}
            for t in tasks:
                tasks_by_page.setdefault(t.page_no, []).append(t)

            part_tables = 0
            part_cells = 0
            part_secs = 0.0
            for page_no in sorted(tasks_by_page):
                page_tasks = tasks_by_page[page_no]
                try:
                    rec = run_one_page(
                        predictor,
                        pdf_path,
                        part_id,
                        page_no,
                        page_tasks,
                        tmp_dir,
                        do_matching=not args.no_match,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"  {part_id} p{page_no}: ERROR {type(e).__name__}: {e}"
                    )
                    continue
                (part_dir / f"page_{page_no:03d}.json").write_text(
                    json.dumps(rec, indent=2)
                )
                page_cells = sum(len(t["cells"]) for t in rec["tables"])
                part_tables += len(rec["tables"])
                part_cells += page_cells
                part_secs += rec["elapsed_secs"]
                total_pages += 1
                print(
                    f"  {part_id} p{page_no}  tables={len(rec['tables'])}  "
                    f"cells={page_cells}  {rec['elapsed_secs']:.1f}s"
                )
            total_tables += part_tables
            total_cells += part_cells
            total_secs += part_secs
            summary_lines.append(
                f"{part_id:<24}  tables={part_tables:>3}  cells={part_cells:>5}  "
                f"elapsed={part_secs:>6.1f}s"
            )

    summary_lines.append("")
    summary_lines.append(
        f"totals: parts={len(part_ids)} pages={total_pages} "
        f"tables={total_tables} cells={total_cells} elapsed={total_secs:.1f}s"
    )
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print()
    print("=== run complete ===")
    print(f"output dir: {output_dir}")
    print(f"  tables:  {tables_dir}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
