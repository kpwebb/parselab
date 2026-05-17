"""Route divergent tables to Qwen 3.6 35B-A3B for authoritative reading.

Third pass on top of the granite + TableFormerV2 hybrid. For each
table, computes three divergence signals; if any fires, crops the
table region from the source PDF and POSTs it to the deployed Qwen
worker (`parselab-qwen36-35b-a3b`) for a fresh read.

Signals:

1. **Image-embedded** — fewer than `--min-tokens` PDF word tokens fall
   inside granite's table bbox. matching can't validate; we need a
   visual second source.
2. **Cell text disagreement** — at least one cell where granite's OTSL
   text and TF-matched PDF text differ by `difflib.SequenceMatcher`
   ratio < `--text-similarity` (default 0.8 → ratio < 0.8 = "different
   enough to flag").
3. **Grid topology mismatch** — granite's `num_rows` / `num_cols` !=
   TF's.

Usage (from `scripts/`):

    uv run python run_qwen_adjudicator.py                              # latest TF run, divergent tables only
    uv run python run_qwen_adjudicator.py --parts ao3400a stm32f411ce
    uv run python run_qwen_adjudicator.py --all-tables                 # every table, not just flagged
    uv run python run_qwen_adjudicator.py --min-tokens 5 --text-similarity 0.85

Prerequisite: the Qwen 3.6 worker is deployed (`modal deploy
qwen36_35b_a3b/app.py`).
"""
from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = WORKSPACE_ROOT / "data" / "corpus"
GRANITE_RUNS_ROOT = WORKSPACE_ROOT / "target" / "granite-docling-runs"
TF_RUNS_ROOT = WORKSPACE_ROOT / "target" / "tableformer-runs"
RUNS_ROOT = WORKSPACE_ROOT / "target" / "qwen-adjudicated-runs"

# Match what the granite + TF runs use, so cropped regions match the
# bboxes those records carry.
RENDER_DPI = 200
DEFAULT_MIN_TOKENS = 5
DEFAULT_TEXT_SIMILARITY = 0.8  # SequenceMatcher.ratio threshold
DEFAULT_PADDING_PX = 24
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONCURRENCY = 4

QWEN_APP = "parselab-qwen36-35b-a3b"
QWEN_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def resolve_qwen_endpoint() -> str:
    """Look up the deployed `parselab-qwen36-35b-a3b` worker's URL via Modal."""
    import modal

    fn = modal.Function.from_name(QWEN_APP, "serve")
    return f"{fn.get_web_url().rstrip('/')}/v1/chat/completions"

QWEN_PROMPT = """\
Read this table image and return its contents as strict JSON only.

Format:
{
  "num_rows": <integer>,
  "num_cols": <integer>,
  "cells": [
    [<string>, <string>, ...],
    ...
  ]
}

Rules:
- Preserve text exactly as it appears, including units (Ω, V, A), \
math symbols (μ, ±, °), subscripts and superscripts (you may render \
subscripts as `_{...}` and superscripts as `^{...}` if useful).
- For cells that span multiple rows or columns, repeat the value in \
each grid position the cell occupies.
- For empty cells, use an empty string "".
- Return ONLY the JSON object. No prose, no markdown fences, no \
explanation.
"""


@dataclass
class Adjudication:
    part_id: str
    page_no: int
    table_idx: int
    flags: list[str] = field(default_factory=list)
    granite_bbox_px: list[float] | None = None
    granite_num_rows: int | None = None
    granite_num_cols: int | None = None
    tf_num_rows: int | None = None
    tf_num_cols: int | None = None
    tokens_in_bbox: int | None = None
    n_cell_disagreements: int | None = None
    qwen_prompt: str | None = None
    qwen_response_raw: str | None = None
    qwen_table: dict | None = None
    elapsed_secs: float | None = None
    error: str | None = None
    skipped_reason: str | None = None


def find_latest(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def parse_granite_run_from_summary(tf_run: Path) -> Path | None:
    """The TF summary.txt contains `granite run: <path>` on a line."""
    summary = tf_run / "summary.txt"
    if not summary.exists():
        return None
    for line in summary.read_text().splitlines():
        line = line.strip()
        if line.startswith("granite run:"):
            path_str = line.split(":", 1)[1].strip()
            p = Path(path_str)
            return p if p.exists() else None
    return None


def load_granite_table(
    granite_run: Path, part_id: str, page_no: int, table_idx: int
) -> dict | None:
    """Pull the granite raw doc's tables[table_idx] for (part, page)."""
    page_path = granite_run / "raw" / part_id / f"page_{page_no:03d}.json"
    if not page_path.exists():
        return None
    try:
        data = json.loads(page_path.read_text())
    except json.JSONDecodeError:
        return None
    tables = (data.get("docling_document") or {}).get("tables") or []
    if 0 <= table_idx < len(tables):
        return tables[table_idx]
    return None


def render_page_with_tokens(pdf_path: Path, page_no_1indexed: int):
    """Render PDF page at RENDER_DPI; return (PIL Image, tokens_with_bboxes_px)."""
    import pymupdf
    from PIL import Image

    scale = RENDER_DPI / 72.0
    with pymupdf.open(pdf_path) as doc:
        page = doc.load_page(page_no_1indexed - 1)
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), alpha=False
        )
        png_bytes = pix.tobytes("png")
        raw_words = page.get_text("words")

    tokens = [
        (w[0] * scale, w[1] * scale, w[2] * scale, w[3] * scale, w[4])
        for w in raw_words
        if w[4]
    ]
    # `.convert("RGB")` forces an eager decode. PIL's lazy load reads
    # the underlying BytesIO on first crop/save, which (a) goes out of
    # scope when this function returns and (b) isn't thread-safe under
    # the threadpool's concurrent crops. Materialize once now.
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return img, tokens


def tokens_inside(
    tokens: list[tuple[float, float, float, float, str]],
    bbox: tuple[float, float, float, float],
) -> int:
    """Count tokens whose center lies inside `bbox`."""
    l, t, r, b = bbox
    n = 0
    for tx0, ty0, tx1, ty1, _ in tokens:
        cx = (tx0 + tx1) * 0.5
        cy = (ty0 + ty1) * 0.5
        if l <= cx <= r and t <= cy <= b:
            n += 1
    return n


def crop_table(image, bbox: tuple[float, float, float, float], padding: float):
    """Crop a PIL image to bbox + padding, clamped to page bounds."""
    l = max(0, bbox[0] - padding)
    t = max(0, bbox[1] - padding)
    r = min(image.width, bbox[2] + padding)
    b = min(image.height, bbox[3] + padding)
    return image.crop((round(l), round(t), round(r), round(b)))


def text_ratio(a: str | None, b: str | None) -> float:
    """SequenceMatcher ratio between two strings (None or empty → 1.0 if both empty)."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def detect_cell_disagreements(
    granite_grid_text: list[list[str | None]],
    tf_cells: list[dict],
    threshold: float,
) -> int:
    """Compare granite OTSL grid text to TF matched_text per (row, col)."""
    if not granite_grid_text or not tf_cells:
        return 0
    n = 0
    for c in tf_cells:
        sr = c.get("start_row")
        sc = c.get("start_col")
        if sr is None or sc is None:
            continue
        if not (0 <= sr < len(granite_grid_text) and 0 <= sc < len(granite_grid_text[sr])):
            continue
        granite_text = granite_grid_text[sr][sc]
        tf_text = c.get("matched_text")
        if text_ratio(granite_text, tf_text) < threshold:
            n += 1
    return n


def build_qwen_request(crop_image, max_tokens: int, prompt: str) -> dict:
    buf = io.BytesIO()
    crop_image.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "model": QWEN_MODEL_ID,
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
        # Qwen3-family supports a "thinking" mode; turn it off to keep
        # latency and tokens bounded for table reading.
        "chat_template_kwargs": {"enable_thinking": False},
    }


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_qwen_json(raw: str) -> dict | None:
    """Strip optional markdown fences and parse strict JSON."""
    if not raw:
        return None
    candidate = raw.strip()
    m = _FENCE_RE.match(candidate)
    if m:
        candidate = m.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try to find the first {...} JSON-shaped substring; the model
        # sometimes prefixes "Here is the JSON:" despite our prompt.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def adjudicate_one(
    client: httpx.Client,
    pdf_path: Path,
    part_id: str,
    page_no: int,
    page_image,
    page_tokens,
    granite_run: Path,
    tf_table: dict,
    table_idx: int,
    args: argparse.Namespace,
    qwen_endpoint: str,
) -> Adjudication:
    bbox_list = tf_table.get("granite_bbox_px") or []
    if len(bbox_list) != 4:
        return Adjudication(
            part_id=part_id,
            page_no=page_no,
            table_idx=table_idx,
            skipped_reason="missing_granite_bbox",
        )
    bbox = (float(bbox_list[0]), float(bbox_list[1]), float(bbox_list[2]), float(bbox_list[3]))

    # Pull granite's grid + structure.
    granite_table = load_granite_table(granite_run, part_id, page_no, table_idx)
    granite_grid = (granite_table or {}).get("data") or {}
    granite_num_rows = granite_grid.get("num_rows")
    granite_num_cols = granite_grid.get("num_cols")
    granite_grid_text = tf_table.get("granite_grid_text") or []  # already stashed by run_tableformer.py

    tf_cells = tf_table.get("cells") or []
    tf_num_rows = tf_table.get("tf_num_rows")
    tf_num_cols = tf_table.get("tf_num_cols")

    # Divergence signals.
    flags: list[str] = []
    n_tokens = tokens_inside(page_tokens, bbox)
    if n_tokens < args.min_tokens:
        flags.append("image_embedded")

    n_disagree = detect_cell_disagreements(
        granite_grid_text, tf_cells, args.text_similarity
    )
    if n_disagree > 0:
        flags.append("text_disagreement")

    if (
        granite_num_rows is not None and tf_num_rows is not None
        and granite_num_rows != tf_num_rows
    ) or (
        granite_num_cols is not None and tf_num_cols is not None
        and granite_num_cols != tf_num_cols
    ):
        flags.append("grid_mismatch")

    adj = Adjudication(
        part_id=part_id,
        page_no=page_no,
        table_idx=table_idx,
        flags=flags,
        granite_bbox_px=list(bbox),
        granite_num_rows=granite_num_rows,
        granite_num_cols=granite_num_cols,
        tf_num_rows=tf_num_rows,
        tf_num_cols=tf_num_cols,
        tokens_in_bbox=n_tokens,
        n_cell_disagreements=n_disagree,
    )

    if not flags and not args.all_tables:
        adj.skipped_reason = "no_divergence"
        return adj

    # Crop + dispatch.
    crop = crop_table(page_image, bbox, args.padding_px)
    body = build_qwen_request(crop, args.max_tokens, QWEN_PROMPT)
    adj.qwen_prompt = QWEN_PROMPT

    started = time.monotonic()
    try:
        # 600s covers the H100 cold start (~3-5 min for the 35B model
        # to mount weights + sglang init) plus any warm steady-state
        # request (typically 30-60s).
        resp = client.post(qwen_endpoint, json=body, timeout=600.0)
        resp.raise_for_status()
        payload = resp.json()
        adj.elapsed_secs = time.monotonic() - started
        raw = payload["choices"][0]["message"]["content"]
        adj.qwen_response_raw = raw
        adj.qwen_table = parse_qwen_json(raw)
    except Exception as e:  # noqa: BLE001
        adj.elapsed_secs = time.monotonic() - started
        adj.error = f"{type(e).__name__}: {e}"
    return adj


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adjudicate divergent tables via Qwen 3.6 35B-A3B"
    )
    p.add_argument("--tf-run", type=Path, default=None,
                   help="target/tableformer-runs/<utc>/ (default: latest)")
    p.add_argument("--granite-run", type=Path, default=None,
                   help="target/granite-docling-runs/<utc>/ (default: read from TF run summary)")
    p.add_argument("--parts", nargs="*", default=[])
    p.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
                   help="image-embedded threshold (default: %(default)s)")
    p.add_argument("--text-similarity", type=float, default=DEFAULT_TEXT_SIMILARITY,
                   help="cell-text agreement ratio cutoff (default: %(default)s)")
    p.add_argument("--padding-px", type=int, default=DEFAULT_PADDING_PX,
                   help="crop padding around granite bbox (default: %(default)s)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="Qwen output token cap (default: %(default)s)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="parallel Qwen dispatches (default: %(default)s)")
    p.add_argument("--all-tables", action="store_true",
                   help="adjudicate every table, not just flagged ones")
    p.add_argument("--endpoint", type=str, default=None,
                   help=("override the Qwen chat-completions endpoint "
                         f"(default: resolved via Modal SDK from {QWEN_APP!r})"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    tf_run = args.tf_run or find_latest(TF_RUNS_ROOT)
    if tf_run is None or not tf_run.exists():
        print(f"no tableformer run found at {TF_RUNS_ROOT}", file=sys.stderr)
        return 2

    granite_run = args.granite_run or parse_granite_run_from_summary(tf_run)
    if granite_run is None or not granite_run.exists():
        print(
            "couldn't resolve granite run; pass --granite-run explicitly",
            file=sys.stderr,
        )
        return 2

    qwen_endpoint = args.endpoint or resolve_qwen_endpoint()

    print(f"tf run:      {tf_run}")
    print(f"granite run: {granite_run}")
    print(f"qwen endpoint: {qwen_endpoint}")
    print(
        f"thresholds: min_tokens={args.min_tokens} "
        f"text_similarity={args.text_similarity} "
        f"all_tables={args.all_tables}"
    )

    # Read manifest for PDF paths.
    import tomllib

    manifest = tomllib.loads((CORPUS_DIR / "manifest.toml").read_text())

    # Collect TF page records.
    tf_tables_root = tf_run / "tables"
    if not tf_tables_root.exists():
        print(f"tf run missing tables/ subdir", file=sys.stderr)
        return 2

    part_ids = sorted([p.name for p in tf_tables_root.iterdir() if p.is_dir()])
    if args.parts:
        unknown = [p for p in args.parts if p not in part_ids]
        if unknown:
            print(f"unknown part_ids: {unknown}", file=sys.stderr)
            return 2
        part_ids = list(args.parts)

    # Collect every (part, page, table) work item.
    work: list[tuple[str, int, dict, int]] = []  # part_id, page_no, tf_table_dict, table_idx
    for part_id in part_ids:
        part_dir = tf_tables_root / part_id
        for page_path in sorted(part_dir.glob("page_*.json")):
            page_no = int(page_path.stem.split("_")[1])
            try:
                rec = json.loads(page_path.read_text())
            except json.JSONDecodeError:
                continue
            for t_idx, table in enumerate(rec.get("tables") or []):
                work.append((part_id, page_no, table, t_idx))

    print(f"\nfound {len(work)} (part, page, table) tuples")

    # Render pages once per (part, page) so we don't re-render for each
    # table on a page.
    render_cache: dict[tuple[str, int], tuple[object, list]] = {}

    def get_page(part_id: str, page_no: int):
        key = (part_id, page_no)
        if key not in render_cache:
            pdf_file = manifest[part_id]["file"]
            pdf_path = CORPUS_DIR / pdf_file
            render_cache[key] = render_page_with_tokens(pdf_path, page_no)
        return render_cache[key]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RUNS_ROOT / run_id
    adj_dir = output_dir / "adjudicated"
    adj_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.txt"

    # Phase 1: classify everything (cheap, no Qwen calls).
    classified: list[Adjudication] = []
    for part_id, page_no, tf_table, t_idx in work:
        page_image, page_tokens = get_page(part_id, page_no)
        # Pre-pass: build a stub Adjudication with flags but no Qwen call,
        # by reusing adjudicate_one logic minus the dispatch. We do that
        # in phase 2 to avoid a needless code split — instead let
        # adjudicate_one early-return on no_divergence.
        classified.append(
            ("__pending__", part_id, page_no, t_idx, tf_table, page_image, page_tokens)
        )

    flagged_count = 0
    skipped_count = 0
    error_count = 0
    elapsed_total = 0.0

    pdf_path_for = lambda pid: CORPUS_DIR / manifest[pid]["file"]

    print(f"\ndispatching to Qwen ({qwen_endpoint})")
    print("(only flagged tables; pass --all-tables to bypass)")

    with httpx.Client(http2=False) as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {}
            for entry in classified:
                _, part_id, page_no, t_idx, tf_table, page_image, page_tokens = entry
                fut = pool.submit(
                    adjudicate_one,
                    client,
                    pdf_path_for(part_id),
                    part_id,
                    page_no,
                    page_image,
                    page_tokens,
                    granite_run,
                    tf_table,
                    t_idx,
                    args,
                    qwen_endpoint,
                )
                futs[fut] = (part_id, page_no, t_idx)

            done = 0
            total = len(futs)
            for fut in as_completed(futs):
                done += 1
                adj = fut.result()
                part_dir = adj_dir / adj.part_id
                part_dir.mkdir(parents=True, exist_ok=True)
                out_path = part_dir / f"page_{adj.page_no:03d}_table_{adj.table_idx:02d}.json"
                out_path.write_text(
                    json.dumps(
                        {
                            "part_id": adj.part_id,
                            "page_no": adj.page_no,
                            "table_idx": adj.table_idx,
                            "flags": adj.flags,
                            "granite_bbox_px": adj.granite_bbox_px,
                            "granite_num_rows": adj.granite_num_rows,
                            "granite_num_cols": adj.granite_num_cols,
                            "tf_num_rows": adj.tf_num_rows,
                            "tf_num_cols": adj.tf_num_cols,
                            "tokens_in_bbox": adj.tokens_in_bbox,
                            "n_cell_disagreements": adj.n_cell_disagreements,
                            "qwen_prompt": adj.qwen_prompt,
                            "qwen_response_raw": adj.qwen_response_raw,
                            "qwen_table": adj.qwen_table,
                            "elapsed_secs": adj.elapsed_secs,
                            "error": adj.error,
                            "skipped_reason": adj.skipped_reason,
                        },
                        indent=2,
                    )
                )
                if adj.skipped_reason:
                    skipped_count += 1
                    continue
                flagged_count += 1
                if adj.error:
                    error_count += 1
                    print(
                        f"  [{done}/{total}] {adj.part_id} p{adj.page_no} t{adj.table_idx}  "
                        f"flags={adj.flags}  ERROR  {adj.error}"
                    )
                else:
                    qt = adj.qwen_table or {}
                    elapsed_total += adj.elapsed_secs or 0.0
                    print(
                        f"  [{done}/{total}] {adj.part_id} p{adj.page_no} t{adj.table_idx}  "
                        f"flags={adj.flags}  qwen={qt.get('num_rows')}×{qt.get('num_cols')}  "
                        f"{adj.elapsed_secs:.1f}s"
                    )

    summary_lines = [
        "Qwen 3.6 35B-A3B table adjudication",
        "=" * 50,
        f"tf run:      {tf_run}",
        f"granite run: {granite_run}",
        f"thresholds:  min_tokens={args.min_tokens} text_similarity={args.text_similarity}",
        f"all_tables:  {args.all_tables}",
        "",
        f"total tables in corpus: {len(work)}",
        f"flagged + dispatched:   {flagged_count}",
        f"skipped (no divergence): {skipped_count}",
        f"errors:                  {error_count}",
        f"sum dispatch elapsed:    {elapsed_total:.1f}s",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print()
    print("=== run complete ===")
    print(f"output dir: {output_dir}")
    print(f"  adjudicated: {adj_dir}")
    print(f"  summary:     {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
