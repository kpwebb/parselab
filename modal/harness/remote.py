"""Run a VLM-endpoint harness from *inside* Modal.

Generic OpenAI-chat-completions throughput harness. Renders the FER-86
corpus inside a Modal container and POSTs each rendered page to an
OpenAI-compatible endpoint. Returns headline metrics. Use to benchmark
any of our deployed VLM workers without paying developer-uplink
network variance.

Scope: throughput measurement only. No per-page envelopes, no DocTags
or layout-JSON parsing — just render → POST → time. For fidelity work,
use the per-model local harness (e.g., `run_granite_docling.py` for
DocTags parsing).

## Built-in presets

Each preset hard-codes the endpoint, model id, and recommended prompt
for one of our deployed workers. Pick a preset with `--preset` or
override individual fields with `--endpoint` / `--model` / `--prompt`.

* `granite`     — ibm-granite/granite-docling-258M (DocTags output)
* `glm-ocr`     — zai-org/GLM-OCR (Markdown output)
* `inf2-flash`  — infly/Infinity-Parser2-Flash (JSON layout)
* `inf2-pro`    — infly/Infinity-Parser2-Pro (JSON layout)
* `qwen36`      — Qwen/Qwen3.6-35B-A3B (general VLM)

## Usage

    modal run modal/harness/remote.py --preset granite
    modal run modal/harness/remote.py --preset inf2-flash --max-tokens 1024
    modal run modal/harness/remote.py --preset inf2-flash --max-tokens 4096
    modal run modal/harness/remote.py --preset inf2-pro --max-tokens 4096

Custom endpoint:

    modal run modal/harness/remote.py \\
        --endpoint https://my-app.modal.run/v1/chat/completions \\
        --model my-org/my-model \\
        --max-tokens 1024
"""
from __future__ import annotations

import base64
import time

import modal

DEFAULT_DPI = 200
DEFAULT_PAGE_LIMIT = 10

# `prompt`: the per-page text prompt sent alongside the image.
# `keep_special_tokens`: granite-docling emits DocTags via special-
#   tokens (`<text>`, `<title>`, etc.) that vLLM/SGLang strip by default
#   (`skip_special_tokens=True`). Force-keep for DocTags-shaped output.
PRESETS: dict[str, dict] = {
    "granite": {
        "endpoint": "https://ferrite-systems--ferrite-granite-docling-serve.modal.run/v1/chat/completions",
        "model": "ibm-granite/granite-docling-258M",
        "prompt": "Convert this page to docling.",
        "keep_special_tokens": True,
    },
    "glm-ocr": {
        "endpoint": "https://ferrite-systems--ferrite-glm-ocr-serve.modal.run/v1/chat/completions",
        "model": "zai-org/GLM-OCR",
        "prompt": "Text Recognition:",
        "keep_special_tokens": False,
    },
    "inf2-flash": {
        "endpoint": "https://ferrite-systems--ferrite-inf2-flash-serve.modal.run/v1/chat/completions",
        "model": "infly/Infinity-Parser2-Flash",
        # The Inf2 layout-extraction prompt from the model card, condensed.
        "prompt": (
            "Extract layout information from the provided PDF image. For each layout "
            "element, output its bbox, category, and the text content within the bbox. "
            "Bbox format: [x1, y1, x2, y2]. Allowed layout categories: 'header', 'title', "
            "'text', 'figure', 'table', 'formula', 'figure_caption', 'table_caption', "
            "'formula_caption', 'figure_footnote', 'table_footnote', 'page_footnote', "
            "'footer'. For 'figure', the text field must be empty. For 'formula', format "
            "text as LaTeX. For 'table', format text as HTML. For all other categories, "
            "format text as Markdown. Sort all layout elements in human reading order. "
            "Final output must be a single JSON object."
        ),
        "keep_special_tokens": False,
    },
    "inf2-pro": {
        "endpoint": "https://ferrite-systems--ferrite-infinity-parser2-serve.modal.run/v1/chat/completions",
        "model": "infly/Infinity-Parser2-Pro",
        "prompt": "Extract layout with bboxes as JSON",
        "keep_special_tokens": False,
    },
    "qwen36": {
        "endpoint": "https://ferrite-systems--ferrite-qwen36-35b-a3b-serve.modal.run/v1/chat/completions",
        "model": "Qwen/Qwen3.6-35B-A3B",
        # Placeholder — Pass 2 use will substitute the layout prompt.
        "prompt": "Extract layout with bboxes as JSON",
        "keep_special_tokens": False,
    },
}

app = modal.App("ferrite-harness-remote")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "httpx>=0.27",
        "pymupdf>=1.24",
    )
    # Bake the FER-86 corpus into the image so the container has the PDFs
    # at hand without any volume mount or download step.
    .add_local_dir(
        "../data/corpus",
        remote_path="/corpus",
    )
)


@app.function(
    image=image,
    timeout=1800,
    # No GPU — this side just renders and POSTs.
    cpu=4.0,
    memory=8192,
)
def run_corpus(
    endpoint: str,
    model: str,
    prompt: str,
    keep_special_tokens: bool,
    max_tokens: int,
    concurrency: int,
    page_limit: int,
    dpi: int,
    disable_thinking: bool = False,
) -> dict:
    """Render → POST → time. Returns headline metrics."""
    import tomllib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import httpx
    import pymupdf

    corpus_dir = Path("/corpus")
    manifest = tomllib.loads((corpus_dir / "manifest.toml").read_text())
    parts = sorted(manifest.items())

    # ── Phase 1: render every page upfront ───────────────────────────
    render_start = time.monotonic()
    tasks: list[tuple[str, int, bytes]] = []  # (part_id, page_no, png_bytes)
    for part_id, body in parts:
        pdf_path = corpus_dir / body["file"]
        with pymupdf.open(pdf_path) as doc:
            n = min(page_limit, doc.page_count)
            scale = dpi / 72.0
            matrix = pymupdf.Matrix(scale, scale)
            for i in range(n):
                pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
                tasks.append((part_id, i + 1, pix.tobytes("png")))
    render_secs = time.monotonic() - render_start
    print(f"rendered {len(tasks)} pages in {render_secs:.1f}s")

    # ── Phase 2: dispatch ────────────────────────────────────────────
    print(
        f"\ndispatching {len(tasks)} chat-completions to {endpoint}, "
        f"model={model}, concurrency={concurrency}, max_tokens={max_tokens}"
    )

    def post_one(client: httpx.Client, part_id: str, page_no: int, png: bytes) -> dict:
        b64 = base64.b64encode(png).decode("ascii")
        body: dict = {
            "model": model,
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
        }
        if keep_special_tokens:
            body["skip_special_tokens"] = False
        if disable_thinking:
            # Qwen3-family models (Qwen3.5-VL, Inf2-Flash, Qwen3.6-A3B, etc.)
            # ship a chat template that honors `enable_thinking`. Setting
            # it false suppresses the <think>...</think> chain-of-thought
            # preamble that we don't want when the task is structured
            # extraction. Works on both vLLM and SGLang OpenAI servers
            # since both pass `chat_template_kwargs` through to the
            # tokenizer's apply_chat_template().
            body["chat_template_kwargs"] = {"enable_thinking": False}

        t0 = time.monotonic()
        try:
            r = client.post(endpoint, json=body, timeout=300.0)
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage") or {}
            choices = data.get("choices") or [{}]
            message = (choices[0] or {}).get("message") or {}
            # vLLM's qwen3 reasoning parser routes output to a field
            # named `reasoning` (sibling of `content`) when no <think>
            # block is present. Inf2-Flash hits this path since its
            # layout-JSON output never wraps in <think>. Capture all
            # three so callers don't need to know the worker's
            # reasoning-parser setting.
            return {
                "part_id": part_id,
                "page_no": page_no,
                "elapsed": time.monotonic() - t0,
                "in_tok": usage.get("prompt_tokens"),
                "out_tok": usage.get("completion_tokens"),
                "content": message.get("content"),
                "reasoning": message.get("reasoning"),
                "reasoning_content": message.get("reasoning_content"),
                "finish_reason": (choices[0] or {}).get("finish_reason"),
                "ok": True,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "part_id": part_id,
                "page_no": page_no,
                "elapsed": time.monotonic() - t0,
                "error": f"{type(e).__name__}: {e}",
                "ok": False,
            }

    dispatch_start = time.monotonic()
    results: list[dict] = []
    with httpx.Client(
        http2=False,
        limits=httpx.Limits(
            max_connections=concurrency * 2,
            max_keepalive_connections=concurrency * 2,
        ),
    ) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(post_one, client, pid, pno, png)
                for (pid, pno, png) in tasks
            ]
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                tag = "OK " if r["ok"] else "ERR"
                extra = (
                    f"in={r['in_tok']} out={r['out_tok']}"
                    if r["ok"]
                    else r["error"]
                )
                print(
                    f"  [{len(results):>3}/{len(tasks)}] {tag} {r['part_id']} "
                    f"p{r['page_no']:>2}  {r['elapsed']:.1f}s  {extra}"
                )
    dispatch_secs = time.monotonic() - dispatch_start

    ok = sum(1 for r in results if r["ok"])
    err = len(results) - ok
    sum_req = sum(r["elapsed"] for r in results)
    in_tok = sum(r.get("in_tok") or 0 for r in results)
    out_tok = sum(r.get("out_tok") or 0 for r in results)
    pages_per_sec = ok / dispatch_secs if dispatch_secs > 0 else 0.0

    summary = {
        "model": model,
        "endpoint": endpoint,
        "pages": len(tasks),
        "ok": ok,
        "errors": err,
        "render_secs": round(render_secs, 2),
        "dispatch_secs": round(dispatch_secs, 2),
        "sum_req_secs": round(sum_req, 1),
        "in_tok_total": in_tok,
        "out_tok_total": out_tok,
        "pages_per_sec": round(pages_per_sec, 3),
        "effective_parallelism": (
            round(sum_req / dispatch_secs, 1) if dispatch_secs > 0 else 0.0
        ),
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "dpi": dpi,
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return {"summary": summary, "results": results}


@app.local_entrypoint()
def main(
    preset: str = "granite",
    endpoint: str = "",
    model: str = "",
    prompt: str = "",
    keep_special_tokens: bool | None = None,
    max_tokens: int = 1024,
    concurrency: int = 16,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    dpi: int = DEFAULT_DPI,
    save_content: bool = False,
    no_thinking: bool = False,
) -> None:
    """Pick a preset and optionally override individual fields.

    With `--save-content`, per-page raw chat-completion content is
    written to `target/quality/<preset>-<utc>/<part_id>_p<NN>.json` so
    Claude Code can judge fidelity in-session.

    With `--no-thinking`, sends `chat_template_kwargs={"enable_thinking":
    false}` to suppress Qwen3-family chain-of-thought preambles. Tag
    propagates into the run_id so output dirs are distinguishable.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    if preset not in PRESETS:
        raise SystemExit(
            f"unknown preset {preset!r}; choices: {sorted(PRESETS)}"
        )
    base = PRESETS[preset]
    resolved_endpoint = endpoint or base["endpoint"]
    resolved_model = model or base["model"]
    resolved_prompt = prompt or base["prompt"]
    resolved_kst = (
        keep_special_tokens
        if keep_special_tokens is not None
        else base["keep_special_tokens"]
    )

    payload = run_corpus.remote(
        endpoint=resolved_endpoint,
        model=resolved_model,
        prompt=resolved_prompt,
        keep_special_tokens=resolved_kst,
        max_tokens=max_tokens,
        concurrency=concurrency,
        page_limit=page_limit,
        dpi=dpi,
        disable_thinking=no_thinking,
    )
    summary = payload["summary"]
    results = payload["results"]

    print("\n(local) headline:")
    print(f"  preset={preset} model={summary['model']}")
    print(
        f"  {summary['ok']}/{summary['pages']} pages, "
        f"dispatch {summary['dispatch_secs']}s, "
        f"{summary['pages_per_sec']} pages/sec, "
        f"effective_parallelism={summary['effective_parallelism']}"
    )

    if save_content:
        # Resolve target/ relative to repo root (this file is at
        # modal/harness/remote.py).
        repo_root = Path(__file__).resolve().parents[2]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag = "-nothink" if no_thinking else ""
        out_dir = repo_root / "target" / "quality" / f"{preset}{tag}-{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Per-page files plus a run-level summary.
        for r in results:
            stem = f"{r['part_id']}_p{r['page_no']:02d}.json"
            (out_dir / stem).write_text(json.dumps(r, indent=2))
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "preset": preset,
                    "prompt": resolved_prompt,
                    **summary,
                },
                indent=2,
            )
        )
        print(f"\nsaved {len(results)} page outputs → {out_dir}")
