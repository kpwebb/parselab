# Parselab Modal workers

Modal apps that host the VLM workers Parselab benchmarks against. Each
worker is a self-contained `app.py` that exposes either:

- **OpenAI-compatible `/v1/chat/completions`** via SGLang or vLLM
  (granite-docling, GLM-OCR, Inf2-Flash, Inf2-Pro, Qwen3.5-9B,
  Qwen3.6-35B-A3B), or
- **A callable Modal class** (docling — IBM's local pipeline, not an
  LLM server).

```
modal/
├── pyproject.toml              # uv-managed driver deps (the worker image
│                               # installs torch/transformers itself)
├── docling/app.py              # docling pipeline worker (callable class)
├── glm_ocr/app.py              # GLM-OCR via SGLang
├── granite_docling/app.py      # granite-docling-258M via vLLM
├── inf2_flash/app.py           # Infinity-Parser2-Flash (2B) via vLLM
├── infinity_parser2/app.py     # Infinity-Parser2-Pro (35B-MoE) via SGLang
├── qwen35_9b/app.py            # Qwen3.5-9B via SGLang
├── qwen36_35b_a3b/app.py       # Qwen3.6-35B-A3B via SGLang
└── harness/
    ├── remote.py               # Modal-side throughput harness (preset-driven)
    ├── run_docling.py          # docling corpus driver (Modal SDK dispatch)
    └── run_granite_docling.py  # granite-docling corpus driver (HTTP dispatch)
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
`pydantic`, `httpx`, `docling-core`). The GPU image installs its own
torch / transformers / SGLang or vLLM at container build time —
nothing extra to do locally.

## Workers at a glance

| Worker | Server | GPU | Output shape | Notes |
| --- | --- | --- | --- | --- |
| `granite_docling/` | vLLM | L40S | DocTags string | `--revision untied`; tiny (258M), L40S kept for cost-comparable numbers |
| `glm_ocr/` | SGLang | L40S | Markdown | 9B; clean prose / TOC, no table structure |
| `inf2_flash/` | vLLM | L40S | JSON layout + bboxes + HTML tables | 2B Qwen3.5-VL; primary Pass 2 candidate |
| `infinity_parser2/` | SGLang | H100 | JSON layout + bboxes + HTML tables | 35B-MoE; needs `enable_thinking=false` for structured output |
| `qwen35_9b/` | SGLang | L40S | text / VLM | text-reasoning post-processing (smaller/cheaper) |
| `qwen36_35b_a3b/` | SGLang | H100 | text / VLM | text-reasoning post-processing (larger); used as table adjudicator |
| `docling/` | (Modal class) | L40S | DoclingDocument | IBM local pipeline; called via Modal SDK, not HTTP |

All workers share one HF-cache volume (`parselab-hf-cache`); first
cold start populates it, subsequent starts mount and skip the
download. All workers run pinned to a single warm container
(`min_containers=1, max_containers=1`) so benchmark numbers are
deterministic.

## Deploy

Each worker is independent:

```sh
cd modal
modal deploy glm_ocr/app.py
modal deploy granite_docling/app.py
modal deploy inf2_flash/app.py
modal deploy infinity_parser2/app.py
modal deploy qwen35_9b/app.py
modal deploy qwen36_35b_a3b/app.py
modal deploy docling/app.py
```

After deploy, the SGLang / vLLM workers expose:

```
POST https://<workspace>--parselab-<worker>-serve.modal.run/v1/chat/completions
```

`docling` exposes a Modal class (`DoclingExtractor`) you reach via
`modal.Cls.from_name("parselab-docling", "DoclingExtractor")`.

## Smoke tests

OpenAI-style workers — POST a single rendered page to the endpoint
(see each worker's docstring for the exact URL).

Docling (Modal class):

```sh
cd modal
modal run docling/app.py --pdf-path ../data/corpus/ao3400a.pdf
modal run docling/app.py --pdf-path ../data/corpus/stm32f411ce.pdf --pages 1,2,3
```

## Benchmarking

The `harness/` directory has three drivers:

- **`harness/remote.py`** — Modal-side throughput harness. Runs *inside*
  a Modal container, renders the corpus with PyMuPDF, and POSTs to a
  selected preset (or a custom endpoint). Use this for any
  cost-per-page number you intend to quote — running outside Modal
  introduces developer-uplink bandwidth as a measurement variable.

  ```sh
  cd modal
  uv run modal run harness/remote.py --preset granite      --max-tokens 2048
  uv run modal run harness/remote.py --preset glm-ocr      --max-tokens 1024
  uv run modal run harness/remote.py --preset inf2-flash   --max-tokens 4096
  uv run modal run harness/remote.py --preset inf2-pro     --max-tokens 4096 --no-thinking
  uv run modal run harness/remote.py --preset qwen36       --max-tokens 4096 --no-thinking
  ```

  Add `--save-content` to write per-page raw chat-completion content
  to `target/quality/<preset>-<utc>/` for downstream fidelity judging.

- **`harness/run_granite_docling.py`** — per-page granite-docling driver
  that POSTs to the deployed worker, parses DocTags via `docling-core`,
  and writes normalized envelopes + raw documents + a CSV + summary.

  ```sh
  cd modal
  uv run python -m harness.run_granite_docling                # whole corpus, 10 pages each
  uv run python -m harness.run_granite_docling --pages 4
  uv run python -m harness.run_granite_docling --parts ao3400a tps562200
  uv run python -m harness.run_granite_docling --concurrency 16
  ```

- **`harness/run_docling.py`** — docling pipeline driver. Dispatches
  via the Modal SDK (`modal.Cls.from_name(...)`) rather than HTTP, since
  docling exposes a class method, not an OpenAI endpoint.

  ```sh
  cd modal
  uv run python -m harness.run_docling                        # whole corpus, 10 pages each
  uv run python -m harness.run_docling --pages 4              # first 4 pages each
  uv run python -m harness.run_docling --parts ao3400a tps562200
  uv run python -m harness.run_docling --concurrency 2
  ```

See [`../BENCHMARKS.md`](../BENCHMARKS.md) for the headline numbers and
methodology these drivers produced.

## Worker-specific notes

### `enable_thinking=false` for Qwen3-family

Qwen3-family VLMs (Inf2-Flash, Inf2-Pro, Qwen3.5-9B, Qwen3.6-A3B) ship
a chat template that supports a chain-of-thought preamble. On
structured-extraction tasks the preamble eats tokens without helping
the output, and on some pages causes the model to produce markdown
analysis instead of the requested JSON. The harness passes
`chat_template_kwargs={"enable_thinking": false}` when `--no-thinking`
is set; this works on both vLLM and SGLang OpenAI servers.

### Granite-docling `--revision untied`

The published `main` branch of `ibm-granite/granite-docling-258M`
ships with tied input/output embeddings, which vLLM rejects per the
model card. The worker pins `--revision untied` to use the
publication-recommended weights.

### `skip_special_tokens=false` for granite-docling

Granite-docling emits DocTags via element-type *special tokens*
(`<text>`, `<title>`, `<table>`, ...). vLLM's default
`skip_special_tokens=true` strips the wrappers and leaves
unparseable bare coordinates. The granite-docling driver and
`remote.py`'s `granite` preset force-keep them.

### GPU class choices

- L40S ($1.95/hr on Modal): granite-docling, GLM-OCR, Inf2-Flash,
  Qwen3.5-9B, docling. Keeps the cost-comparison axis sane; the small
  workers don't need an H100.
- H100 ($3.50/hr): Inf2-Pro (~35B-MoE bf16 ≈ 70 GB), Qwen3.6-35B-A3B
  (same scale). L40S is too small.
