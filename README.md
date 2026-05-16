# Parselab

Evaluation environment for vision-language models doing PDF data
extraction. Modal-deployed VLM workers, a small corpus, matched-cap
benchmark methodology, and (in progress) a purpose-built UI, based on
Zed's GPUI framework for exploring model performance.

## What this is

A small, opinionated harness for answering one question well:
**"Given the VLMs available today, which one should I use to extract
structured data from these PDFs?"**

Parselab ships:

- **Workers** for a handful of open-weights document-VLMs deployed on
  [Modal](https://modal.com): `granite-docling-258M`, `GLM-OCR`,
  `Infinity-Parser2-Flash` (2B), `Infinity-Parser2-Pro` (35B-MoE),
  `Qwen3.6-35B-A3B`, and others.
- **A reproducible eval methodology** — matched output-token budgets,
  warm-container measurements, Modal-side dispatch to remove client
  network from the loop, and per-page content capture for downstream
  fidelity judging.
- **A 12-datasheet electronics corpus** spanning passive components,
  discretes, MCUs, USB-PD, and connectors — the original motivating
  workload (see "Origin story" below). Easy to swap or extend with
  your own PDFs.
- **A case-study writeup** in [`BENCHMARKS.md`](BENCHMARKS.md) covering
  cost / throughput / fidelity across the model axis and a strategy
  synthesis for one-pass vs two-pass structured-document extraction.

What it *doesn't* ship yet: a polished UI for exploring model output.
That's actively being built on top of Zed's GPUI framework — see
"Roadmap."

## Origin story

Parselab spun out of work on a desktop EDA tool that treats vendor
datasheet PDFs as a live component library — extracting electrical,
thermal, mechanical, and packaging facets from datasheets so the
designer doesn't have to hand-curate component data. Presented at the
[Raleigh AI Tinkerers
meetup](https://raleigh.aitinkerers.org), May 06 2025:

> Kevin Webb presented a VLM-powered "electronics design" workflow
> that treats vendor datasheet PDFs as a live component library. The
> demo built a document extraction ETL pipeline with open-weights VLMs
> deployed on Modal, then used a Zed IDE UI on top to browse and
> validate the extracted models. Instead of manually curating
> libraries, it turns technical docs into structured parts you can
> query.

The component-extraction app remains the largest single consumer of
this work, but the **methodology and infrastructure** generalize to
any PDF extraction problem where you want to compare multiple VLMs
under controlled conditions. Parselab is the methodology and
infrastructure, factored out and made standalone.

## What's interesting

A few things in here are worth pointing at directly:

**Matched-cap benchmarking.** Different VLMs have different "natural"
output verbosities — granite-docling at 1024 tokens is comparing
apples to a model that needs 4096 to finish one page. The harness lets
you set the cap explicitly, surface `finish_reason=length`, and avoid
the easy mistake of comparing a truncated cheap model against an
unbounded expensive one.

**Modal-side dispatch (`modal/harness/remote.py`).** The harness runs
*inside* a Modal container and POSTs to the deployed VLM endpoint, so
the network path is intra-datacenter rather than residential uplink.
Eliminates a class of "is it the model or my Wi-Fi?" debugging that
ate half a session before we figured it out.

**`enable_thinking=false`.** Qwen3-family VLMs (Inf2-Flash, Inf2-Pro,
Qwen3.6-A3B) ship a chat template that supports a chain-of-thought
preamble. For structured extraction tasks, the preamble is pure
overhead and on some pages causes the model to produce markdown
analysis instead of the requested JSON. The harness wires the right
kwarg to disable it.

**LLM-as-judge for fidelity.** The harness can capture per-page raw
chat-completion content to disk; a separate session of Claude Code
(or any LLM agent) reads the source PDF page + each model's output
side-by-side and scores fidelity. No special API plumbing in the
harness itself.

## Headline findings

Full numbers and methodology in
[`BENCHMARKS.md`](BENCHMARKS.md). Headline:

- **Granite-docling-258M** is the cheapest base pass on L40S
  (~$0.00016/page), runs on Apple Silicon via MLX (~7s/page on M1
  Pro), and emits structured DocTags. Truncates ~15% of dense pages
  at 2048 cap — but that signal is clean and routable.
- **Inf2-Flash (2B)** delivers complete structured JSON layout +
  bboxes + HTML tables at ~$0.00043/page on L40S. Compared to its
  larger sibling Inf2-Pro (~$0.0024/page on H100), it's ~3× faster
  and ~5.6× cheaper at matched output budget with ~1-2% accuracy
  drop on the model card's benchmarks.
- **GLM-OCR (~9B)** wins on clean markdown extraction of TOCs and
  prose, loses table structure entirely (no `|` separators).
- **The smallest model wins on small-glyph fidelity** — granite
  correctly preserves `θ` in subscripts where both GLM and Inf2-Flash
  misread as `0`. Counter-intuitive but consistent.
- **Qwen3-family models default to chain-of-thought** on structured
  extraction tasks unless you explicitly disable thinking. ~50% of
  Inf2-Pro pages emit prose instead of JSON without the kwarg fix.

## Run it yourself

### Prerequisites

- [Modal](https://modal.com) account and CLI authenticated.
- Python 3.12+ and [uv](https://docs.astral.sh/uv/).
- A few PDFs you want to evaluate. The shipped corpus is
  electronics-datasheet-shaped; the harness doesn't assume domain.

### Deploy the workers

Each model has its own self-contained Modal app:

```sh
cd modal
modal deploy glm_ocr/app.py
modal deploy granite_docling/app.py
modal deploy inf2_flash/app.py
# ... etc.
```

See [`modal/README.md`](modal/README.md) for per-worker details, GPU
class choices, and worker-specific tuning notes.

### Run a benchmark

```sh
cd modal
uv run modal run harness/remote.py --preset granite --max-tokens 2048
uv run modal run harness/remote.py --preset inf2-flash --max-tokens 4096
uv run modal run harness/remote.py --preset inf2-pro \
    --max-tokens 4096 --no-thinking --save-content
```

`--save-content` writes per-page raw chat-completion content to
`target/quality/<preset>-<utc>/` so you can do the fidelity readthrough
yourself (or feed it to a judge).

### Run granite locally on Apple Silicon

```sh
cd scripts
uv sync
uv run python run_granite_docling_mlx.py --pages 4
```

No Modal account needed, no marginal cost per page. About 7s per page
sequential on M1 Pro.

## Layout

```
parselab/
├── README.md              # this file
├── BENCHMARKS.md          # current measurements + strategy synthesis
├── LICENSE                # MIT
├── modal/                 # Modal worker definitions + harness scripts
├── scripts/               # Apple Silicon / darwin-arm64 dev tools
│                          # (MLX driver, TableFormer hybrid, Qwen
│                          #  adjudicator)
├── data/
│   └── corpus/            # 12-datasheet electronics corpus
├── ir/                    # (TBD) document IR types — multi-pass
│                          # extraction store, bbox'd content blocks
└── ui/                    # (TBD) exploration UI on Zed's GPUI
```

## Roadmap

- **UI for exploring VLM output** on Zed's GPUI framework: side-by-side
  source PDF + model output, prompt iteration, region-select → re-extract.
- **Agentic VLM tooling** — structured tool-call wrappers around the
  deployed workers, multi-model orchestration, eval-as-you-go feedback
  loops.
- **Document IR + bbox'd content extraction blocks** — multi-pass
  extraction state representation that survives prompt changes,
  model swaps, and re-extractions.
- **Multi-corpus support** — datasheets are one shape of PDF; research
  papers, contracts, scanned forms, slide decks all expose different
  failure modes worth measuring against.

## License

MIT — see [`LICENSE`](LICENSE).

The shipped corpus PDFs are vendor datasheets reproduced from their
public source pages, used here under fair-use for evaluation. The
models evaluated have their own (mostly Apache 2.0) licenses; see
each worker's `app.py` for model-card links.

## Author

Kevin Webb · [@kpwebb](https://github.com/kpwebb) · MTS at
[dottxt](https://dottxt.co).
