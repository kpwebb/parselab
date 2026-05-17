# Benchmarks

A reproducible head-to-head of open-weights vision-language models on
the same PDF extraction workload. Measured 2025-05-16 against a
12-document electronics datasheet corpus (101 pages, first-N from each
PDF).

This file answers two questions:

1. **Cost vs performance.** Across the VLMs available today —
   `granite-docling-258M`, `GLM-OCR` (~9B), `Infinity-Parser2-Flash`
   (~2B), `Infinity-Parser2-Pro` (~35B-MoE), `Qwen3.6-35B-A3B` —
   what does each cost per page, what does each return, and what's
   the throughput?
2. **How to build toward structured documents.** Is the canonical
   two-pass design (cheap markdown base pass + expensive structured
   on-demand) still the right shape? Or do today's models support a
   single-pass structured workflow?

## Headline

**Cost vs performance.** All measurements through the same
Modal-side dispatch harness on warm containers. L40S workers at
Modal's $1.95/hr; H100 workers at $3.50/hr. Concurrency 16 client-side.

| Model | GPU | Output shape | Pages/sec | Cost / page | Truncation rate |
| --- | --- | --- | --- | --- | --- |
| **granite-docling-258M @ 2048** | L40S | DocTags → markdown + bboxes | **3.32** | **$0.000164** | 14.9% |
| GLM-OCR @ 1024 | L40S | Markdown | 2.02 | $0.000268 | low |
| **Inf2-Flash @ 4096** | L40S | JSON layout + bboxes + HTML tables | **1.28** | **$0.000425** | 0% |
| Inf2-Pro @ 4096 (no-think) | H100 | JSON layout | **0.53** | **$0.00183** | 1% |
| Qwen3.6-35B-A3B @ 4096 (no-think) | H100 | quasi-JSON (wrong schema, see below) | 0.39 | $0.00249 | 28.7% |

The "@N" suffix is the model's `max_tokens` setting. Different models
have different natural verbosities: granite wants ~1500–2000 tokens to
finish a typical page, Inf2-Flash wants ~1200–1700, Inf2-Pro and
Qwen36 want 1700–3500. Capping below natural verbosity *biases the
comparison* by truncating one model's output while another finishes
cleanly. The right comparison is **matched-completion**, not
matched-cap. (See "Method note: token caps" below.)

**Strategy.** Three concrete shapes are on the table:

- **A — Inf2-Flash single-pass structured:** every page through one
  model that emits complete layout JSON. ~3× more expensive than the
  cheapest base pass but no orchestration logic, no second pass.
- **B — Granite single-pass with truncation as routing signal:** the
  cheapest path, plus an Apple-Silicon-local fallback via the
  MLX-converted weights. The ~15% of pages that fill the budget are
  exactly the pages where a second pass is warranted; the truncation
  signal is the routing signal.
- **C — Two-pass: GLM markdown base + Inf2-Flash on selected pages.**
  Minimum disruption from canonical "cheap markdown + expensive
  structured" pipeline designs. Requires a page selector.

Details + my read on each option in "Strategy" below.

## Setup

### Corpus

`data/corpus/` ships 12 electronics datasheets — passives, discretes,
power converters, MCUs, USB-PD, connectors — totaling ~280 pages. Each
harness run extracts the first 10 pages per PDF = 101 page-extractions.
The shipped corpus is electronics-shaped (parameter tables, MCU
pinouts, package drawings, footnotes with subscripted symbols). Add
your own PDFs to `data/corpus/` and update `manifest.toml` to point at
a different workload.

### Hardware

All measurements through Modal. Per-worker GPU class:

| Worker | GPU | Why |
| --- | --- | --- |
| granite-docling-258M | L40S | 258M params; could run smaller, L40S kept for consistency with the SGLang workers |
| GLM-OCR | L40S | ~9B params; tight on smaller GPUs |
| Inf2-Flash | L40S | 2B params; L40S has KV headroom to spare |
| Inf2-Pro | H100 | ~35B-MoE params; bf16 weights ~70 GB |
| Qwen3.6-35B-A3B | H100 | ~35B-MoE; H100 for the same reason |

All workers pinned to a single warm container (`max_containers=1,
min_containers=1`) to keep measurements deterministic and avoid
cold-start contamination.

### Eval harness

`modal/harness/remote.py` — a preset-driven dispatch loop that runs
*inside* a Modal container, renders pages with PyMuPDF, and POSTs to
the deployed VLM endpoint. Preset selection picks the model + endpoint
+ recommended prompt:

```sh
modal run modal/harness/remote.py --preset granite     --max-tokens 2048
modal run modal/harness/remote.py --preset glm-ocr     --max-tokens 1024
modal run modal/harness/remote.py --preset inf2-flash  --max-tokens 4096
modal run modal/harness/remote.py --preset inf2-pro    --max-tokens 4096 --no-thinking
modal run modal/harness/remote.py --preset qwen36      --max-tokens 4096 --no-thinking
```

Running the harness on Modal eliminates the developer's uplink as a
measurement variable. (A slow client uplink at concurrency 16 cost us
~10× throughput during early measurements before we identified the
network as the bottleneck. The harness defaults to Modal-side.)

`--save-content` writes per-page raw chat-completion content to
`target/quality/<preset>-<utc>/` so you can do the fidelity comparison
in a separate pass — e.g., feed the page images + per-model outputs to
an LLM agent and ask it to score.

### Method notes

**Token caps.** Different VLMs have different natural output
verbosities. The fair comparison isn't "all models at 1024" — it's "each
model at a cap where it doesn't truncate on most of the corpus."
Surface `finish_reason=length` and report the truncation rate alongside
the throughput number. Truncated output isn't comparable to complete
output even if the wall-time is faster.

**`enable_thinking=false` for Qwen3-family.** Qwen3-family VLMs
(Inf2-Flash, Inf2-Pro, Qwen3.6-A3B) ship a chat template that supports
a chain-of-thought preamble. On structured-extraction tasks the
preamble eats tokens without helping the output; on some pages it
causes the model to produce markdown analysis instead of the requested
JSON. The harness passes `chat_template_kwargs={"enable_thinking":
false}` when `--no-thinking` is set; this works on both vLLM and SGLang
servers via the standard chat-completion request body.

**Modal-side dispatch.** As noted above, this removes
developer-network bandwidth from the measurement. Tell that you've
re-introduced it: SGLang's `#queue-req: 0` plus `#running-req: 1`
simultaneously — requests aren't queueing at the server, they're
trickling in slowly from the client. Run on Modal-side for any
cost-per-page number you intend to quote.

All numbers below are 3-run means on a warm container unless noted,
run within the same ~15-minute window.

## Cost vs performance — details

### Pass 1 axis (cheap base extraction)

Three candidates for the "give me everything on the page cheaply" role:

| Model | Cap | Pages/sec (warm) | Sum_req | Cost/page | Per-page latency | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| granite @ 1024 | 1024 | 3.80 | 410s | $0.000142 | 4.1s | 0/101 |
| **granite @ 2048** | **2048** | **3.32** | **470s** | **$0.000164** | **4.7s** | **0/101** |
| GLM @ 1024 | 1024 | 2.02 | 755s | $0.000268 | 7.5s | 0/101 |
| Inf2-Flash @ 1024 | 1024 | 1.80 | 845s | $0.000302 | 8.5s | 0/101 (all truncated) |

Granite at 1024 cap is the cheapest by raw cost-per-page but truncates
~80% of dense sampled pages. At 2048, throughput drops 13% but
truncation rate falls to 14.9% corpus-wide. **For a cheap base pass,
granite-at-2048 is the right cap.**

GLM-OCR is the most consistent at 1024 — fits naturally inside 1024 on
~99% of pages. Best for clean text / TOC / prose, worst for table
structure (emits flat space-separated rows, no `|` separators).

Inf2-Flash at 1024 truncates every page (its natural output is 1200+
tokens on simple pages, much more on dense ones). Move it up to 4096
and it's a Pass 2 candidate rather than a Pass 1 candidate; see below.

### Pass 2 axis (structured-with-bboxes extraction)

Three candidates for the "give me complete structured JSON layout with
per-element bboxes" role:

| Model | GPU | Cap | Pages/sec (warm) | Cost/page | Truncation rate | Schema |
| --- | --- | --- | --- | --- | --- | --- |
| **Inf2-Flash @ 4096** | L40S | 4096 | **1.28** | **$0.000425** | **0%** | requested (`bbox` + `category` + `text`) |
| Inf2-Pro @ 4096 (no-think) | H100 | 4096 | 0.53 | $0.00183 | 1% | requested |
| Qwen3.6-A3B @ 4096 (no-think) | H100 | 4096 | 0.39 | $0.00249 | 28.7% | **wrong** (`bbox_2d` + `text_content`) |

**Inf2-Flash dominates Inf2-Pro on cost / throughput** by ~4.3× and
~2.4× respectively, with the model card's stated ~1-2% accuracy drop
on olmOCR-Bench and ParseBench. The L40S vs H100 GPU-class difference
compounds the throughput advantage.

**Qwen3.6-35B-A3B underperforms Inf2-Pro on this task** even on the
same hardware. Two reasons:

1. **Wrong schema.** Qwen36 defaults to `{bbox_2d, text_content}`
   regardless of what the prompt asks for. ~90% of Qwen36 pages emit
   this schema; downstream Pass 2 parsers expecting `{bbox, category,
   text}` will silently produce empty results.
2. **Verbosity.** Qwen36 emits 2400 mean output tokens per page (with
   thinking disabled) and still truncates 29% of pages at 4096 cap.

Inf2-Pro is fine-tuned for document layout extraction; Qwen36 is a
base instruct model being used out-of-distribution. The size advantage
doesn't compensate for the lack of task tuning.

### The `enable_thinking` finding

The first Inf2-Pro measurement (without `--no-thinking`) showed
**50/101 pages emitting markdown analysis prose instead of JSON.**
Inf2-Pro is fine-tuned from Qwen3.5; its chat template supports
chain-of-thought; on complex pages the model enters CoT mode and the
prose answer never converges to JSON. Same root cause as Qwen36 above.

Setting `chat_template_kwargs={"enable_thinking": false}` flips the
behavior across the corpus:

| Metric | Inf2-Pro default | Inf2-Pro `no-thinking` |
| --- | --- | --- |
| Pages producing JSON | 51/101 | **101/101** |
| Pages producing prose | 50/101 | **0/101** |
| Mean output tokens | 1861 | 1287 (-31%) |
| Truncation rate | 9% | 1% |
| Pages/sec (warm) | 0.40 | **0.53** (+33%) |
| Cost/page (H100) | $0.0024 | **$0.00183** (-24%) |

**Lesson:** for *any* Qwen3-family model on a structured extraction
task, always set `enable_thinking=false`. Without it, you'll get
silently-degraded output on a meaningful fraction of pages, plus pay
~30% more in tokens and ~25% more in wall-time for output that's
demonstrably worse.

The same kwarg run on Qwen3.6-A3B improved its numbers too — truncation
rate fell from 46.5% → 28.7%, mean output tokens 3418 → 2411 — but
didn't fix the wrong-schema problem (Qwen36 emits `bbox_2d` +
`text_content` regardless of the prompt).

## Fidelity findings

Full per-page readthrough lives in
[`data/tests/quality/`](data/tests/quality/) — the committed snapshot
of the run dirs the headline findings here are derived from. The
[`2026-05-16-comparison.md`](data/tests/quality/2026-05-16-comparison.md)
file is the per-page LLM-as-judge readthrough; the per-`<preset>-<utc>/`
subdirectories carry raw chat-completion content per page for
independent re-judging. Fresh runs (via `remote.py --save-content`)
land in `target/quality/<preset>-<utc>/` locally. Headline patterns
from 5 representative pages across the model axis:

### Pass 1 axis

- **Granite at 2048**: complete on 86/101 corpus pages. Captures
  structure where it completes; OTSL tables with correct cell topology.
  **Strongest at small-glyph fidelity** — preserves `θ` subscript in
  `R_{θJB}` where both GLM-OCR and Inf2-Flash misread as `0`. The 15%
  that truncate cluster on TOC pages with dotted leaders — the model
  transcribes the dots literally inside a table cell, eating budget
  without ever producing useful structure. Unfixable by budget alone.
- **GLM-OCR**: most consistent at 1024 cap (truncated on 1/5 sampled).
  Best for clean text / TOC / prose — cheapest in output tokens
  per-page, best hierarchical preservation. Loses table structure
  entirely (no `|` separators, no HTML).
- **Inf2-Flash @ 4096**: complete on 5/5 sampled. Best for layout-aware
  tasks — every element has bbox + category, tables emit HTML with
  proper `colspan`. Most expensive in output tokens. Math in LaTeX
  form, slightly heavier than GLM's `$...$`.

### Pass 2 axis

- **Inf2-Pro (no-thinking)**: clean JSON layout. Schema matches the
  prompt spec. Notably **better small-glyph fidelity** than Inf2-Flash
  — preserves `θ` subscripts and emits properly-LaTeX'd math
  (`\(R_{\theta JB}\)`, `\(V_{IN}\)`) where Inf2-Flash misreads `θ` as
  `0`. On dense revision-history / TOC pages, also emits **deeper
  table nesting** — each revision-section becomes its own `<table>`
  with proper rows. ~4.3× the cost of Inf2-Flash, but the quality
  delta is real on small-glyph-heavy and dense-table pages.
- **Inf2-Flash**: same schema, broad coverage, ~4× cheaper. **Weak on
  the `θ → 0` OCR error class** — same Qwen3.5 vision encoder as
  Inf2-Pro but the 2B decoder doesn't disambiguate small subscript
  glyphs reliably. Sufficient on prose / non-small-glyph pages.
- **Qwen3.6-A3B (no-thinking)**: wrong schema (`bbox_2d` /
  `text_content`). Captures more fine-grained word-level bboxes than
  Inf2-Flash (every column header becomes its own bbox, every value
  its own bbox) but loses table semantics entirely — no HTML
  structure, just a flat list of bbox + text entries. Preserves `θ`
  correctly (35B helps), but the schema mismatch makes downstream
  parsing impractical. Useful as a "word locator" but not as a
  structured-layout extractor.

### Tables specifically

For datasheet parameter tables — the load-bearing content for
electronics extraction:

| Rank | Model | Format | Notes |
| --- | --- | --- | --- |
| 1 | Inf2-Flash | HTML w/ `colspan`/`rowspan` | Directly parseable; merged cells correct |
| 2 | Inf2-Pro | HTML similar | Same family, slightly different rendering |
| 3 | Granite | OTSL | Correct topology when not truncated; uses domain-specific markers |
| 4 | GLM-OCR | Flat space-separated text | Content captured but column structure lost |
| 5 | Qwen3.6-A3B | Per-cell bbox+text | No table structure; each cell is a standalone item |

This is the single biggest fidelity asymmetry across the model axis.

## Strategy — 1-pass vs 2-pass for structured documents

Three concrete shapes on the table. Cost example uses a "typical
30-page datasheet" through each option.

### Option A — Inf2-Flash single-pass structured

Every page through Inf2-Flash @ 4096. Complete structured JSON layout
with bboxes + HTML tables everywhere. ~1.28 pages/sec on a single L40S,
~$0.000425/page, $0.013 per 30-page datasheet.

- **Pros:** Single model, single deploy, single output format. No page
  selector needed. UI cross-highlighting wired directly to image-pixel
  bboxes. Table structure preserved everywhere.
- **Cons:** Most expensive of the three options. ~3× the per-page cost
  of granite at 2048. Loses granite's small-glyph fidelity. No
  Apple-Silicon-local fallback.

### Option B — Granite single-pass + truncation routing

Every page through granite @ 2048. ~85% of pages produce complete
DocTags; ~15% fill the budget. Use `finish_reason=length` as a
routing signal: truncated pages get escalated to Pass 2 (Inf2-Flash or
similar).

Cost: ~$0.000164/page baseline (~$0.005 per 30-page datasheet baseline)
plus ~$0.000425/page on escalations (~$0.004 if 10 pages of 30
escalate). **Total: ~$0.009 per 30-page datasheet.**

- **Pros:** Cheapest. Apple-Silicon-local extraction works today via
  the MLX-converted weights (~7s/page interactive). Truncation is
  self-signaling — no separate page-selector model.
- **Cons:** The TOC-style failure cluster is genuinely broken on
  granite (dotted leaders). For pages where granite fails, the Pass 2
  escalation has to carry the full content, not just the structure.

### Option C — Two-pass: GLM markdown + Inf2-Flash escalations

Pass 1: GLM-OCR markdown on every page. Pass 2: Inf2-Flash on pages
the selector flags. Closest to the canonical "cheap markdown base +
expensive structured" pipeline design.

Cost: ~$0.008 baseline (30 × $0.000268) + ~$0.004 on escalations =
**~$0.012 per 30-page datasheet.**

- **Pros:** Smallest delta from existing two-pass designs. Pass 1 is
  cheap clean markdown with hierarchical preservation. Pass 2 quality
  matches Inf2-Pro at a fraction of the cost.
- **Cons:** Requires a page-selector to make Pass 2 routing
  cost-effective. GLM has no MLX path. Table structure on Pass 1 is
  lost; downstream consumers wait for Pass 2 to get structured tables.

### Which one

Differences are small in absolute terms — $0.009 vs $0.012 vs $0.013 per
30-page datasheet. At scale (thousands of datasheets) the cost gap
widens linearly, but quality / feature differences should drive the
call at our current scale.

- **Option B is the most interesting strategically.** Cheapest, only
  option with an offline / desktop story (MLX-local), and the
  truncation signal removes the need for a separate page selector. The
  TOC-failure cluster is a real concern but those pages are also where
  the escalation pass was always going to do the work.
- **Option C is the safest.** Minimum disruption from canonical
  two-pass pipeline designs; everything already works in published
  reference architectures.
- **Option A is the riskiest.** Putting all your eggs in Inf2-Flash
  means no offline path and accepting the θ-subscript OCR error class.
  Worth keeping on the radar; not where I'd start.

### Validation steps before committing

1. Probe the `θ → 0` OCR error class at higher DPI (300 instead of 200).
   If it disappears at better input quality, granite's small-glyph
   advantage collapses and the case for Option B weakens.
2. Test a "no dotted leaders" prompt mitigation on granite. Cheap to
   try; if it works, the TOC failure cluster stops being a failure and
   Option B looks even stronger.
3. Re-measure on a non-datasheet corpus (research papers, contracts,
   scanned forms). The strategy synthesis is shaped by the failure
   modes specific to electronics datasheets; other corpora will shift
   the picture.

## What's not measured here

- **Granite-MLX corpus run** — only the smoke-test result is in. A
  full 101-page run would confirm fidelity parity with the Modal
  granite path.
- **Higher-DPI re-measurement** — see "Validation steps" above.
- **Non-datasheet corpora** — the strategy synthesis is shaped by the
  failure modes specific to electronics datasheets.

## Run dirs

* [`data/tests/quality/`](data/tests/quality/) — committed snapshot of
  the fidelity-judging artifacts that produced the findings above.
  Per-preset subdirs (one per run) hold raw chat-completion content
  per page; `2026-05-16-comparison.md` is the per-page readthrough.
* `target/quality/<preset>-<utc>/` — destination for fresh local runs
  (Modal-side `remote.py` with `--save-content`). Regeneratable; not
  committed.
