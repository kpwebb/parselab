# Benchmarks

Measured 2026-05-16 against the FER-86 corpus (12 datasheets, 101 pages
first-N from each). Replaces the earlier May-3 / FER-80 digest — the
model axis and the eval process have both shifted enough that a fresh
write-up is cleaner than incremental edits.

This file answers two questions:

1. **Cost vs performance.** Across the candidates we have on hand today
   — `granite-docling-258M`, `GLM-OCR` (~9B), `Infinity-Parser2-Flash`
   (~2B) — what does each cost per page, what does each return, and
   what's the throughput?
2. **How to build toward structured documents.** The original FER-83
   design is two-pass (cheap markdown base + expensive structured
   on-demand). Today's data argues for revisiting that decomposition.

Inf2-Pro (Pass 2 candidate, H100) and Qwen3.6-35B-A3B (adjudicator
candidate, H100) are referenced where useful but not freshly measured
this session — see "Deferred to Phase 2."

## Headline answers

**Cost vs performance.** On L40S at $1.95/hr Modal pricing, with all
three measured through the same Modal-side dispatch harness on the same
warm containers:

| Model | Output shape | Warm pages/sec | Cost / page | Truncation rate (corpus) |
| --- | --- | --- | --- | --- |
| granite @ 1024 | DocTags → markdown + bboxes | 3.80 | $0.000142 | ~80% of dense pages |
| **granite @ 2048** | DocTags → markdown + bboxes | **3.32** | **$0.000164** | **14.9%** |
| GLM-OCR @ 1024 | Markdown | 2.02 | $0.000268 | low — fits 1024 on most pages |
| Inf2-Flash @ 1024 | JSON layout (truncated) | 1.80 | $0.000302 | ~100% — natural output > 1024 |
| **Inf2-Flash @ 4096** | **JSON layout + bboxes + HTML tables** | **1.28** | **$0.000425** | **0%** |

The "@ N" suffix is the model's `max_tokens` setting. Granite needs
2048 to stop truncating most pages; Inf2-Flash needs 4096 (its natural
output is longer than 1024). GLM finishes naturally inside 1024 on most
pages and is closest to a fair single-setting comparison.

**Structured-document strategy.** The 2-pass design (cheap markdown +
expensive structured-on-demand) made sense in May when Pass 2 was
~6.5× more expensive than Pass 1 and required a beefier GPU class. The
2-pass design is **weakened today** because:

- Inf2-Flash at 4096 is ~3× the cost of granite-at-2048 — small enough
  that "always-structured base pass" is now in the conversation,
  especially for the parameter-table heavy pages where Pass 2 has been
  carrying the structure burden.
- Granite at 2048 with `finish_reason=length` as a routing signal gives
  a clean 2-pass shape *without* needing a separate page-selector
  model: the ~15% of pages that fill the budget are exactly the dense
  parameter-table and TOC pages where Pass 2 / Inf2-Flash would have
  to step in anyway.
- The granite-MLX path (FER-128, validated locally on M1 Pro this
  session) makes the cheap base pass viable *offline*. That changes the
  product story for the desktop app.

See "Strategy" section below for the three concrete options.

## Setup

* **Corpus** — `data/corpus/manifest.toml`, 12 datasheets across vendors
  (passives, discretes, power, MCU, USB-PD, connector); ~280 pages
  total. Each harness run extracts first 10 pages per PDF = 101
  page-extractions.
* **Hardware** — All three models deployed on **L40S** (Modal $1.95/hr).
  Granite uses vLLM via the `granite_docling` worker; GLM uses SGLang
  via `glm_ocr`; Inf2-Flash uses vLLM via `inf2_flash`. All workers
  pinned to `max_containers=1, min_containers=1`, single warm container
  each.
* **Eval harness** — `modal/harness/remote.py`, a Modal-side dispatch
  loop that renders pages with PyMuPDF inside a Modal container and
  POSTs to the deployed VLM endpoint. Running it from Modal eliminates
  developer-uplink network bandwidth as a variable (a slow client
  uplink at concurrency 16 cost us ~10× throughput in this morning's
  earlier runs before we figured out the network was the bottleneck).
  Preset-driven: `--preset granite|glm-ocr|inf2-flash|inf2-pro|qwen36`.
* **Eval judge** — Claude Code in-session, per
  [`feedback_eval_judge.md`](.claude/projects/-Users-kpwferrite-workspace/memory/feedback_eval_judge.md):
  the harness writes per-page raw chat-completion content to
  `target/quality/<preset>-<utc>/` with `--save-content`; in-session
  Claude reads the source PDF page + each model's output side-by-side
  and scores fidelity. No Anthropic API plumbing in the Rust crate.
  Five representative pages span failure-mode classes; findings in
  `target/quality/2026-05-16-comparison.md` (session-local, not
  tracked in git — regenerable by re-running the content-capture
  harness and re-judging in a Claude Code session).

All numbers below are 3-run means on a warm container unless noted, run
within the same ~15-minute window on the same network.

## Cost vs performance — details

### Per-model summary

**granite-docling-258M** (Idefics3 — siglip2-base-patch16-512 vision +
Granite-165M decoder)

- 258M params total, ~520MB at bf16. Tiny relative to the L40S budget.
- Output: DocTags string → parsed via `docling-core` to a
  `DoclingDocument` that round-trips to markdown + items + bboxes.
- Strong at small-glyph fidelity (correctly preserves θ subscript in
  `R_{θJB}` where GLM and Inf2-Flash both misread as `0`).
- Weak failure modes: at 1024 cap truncates on ~80% of dense pages;
  even at 2048 cap, 14.9% of pages truncate. The TOC pages with dotted
  leaders (stm32f411ce p3-p10) are unfixable by token budget — the
  model transcribes dots literally until the cap is reached.
- **Apple Silicon MLX viable** via `ibm-granite/granite-docling-258M-mlx`.
  Validated on M1 Pro at ~7s/page sequential (FER-128).

**GLM-OCR** (`zai-org/GLM-OCR`, ~9B GLM-4.6 OCR variant)

- Output: clean markdown. No bboxes, no explicit table structure
  (tables render as space-separated text rows — content captured but
  column alignment lost).
- Best at TOC / prose / hierarchical content. Cheapest in output tokens
  on most page types (442 tokens for stm32f411ce p3 vs granite's 1024
  truncated + Inf2-Flash's 706).
- Subscript/special-char fidelity is generally good (V_{CEO}, ®, °C,
  ±, Ω, em-dash all preserved), with the θ→0 OCR error on small
  subscripts as a known weak spot.
- No published MLX port; 9B at fp16 doesn't fit 16 GB unified memory
  and `--trust-remote-code` custom modeling complicates conversion.

**Infinity-Parser2-Flash** (`infly/Infinity-Parser2-Flash`, 2B Qwen3.5
based)

- Output: JSON layout with per-element bbox + category + content. HTML
  tables with `colspan`/`rowspan` for merged cells. LaTeX math notation.
- Always completes at 4096 cap — never truncated on this corpus.
- Best structural fidelity: directly-usable bboxes in image-pixel
  coordinates, explicit category labels (`title`, `text`, `table`,
  `figure`, `header`, `footer`, etc.), proper table cell topology.
- Largest output token count of the three (1700–2500 tokens/page mean
  vs granite's 700–1500 and GLM's 400–700).
- vLLM-deployed with `--reasoning-parser qwen3`; output lands in
  `message.reasoning` field rather than `message.content` — harness
  captures both.

### Throughput / cost in detail

| Run | Pages/sec (3-run mean) | Sum_req | Cost/page | Per-page mean latency | Errors |
| --- | --- | --- | --- | --- | --- |
| granite @ 1024 | 3.80 | 410s | $0.000142 | 4.1s | 0/101 |
| granite @ 2048 | 3.32 | 470s | $0.000164 | 4.7s | 0/101 |
| GLM @ 1024 | 2.02 | 755s | $0.000268 | 7.5s | 0/101 |
| Inf2-Flash @ 1024 | 1.80 | 845s | $0.000302 | 8.5s | 0/101 (all truncated) |
| Inf2-Flash @ 4096 | 1.28 | 1185s | $0.000425 | 11.7s | 0/101 |

All three workers can hold concurrency 16 with `effective_parallelism
≈ 15`. Earlier-in-session experiments with `concurrency=64` on the
granite worker found no throughput win — vLLM saturates at the same
~3.8 dispatch p/s and per-request latency just inflates. c=16 is the
right default on tail-latency grounds.

## Fidelity findings (summary)

Full per-page readthrough at `target/quality/2026-05-16-comparison.md`
(session-local). Headline patterns from 5 representative pages:

**Granite at 1024**: truncated on 4/5 sampled pages. Captured structure
where it completed; OTSL tables with correct cell topology when not
truncated. **Strongest at small-glyph fidelity** (caught θ where the
larger models missed it).

**Granite at 2048**: complete on 3/5 sampled pages; the remaining
truncations (stm32f411ce p3, yageo_rc0805 p2) doubled their content
volume but didn't finish. The stm32f411ce cluster is the unfixable
case — dotted leaders.

**GLM-OCR**: most consistent at 1024 cap (truncated on 1/5 sampled).
**Best for clean text / TOC / prose** — cheapest in output tokens, best
hierarchical preservation. Loses table structure entirely (no `|`
separators, no HTML).

**Inf2-Flash @ 4096**: complete on 5/5 sampled. **Best for layout-aware
tasks** — every element has bbox + category, tables emit HTML with
proper `colspan`. Most expensive in output tokens. Math in LaTeX form,
slightly heavier than GLM's `$...$`.

### Tables specifically

For datasheet parameter tables (the load-bearing content of FER-86):

| Rank | Model | Format | Notes |
| --- | --- | --- | --- |
| 1 | Inf2-Flash | HTML w/ `colspan`/`rowspan` | Directly parseable, merged cells correct |
| 2 | Granite | OTSL | Correct topology when not truncated; uses domain-specific markers |
| 3 | GLM-OCR | Flat space-separated text | Content captured but column structure lost |

This is the single biggest fidelity asymmetry across the three.

## Strategy — 1-pass vs 2-pass for structured documents

The Ferrite IR (FER-101) is structure-aware: per-page extraction store
with typed regions, ToC entries, and component facets. Whatever
extraction stack we pick needs to feed that. Three concrete shapes are
on the table.

### Option A — Inf2-Flash single-pass structured

**Replace both Pass 1 and Pass 2 with Inf2-Flash @ 4096.** Every page
produces complete JSON layout with bboxes + HTML tables. ~$0.000425
per page, ~1.28 pages/sec on a single L40S.

* Pros: Single model, single deploy, single output format. No page
  selector needed. UI cross-highlighting wired directly to image-pixel
  bboxes. Table structure preserved everywhere.
* Cons: Most expensive option. ~3× the per-page cost of granite at
  2048. Loses granite's small-glyph fidelity (θ→0 OCR error). No
  Apple-Silicon-local fallback — Inf2-Flash has no MLX port and
  conversion of the Qwen3.5 VL stack is non-trivial.
* When this wins: scale-bound workloads where structure-on-every-page
  matters more than per-page cost; or projects where you want one
  model end-to-end and don't need an offline desktop story.

### Option B — Granite single-pass structured (at 2048 cap)

**Replace Pass 1 with granite-docling at 2048 cap.** ~$0.000164/page on
L40S, plus a free MLX local fallback (~7s/page on M1 Pro, $0 marginal
cost). Use `finish_reason=length` as the routing signal: the 15% of
pages that hit the cap get escalated to Pass 2.

* Pros: Cheapest option per page. Best small-glyph fidelity.
  Apple-Silicon-local extraction works today (FER-128). Truncation is
  self-signaling — no separate page selector needed (FER-104 becomes
  optional).
* Cons: The 8-page stm32f411ce cluster is genuinely broken on this
  model (dotted leaders). For pages where granite fails, Pass 2 has to
  carry the full content, not just the structure. Total cost per page
  on dense documents like the STM32 datasheet is dominated by Pass 2
  escalations.
* When this wins: corpus where most pages are simple enough to fit in
  granite's natural budget; offline / desktop use cases; cost-bound
  workloads.

### Option C — GLM-OCR Pass 1 + Inf2-Flash Pass 2 (today's two-pass)

**Keep the existing Pass 1 (GLM markdown) but swap Inf2-Pro for
Inf2-Flash on Pass 2.** ~$0.000268/page Pass 1 + Inf2-Flash escalations
at $0.000425/page only on pages the selector flags.

* Pros: Smallest delta from the May-3 design. All existing tooling
  (FER-103 ToC builder, IR `Content::Markdown`, inspector pane,
  ad-hoc extractor) keeps working unchanged on Pass 1. Pass 2 quality
  goes up vs Inf2-Pro on accuracy and gets cheaper on throughput.
* Cons: Still requires the page selector (FER-104) for Pass 2 routing
  to be cost-effective. GLM has no MLX path — desktop app stays
  network-dependent unless we add granite as a secondary local option.
  Table structure on Pass 1 is lost (GLM flat-text tables); the IR has
  to wait for Pass 2 to get structured tables.
* When this wins: minimum-disruption path forward; keeps the project
  on a known-good architecture and just upgrades the Pass 2 model
  swap-out.

### Comparison at typical workload sizes

For a "typical 30-page datasheet" through each option:

| Option | Pass 1 cost | Pass 2 cost (assume ~10 pages escalate) | Total | Offline-capable? |
| --- | --- | --- | --- | --- |
| A — Inf2-Flash only | $0.013 (30 × 0.000425) | — | **$0.013** | No |
| B — granite @ 2048 + (Inf2-Flash escalations) | $0.005 (30 × 0.000164) | $0.004 (10 × 0.000425) | **$0.009** | **Yes (granite-MLX)** |
| C — GLM-OCR + (Inf2-Flash escalations) | $0.008 (30 × 0.000268) | $0.004 (10 × 0.000425) | **$0.012** | No (no MLX) |

Differences are small in absolute terms (~$0.013 vs $0.009 per
datasheet). At scale (thousands of datasheets) the cost gap widens
linearly, but quality/feature differences should drive the call, not
per-page-cost-at-this-scale.

### My read

**Option B is the most interesting.** It's the cheapest, gives us
offline-capability via MLX, and the natural truncation signal removes
the need for a separate page selector. The 8-page stm32f411ce-style
failure cluster is a real concern but those pages are also where Pass
2 was always going to do the work — the failure isn't "we lost the
content," it's "we noticed early and routed sooner."

**Option C is the safest.** Minimum disruption; everything already
works; just swap the Pass 2 model.

**Option A is the riskiest.** Putting all your eggs in Inf2-Flash means
no offline path and accepting the θ-subscript OCR error class. Worth
keeping on the radar as Inf2-Flash improves, but not where I'd
start today.

Concrete next steps to validate before committing:

1. Measure Inf2-Pro on the Pass 2 axis today (rather than trusting the
   May-3 BENCHMARKS.md numbers). Settles whether Inf2-Flash really is
   ~5× cheaper than Inf2-Pro on H100, or whether that's stale.
2. Run granite-MLX through the same 5 representative pages on M1 Pro
   to confirm fidelity parity with Modal granite. If MLX granite has
   any divergence we don't know about, the Option B case weakens.
3. Probe the θ→0 OCR failure in GLM / Inf2-Flash at higher DPI (300
   instead of 200). If the error goes away with better input image,
   the asymmetry collapses and granite's small-glyph advantage is no
   longer load-bearing.
4. Test the "no dotted leaders" prompt mitigation on granite. Cheap to
   try; if it works, the 8-page stm32f411ce cluster stops being a
   failure and Option B looks even stronger.

## Deferred to Phase 2

* **Inf2-Pro head-to-head Pass 2 comparison.** The H100 worker is
  coded for cu130 (FER-127) but not redeployed since this morning's
  shutdown. ~$80/day idle once warm, so worth batching with the qwen36
  redeploy. Same harness shape as Inf2-Flash (`--preset inf2-pro`).
* **Qwen3.6-35B-A3B as Pass 2 / adjudicator candidate.** Originally
  filed for FER-113 post-processing eval; an "is bigger better?"
  control against Inf2-Flash for layout extraction. Needs a custom
  prompt — its current worker docstring says "used here for text-only
  post-processing" and adapting to image-in layout extraction is the
  spike question.
* **Pin SGLang / vLLM versions** ([FER-130](https://linear.app/ferrite/issue/FER-130)).
  Float-by-default `sglang[all]` / `vllm` is what made this morning's
  network-bandwidth bug initially read as a "SGLang regression." Pin
  to remove that class of confusion.

## Run dirs

* `target/harness-runs/<utc>/` — GLM via Rust extractor-harness.
* `target/granite-docling-runs/<utc>/` — granite via local Python
  harness.
* `target/quality/<preset>-<utc>/` — Modal-side `remote.py` with
  `--save-content`; per-page raw chat-completion outputs for fidelity
  judging.
* `target/quality/2026-05-16-comparison.md` — fidelity readthrough.
