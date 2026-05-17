# Fidelity comparison: granite vs GLM-OCR vs Inf2-Flash

Judge: Claude Code in-session, 2026-05-16, against the datasheet corpus.

## Setup

All three models extracted the same 101 pages of the datasheet corpus
through the parameterized `modal/harness/remote.py` Modal-side harness.
Caps are at each model's natural sweet spot (Inf2-Flash needs more
output budget than the other two; matched-cap testing was unfair to
it):

| Model | Cap | Output dir |
|---|---|---|
| granite-docling-258M | 1024 | [`granite-20260516T202321Z/`](granite-20260516T202321Z/) |
| GLM-OCR | 1024 | [`glm-ocr-20260516T202438Z/`](glm-ocr-20260516T202438Z/) |
| Inf2-Flash | 4096 | [`inf2-flash-20260516T203020Z/`](inf2-flash-20260516T203020Z/) |

Five representative pages spanning failure-mode classes were read
side-by-side against the source PDF:

| Page | Class |
|---|---|
| mmbt3904 p1 | Transistor cover: prose + 2 small tables + figures |
| tps562200 p2 | Dense TOC + revision history (2-column TOC + long bulleted list) |
| stm32f411ce p3 | Hierarchical TOC continuation (4 levels deep, ~25 entries, dotted leaders) |
| yageo_rc0805 p2 | Ordering-info breakdown: many small key-value sub-tables + sidebar |
| gct_usb4125 p1 | Cover sheet: complex header table with merged cells + 3 figures |

## Per-page summary

### mmbt3904 p1 — transistor cover

| Model | Finish | Out tokens | Coverage |
|---|---|---|---|
| Granite | **length** | 1024 | ~60% — truncated mid-page; missed MARKING DIAGRAM body + ORDERING INFORMATION table |
| GLM-OCR | stop | 691 | ~95% — captured all sections, but tables flattened to space-separated text (column alignment lost) |
| Inf2-Flash | stop | 2253 | 100% — every element with bbox + category + HTML tables + LaTeX math |

Notes:
- All three preserved `V_{CEO}`-style subscripts, ° symbols, em-dashes.
- Math notation format varies: Granite `V$_{CEO}$`, GLM `$V_{CEO}$`, Inf2 `\(V_{CEO}\)`.
- Inf2 distinguishes `title` from `table_caption` semantically; granite and GLM don't.

### tps562200 p2 — TOC + revision history

| Model | Finish | Out tokens | Coverage |
|---|---|---|---|
| Granite | **length** | 1024 | ~65% — modeled the TOC as `<otsl>` tables and the revision-history bullets as more tables; truncated mid-third-section |
| GLM-OCR | **length** | 1024 | ~85% — clean indented TOC, truncated mid-bullet in the third change-set |
| Inf2-Flash | stop | 1678 | 100% |

Notes:
- **Subscript θ fidelity**: granite correctly preserved `R$_{θJB}$`; GLM and Inf2 both misread the θ as `0` → `R_{0JB}` / `R_{0JB}`. **Granite wins this OCR-level test.**
- Granite's choice to model the TOC as a table is wasteful (otsl preamble + many `<fcel>` cells per entry).
- Inf2 keeps the 2-column TOC as 2 text blocks with embedded newlines — matches visual layout but loses semantic hierarchy.

### stm32f411ce p3 — hierarchical TOC continuation

| Model | Finish | Out tokens | Coverage |
|---|---|---|---|
| Granite | **length** | 1024 | ~25% — truncated very early. Started an `<otsl>` table for the TOC; dotted leaders (`. . . . . .`) consumed enormous tokens. **This is the "silent-empty" failure** — docling-core can't parse an unclosed otsl, returns `items=1, md_chars=0` |
| GLM-OCR | stop | 442 | 100% — cleanest output: indented markdown TOC with all 20 sections + 6 sub-items, `$V_{BAT}$` and `®` preserved |
| Inf2-Flash | stop | 706 | 100% — one big text block containing all entries; hierarchical indentation lost; footer + figure placeholder captured |

Notes:
- This is the page that drove all 8 of the stm32f411ce "silent-empty" envelopes we saw in this morning's runs. Root cause: granite tries to model the TOC's dotted leaders as table cells; the dots are repeated literally; token budget evaporates before the table closes.
- **GLM is the clear winner here on every dimension.** Lowest token cost (442 vs 706 Inf2) and best structural fidelity (preserves the visual hierarchy).

### yageo_rc0805 p2 — ordering-info breakdown

| Model | Finish | Out tokens | Coverage |
|---|---|---|---|
| Granite | **length** | 1024 | ~50% — captured up to `(4) TAPING REEL`; missed `(5)`, `(6)`, the ORDERING EXAMPLE sidebar, the 3 NOTES |
| GLM-OCR | stop | 420 | 100% — `±%`, `Ω`, `–`, smart-quotes all correct |
| Inf2-Flash | stop | 1695 | 100% — `$\pm$1%`, `$56 \Omega$` in LaTeX; every element categorized |

Notes:
- This is a pure "many small text blocks" page — the type where Inf2's structural overhead hurts most: 1695 tokens vs GLM's 420 for essentially the same content.
- Granite's per-element decomposition (separate `<text>` nodes for each `(1)(2)(3)(4)(5)(6)` annotation marker) is wasteful and likely contributed to the truncation.

### gct_usb4125 p1 — cover sheet w/ merged-cell header

| Model | Finish | Out tokens | Coverage |
|---|---|---|---|
| Granite | stop | 270 | 90% — OTSL table with `<lcel>`/`<ecel>` colspan markers; one cell-routing issue (`CC` lands in wrong column); 3 figure placeholders captured |
| GLM-OCR | stop | 94 | 70% — table flattened to space-separated text; figures+captions captured but no logo/footer |
| Inf2-Flash | stop | 546 | 100% — HTML table with explicit `colspan="2"` and `colspan="5"` correctly representing the merged-cell layout; logo and footer captured |

Notes:
- **Inf2-Flash is best on table structure with merged cells.** The HTML colspan semantics map directly to downstream parsing.
- Granite's OTSL gets the cell topology mostly right but with a small attribution error.
- GLM loses the table structure entirely on this kind of page.

## Cross-cutting findings

### 1. Granite at 1024 cap is truncated more often than not

On the 5 pages reviewed, granite hit `finish_reason=length` on 4/5 (80%). For the role of "cheap base pass producing structured output at 1024 cap" it's unreliable — the matched-budget benchmark advantage we measured is partly an artifact of producing less content. Granite needs either a larger budget (1536–2048) or a different role (interactive single-page extraction where 1024 is fine on typical content).

### 2. GLM-OCR is the best plain-text / markdown extractor

For pages where you want **clean readable text content** (TOCs, prose, simple text-with-tables) GLM is consistently:
- Cheapest in output tokens (94 / 420 / 442 / 691 / 1024 across the 5 pages)
- Most natural markdown formatting
- Best at hierarchy preservation (indented TOCs)

Where it fails: anything that needs **table cell structure** — GLM emits flat text with no `|` separators or HTML, so column boundaries disappear. For datasheet parameter tables that's a real loss.

### 3. Inf2-Flash produces the richest structural metadata

Inf2-Flash output is the only one with:
- Per-element bboxes in image-pixel coordinates (directly usable for UI cross-highlighting)
- Explicit category labels (`title`, `text`, `table`, `figure`, `figure_caption`, `header`, `footer`, etc.)
- HTML tables with `colspan` / `rowspan` for merged cells
- LaTeX math notation in `\( \)` form

Cost: 2-5× more output tokens than GLM on simple pages, ~1.5× more than granite when granite isn't truncated. At the natural 4096 cap on L40S it's ~$0.000425/page.

### 4. Subscript θ is a granite-only strength

Counterintuitively, the smallest model preserved the θ subscript in `R_{θJB}` correctly while both larger models (GLM ~9B, Inf2 2B) misread it as `0`. Granite's vision encoder (siglip2-base-patch16-512) appears to be more careful with small glyph distinctions than the Qwen vision towers used by GLM and Inf2.

### 5. Table-structure fidelity ranking

For datasheet parameter tables specifically:

| Rank | Model | Format | Notes |
|---|---|---|---|
| 1 | Inf2-Flash | HTML w/ colspan | Directly parseable, merged cells correct |
| 2 | Granite | OTSL | Correct topology when not truncated; uses domain-specific markers |
| 3 | GLM-OCR | Flat text | Content captured but column structure lost |

### 6. Math-notation fidelity is roughly tied

All three preserve `V_{CEO}`, `T_J`, `R_{θJA}` style subscripts. Format differs:

| Model | Style | Notes |
|---|---|---|
| Granite | `V$_{CEO}$` | LaTeX inside DocTags text field; awkward |
| GLM | `$V_{CEO}$` | Standard inline LaTeX in markdown |
| Inf2 | `\(V_{CEO}\)` | LaTeX display syntax |

All three preserve `°`, `±`, `Ω`, `®`, em-dash correctly when not truncated.

## Role-based recommendations

### Cheap base pass producing markdown (current Pass 1 role)

**Pick GLM-OCR.** Cheapest output, cleanest markdown, best at hierarchical structure (TOCs). Easiest to consume downstream — plain markdown maps to most existing tooling without bespoke parsers. Its weakness — flat-text tables — is exactly what Pass 2 was designed to handle on demand.

### Cheap base pass producing structured layout (new role)

**Pick Inf2-Flash at 4096 cap.** Always completes, always emits structured JSON layout with bboxes + categories + HTML tables. ~3× more expensive per page than GLM but you get the layout metadata for free (no Pass 2 needed for routine pages). The granite-as-Pass-1 case (see `BENCHMARKS.md` base-pass section) is weakened by today's truncation findings — granite at 1024 is unreliable, and granite at 1536+ is no longer cheaper than Inf2-Flash at 4096.

### Pass 2 structured layout (replace inf2-pro?)

**Inf2-Flash is the strong candidate.** Same output shape as inf2-pro, ~1-2% accuracy drop per the model card's olmOCR/ParseBench numbers, but ~2.8× faster and ~5.6× cheaper than inf2-pro per BENCHMARKS.md's old Pass 2 numbers. The Pass 2 axis comparison (inf2-pro vs inf2-flash on the same prompt) is the next obvious test once inf2-pro is redeployed.

### Interactive desktop single-page extraction

**Granite still wins for offline / MLX use cases.** The MLX path is validated locally (`scripts/run_granite_docling_mlx.py`); Inf2-Flash has no MLX port and the 2B Qwen3.5 base would need conversion work. For "click a page, get extraction in ~7s with no network," granite-MLX is the only candidate on the board.

## Granite at 2048 cap — followup measurement

After the initial 1024 run showed granite truncating on 4/5 representative
pages, re-ran the same corpus at `--max-tokens 2048` to see if the
truncation went away.

**Corpus-wide:** 86/101 pages finish naturally; **15/101 still truncate
(14.9%)**. The persistent truncations cluster:

| Cluster | Pages | Why |
|---|---|---|
| stm32f411ce p3–p10 | 8 | TOC + functional-overview hierarchical pages — granite renders dotted leaders character-by-character inside `<otsl>` table cells. Doubling the budget just gets you more dots; the model never recognizes "dotted leader" semantics. **No amount of token budget will fix these.** |
| Dense parameter / ordering tables | 7 | mmbt3904 p2 + p8, gct_usb4125 p10, coilcraft_xal7030 p1, murata_grm21 p2, pmeg3010eb p7, yageo_rc0805 p2. Content-genuinely-long pages; would likely finish at 3072–4096 cap. |

**Effect on representative pages:**

| Page | 1024 result | 2048 result | Change |
|---|---|---|---|
| mmbt3904 p1 | trunc, 60% coverage | **stop, 1428 tok, complete** | ✓ fixed |
| tps562200 p2 | trunc, 65% coverage | **stop, 1599 tok, complete** | ✓ fixed |
| stm32f411ce p3 | trunc, 25% coverage | trunc, ~50% (4241 chars) | still failing |
| yageo_rc0805 p2 | trunc, 50% coverage | trunc, ~80% (5413 chars) | partial |
| gct_usb4125 p1 | stop, 270 tok | stop, 285 tok | unchanged |

**Throughput / cost:**

| Cap | Pages/sec (warm) | Cost/page | Truncation rate (corpus) |
|---|---|---|---|
| 1024 | 3.80 | ~$0.000142 | "high" — 4/5 sampled pages |
| **2048** | **3.32** | **~$0.000164** | **14.9%** |
| (Inf2-Flash @ 4096 reference) | 1.275 | ~$0.000425 | 0% |

Granite at 2048 is ~15% more expensive per page than at 1024, but still
**~2.6× cheaper than Inf2-Flash at 4096** and produces complete output
on ~85% of the corpus.

**Revised recommendation.** Granite at 2048 is a credible
cheap-structured-base-pass candidate when paired with a routing signal
that escalates the ~15% of pages that still truncate. The
`finish_reason=length` signal is clean and reliable — the harness can
flag truncated envelopes as "needs Pass 2 / Qwen3.6 adjudication"
without inspecting content. This fits naturally with the page-selector
and Qwen3.6-adjudicator follow-ons.

The 8-page stm32f411ce cluster is the load-bearing exception. Those
pages need a different model (GLM-OCR completes them all cleanly in
~700 tokens) or a prompt change that prevents granite from modeling
dotted leaders as table content.

## Open questions

- **Subscript θ failure mode in larger models.** Why do GLM (~9B) and Inf2-Flash (2B) both miss θ that granite-258M gets right? Worth a deeper look — could be a render-quality issue at our 200 DPI, or a tokenizer specificity issue. If it generalizes, it's a real concern for any datasheet pipeline (θJA, θJB are common thermal-resistance symbols).
- **Inf2-Flash on Pass 1-style markdown task.** The model card mentions `task_type="doc2md"` which directly emits markdown instead of layout JSON. Worth measuring once for completeness, though the layout-JSON output is more useful downstream anyway.
- **Granite dotted-leader prompt mitigation.** Can a system prompt addition like "do not transcribe dotted leaders" steer granite away from the literal-dots failure mode? Cheap to test; might unlock the stm32f411ce pages.
