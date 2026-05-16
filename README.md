# Parselab

Evaluation environment for vision-language models doing PDF data
extraction. Modal-deployed VLM workers, a small corpus, matched-cap
benchmark methodology, and (in progress) a purpose-built UI for
exploring model performance.

## Why this exists

Parselab spun out of the [Ferrite](https://linear.app/ferrite) EDA
project — a desktop tool for working with electronic-component
datasheets. The bulk of the Modal workers and the eval methodology
were built to answer "which VLM should drive Ferrite's PDF extraction
pipeline?" The infrastructure built to answer that question is
generally useful for **any project that needs to evaluate VLMs against
PDFs**, so this is that infrastructure factored out.

Ferrite remains the original consumer; Parselab also serves any future
PDF-extraction project that wants:

* Deployed VLM workers behind a stable OpenAI-compatible HTTP surface.
* A repeatable methodology for measuring per-page cost / throughput
  with the same prompt + token budget across models.
* A way to inspect model output side-by-side against the source page,
  with optional in-session LLM-as-judge for fidelity comparison.

## Tracking

Linear project: **Parselab**, in the Ferrite team. Issues prefixed `FER-`
for now; cross-link to upstream Ferrite issues where relevant.

## Status

Bootstrap in progress. Initial migration brings over the work that
landed in Ferrite through 2026-05-16. See `BENCHMARKS.md` for the
current model axis and matched-cap measurements.

## Layout

```
parselab/
├── README.md              # this file
├── BENCHMARKS.md          # current measurements + strategy synthesis
├── modal/                 # Modal worker definitions + harness scripts
├── scripts/               # darwin/arm64-only dev tools (MLX driver,
│                          # TableFormer hybrid, Qwen adjudicator)
├── data/
│   └── corpus/            # the FER-86 datasheet corpus
├── ir/                    # (TBD) document IR types — multi-pass
│                          # extraction store, bbox'd content blocks
└── ui/                    # (TBD) exploration UI + agentic tools
```

## Initial scope

* [x] Repo bootstrap
* [x] Migrate Modal workers / harness / corpus / BENCHMARKS.md
* [ ] Decide UI stack (gpui / web / Tauri / Streamlit)
* [ ] Scope the agentic VLM tooling
* [ ] Migrate document IR + bbox'd content extraction blocks from
      Ferrite
* [ ] Decide long-term Modal-worker ownership (Parselab owns, Ferrite
      consumes via deployed URLs)

## Out of scope (for now)

* Open-sourcing or public-facing distribution. Initially an internal
  tool.
* Replacing Ferrite's Rust `extractor-client` crate. Parselab and
  Ferrite coexist; the Rust client continues to call deployed worker
  URLs.
* Production extraction pipelines for users outside the EDA use case.
  Those become consumers of the Parselab harness, not part of Parselab
  itself.
