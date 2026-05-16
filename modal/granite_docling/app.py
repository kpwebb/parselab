"""Granite-Docling-258M Modal worker — vLLM OpenAI-compatible server.

Pure infrastructure: hosts IBM's `ibm-granite/granite-docling-258M`
(Idefics3-based VLM, siglip2-base-patch16-512 vision encoder + Granite
165M LLM decoder) behind vLLM's OpenAI `/v1/chat/completions` endpoint.
The Python eval harness renders pages with PyMuPDF and POSTs image +
prompt; the response body is a DocTags string parsed downstream via
`docling-core`.

vLLM (not SGLang) for this model. SGLang's `transformers_auto`
multimodal processor (the generic fallback for Idefics3-via-AutoModel)
hits an `AttributeError: type object 'Modality' has no attribute
'MULTI_IMAGES'` during request handling — internal SGLang bug. The
model card explicitly recommends vLLM with `--revision untied`, so we
use that path here. Same OpenAI HTTP surface as the SGLang workers, so
the harness driver is unchanged.

L40S to keep this directly comparable to the GLM-OCR Pass 1 worker —
overkill for a 258M model, but the per-page cost number lines up with
the existing BENCHMARKS.md table without GPU-class apples-to-oranges.

Tracking issue: FER-126.

Note on `--revision untied`: the published main branch ships with tied
input/output embeddings, which vLLM rejects per the model card.

Deploy:

    cd modal
    modal deploy granite_docling/app.py

After deploy, hit as a standard OpenAI client:

    POST https://ferrite-systems--ferrite-granite-docling-serve.modal.run/v1/chat/completions
"""
from __future__ import annotations

import subprocess

import modal

MODEL_ID = "ibm-granite/granite-docling-258M"
MODEL_REVISION = "untied"
VLLM_PORT = 8000

app = modal.App("ferrite-granite-docling")

image = (
    # cu13 + python 3.12 + libnuma1 — same base image as the other
    # workers (FER-127). cu13 matches current vLLM kernel wheels.
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("libnuma1")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "vllm",
        # granite-docling is Idefics3-based; recent transformers carries
        # the architecture. Pin a floor that includes Idefics3 +
        # siglip2 support.
        "transformers>=4.50",
        "accelerate>=1.0",
        "pillow>=10.0",
    )
)

# Shared HF model cache volume — granite-docling weights are tiny
# (~1GB) but reusing the existing volume keeps the cache layout
# consistent across all workers.
model_cache = modal.Volume.from_name("ferrite-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=120,
    # Single warm container — predictable cost/latency. vLLM's
    # continuous batching handles concurrency in-server.
    max_containers=1,
    # Keep one container warm so eval runs never pay cold-start. 258M
    # weights load fast, but image pull + vLLM init still costs ~1min.
    min_containers=1,
)
# 16. Tried 64 (2026-05-16) thinking the L40S had idle capacity at
# c=16 — measured both locally and from a Modal-side harness and found
# throughput essentially identical (~3.8 pages/sec dispatch-only at
# either c). The per-request latency *grew* with concurrency (4s → 12s)
# without a wall-time win, so c=16 dominates on tail latency without
# costing throughput. Likely cause: granite's vision encoder
# (siglip2-base-patch16-512) doesn't batch well across requests in
# vLLM at this model size; prefill serializes even with KV cache room.
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve() -> None:
    """Launch vLLM's OpenAI-compatible server on `VLLM_PORT`."""
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"

    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        # Pinned to the untied-weights revision — vLLM rejects the
        # tied-weights main branch per the model card.
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Idefics3 + siglip2 support lives in transformers; trust the
        # HF remote code path.
        "--trust-remote-code",
        # Static GPU memory fraction. 258M weights + KV pool fit easily
        # on an L40S; matching the SGLang workers' 0.85 leaves headroom.
        "--gpu-memory-utilization",
        "0.85",
        # Cap context. DocTags output for a dense datasheet page tends
        # to land under 4K tokens; 8K leaves headroom without paying
        # for unused KV.
        "--max-model-len",
        "8192",
    ]
    print(f"starting vllm: {' '.join(cmd)}")
    subprocess.Popen(cmd)
