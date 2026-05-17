"""Infinity-Parser2-Flash Modal worker — vLLM OpenAI-compatible server.

The 2B Qwen3.5-based sibling of Infinity-Parser2-Pro. Same image-in /
JSON-layout-out surface, ~3.68× faster single-stream throughput per the
model card, ~1-2% accuracy drop on olmOCR-Bench and ParseBench. Fits
comfortably on L40S (2B params bf16 ≈ 4 GB; vs Pro's 35B / H100).

Why we want this worker:

* **Pass 1 axis** — at matched 1024-token cap, compare cost/throughput
  vs GLM-OCR and granite-docling as a cheap structured base pass.
* **Pass 2 axis** — at 4096-token cap, compare vs Infinity-Parser2-Pro
  (same task, smaller / cheaper model) and the Qwen3.6-35B-A3B
  adjudicator candidate.

vLLM (not SGLang) per the model card recommendation. `--reasoning-parser
qwen3` strips any `<think>...</think>` traces from output.
`--enable-prefix-caching` lets the model reuse the prompt prefix's KV
cache across requests on a shared prompt (we use a single prompt for
all pages, so this is a meaningful win).

Deploy:

    cd modal
    modal deploy inf2_flash/app.py

Endpoint:

    https://<workspace>--parselab-inf2-flash-serve.modal.run/v1/chat/completions

(Resolved at runtime via `modal/harness/endpoints.py`.)
"""
from __future__ import annotations

import subprocess

import modal

MODEL_ID = "infly/Infinity-Parser2-Flash"
VLLM_PORT = 8000

app = modal.App("parselab-inf2-flash")

image = (
    # cu13 base — same as the other workers.
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
        "transformers>=4.50",
        "accelerate>=1.0",
        "qwen-vl-utils>=0.0.10",
        "pillow>=10.0",
    )
)

# Shared HF model cache volume.
model_cache = modal.Volume.from_name("parselab-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=120,
    max_containers=1,
    # Keep one container warm so eval runs never pay cold-start.
    min_containers=1,
)
# 16 matches the Pass 1 dispatch concurrency we use elsewhere. The
# learnings from the granite c=64 experiment apply here too: pushing
# concurrency higher than the server's natural batching ceiling buys
# nothing but tail latency.
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=900)
def serve() -> None:
    """Launch vLLM's OpenAI-compatible server on `VLLM_PORT`."""
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"

    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Idefics/Qwen3 multimodal stack — trust the HF remote code path.
        "--trust-remote-code",
        # Strip <think>...</think> traces (Qwen3 reasoning parser).
        "--reasoning-parser",
        "qwen3",
        # Single-GPU. Model card example uses TP=2; we don't need it on
        # L40S for a 2B model.
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        # Model trains to 65536; we cap at 8192 to leave more KV budget
        # for concurrent requests. Image + prompt + 4096 output well
        # under 8K.
        "--max-model-len",
        "8192",
        # Reuse the shared prompt-prefix KV across requests. Big win
        # since every page in our harness sends the same layout prompt.
        "--enable-prefix-caching",
    ]
    print(f"starting vllm: {' '.join(cmd)}")
    subprocess.Popen(cmd)
