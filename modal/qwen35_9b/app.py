"""Qwen3.5-9B Modal worker — stock SGLang OpenAI-compatible server.

Used by the FER-113 post-processing model evaluation as the "smaller +
cheaper" candidate for FER-103/89/90/91 text reasoning tasks. Hybrid
Gated DeltaNet + sparse MoE arch (similar concurrency story to the
Mamba-based Inf2 worker — applying the same tuning playbook).

Multimodal-capable but used here for text-only post-processing of
already-extracted page content from Pass 1 (markdown) or Pass 2 (JSON).
Thinking mode is disabled per-request via `enable_thinking=false` in
chat_template_kwargs from the client.

Deploy:

    cd modal
    modal deploy qwen35_9b/app.py

Endpoint:

    https://ferrite-systems--ferrite-qwen35-9b-serve.modal.run/v1/chat/completions
"""
from __future__ import annotations

import subprocess

import modal

MODEL_ID = "Qwen/Qwen3.5-9B"
SGLANG_PORT = 30000

app = modal.App("ferrite-qwen35-9b")

image = (
    # cu13 base — see FER-127 for the cu128 → cu13 rationale.
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
        "sglang[all]",
        "transformers>=4.50",
        "accelerate>=1.0",
        "qwen-vl-utils>=0.0.10",
        "pillow>=10.0",
        "einops",
    )
)

model_cache = modal.Volume.from_name("ferrite-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=120,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=SGLANG_PORT, startup_timeout=600)
def serve() -> None:
    """Launch SGLang's OpenAI-compatible server. 9B fits comfortably on
    L40S; tuning mirrors what we learned from the Inf2 worker (FER-80)."""
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    os.environ["TRITON_CACHE_DIR"] = "/cache/triton"

    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(SGLANG_PORT),
        "--mem-fraction-static",
        "0.85",
        "--trust-remote-code",
        # Same tuning playbook as the Inf2 / GLM-OCR workers.
        "--enable-mixed-chunk",
        "--schedule-conservativeness",
        "0.3",
        # Qwen3.5 has its own reasoning parser; we disable thinking
        # per-request from the client, but the parser is still useful
        # for any thinking traces that do come back.
        "--reasoning-parser",
        "qwen3",
    ]
    print(f"starting sglang: {' '.join(cmd)}")
    subprocess.Popen(cmd)
