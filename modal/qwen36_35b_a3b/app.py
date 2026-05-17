"""Qwen3.6-35B-A3B Modal worker — stock SGLang OpenAI-compatible server.

Larger / more expensive text-reasoning candidate alongside the
Qwen3.5-9B worker. 35B total / 3B activated MoE — same scale as Inf2
(which is ~35B Qwen3.5-MoE under the hood, fine-tuned for document
layout). This worker uses the BASE instruct model so we can tell apart
"size matters" from "domain fine-tuning matters."

Multimodal but used here primarily for text-only post-processing /
table adjudication on already-extracted page content. Thinking mode
disabled per-request via `enable_thinking=false`.

The model card recommends 8 GPUs (TP=8), but we run on H100×1 to match
our existing Inf2 deployment — the practical scale we're at doesn't
need maxed-out TP. May need lower mem-fraction-static to fit.

Deploy:

    cd modal
    modal deploy qwen36_35b_a3b/app.py

Endpoint:

    https://<workspace>--parselab-qwen36-35b-a3b-serve.modal.run/v1/chat/completions

(Resolved at runtime via `modal/harness/endpoints.py`.)
"""
from __future__ import annotations

import subprocess

import modal

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
SGLANG_PORT = 30000

app = modal.App("parselab-qwen36-35b-a3b")

image = (
    # cu13 base — matches the SGLang kernel wheel ABI.
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

model_cache = modal.Volume.from_name("parselab-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=120,
    max_containers=1,
    # Keep one container warm so eval harnesses and adjudication runs
    # don't pay the H100 cold-start.
    min_containers=1,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=SGLANG_PORT, startup_timeout=900)
def serve() -> None:
    """Launch SGLang's OpenAI-compatible server. 35B-A3B on H100 is
    tight (model is 35B × bf16 = ~70GB; H100 has 80GB) — leaving little
    room for KV / state cache. If we OOM during load, drop to
    --mem-fraction-static=0.92 → 0.85 → 0.80 in that order."""
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
        # Slightly less memory budget than Inf2 since the base model is
        # bigger relative to fine-tuned variants. Bump down if it OOMs.
        "--mem-fraction-static",
        "0.92",
        "--trust-remote-code",
        # Tuning shared with the GLM-OCR / Inf2 workers.
        "--enable-mixed-chunk",
        "--schedule-conservativeness",
        "0.3",
        # Qwen3 reasoning parser for any thinking traces that come back.
        "--reasoning-parser",
        "qwen3",
    ]
    print(f"starting sglang: {' '.join(cmd)}")
    subprocess.Popen(cmd)
