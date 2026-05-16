"""Infinity-Parser2-Pro Modal worker — stock SGLang OpenAI-compatible server.

Pure infrastructure: hosts Infinity-Parser2-Pro behind SGLang's standard
OpenAI `/v1/chat/completions` endpoint. PDF rendering, image encoding,
prompt construction (including `enable_thinking=False` for Qwen3.5-VL),
JSON layout parsing (the old `_parse_layout_json` / `LayoutParseError`),
metrics, and IR construction live in the Rust `extractor-client`
(FER-83). The prior in-worker Python (custom extract method, FER-82
envelope, per-page error handling, image preprocessing constants) was
retired when we switched from HF transformers to SGLang — empirical
probe (2026-05-03) showed SGLang gives ~10-15× throughput on the same
H100 vs HF + SDPA. See git history for the working HF-transformers
worker if needed.

Note: the Infinity-Parser2 official repo only ships HF + vLLM backends,
not SGLang. Our probe confirmed SGLang serves this fine-tuned Qwen3.5-VL
variant correctly (0 errors, valid output across concurrencies 1/2/4/8).

Deploy:

    cd modal
    modal deploy infinity_parser2/app.py

After deploy, hit as a standard OpenAI client:

    POST https://ferrite-systems--ferrite-infinity-parser2-serve.modal.run/v1/chat/completions
"""
from __future__ import annotations

import subprocess

import modal

MODEL_ID = "infly/Infinity-Parser2-Pro"
SGLANG_PORT = 30000

app = modal.App("ferrite-infinity-parser2")

image = (
    # cu13 + python 3.12 + libnuma1 — same base ABI as the GLM-OCR worker.
    # See FER-127 for the cu128 → cu13 rationale.
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
        # Qwen3.5-VL utility helpers; sglang uses them internally.
        "qwen-vl-utils>=0.0.10",
        "pillow>=10.0",
        "einops",
    )
)

# Shared HF model cache volume; first cold start downloads ~70GB of
# weights, subsequent starts mount and skip.
model_cache = modal.Volume.from_name("ferrite-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=120,
    # Pin to a single container for predictable cost/latency. SGLang's
    # in-server continuous batching handles concurrent requests up to its
    # memory budget; H100 + 35B MoE saturates at ~4 concurrent requests
    # per probe (2026-05-03), well within one container's capacity.
    max_containers=1,
    # Keep one container warm so Pass 2 dispatches from the desktop app
    # don't pay the ~3-5min H100 cold-start (image pull + 35B weights
    # mount + SGLang init).
    min_containers=1,
)
# Required to let multiple HTTP requests reach SGLang concurrently —
# Modal's default is 1, which serializes everything at the container
# boundary. 16 is well above the empirical saturation point (c=4) for
# the 35B-MoE model, leaving headroom for transient bursts.
@modal.concurrent(max_inputs=16)
@modal.web_server(port=SGLANG_PORT, startup_timeout=900)
def serve() -> None:
    """Launch SGLang's OpenAI-compatible server. 35B MoE model + cold
    cache = up to ~5 min to ready, hence the longer `startup_timeout`."""
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    # Persist Triton's JIT cache so first-call autotune cost is paid once.
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
        # Bumped from 0.85 → 0.92 to expand the KV cache pool. Empirical
        # token usage held at 0.40-0.50 so we have headroom; growing
        # mem-fraction gives more KV tokens per concurrent request.
        "--mem-fraction-static",
        "0.92",
        # Override SGLang's conservative auto-calculated mamba cache cap
        # (was 9 → ~3-4 concurrent in no_buffer). Each request needs 2
        # mamba slots; 32 slots gives the aggressive scheduler room to
        # pack up to ~16 concurrent if KV permits.
        "--max-mamba-cache-size",
        "32",
        # Explicit running-request cap, raised in lockstep with the
        # mamba pool. Actual ceiling will still be the min of (mamba,
        # KV, max-running-requests).
        "--max-running-requests",
        "16",
        # Pack the running batch more aggressively — default 1.0 leaves
        # idle slots even when capacity is available (we observed
        # #running-req: 5 with mamba room for 8 at conservativeness=1.0).
        # 0.3 matches the Pass 1 worker's setting.
        "--schedule-conservativeness",
        "0.3",
        "--trust-remote-code",
    ]
    print(f"starting sglang: {' '.join(cmd)}")
    subprocess.Popen(cmd)
