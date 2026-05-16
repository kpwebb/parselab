"""GLM-OCR Modal worker — stock SGLang OpenAI-compatible server.

Pure infrastructure: hosts the GLM-OCR model behind SGLang's standard
OpenAI `/v1/chat/completions` endpoint. PDF rendering, image encoding,
request orchestration, metrics, and IR construction live in the Rust
`extractor-client` (FER-83). The prior in-worker Python (custom extract
methods, FER-82 envelope construction, per-page error handling, the
`cuda_poisoned` self-exit recovery) was retired when we switched from
HF transformers to SGLang — empirical probe (2026-05-03) showed SGLang
gives ~6-8× throughput on the same L40S, and continuous batching
naturally bounds the loop failure mode that motivated all the worker-
side mitigations. See git history for the working HF-transformers worker.

Deploy:

    cd modal
    modal deploy glm_ocr/app.py

After deploy, hit as a standard OpenAI client:

    POST https://ferrite-systems--ferrite-glm-ocr-serve.modal.run/v1/chat/completions
"""
from __future__ import annotations

import subprocess

import modal

MODEL_ID = "zai-org/GLM-OCR"
SGLANG_PORT = 30000

app = modal.App("ferrite-glm-ocr")

image = (
    # cu13 + python 3.12 + libnuma1. cu13 is required by current
    # sgl-kernel wheels on PyPI (FER-127); earlier cu128 builds hit
    # `libnvrtc.so.13: cannot open shared object file` on fresh
    # deploys. cu124 attempts before that hit sgl_kernel arch
    # mismatches (SM89 / L40S) — see git history.
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
        "transformers>=4.46",
        "accelerate>=1.0",
        "pillow>=10.0",
    )
)

# Shared HF model cache volume; first cold start downloads ~1.8GB of
# weights, subsequent starts mount and skip.
model_cache = modal.Volume.from_name("ferrite-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=120,
    # Pin to a single container for predictable cost/latency. SGLang's
    # in-server continuous batching handles concurrent requests up to its
    # memory budget; we don't need horizontal scaling at the corpus
    # workload sizes we're running today.
    max_containers=1,
    # Keep one container warm 24/7 so the desktop app and eval harnesses
    # never pay GLM-OCR's ~60s cold-start. With max_containers=1 this is
    # at most one L40S idle.
    min_containers=1,
)
# Modal's default per-container input limit is 1 — without this, every
# HTTP request is serialized at the Modal layer before reaching SGLang,
# defeating SGLang's continuous batching (we saw `#running-req: 1` in
# server logs and 11s/14s queue ratio per request). 32 is well above
# what SGLang needs to pack a full batch (probe saturated at c=16) and
# leaves headroom for transient spikes.
@modal.concurrent(max_inputs=32)
@modal.web_server(port=SGLANG_PORT, startup_timeout=600)
def serve() -> None:
    """Launch SGLang's OpenAI-compatible server on `SGLANG_PORT`. Modal
    polls the port until it's listening, then routes external HTTP traffic
    to it. The subprocess keeps serving in the background after this
    function returns until the container scales down."""
    import os

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"

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
        # Interleave prefill chunks with running decodes so a new request
        # arriving mid-decode doesn't have to wait for the running batch
        # to finish before its prefill starts.
        "--enable-mixed-chunk",
        # Pack the running batch more aggressively — default 1.0 leaves
        # memory headroom we don't need (workload token usage holds at
        # 0.01-0.10). Pushes SGLang to admit more concurrent requests.
        "--schedule-conservativeness",
        "0.3",
        # Multimodal request-pipeline flags. Added speculatively while
        # debugging an apparent throughput regression after the cu13
        # rebuild (SGLang reported `#running-req: 1` despite c=16 client
        # dispatch). The actual regression turned out to be client-side
        # network bandwidth (200-DPI base64 image uploads bottlenecking
        # the residential uplink at c=16), not server-side preprocessing
        # serialization — switching uplinks restored 1.85 pages/sec mean
        # (vs the 1.46 May 3 baseline). These flags are therefore
        # **unverified** — they didn't hurt, and they're plausibly useful
        # on this workload (image-heavy chat completions), but we don't
        # have a clean A/B. Worth ripping out if they ever cause surprise.
        # `--enable-prefix-mm-cache` was tried here but rejected by
        # SGLang ("requires --encoder-only" / encoder-disaggregation
        # mode) — skip in single-container deploys.
        "--enable-tokenizer-batch-encode",
        "--keep-mm-feature-on-device",
    ]
    print(f"starting sglang: {' '.join(cmd)}")
    subprocess.Popen(cmd)
