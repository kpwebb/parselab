"""Resolve deployed worker endpoint URLs at runtime via the Modal SDK.

Avoids hardcoding workspace-prefixed URLs (which vary per Modal
account) and lets the harness work for anyone who's deployed the
workers without editing source. Looks up the `serve` function of each
deployed app by name and reads its public web URL.

Usage:

    from harness.endpoints import chat_completions_url_for_preset

    url = chat_completions_url_for_preset("inf2-flash")
    # → "https://<your-workspace>--parselab-inf2-flash-serve.modal.run/v1/chat/completions"

The lookup talks to Modal's control plane; it does NOT cold-start the
worker. Cached for the lifetime of the process so repeated calls are
free.
"""
from __future__ import annotations

from functools import lru_cache

import modal


# Preset name → deployed Modal app name. Function name on every web
# worker is `serve` (see each worker's @modal.web_server-decorated
# function).
WORKER_APPS: dict[str, str] = {
    "granite":    "parselab-granite-docling",
    "glm-ocr":    "parselab-glm-ocr",
    "inf2-flash": "parselab-inf2-flash",
    "inf2-pro":   "parselab-infinity-parser2",
    "qwen35":     "parselab-qwen35-9b",
    "qwen36":     "parselab-qwen36-35b-a3b",
}


@lru_cache(maxsize=None)
def chat_completions_url(app_name: str, function_name: str = "serve") -> str:
    """Return `<webserver-url>/v1/chat/completions` for a deployed worker.

    Raises `modal.exception.NotFoundError` if the app/function isn't
    deployed under the caller's current Modal workspace — fix by
    running `modal deploy modal/<worker>/app.py`.
    """
    fn = modal.Function.from_name(app_name, function_name)
    return f"{fn.get_web_url().rstrip('/')}/v1/chat/completions"


def chat_completions_url_for_preset(preset: str) -> str:
    """Look up an endpoint by preset name (see WORKER_APPS)."""
    try:
        app_name = WORKER_APPS[preset]
    except KeyError as e:
        raise KeyError(
            f"unknown preset {preset!r}; known: {sorted(WORKER_APPS)}"
        ) from e
    return chat_completions_url(app_name)
