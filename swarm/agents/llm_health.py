"""Health check utilities for LLM providers."""

import logging
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def check_llama_server(
    base_url: str = "http://localhost:8080",
    timeout: float = 5.0,
) -> bool:
    """Ping llama-server /health endpoint.

    Args:
        base_url: Base URL of the llama-server (without /v1 suffix).
        timeout: Request timeout in seconds.

    Returns:
        True if the server is reachable and healthy.
    """
    # Restrict to http/https to prevent SSRF via file:// etc.
    parsed = urlparse(base_url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        logger.warning(
            "Refusing health check: scheme %r not in %s",
            parsed.scheme,
            _ALLOWED_SCHEMES,
        )
        return False

    # Strip /v1 suffix if present so we hit the root health endpoint
    health_url = base_url.rstrip("/")
    if health_url.endswith("/v1"):
        health_url = health_url[: -len("/v1")]
    health_url += "/health"

    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, OSError) as exc:
        logger.debug("llama-server health check failed: %s", exc)
        return False


def require_llama_server(
    base_url: str = "http://localhost:8080",
    timeout: float = 5.0,
    model_hint: Optional[str] = None,
) -> None:
    """Raise with a helpful message if llama-server is unreachable.

    Call this at scenario startup so users get a clear error instead of
    a timeout mid-simulation.
    """
    if check_llama_server(base_url, timeout):
        return

    hint = ""
    if model_hint:
        hint = f" with model '{model_hint}'"
    raise ConnectionError(
        f"Cannot reach llama-server at {base_url}{hint}. "
        "Start it with: llama-server -m <model.gguf> --port 8080\n"
        "Or run: ./scripts/llama-server-setup.sh start"
    )
