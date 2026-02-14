"""API key authentication and rate-limiting middleware."""

import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# In-memory API key store: key -> agent_id
_api_keys: dict[str, str] = {}

# Rate-limit tracking: key -> list of request timestamps
_rate_limit_windows: dict[str, list[float]] = defaultdict(list)

# Per-key resource quotas (set when key is issued)
_key_quotas: dict[str, dict] = {}

# Trusted keys allowed to publish public results
_trusted_keys: set[str] = set()

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def register_api_key(
    api_key: str,
    agent_id: str,
    *,
    trusted: bool = False,
    max_epochs: int = 100,
    max_steps: int = 100,
    max_concurrent: int = 5,
    max_runtime_seconds: int = 600,
) -> None:
    """Register an API key with its associated agent and quotas."""
    _api_keys[api_key] = agent_id
    _key_quotas[api_key] = {
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "max_concurrent": max_concurrent,
        "max_runtime_seconds": max_runtime_seconds,
    }
    if trusted:
        _trusted_keys.add(api_key)


def is_trusted(api_key: str) -> bool:
    """Check whether a key is trusted (can publish public results)."""
    return api_key in _trusted_keys


def get_quotas(api_key: str) -> dict:
    """Return the resource quotas for a key."""
    return _key_quotas.get(api_key, {})


def _extract_bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Extract token from 'Bearer <token>' header value."""
    if header_value is None:
        return None
    if header_value.startswith("Bearer "):
        return header_value[7:]
    return header_value


async def require_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(_api_key_header),
) -> str:
    """FastAPI dependency that validates the API key.

    Returns the agent_id associated with the key.
    """
    token = _extract_bearer_token(api_key_header)
    if token is None or token not in _api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Rate limiting
    rate_limit = getattr(request.app, "_swarm_rate_limit", 100)
    now = time.monotonic()
    window = _rate_limit_windows[token]

    # Prune entries older than 60 seconds
    cutoff = now - 60
    _rate_limit_windows[token] = [t for t in window if t > cutoff]
    window = _rate_limit_windows[token]

    if len(window) >= rate_limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    window.append(now)
    return _api_keys[token]
