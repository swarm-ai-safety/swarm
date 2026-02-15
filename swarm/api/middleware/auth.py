"""API key authentication and rate-limiting middleware."""

import hashlib
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# In-memory API key store: key_hash -> agent_id
# We store hashes of keys rather than plaintext to limit exposure if
# the process memory is ever dumped.
_api_keys: dict[str, str] = {}

# Rate-limit tracking: key_hash -> list of request timestamps
_rate_limit_windows: dict[str, list[float]] = defaultdict(list)

# Lock protecting _rate_limit_windows for thread safety (security fix 4.4).
_rate_limit_lock = threading.Lock()

# Per-key resource quotas (set when key is issued)
_key_quotas: dict[str, dict] = {}

# Trusted keys allowed to publish public results
_trusted_keys: set[str] = set()

# Global caps to prevent unbounded growth
MAX_REGISTERED_KEYS = 10_000
MAX_RATE_LIMIT_ENTRIES_PER_KEY = 200

# PBKDF2 parameters for API key hashing.
# Fixed application-level salt (keys are high-entropy UUIDs, so per-key
# salt is unnecessary; fixed salt provides domain separation).
_KDF_SALT = b"swarm-api-key-v1"
_KDF_ITERATIONS = 100_000

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _hash_key(raw_key: str) -> str:
    """Hash an API key using PBKDF2-HMAC-SHA256.

    Uses a computationally expensive KDF rather than plain SHA-256 so that
    brute-forcing dumped hashes is infeasible even for shorter keys.
    """
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        raw_key.encode("utf-8"),
        _KDF_SALT,
        _KDF_ITERATIONS,
    )
    return dk.hex()


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
    if len(_api_keys) >= MAX_REGISTERED_KEYS:
        raise RuntimeError("Maximum number of registered API keys reached")

    key_hash = _hash_key(api_key)
    _api_keys[key_hash] = agent_id
    _key_quotas[key_hash] = {
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "max_concurrent": max_concurrent,
        "max_runtime_seconds": max_runtime_seconds,
    }
    if trusted:
        _trusted_keys.add(key_hash)


def is_trusted(api_key: str) -> bool:
    """Check whether a raw key is trusted (can publish public results).

    Prefer ``is_trusted_hash`` with a pre-computed hash when the hash
    is already available (e.g. from ``get_request_key_hash``).
    """
    return _hash_key(api_key) in _trusted_keys


def is_trusted_hash(key_hash: str) -> bool:
    """Check whether a pre-computed key hash is trusted."""
    return key_hash in _trusted_keys


def get_quotas(api_key: str) -> dict:
    """Return the resource quotas for a raw key.

    Prefer ``get_quotas_hash`` with a pre-computed hash when the hash
    is already available (e.g. from ``get_request_key_hash``).
    """
    return _key_quotas.get(_hash_key(api_key), {})


def get_quotas_hash(key_hash: str) -> dict:
    """Return the resource quotas for a pre-computed key hash."""
    return _key_quotas.get(key_hash, {})


def get_request_key_hash(request: Request) -> str:
    """Retrieve the key hash stashed by ``require_api_key``.

    This avoids re-computing the expensive PBKDF2 hash on every call
    to ``is_trusted`` / ``get_quotas`` within the same request.
    """
    key_hash: str = request.state._swarm_key_hash  # type: ignore[attr-defined]
    return key_hash


def _extract_bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Extract token from 'Bearer <token>' header value."""
    if header_value is None:
        return None
    if header_value.startswith("Bearer "):
        return header_value[7:]
    return header_value


def _lookup_key(token: str) -> tuple[Optional[str], str]:
    """Look up an API key by its PBKDF2 hash.

    Returns (agent_id_or_None, key_hash) so the caller can reuse
    the expensive hash without recomputing.
    """
    token_hash = _hash_key(token)
    return _api_keys.get(token_hash), token_hash


async def require_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(_api_key_header),
) -> str:
    """FastAPI dependency that validates the API key.

    Returns the agent_id associated with the key.  The computed key hash
    is stashed on ``request.state._swarm_key_hash`` so downstream code
    can call ``get_request_key_hash(request)`` instead of re-hashing.
    """
    token = _extract_bearer_token(api_key_header)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    agent_id, token_hash = _lookup_key(token)
    if agent_id is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Stash the hash so the router can call is_trusted_hash / get_quotas_hash
    # without re-computing the expensive PBKDF2.
    request.state._swarm_key_hash = token_hash

    # Rate limiting (thread-safe via _rate_limit_lock, security fix 4.4)
    rate_limit = getattr(request.app, "_swarm_rate_limit", 100)
    now = time.monotonic()

    with _rate_limit_lock:
        window = _rate_limit_windows[token_hash]

        # Prune entries older than 60 seconds, but cap stored entries to prevent
        # memory abuse from rapid-fire requests that don't get pruned fast enough.
        cutoff = now - 60
        pruned = [t for t in window if t > cutoff]
        if len(pruned) > MAX_RATE_LIMIT_ENTRIES_PER_KEY:
            pruned = pruned[-MAX_RATE_LIMIT_ENTRIES_PER_KEY:]

        if len(pruned) >= rate_limit:
            _rate_limit_windows[token_hash] = pruned
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        pruned.append(now)
        _rate_limit_windows[token_hash] = pruned

    return agent_id
