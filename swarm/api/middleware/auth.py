"""API key authentication and rate-limiting middleware."""

import hashlib
import hmac
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# In-memory API key store: key_hash -> agent_id
# We store hashes of keys rather than plaintext to limit exposure if
# the process memory is ever dumped.
_api_keys: dict[str, str] = {}

# Reverse map for quick lookup: key_hash -> raw key needed for hmac comparison
# (This is belt-and-suspenders — ideally we'd only store hashes, but we need
# constant-time comparison against the incoming token.)
_api_key_hashes: dict[str, str] = {}  # raw_key -> key_hash

# Rate-limit tracking: key_hash -> list of request timestamps
_rate_limit_windows: dict[str, list[float]] = defaultdict(list)

# Per-key resource quotas (set when key is issued)
_key_quotas: dict[str, dict] = {}

# Trusted keys allowed to publish public results
_trusted_keys: set[str] = set()

# Global caps to prevent unbounded growth
MAX_REGISTERED_KEYS = 10_000
MAX_RATE_LIMIT_ENTRIES_PER_KEY = 200

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _hash_key(raw_key: str) -> str:
    """Produce a deterministic hash of an API key for storage."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


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
    _api_key_hashes[api_key] = key_hash
    _key_quotas[key_hash] = {
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "max_concurrent": max_concurrent,
        "max_runtime_seconds": max_runtime_seconds,
    }
    if trusted:
        _trusted_keys.add(key_hash)


def is_trusted(api_key: str) -> bool:
    """Check whether a key is trusted (can publish public results)."""
    return _hash_key(api_key) in _trusted_keys


def get_quotas(api_key: str) -> dict:
    """Return the resource quotas for a key."""
    return _key_quotas.get(_hash_key(api_key), {})


def _extract_bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Extract token from 'Bearer <token>' header value."""
    if header_value is None:
        return None
    if header_value.startswith("Bearer "):
        return header_value[7:]
    return header_value


def _constant_time_key_lookup(token: str) -> Optional[str]:
    """Look up an API key using constant-time comparison.

    Returns the agent_id if valid, None otherwise.
    """
    # First, compute the hash to narrow down the candidate.
    # SHA-256 is not timing-sensitive for our purposes.
    token_hash = _hash_key(token)

    # Use hmac.compare_digest for the hash comparison to prevent
    # timing side-channels on the hash value itself.
    for stored_hash, agent_id in _api_keys.items():
        if hmac.compare_digest(token_hash, stored_hash):
            return agent_id
    return None


async def require_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(_api_key_header),
) -> str:
    """FastAPI dependency that validates the API key.

    Returns the agent_id associated with the key.
    """
    token = _extract_bearer_token(api_key_header)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    agent_id = _constant_time_key_lookup(token)
    if agent_id is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Rate limiting
    rate_limit = getattr(request.app, "_swarm_rate_limit", 100)
    token_hash = _hash_key(token)
    now = time.monotonic()
    window = _rate_limit_windows[token_hash]

    # Prune entries older than 60 seconds, but cap stored entries to prevent
    # memory abuse from rapid-fire requests that don't get pruned fast enough.
    cutoff = now - 60
    pruned = [t for t in window if t > cutoff]
    if len(pruned) > MAX_RATE_LIMIT_ENTRIES_PER_KEY:
        pruned = pruned[-MAX_RATE_LIMIT_ENTRIES_PER_KEY:]
    _rate_limit_windows[token_hash] = pruned
    window = _rate_limit_windows[token_hash]

    if len(window) >= rate_limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    window.append(now)
    return agent_id
