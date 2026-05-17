"""SWARM API middleware."""

from swarm.api.middleware.auth import (
    DEFAULT_SCOPES,
    Scope,
    _api_keys,
    _key_quotas,
    _key_scopes,
    _rate_limit_lock,
    _rate_limit_windows,
    _trusted_keys,
    get_quotas,
    get_quotas_hash,
    get_request_key_hash,
    get_scopes_hash,
    is_trusted,
    is_trusted_hash,
    register_api_key,
    require_api_key,
    require_scope,
)
from swarm.api.middleware.trace import TraceIDMiddleware

__all__ = [
    "DEFAULT_SCOPES",
    "Scope",
    "TraceIDMiddleware",
    "_api_keys",
    "_key_quotas",
    "_key_scopes",
    "_rate_limit_lock",
    "_rate_limit_windows",
    "_trusted_keys",
    "get_quotas",
    "get_quotas_hash",
    "get_request_key_hash",
    "get_scopes_hash",
    "is_trusted",
    "is_trusted_hash",
    "register_api_key",
    "require_api_key",
    "require_scope",
]
