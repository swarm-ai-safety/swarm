"""Unit tests for dispatch_path construction in AWM adapter_server.

Tests focus on the trailing-slash preservation fix and query-string
validation logic introduced alongside the SSRF-hardening changes.
"""

import posixpath

import pytest


def _build_dispatch_path(rendered_path: str) -> str | None:
    """Mirror the dispatch_path construction logic from adapter_server.call_tool.

    Returns the dispatch_path string, or None if validation should reject it.
    """
    path_only, sep, query = rendered_path.partition("?")
    normalized = posixpath.normpath(path_only)
    if not normalized.startswith("/"):
        return None

    allowed_path_chars = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "/._-~"
    )
    if any(ch not in allowed_path_chars for ch in normalized):
        return None

    if query:
        allowed_query_chars = allowed_path_chars | set("=&+%@!$'()*,;:")
        if any(ch not in allowed_query_chars for ch in query):
            return None

    # Preserve trailing slash from the original path
    trailing_slash = "/" if path_only.endswith("/") and normalized != "/" else ""
    return normalized + trailing_slash + sep + query


class TestDispatchPathTrailingSlash:
    """Trailing slash is preserved when building dispatch_path."""

    def test_path_without_trailing_slash_unchanged(self):
        assert _build_dispatch_path("/foo/bar") == "/foo/bar"

    def test_path_with_trailing_slash_preserved(self):
        assert _build_dispatch_path("/foo/bar/") == "/foo/bar/"

    def test_root_slash_not_doubled(self):
        # "/" normpath → "/" — must not become "//"
        assert _build_dispatch_path("/") == "/"

    def test_single_segment_with_trailing_slash(self):
        assert _build_dispatch_path("/items/") == "/items/"

    def test_nested_path_with_trailing_slash(self):
        assert _build_dispatch_path("/api/v1/users/") == "/api/v1/users/"

    def test_query_string_preserved_with_trailing_slash(self):
        assert _build_dispatch_path("/items/?page=2") == "/items/?page=2"

    def test_query_string_preserved_without_trailing_slash(self):
        assert _build_dispatch_path("/items?page=2") == "/items?page=2"

    def test_normpath_collapses_dot_segments(self):
        # /foo/./bar/ → /foo/bar/ (dot collapsed, trailing slash preserved)
        assert _build_dispatch_path("/foo/./bar/") == "/foo/bar/"


class TestDispatchPathQueryValidation:
    """Query strings with invalid characters are rejected."""

    def test_valid_query_accepted(self):
        assert _build_dispatch_path("/items?id=1&sort=asc") == "/items?id=1&sort=asc"

    def test_query_with_newline_rejected(self):
        assert _build_dispatch_path("/items?\nid=1") is None

    def test_query_with_space_rejected(self):
        assert _build_dispatch_path("/items?foo bar=1") is None

    def test_query_with_control_char_rejected(self):
        assert _build_dispatch_path("/items?foo\x00=1") is None


class TestDispatchPathCharWhitelist:
    """Path character whitelist is applied to the normalized path."""

    def test_valid_path_accepted(self):
        assert _build_dispatch_path("/valid-path_1.2~3") == "/valid-path_1.2~3"

    def test_path_with_space_rejected(self):
        assert _build_dispatch_path("/foo bar") is None

    def test_path_with_hash_rejected(self):
        assert _build_dispatch_path("/foo#bar") is None
