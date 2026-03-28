"""MkDocs hook: wrap glossary terms in <dfn> tags and arxiv references in <cite> tags."""

import re

# Terms to wrap in <dfn> — ordered longest-first to avoid partial matches
_GLOSSARY_TERMS = sorted([
    "distributional safety",
    "soft label",
    "soft labels",
    "toxicity rate",
    "quality gap",
    "adverse selection",
    "conditional loss",
    "incoherence index",
    "signal-action divergence",
    "circuit breaker",
    "circuit breakers",
    "transaction tax",
    "reputation decay",
    "collusion detection",
    "random audit",
    "random audits",
    "purity paradox",
    "trust-then-exploit",
    "governance latency",
    "variance amplification",
    "information asymmetry",
    "externality internalization",
    "staking",
], key=len, reverse=True)

# Build regex: match terms not already inside tags
_TERM_PATTERN = re.compile(
    r'(?<![<\w/])(?<!</)\b(' + '|'.join(re.escape(t) for t in _GLOSSARY_TERMS) + r')\b(?![^<]*>)',
    re.IGNORECASE,
)

# Arxiv citation pattern
_ARXIV_PATTERN = re.compile(
    r'(?<!</a>)(?<!">)(arXiv[:\s]+\d{4}\.\d{4,5})',
    re.IGNORECASE,
)


def _skip_code_blocks(html: str) -> list[tuple[str, bool]]:
    """Split HTML into (segment, is_code) pairs to avoid modifying code blocks."""
    parts: list[tuple[str, bool]] = []
    code_pattern = re.compile(r'(<(?:code|pre|script|style)[^>]*>.*?</(?:code|pre|script|style)>)', re.DOTALL | re.IGNORECASE)
    last_end = 0
    for m in code_pattern.finditer(html):
        if m.start() > last_end:
            parts.append((html[last_end:m.start()], False))
        parts.append((m.group(0), True))
        last_end = m.end()
    if last_end < len(html):
        parts.append((html[last_end:], False))
    return parts


def _wrap_terms_in_segment(segment: str) -> str:
    """Wrap first occurrence of each glossary term in <dfn> tags."""
    seen: set[str] = set()

    def _wrap_term(match: re.Match, _seen: set[str] = seen, _seg: str = segment) -> str:
        term = match.group(0)
        term_lower = term.lower()
        if term_lower in _seen:
            return term
        prefix = _seg[:match.start()]
        if '<dfn' in prefix[max(0, len(prefix) - 100):]:
            return term
        _seen.add(term_lower)
        return f'<dfn>{term}</dfn>'

    return _TERM_PATTERN.sub(_wrap_term, segment)


def on_page_content(html: str, **kwargs) -> str:
    """Post-process rendered HTML to add semantic tags."""
    parts = _skip_code_blocks(html)
    result: list[str] = []

    for segment, is_code in parts:
        if is_code:
            result.append(segment)
            continue

        segment = _wrap_terms_in_segment(segment)

        # Wrap arxiv references in <cite>
        segment = _ARXIV_PATTERN.sub(r'<cite>\1</cite>', segment)

        result.append(segment)

    return ''.join(result)
