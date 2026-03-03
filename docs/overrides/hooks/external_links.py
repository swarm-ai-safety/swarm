"""MkDocs hook: add rel="noopener noreferrer" and target="_blank" to external links."""

import re


def on_page_content(html: str, **kwargs) -> str:
    """Post-process rendered HTML to add noopener to external links."""
    def _add_attrs(match: re.Match) -> str:
        tag = match.group(0)
        if 'rel=' in tag:
            return tag
        return tag.replace('>', ' rel="noopener noreferrer" target="_blank">', 1)

    return re.sub(r'<a\s+[^>]*href="https?://[^"]*"[^>]*>', _add_attrs, html)
