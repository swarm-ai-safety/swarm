"""MkDocs hook: regenerate the KB graph and inject per-page backlinks.

- on_pre_build: rebuild docs/assets/kb_graph.json so the /graph page is fresh.
- on_page_content: append a "Linked from" section listing inbound links — the
  reverse edges the markdown corpus doesn't surface on its own.

Graph-building logic lives in scripts/build_kb_graph.py (shared with the CLI).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_kb_graph  # noqa: E402

_GRAPH: dict | None = None
_NODES: dict[str, dict] = {}


def on_pre_build(config, **kwargs) -> None:
    global _GRAPH, _NODES
    _GRAPH = build_kb_graph.build_graph()
    build_kb_graph.write_graph(_GRAPH)
    _NODES = {n["id"]: n for n in _GRAPH["nodes"]}


def _rel_link(page_url: str, target_url: str) -> str:
    """Relative href from the current rendered page to another page's URL.

    Material serves directory URLs (e.g. ``concepts/soft-labels/``), so depth is
    the number of path segments in the current page's URL.
    """
    prefix = "../" * page_url.count("/")
    return prefix + target_url


def on_page_content(html: str, page=None, **kwargs) -> str:
    if page is None or not _NODES:
        return html
    node = _NODES.get(page.file.src_uri)
    if not node or not node["in"]:
        return html

    page_url = page.url  # e.g. "concepts/soft-labels/"
    inbound = sorted(
        (_NODES[i] for i in node["in"] if i in _NODES),
        key=lambda n: (n["section"], n["title"].lower()),
    )
    items = "\n".join(
        f'<li><a href="{_rel_link(page_url, n["url"])}">{n["title"]}</a>'
        f' <span class="kb-backlink-section">{n["section"]}</span></li>'
        for n in inbound
    )
    block = (
        '\n<aside class="kb-backlinks" markdown="0">\n'
        f'<h2>Linked from <span class="kb-backlink-count">{len(inbound)}</span></h2>\n'
        f"<ul>\n{items}\n</ul>\n"
        f'<p><a href="{_rel_link(page_url, "graph/")}">Open the knowledge graph →</a></p>\n'
        "</aside>\n"
    )
    return html + block
