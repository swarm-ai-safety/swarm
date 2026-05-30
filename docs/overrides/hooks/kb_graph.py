"""MkDocs hook: regenerate the multi-source KB graph and inject backlinks.

- on_pre_build: rebuild docs/assets/kb_graph.json so the /graph page is fresh.
- on_page_content: append a "Linked from" section on rendered doc pages listing
  every inbound edge — including from scenarios, slash commands, agents, roles,
  and (when checked out locally) artifacts notes/papers. Non-doc nodes link out
  to their source on GitHub.

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

_KIND_LABEL = {
    "doc": "docs",
    "scenario": "scenario",
    "command": "command",
    "agent": "agent",
    "role": "role",
    "paper-art": "paper",
    "research-art": "research",
    "note-art": "note",
    "code": "source",
}


def on_pre_build(config, **kwargs) -> None:
    global _GRAPH, _NODES
    _GRAPH = build_kb_graph.build_graph()
    build_kb_graph.write_graph(_GRAPH)
    _NODES = {n["id"]: n for n in _GRAPH["nodes"]}


def _href_for(page_url: str, target: dict) -> str:
    """Return an href from the current rendered doc page to any node.

    Doc nodes resolve via mkdocs directory URLs (relative).
    Non-doc nodes link to their GitHub source (absolute).
    """
    if target["kind"] == "doc" and target.get("url"):
        prefix = "../" * page_url.count("/")
        return prefix + target["url"]
    return target.get("external_url") or "#"


def on_page_content(html: str, page=None, **kwargs) -> str:
    if page is None or not _NODES:
        return html
    node = _NODES.get(page.file.src_uri)
    if not node or not node["in"]:
        return html

    page_url = page.url
    inbound = sorted(
        (_NODES[i] for i in node["in"] if i in _NODES),
        key=lambda n: (n["kind"], n["section"], n["title"].lower()),
    )
    items = []
    for n in inbound:
        kind_label = _KIND_LABEL.get(n["kind"], n["kind"])
        external = ' target="_blank" rel="noopener"' if n["kind"] != "doc" else ""
        items.append(
            f'<li><a href="{_href_for(page_url, n)}"{external}>{n["title"]}</a>'
            f' <span class="kb-backlink-section">{kind_label}</span></li>'
        )

    graph_href = "../" * page_url.count("/") + "graph/"
    block = (
        '\n<aside class="kb-backlinks" markdown="0">\n'
        f'<h2>Linked from <span class="kb-backlink-count">{len(inbound)}</span></h2>\n'
        "<ul>\n" + "\n".join(items) + "\n</ul>\n"
        f'<p><a href="{graph_href}">Open the knowledge graph →</a></p>\n'
        "</aside>\n"
    )
    return html + block
