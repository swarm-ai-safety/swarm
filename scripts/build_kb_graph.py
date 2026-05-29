#!/usr/bin/env python3
"""Build a knowledge-graph JSON from the docs/ markdown corpus.

Nodes are pages; edges are explicit links between them ([](x.md) and [[wikilinks]]).
The output (docs/assets/kb_graph.json) powers the interactive /graph page and the
per-page backlinks injected by docs/overrides/hooks/kb_graph.py.

This module is imported by the mkdocs hook (so the graph stays fresh on every
build) and is also runnable as a CLI for a connectivity report:

    python scripts/build_kb_graph.py            # write json + print orphan report
    python scripts/build_kb_graph.py --check     # exit 1 if orphans exist (CI gate)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pyyaml ships with mkdocs; degrade gracefully if missing
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUT_PATH = DOCS_DIR / "assets" / "kb_graph.json"

# Built/vendored areas that are not authored knowledge.
EXCLUDE_DIRS = {"game-app", "javascripts", "stylesheets", "overrides", "assets",
                "images", "figures", "sprites", ".well-known"}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# [text](path.md) and [text](path.md#anchor) — capture the target.
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)")
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def _iter_md_files() -> list[Path]:
    out = []
    for p in DOCS_DIR.rglob("*.md"):
        rel_parts = p.relative_to(DOCS_DIR).parts
        if any(part in EXCLUDE_DIRS for part in rel_parts[:-1]):
            continue
        out.append(p)
    return sorted(out)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    body = text[m.end():]
    if yaml is None:
        return {}, body
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        return (meta if isinstance(meta, dict) else {}), body
    except yaml.YAMLError:
        return {}, body


def _title_for(meta: dict, body: str, rel: str) -> str:
    if meta.get("title"):
        return str(meta["title"])
    h1 = H1_RE.search(body)
    if h1:
        return h1.group(1).strip()
    return Path(rel).stem.replace("-", " ").replace("_", " ").title()


def _tags_for(meta: dict) -> list[str]:
    tags: list[str] = []
    for key in ("tags", "keywords", "defined_terms"):
        val = meta.get(key)
        if isinstance(val, list):
            tags.extend(str(v) for v in val)
        elif isinstance(val, str):
            tags.append(val)
    # de-dupe, preserve order
    seen = set()
    out = []
    for t in tags:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _section_for(rel: str) -> str:
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else "root"


def _clean_target(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1]
    # drop anchor / query / surrounding title text after a space
    raw = raw.split(" ", 1)[0]
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    return raw


def build_graph() -> dict:
    files = _iter_md_files()
    # rel-path id (posix), e.g. "concepts/soft-labels.md"
    id_for: dict[Path, str] = {f: f.relative_to(DOCS_DIR).as_posix() for f in files}
    valid_ids = set(id_for.values())
    # stem -> id, for resolving [[wikilinks]] and bare links; title -> id too
    stem_to_id: dict[str, str] = {}
    title_to_id: dict[str, str] = {}

    nodes: dict[str, dict] = {}
    raw_bodies: dict[str, str] = {}

    for f in files:
        nid = id_for[f]
        text = f.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(text)
        title = _title_for(meta, body, nid)
        nodes[nid] = {
            "id": nid,
            "title": title,
            "section": _section_for(nid),
            "description": str(meta.get("description", "")).strip(),
            "tags": _tags_for(meta),
            "out": [],
            "in": [],
            "url": nid[:-3] + "/" if nid.endswith(".md") else nid,
        }
        raw_bodies[nid] = body
        stem_to_id.setdefault(f.stem.lower(), nid)
        title_to_id.setdefault(title.lower(), nid)

    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()

    def add_edge(src: str, dst: str, kind: str) -> None:
        if dst not in nodes or src == dst:
            return
        key = (src, dst)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": src, "target": dst, "kind": kind})
        nodes[src]["out"].append(dst)
        nodes[dst]["in"].append(src)

    for nid, body in raw_bodies.items():
        src_path = DOCS_DIR / nid
        # explicit markdown links to .md targets
        for raw in MD_LINK_RE.findall(body):
            target = _clean_target(raw)
            if not target.endswith(".md"):
                continue
            if target.startswith("http") or target.startswith("//"):
                continue
            resolved = (src_path.parent / target).resolve()
            try:
                rel = resolved.relative_to(DOCS_DIR).as_posix()
            except ValueError:
                continue
            if rel in valid_ids:
                add_edge(nid, rel, "link")
        # [[wikilinks]] resolved by stem or title
        for raw in WIKILINK_RE.findall(body):
            key = raw.strip().lower()
            dst = stem_to_id.get(key) or title_to_id.get(key)
            if dst:
                add_edge(nid, dst, "wikilink")

    # finalize degree + orphan flags
    for n in nodes.values():
        n["outdegree"] = len(n["out"])
        n["indegree"] = len(n["in"])
        n["orphan"] = (n["indegree"] == 0)

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "orphan_count": sum(1 for n in nodes.values() if n["orphan"]),
        },
    }


def write_graph(graph: dict | None = None) -> Path:
    graph = graph or build_graph()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return OUT_PATH


def _report(graph: dict) -> None:
    s = graph["stats"]
    print(f"KB graph: {s['node_count']} nodes, {s['edge_count']} edges, "
          f"{s['orphan_count']} orphans (no inbound links)")
    orphans = [n for n in graph["nodes"] if n["orphan"]]
    if orphans:
        print("\nOrphans / weakly-linked pages (densify these for a connected graph):")
        for n in sorted(orphans, key=lambda x: x["section"]):
            print(f"  [{n['section']}] {n['id']}  (out:{n['outdegree']})")


if __name__ == "__main__":
    g = build_graph()
    write_graph(g)
    _report(g)
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")
    if "--check" in sys.argv and g["stats"]["orphan_count"]:
        sys.exit(1)
