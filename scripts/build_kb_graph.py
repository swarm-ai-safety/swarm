#!/usr/bin/env python3
"""Build a knowledge-graph JSON spanning multiple sources in this repo.

Source providers contribute nodes + edges to a shared graph:
  - docs        : docs/**/*.md (rendered by mkdocs)            kind=doc
  - scenarios   : scenarios/*.yaml                              kind=scenario
  - commands    : .claude/commands/*.md (slash commands)        kind=command
  - claude-agents: .claude/agents/*.md (specialist roles)       kind=agent
  - roles       : agents/**/*.md (CEO/engineer/etc. role docs)  kind=role
  - artifacts   : swarm-artifacts/{papers,research,notes}/*.md  kind=paper-art|research-art|note-art
                  (local-only — only included if the artifacts repo is
                  checked out alongside this one)

Edges are inferred from:
  - explicit markdown links to .md / .yaml targets that resolve to a node id
  - [[wikilinks]] resolved by filename stem
  - /slash-command mentions resolving to command nodes
  - bare file-path mentions (scenarios/foo.yaml, .claude/commands/bar.md, etc.)

The output (docs/assets/kb_graph.json) powers the interactive /graph page and
the per-page backlinks injected by docs/overrides/hooks/kb_graph.py.

Run as CLI for a connectivity report:
    python scripts/build_kb_graph.py
    python scripts/build_kb_graph.py --check   # exit 1 if any orphans
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pyyaml ships with mkdocs
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUT_PATH = DOCS_DIR / "assets" / "kb_graph.json"

ARTIFACTS_DIR = Path(os.environ.get("SWARM_ARTIFACTS", REPO_ROOT.parent / "swarm-artifacts"))

GITHUB_BLOB_REPO = "https://github.com/swarm-ai-safety/swarm/blob/main"
GITHUB_BLOB_ARTIFACTS = "https://github.com/swarm-ai-safety/swarm-artifacts/blob/main"

# Excluded subdirs under docs/ that hold built/vendored assets.
DOC_EXCLUDE = {"game-app", "javascripts", "stylesheets", "overrides", "assets",
               "images", "figures", "sprites", ".well-known"}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)")
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
SLASHCMD_RE = re.compile(r"(?<![A-Za-z0-9_/])/([a-z][a-z0-9_-]+)\b")
# mkdocstrings directive in api docs, e.g.  "::: swarm.core.proxy.ProxyComputer"
MKDOCSTRINGS_RE = re.compile(r"^:::\s*([a-z_][a-z0-9_.]+)", re.MULTILINE)
PATH_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9_/])"
    r"((?:scenarios|\.claude/commands|\.claude/agents|agents|docs)/[A-Za-z0-9_./-]+\.(?:md|yaml))"
)


@dataclass
class Node:
    id: str
    kind: str            # doc | scenario | command | agent | role | paper-art | research-art | note-art
    title: str
    section: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    url: str | None = None              # internal mkdocs URL if rendered
    external_url: str | None = None     # GitHub blob URL for non-rendered nodes
    source_path: str = ""               # repo-relative
    source_repo: str = "this"           # "this" or "artifacts"
    text: str = field(default="", repr=False)  # body kept on the node for edge inference, stripped before JSON


# ---------- frontmatter / title helpers ----------

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


def _title_from(meta: dict, body: str, fallback: str) -> str:
    if meta.get("title"):
        return str(meta["title"])
    h1 = H1_RE.search(body)
    if h1:
        return h1.group(1).strip()
    return Path(fallback).stem.replace("-", " ").replace("_", " ").title()


def _collect_tags(meta: dict) -> list[str]:
    tags: list[str] = []
    for key in ("tags", "keywords", "defined_terms", "motif"):
        val = meta.get(key)
        if isinstance(val, list):
            tags.extend(str(v) for v in val)
        elif isinstance(val, str):
            tags.append(val)
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


# ---------- source providers ----------

def _docs_nodes() -> list[Node]:
    nodes: list[Node] = []
    for p in sorted(DOCS_DIR.rglob("*.md")):
        rel = p.relative_to(DOCS_DIR).as_posix()
        if any(part in DOC_EXCLUDE for part in Path(rel).parts[:-1]):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(text)
        section = Path(rel).parts[0] if "/" in rel else "root"
        nodes.append(Node(
            id=rel, kind="doc",
            title=_title_from(meta, body, rel),
            section=section,
            description=str(meta.get("description", "")).strip(),
            tags=_collect_tags(meta),
            url=rel[:-3] + "/" if rel.endswith(".md") else rel,
            source_path=f"docs/{rel}",
            text=body,
        ))
    return nodes


def _scenario_nodes() -> list[Node]:
    sdir = REPO_ROOT / "scenarios"
    if not sdir.is_dir():
        return []
    out: list[Node] = []
    for p in sorted(sdir.glob("*.yaml")):
        text = p.read_text(encoding="utf-8", errors="replace")
        meta = {}
        if yaml is not None:
            try:
                meta = yaml.safe_load(text) or {}
                if not isinstance(meta, dict):
                    meta = {}
            except yaml.YAMLError:
                meta = {}
        rel = f"scenarios/{p.name}"
        title = str(meta.get("scenario_id") or meta.get("name") or p.stem).strip()
        out.append(Node(
            id=rel, kind="scenario",
            title=title,
            section="scenarios",
            description=str(meta.get("description", "")).strip(),
            tags=_collect_tags(meta),
            external_url=f"{GITHUB_BLOB_REPO}/{rel}",
            source_path=rel,
            text=text,
        ))
    return out


def _command_nodes() -> list[Node]:
    cdir = REPO_ROOT / ".claude" / "commands"
    if not cdir.is_dir():
        return []
    out: list[Node] = []
    for p in sorted(cdir.glob("*.md")):
        text = p.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(text)
        rel = f".claude/commands/{p.name}"
        out.append(Node(
            id=f"cmd/{p.stem}", kind="command",
            title=f"/{p.stem}",
            section="commands",
            description=str(meta.get("description", "")).strip() or _first_para(body),
            tags=_collect_tags(meta),
            external_url=f"{GITHUB_BLOB_REPO}/{rel}",
            source_path=rel,
            text=body,
        ))
    return out


def _agent_nodes() -> list[Node]:
    adir = REPO_ROOT / ".claude" / "agents"
    if not adir.is_dir():
        return []
    out: list[Node] = []
    for p in sorted(adir.glob("*.md")):
        text = p.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(text)
        rel = f".claude/agents/{p.name}"
        out.append(Node(
            id=f"agent/{p.stem}", kind="agent",
            title=_title_from(meta, body, p.stem),
            section="agents",
            description=str(meta.get("description", "")).strip() or _first_para(body),
            tags=_collect_tags(meta),
            external_url=f"{GITHUB_BLOB_REPO}/{rel}",
            source_path=rel,
            text=body,
        ))
    return out


def _role_nodes() -> list[Node]:
    rdir = REPO_ROOT / "agents"
    if not rdir.is_dir():
        return []
    out: list[Node] = []
    for p in sorted(rdir.rglob("*.md")):
        rel = p.relative_to(REPO_ROOT).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(text)
        # role/<agent>/<filename-no-ext>
        nid = "role/" + p.relative_to(rdir).with_suffix("").as_posix()
        out.append(Node(
            id=nid, kind="role",
            title=_title_from(meta, body, p.stem),
            section="roles",
            description=str(meta.get("description", "")).strip() or _first_para(body),
            tags=_collect_tags(meta),
            external_url=f"{GITHUB_BLOB_REPO}/{rel}",
            source_path=rel,
            text=body,
        ))
    return out


def _artifacts_nodes() -> list[Node]:
    if not ARTIFACTS_DIR.is_dir():
        return []
    out: list[Node] = []
    for sub, kind in (("papers", "paper-art"),
                      ("research", "research-art"),
                      ("notes", "note-art")):
        d = ARTIFACTS_DIR / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            text = p.read_text(encoding="utf-8", errors="replace")
            meta, body = _parse_frontmatter(text)
            rel = f"{sub}/{p.name}"
            out.append(Node(
                id=f"art/{rel}", kind=kind,
                title=_title_from(meta, body, p.stem),
                section=f"artifacts:{sub}",
                description=str(meta.get("description", "")).strip(),
                tags=_collect_tags(meta),
                external_url=f"{GITHUB_BLOB_ARTIFACTS}/{rel}",
                source_path=f"swarm-artifacts/{rel}",
                source_repo="artifacts",
                text=body,
            ))
    return out


def _first_para(body: str) -> str:
    """First non-heading paragraph, used as a fallback description."""
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if chunk and not chunk.startswith(("#", "```", "---", "<")):
            return " ".join(chunk.split())[:300]
    return ""


# ---------- edge inference ----------

def _resolve_md_link(src_node: Node, raw: str, by_doc_rel: dict[str, str],
                     by_path: dict[str, str]) -> str | None:
    """Resolve a markdown-link target to a node id, if it points at one."""
    raw = raw.strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1]
    raw = raw.split(" ", 1)[0].split("#", 1)[0].split("?", 1)[0]
    if not raw or raw.startswith(("http://", "https://", "mailto:", "//")):
        return None

    # If the source is a doc, treat links as relative to docs/<src dir>.
    if src_node.kind == "doc":
        if not (raw.endswith(".md") or raw.endswith(".yaml")):
            return None
        src_dir = Path(src_node.source_path).parent
        target = (REPO_ROOT / src_dir / raw).resolve()
        try:
            rel = target.relative_to(DOCS_DIR).as_posix()
        except ValueError:
            try:
                rel_repo = target.relative_to(REPO_ROOT).as_posix()
                return by_path.get(rel_repo)
            except ValueError:
                return None
        return by_doc_rel.get(rel)

    # For non-doc sources, try interpreting as a repo-relative path.
    return by_path.get(raw)


def _resolve_python_path(dotted: str) -> Path | None:
    """Map a dotted name (swarm.core.proxy.ProxyComputer) to a source file.

    Tries progressively shorter dotted prefixes, stopping at the first one
    that resolves to either <prefix>.py or <prefix>/__init__.py under the
    repo root. Returns None if nothing resolves (likely a typo in a directive).
    """
    parts = dotted.split(".")
    while parts:
        as_file = REPO_ROOT.joinpath(*parts).with_suffix(".py")
        as_pkg = REPO_ROOT.joinpath(*parts, "__init__.py")
        if as_file.is_file():
            return as_file
        if as_pkg.is_file():
            return as_pkg
        parts.pop()
    return None


def _code_pass(nodes: list[Node]) -> list[tuple[str, str]]:
    """Mine api docs for mkdocstrings directives.

    Returns the explicit (src_doc_id, code_node_id) edges to add. Mutates
    ``nodes`` in place to register a new code node per unique source file
    referenced. Code nodes are linked to GitHub source for the *file* (not
    the dotted symbol), which is what the graph actually navigates between.
    """
    by_id = {n.id: n for n in nodes}
    edges: list[tuple[str, str]] = []
    for n in nodes:
        if n.kind != "doc" or n.section != "api":
            continue
        for dotted in MKDOCSTRINGS_RE.findall(n.text):
            src = _resolve_python_path(dotted)
            if not src:
                continue
            rel = src.relative_to(REPO_ROOT).as_posix()
            cid = f"code/{rel}"
            if cid not in by_id:
                code_node = Node(
                    id=cid, kind="code",
                    title=rel,
                    section="code",
                    description=_module_doc(src),
                    external_url=f"{GITHUB_BLOB_REPO}/{rel}",
                    source_path=rel,
                )
                nodes.append(code_node)
                by_id[cid] = code_node
            edges.append((n.id, cid))
    return edges


def _module_doc(src: Path) -> str:
    """Best-effort: the module-level docstring of a python file, single line."""
    try:
        text = src.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.match(r'\s*("""|\'\'\')(.*?)\1', text, re.DOTALL)
    if not m:
        return ""
    return " ".join(m.group(2).strip().split())[:300]


def build_graph() -> dict:
    nodes: list[Node] = []
    for fn in (_docs_nodes, _scenario_nodes, _command_nodes,
               _agent_nodes, _role_nodes, _artifacts_nodes):
        nodes.extend(fn())

    explicit_edges = _code_pass(nodes)
    by_id: dict[str, Node] = {n.id: n for n in nodes}

    # Lookup tables for resolution.
    by_doc_rel: dict[str, str] = {n.id: n.id for n in nodes if n.kind == "doc"}
    by_path: dict[str, str] = {n.source_path: n.id for n in nodes}
    by_stem: dict[str, str] = {}
    for n in nodes:
        stem = Path(n.source_path).stem.lower()
        by_stem.setdefault(stem, n.id)
    by_cmd: dict[str, str] = {n.id.split("/", 1)[1].lower(): n.id
                              for n in nodes if n.kind == "command"}

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    out_by: dict[str, list[str]] = {n.id: [] for n in nodes}
    in_by: dict[str, list[str]] = {n.id: [] for n in nodes}

    def add(src: str, dst: str, kind: str) -> None:
        if dst not in by_id or src == dst:
            return
        key = (src, dst)
        if key in seen:
            return
        seen.add(key)
        edges.append({"source": src, "target": dst, "kind": kind})
        # Semantic suggestions are TF-IDF hints, not real corpus links. They
        # render in the viz (from `edges`) but must stay out of per-node
        # in/out lists so they never appear as "Linked from" backlinks and
        # never mask a real orphan from --check / "Show orphans".
        if kind == "semantic":
            return
        out_by[src].append(dst)
        in_by[dst].append(src)

    # explicit code edges from the api-docs mkdocstrings pass
    for src, dst in explicit_edges:
        add(src, dst, "code")

    for n in nodes:
        body = n.text
        # explicit markdown / yaml links
        for raw in MD_LINK_RE.findall(body):
            tgt = _resolve_md_link(n, raw, by_doc_rel, by_path)
            if tgt:
                add(n.id, tgt, "link")
        # [[wikilinks]] resolved by stem
        for raw in WIKILINK_RE.findall(body):
            tgt = by_stem.get(raw.strip().lower())
            if tgt:
                add(n.id, tgt, "wikilink")
        # /slash-command mentions resolve to command nodes
        for cmd in SLASHCMD_RE.findall(body):
            tgt = by_cmd.get(cmd.lower())
            if tgt:
                add(n.id, tgt, "slashcmd")
        # bare repo-relative path mentions
        for raw in PATH_MENTION_RE.findall(body):
            tgt = by_path.get(raw)
            if tgt:
                add(n.id, tgt, "mention")

    # suggest semantic edges where no explicit link exists yet
    for src, dst in _semantic_edges(nodes, seen):
        add(src, dst, "semantic")

    # finalize + drop body text
    out_nodes = []
    for n in nodes:
        out_nodes.append({
            "id": n.id, "kind": n.kind, "title": n.title, "section": n.section,
            "description": n.description, "tags": n.tags,
            "url": n.url, "external_url": n.external_url,
            "source_path": n.source_path, "source_repo": n.source_repo,
            "in": in_by[n.id], "out": out_by[n.id],
            "indegree": len(in_by[n.id]), "outdegree": len(out_by[n.id]),
            "orphan": len(in_by[n.id]) == 0,
        })

    return {
        "nodes": out_nodes,
        "edges": edges,
        "stats": {
            "node_count": len(out_nodes),
            "edge_count": len(edges),
            "orphan_count": sum(1 for n in out_nodes if n["orphan"]),
            "by_kind": _count_by(out_nodes, "kind"),
            "artifacts_included": any(n["source_repo"] == "artifacts" for n in out_nodes),
        },
    }


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOPWORDS = frozenset("""
the and for with this that from your you our are not has have was were
which what where when who how all any can use using used into out
about other than then them they our its our some such only most more
will would could should may might must shall use one two three
swarm scenario agent agents page docs doc note notes paper papers
the_ but each etc i_e if it_ no on or so to up via was via vs
""".split())


def _tfidf_vectors(nodes: list[Node]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    import math
    from collections import Counter

    texts: dict[str, list[str]] = {}
    for n in nodes:
        body = " ".join([n.title, n.description, n.text[:1500], " ".join(n.tags)])
        toks = [t.lower() for t in _TOKEN_RE.findall(body)]
        toks = [t for t in toks if t not in _STOPWORDS]
        texts[n.id] = toks

    df: Counter[str] = Counter()
    for toks in texts.values():
        df.update(set(toks))
    n_docs = max(1, len(texts))
    idf = {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}

    vectors: dict[str, dict[str, float]] = {}
    norms: dict[str, float] = {}
    for nid, toks in texts.items():
        tf = Counter(toks)
        if not tf:
            vectors[nid] = {}
            norms[nid] = 0.0
            continue
        v = {t: (c / len(toks)) * idf.get(t, 0.0) for t, c in tf.items()}
        # keep only the top 60 dims per doc for speed + a denoising effect
        if len(v) > 60:
            top = sorted(v.items(), key=lambda kv: kv[1], reverse=True)[:60]
            v = dict(top)
        norms[nid] = math.sqrt(sum(x * x for x in v.values()))
        vectors[nid] = v
    return vectors, norms


def _semantic_edges(nodes: list[Node], existing: set[tuple[str, str]],
                    k: int = 3, threshold: float = 0.18) -> list[tuple[str, str]]:
    """Suggest up to k semantic edges per node where no explicit link exists.

    Uses a simple TF-IDF cosine over title+description+body snippet+tags. Skips
    code nodes as targets (they already have precise links) and skips pairs
    that already share an explicit edge in either direction.
    """
    vectors, norms = _tfidf_vectors(nodes)
    by_id = {n.id: n for n in nodes}
    out: list[tuple[str, str]] = []
    # Build a token -> nodes index for sparse candidate generation.
    inv: dict[str, list[str]] = {}
    for nid, v in vectors.items():
        for t in v:
            inv.setdefault(t, []).append(nid)

    for n in nodes:
        if not vectors[n.id] or norms[n.id] == 0:
            continue
        # candidate set: any node sharing at least one token (sparse)
        cand: set[str] = set()
        for t in vectors[n.id]:
            cand.update(inv.get(t, ()))
        cand.discard(n.id)

        scored: list[tuple[float, str]] = []
        v1, n1 = vectors[n.id], norms[n.id]
        for cid in cand:
            tgt = by_id.get(cid)
            if not tgt or tgt.kind == "code":
                continue
            if (n.id, cid) in existing or (cid, n.id) in existing:
                continue
            v2, n2 = vectors[cid], norms[cid]
            if n2 == 0:
                continue
            # cosine via sparse dict dot product
            if len(v1) > len(v2):
                small, big = v2, v1
            else:
                small, big = v1, v2
            dot = sum(w * big.get(t, 0.0) for t, w in small.items())
            sim = dot / (n1 * n2)
            if sim >= threshold:
                scored.append((sim, cid))
        scored.sort(reverse=True)
        for _, cid in scored[:k]:
            out.append((n.id, cid))
    return out


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it[key]] = out.get(it[key], 0) + 1
    return out


def write_graph(graph: dict | None = None) -> Path:
    graph = graph or build_graph()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return OUT_PATH


def _report(graph: dict) -> None:
    s = graph["stats"]
    print(f"KB graph: {s['node_count']} nodes, {s['edge_count']} edges, "
          f"{s['orphan_count']} orphans")
    print("By kind:", ", ".join(f"{k}={v}" for k, v in sorted(s["by_kind"].items())))
    print(f"Artifacts included: {s['artifacts_included']}")
    orphans = [n for n in graph["nodes"] if n["orphan"]]
    if orphans:
        # Surface only the first 30 to keep CLI output readable.
        print(f"\nFirst {min(30, len(orphans))} orphans (no inbound):")
        for n in sorted(orphans, key=lambda x: (x["kind"], x["section"]))[:30]:
            print(f"  [{n['kind']}/{n['section']}] {n['id']}  (out:{n['outdegree']})")


if __name__ == "__main__":
    g = build_graph()
    write_graph(g)
    _report(g)
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")
    if "--check" in sys.argv and g["stats"]["orphan_count"]:
        sys.exit(1)
