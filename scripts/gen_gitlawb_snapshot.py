#!/usr/bin/env python3
"""Generate a backfill snapshot for the gitlawb live dashboard.

The dashboard at docs/bridges/gitlawb.md cannot fetch node.gitlawb.com directly
from the browser: the site CSP (`connect-src`) and the node's missing CORS
headers both block a cross-origin request. The WebSocket stream still delivers
*live* events, but the page would otherwise load with no history.

This script runs server-side (no CORS/CSP), queries the same backfill data, and
writes it next to the built page so the browser can load it same-origin.

It has two scoring tiers:

  * Scored (option C): if the ``swarm`` package is importable, each event is run
    through the real ``GitlawbMapper``/``GitlawbMetrics`` bridge so every record
    carries a SoftMetrics quality probability ``p`` and the snapshot includes an
    aggregate metrics block. This is what the scheduled GitHub Action produces.
  * Raw fallback: in a minimal environment without ``swarm`` (e.g. the Vercel
    docs build venv), events are emitted without ``p`` and the page falls back
    to its in-browser heuristic.

It is deliberately fail-safe: any network error produces an empty-but-valid
snapshot so a deploy never breaks.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HTTP_URL = "https://node.gitlawb.com/graphql"
QUERY = (
    "query { "
    "refUpdates(limit: 20) { repo refName oldSha newSha pusherDid nodeDid timestamp } "
    "tasks(limit: 20) { id status delegatorDid assigneeDid createdAt } "
    "}"
)
TIMEOUT_S = 15


def fetch_backfill() -> dict:
    body = json.dumps({"query": QUERY}).encode("utf-8")
    req = urllib.request.Request(
        HTTP_URL,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") or {}
    return {
        "refUpdates": data.get("refUpdates") or [],
        "tasks": data.get("tasks") or [],
    }


def score_with_bridge(ref_updates: list[dict], tasks: list[dict]) -> dict | None:
    """Score events via the real SWARM gitlawb bridge.

    Returns a dict with per-event ``p`` attached and an aggregate ``metrics``
    block, or ``None`` if the swarm package is not importable (raw fallback).
    """
    # Ensure the repo root is importable even when run as `python scripts/...`
    # (which puts scripts/ — not the repo root — on sys.path).
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import asyncio

        from swarm.bridges.gitlawb.config import GitlawbConfig
        from swarm.bridges.gitlawb.mapper import GitlawbMapper
        from swarm.bridges.gitlawb.metrics import GitlawbMetrics
    except ImportError:
        return None

    # Heuristic scoring only — deterministic, no API key / network judge.
    config = GitlawbConfig(use_llm_judge=False)
    mapper = GitlawbMapper(config)
    metrics = GitlawbMetrics(config)

    async def _run() -> dict:
        interactions = []
        scored_refs = []
        for event in ref_updates:
            interaction = await mapper.enrich(mapper.map_ref_update(event))
            interactions.append(interaction)
            scored_refs.append({**event, "p": round(interaction.p, 4)})

        scored_tasks = []
        for task in tasks:
            interaction = await mapper.enrich(mapper.map_task_creation(task))
            interactions.append(interaction)
            scored_tasks.append({**task, "p": round(interaction.p, 4)})

        report = metrics.compute(interactions)
        return {
            "refUpdates": scored_refs,
            "tasks": scored_tasks,
            "metrics": report.to_dict(),
        }

    return asyncio.run(_run())


def main() -> int:
    out_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("site/bridges/gitlawb_snapshot.json")
    )

    snapshot: dict = {
        "refUpdates": [],
        "tasks": [],
        "metrics": None,
        "scored": False,
        "generatedAt": None,
        "ok": False,
    }
    try:
        raw = fetch_backfill()
        scored = score_with_bridge(raw["refUpdates"], raw["tasks"])
        if scored is not None:
            snapshot["refUpdates"] = scored["refUpdates"]
            snapshot["tasks"] = scored["tasks"]
            snapshot["metrics"] = scored["metrics"]
            snapshot["scored"] = True
            tier = "scored (SoftMetrics)"
        else:
            snapshot["refUpdates"] = raw["refUpdates"]
            snapshot["tasks"] = raw["tasks"]
            tier = "raw (swarm not importable)"
        snapshot["ok"] = True
        n = len(snapshot["refUpdates"]) + len(snapshot["tasks"])
        print(f"gitlawb snapshot: {tier} — {n} events from {HTTP_URL}")
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        # Never fail the build on a node outage — ship an empty snapshot.
        print(
            f"gitlawb snapshot: backfill unavailable ({e!r}); writing empty snapshot",
            file=sys.stderr,
        )

    snapshot["generatedAt"] = datetime.now(timezone.utc).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot), encoding="utf-8")
    print(f"gitlawb snapshot: wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
