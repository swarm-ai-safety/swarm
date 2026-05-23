#!/usr/bin/env python3
"""Generate a scored backfill snapshot for the gitlawb live dashboard.

The dashboard at docs/bridges/gitlawb.md cannot fetch node.gitlawb.com directly
from the browser: the site CSP (`connect-src`) and the node's missing CORS
headers both block a cross-origin request. The WebSocket stream still delivers
*live* events, but the page would otherwise load with no history.

This script runs server-side (no CORS/CSP) at deploy time, queries the same
backfill data, scores each event, and writes it next to the built page so the
browser can load it same-origin.

Scoring mirrors the SWARM gitlawb bridge's heuristic
(`swarm.bridges.gitlawb.mapper._heuristic_score`) exactly, but is reimplemented
here with the standard library only. The bridge's heuristic *is* its scoring
method when no LLM judge is configured, so this is faithful to it — and avoids
installing/importing the full swarm package (numpy/pandas/pydantic/gql) in the
docs build, which proved brittle across dependency versions. The page reads the
per-event ``p`` directly (see scoreEvent in gitlawb.md).

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


def _score_ref_update(event: dict) -> float:
    """Mirror of GitlawbMapper heuristic for a ref-update (push)."""
    base = 0.7
    ref = str(event.get("refName") or "").lower()
    if "temp" in ref or "test" in ref or "wip" in ref:
        base -= 0.15
    old_sha = str(event.get("oldSha") or "")
    if old_sha and all(c == "0" for c in old_sha):
        base -= 0.2  # force push / branch creation
    return round(max(0.1, min(0.95, base)), 4)


def _score_task(event: dict) -> float:
    """Mirror of GitlawbMapper heuristic for a task event.

    The bridge keys off the task status (`_heuristic_score` in
    swarm/bridges/gitlawb/mapper.py): completed is a positive signal, failed a
    strong negative. The backfill query returns each task's current ``status``,
    which is the snapshot-equivalent of the live event's ``new_status``.
    """
    status = str(event.get("status") or "").lower()
    if status == "completed":
        return 0.8
    if status == "failed":
        return 0.2
    if status == "claimed":
        return 0.5
    return 0.5


def score(ref_updates: list[dict], tasks: list[dict]) -> dict:
    scored_refs = [{**e, "p": _score_ref_update(e)} for e in ref_updates]
    scored_tasks = [{**t, "p": _score_task(t)} for t in tasks]

    ps = [e["p"] for e in scored_refs] + [t["p"] for t in scored_tasks]
    n = len(ps)
    metrics = {
        "interaction_count": n,
        "average_quality": round(sum(ps) / n, 4) if n else 0.0,
        "toxicity_rate": round(sum(1 for p in ps if p < 0.3) / n, 4) if n else 0.0,
    }
    return {"refUpdates": scored_refs, "tasks": scored_tasks, "metrics": metrics}


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
        result = score(raw["refUpdates"], raw["tasks"])
        snapshot.update(result)
        snapshot["scored"] = True
        snapshot["ok"] = True
        n = len(snapshot["refUpdates"]) + len(snapshot["tasks"])
        print(f"gitlawb snapshot: scored {n} events from {HTTP_URL}")
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
