#!/usr/bin/env python3
"""Generate a same-origin backfill snapshot for the gitlawb live dashboard.

The dashboard at docs/bridges/gitlawb.md cannot fetch node.gitlawb.com directly
from the browser: the site's CSP (`connect-src`) and the node's missing CORS
headers both block a cross-origin request. The WebSocket stream still delivers
*live* events once CSP allows the origin, but the page would otherwise load with
no history.

This script runs at deploy time (server-side, no CORS/CSP), queries the same
backfill data the page used to fetch, and writes it next to the built page so the
browser can load it same-origin. It is deliberately fail-safe: any network error
produces an empty-but-valid snapshot so the deploy never breaks.
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


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("site/bridges/gitlawb_snapshot.json")

    snapshot = {"refUpdates": [], "tasks": [], "generatedAt": None, "ok": False}
    try:
        result = fetch_backfill()
        snapshot.update(result)
        snapshot["ok"] = True
        n = len(snapshot["refUpdates"]) + len(snapshot["tasks"])
        print(f"gitlawb snapshot: fetched {n} events from {HTTP_URL}")
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        # Never fail the build on a node outage — ship an empty snapshot.
        print(f"gitlawb snapshot: backfill unavailable ({e!r}); writing empty snapshot", file=sys.stderr)

    snapshot["generatedAt"] = datetime.now(timezone.utc).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot), encoding="utf-8")
    print(f"gitlawb snapshot: wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
