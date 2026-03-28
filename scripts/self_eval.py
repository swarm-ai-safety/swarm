#!/usr/bin/env python3
"""Self-evaluation summary generator for Paperclip agent heartbeats.

Queries the Paperclip API for an agent's recent task history and generates
a concise self-evaluation summary. When PerformanceTracker JSONL logs are
available (from SWA-51), incorporates those metrics too.

Usage:
    python scripts/self_eval.py                    # uses env vars
    python scripts/self_eval.py --last-n 10        # last 10 completed tasks
    python scripts/self_eval.py --tracker-log path  # include PerformanceTracker data

Environment variables (auto-injected by Paperclip adapter):
    PAPERCLIP_API_URL, PAPERCLIP_API_KEY, PAPERCLIP_AGENT_ID, PAPERCLIP_COMPANY_ID
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def api_get(url: str, api_key: str) -> Any:
    """GET request to Paperclip API."""
    req = Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_recent_issues(
    api_url: str, api_key: str, company_id: str, agent_id: str, status: str,
) -> list[dict]:
    """Fetch issues for this agent with given status."""
    url = (
        f"{api_url}/api/companies/{company_id}/issues"
        f"?assigneeAgentId={agent_id}&status={status}"
    )
    return api_get(url, api_key)


def load_tracker_log(path: Path, last_n: int) -> list[dict]:
    """Load the last N entries from a PerformanceTracker JSONL file."""
    if not path.exists():
        return []
    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries[-last_n:]


def compute_summary(
    done_issues: list[dict],
    blocked_issues: list[dict],
    in_progress_issues: list[dict],
    tracker_entries: list[dict],
    last_n: int,
) -> str:
    """Generate a markdown self-evaluation summary."""
    lines: list[str] = []
    lines.append("## Self-Evaluation (last {} tasks)\n".format(last_n))

    # --- Completion stats ---
    recent_done = done_issues[:last_n]
    if recent_done:
        durations: list[float] = []
        for issue in recent_done:
            started = issue.get("startedAt")
            completed = issue.get("completedAt")
            if started and completed:
                t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                durations.append((t1 - t0).total_seconds() / 3600)

        lines.append(f"**Completed:** {len(recent_done)} tasks")
        if durations:
            avg_hrs = sum(durations) / len(durations)
            lines.append(f"**Avg time-to-close:** {avg_hrs:.1f}h")
        lines.append("")

        # Priority breakdown
        priority_counts: dict[str, int] = {}
        for issue in recent_done:
            p = issue.get("priority", "unknown")
            priority_counts[p] = priority_counts.get(p, 0) + 1
        if priority_counts:
            breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(priority_counts.items()))
            lines.append(f"**Priority mix:** {breakdown}")
            lines.append("")
    else:
        lines.append("**Completed:** 0 tasks in recent history")
        lines.append("")

    # --- Blockers ---
    if blocked_issues:
        lines.append(f"**Currently blocked:** {len(blocked_issues)} task(s)")
        for issue in blocked_issues[:3]:
            lines.append(f"- {issue.get('identifier', '?')}: {issue.get('title', '?')}")
        lines.append("")
    else:
        lines.append("**Currently blocked:** none")
        lines.append("")

    # --- In-progress ---
    if in_progress_issues:
        lines.append(f"**In progress:** {len(in_progress_issues)} task(s)")
        for issue in in_progress_issues[:3]:
            lines.append(f"- {issue.get('identifier', '?')}: {issue.get('title', '?')}")
        lines.append("")

    # --- PerformanceTracker data ---
    if tracker_entries:
        lines.append("### PerformanceTracker Trends\n")
        # Extract metric trends from the last N entries
        metric_keys: set[str] = set()
        for entry in tracker_entries:
            metrics = entry.get("metrics", {})
            metric_keys.update(metrics.keys())

        for key in sorted(metric_keys):
            values = [
                e.get("metrics", {}).get(key)
                for e in tracker_entries
                if e.get("metrics", {}).get(key) is not None
            ]
            if len(values) >= 2:
                first_half = values[: len(values) // 2]
                second_half = values[len(values) // 2 :]
                avg_first = sum(first_half) / len(first_half)
                avg_second = sum(second_half) / len(second_half)
                if avg_first > 0:
                    change_pct = ((avg_second - avg_first) / avg_first) * 100
                    direction = "+" if change_pct > 0 else ""
                    lines.append(
                        f"- **{key}:** {avg_second:.2f} ({direction}{change_pct:.0f}% vs prior)"
                    )
                else:
                    lines.append(f"- **{key}:** {avg_second:.2f}")
            elif values:
                lines.append(f"- **{key}:** {values[-1]:.2f}")
        lines.append("")
    else:
        lines.append(
            "*PerformanceTracker log not yet available. "
            "Self-eval based on Paperclip task history only.*"
        )
        lines.append("")

    # --- Reflection prompts ---
    lines.append("### Reflection\n")
    lines.append("Based on the data above, consider:")
    lines.append("1. **What worked well** in recent tasks?")
    lines.append("2. **What caused blockers** or delays?")
    lines.append("3. **What should change** in the next few heartbeats?")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent self-evaluation summary")
    parser.add_argument("--last-n", type=int, default=10,
                        help="Number of recent tasks to analyze (default: 10)")
    parser.add_argument("--tracker-log", type=str, default=None,
                        help="Path to PerformanceTracker JSONL log")
    args = parser.parse_args()

    api_url = os.environ.get("PAPERCLIP_API_URL", "")
    api_key = os.environ.get("PAPERCLIP_API_KEY", "")
    agent_id = os.environ.get("PAPERCLIP_AGENT_ID", "")
    company_id = os.environ.get("PAPERCLIP_COMPANY_ID", "")

    if not all([api_url, api_key, agent_id, company_id]):
        print("Error: Missing required env vars (PAPERCLIP_API_URL, PAPERCLIP_API_KEY, "
              "PAPERCLIP_AGENT_ID, PAPERCLIP_COMPANY_ID)", file=sys.stderr)
        sys.exit(1)

    try:
        done_issues = fetch_recent_issues(api_url, api_key, company_id, agent_id, "done")
        blocked_issues = fetch_recent_issues(api_url, api_key, company_id, agent_id, "blocked")
        in_progress = fetch_recent_issues(api_url, api_key, company_id, agent_id, "in_progress")
    except URLError as e:
        print(f"Error: Could not reach Paperclip API: {e}", file=sys.stderr)
        sys.exit(1)

    tracker_entries: list[dict] = []
    if args.tracker_log:
        tracker_entries = load_tracker_log(Path(args.tracker_log), args.last_n)

    summary = compute_summary(done_issues, blocked_issues, in_progress, tracker_entries, args.last_n)
    print(summary)


if __name__ == "__main__":
    main()
