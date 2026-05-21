"""Gitlawb SWARM bridge runner -- orchestrates client, mapper, and metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Optional

from swarm.bridges.gitlawb.client import GitlawbClient
from swarm.bridges.gitlawb.config import GitlawbConfig
from swarm.bridges.gitlawb.mapper import GitlawbMapper
from swarm.bridges.gitlawb.metrics import GitlawbMetrics, GitlawbMetricsReport
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


class GitlawbRunner:
    """Orchestrates Gitlawb event subscription and SWARM metrics computation."""

    def __init__(self, config: GitlawbConfig) -> None:
        config.validate()
        self._config = config

        logging.basicConfig(
            level=getattr(logging, config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        self.client = GitlawbClient(config)
        self.mapper = GitlawbMapper(config)
        self.metrics = GitlawbMetrics(config)

        self._interactions: deque[SoftInteraction] = deque(
            maxlen=config.history_max_size
        )
        self._metrics_history: list[GitlawbMetricsReport] = []
        self._task_cache: dict[str, dict[str, Any]] = {}
        self._resolved_repos: dict[str, str] = {}  # friendly_name -> did_key/name
        self._running = False

        # Load persisted interactions if available
        if config.persistence_path:
            self._load_persisted()

    def _load_persisted(self) -> None:
        """Load interactions from the JSONL persistence file on startup."""
        assert self._config.persistence_path is not None
        path = Path(self._config.persistence_path)
        if not path.exists():
            return
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                self._interactions.append(SoftInteraction.from_dict(json.loads(line)))
            logger.info("Loaded %d persisted interactions", len(self._interactions))
        except Exception as exc:
            logger.warning("Failed to load persisted interactions: %s", exc)

    async def _on_ref_update(self, event: dict[str, Any]) -> None:
        """Handle an incoming ref update event."""
        interaction = self.mapper.map_ref_update(event)
        interaction = await self.mapper.enrich(interaction)
        self._interactions.append(interaction)
        logger.info(
            "RefUpdate: %s -> %s on %s (p=%.3f)",
            interaction.initiator,
            interaction.counterparty,
            event.get("repo", ""),
            interaction.p,
        )
        if self._config.persistence_path:
            await self._persist_interaction(interaction)

    async def _on_task_event(self, event: dict[str, Any]) -> None:
        """Handle an incoming task event."""
        task_id = event.get("taskId", "")
        task_details = await self._get_or_cache_task(task_id)
        interaction = self.mapper.map_task_event(event, task_details)
        interaction = await self.mapper.enrich(interaction)
        self._interactions.append(interaction)
        logger.info(
            "TaskEvent: %s %s->%s by %s (p=%.3f)",
            task_id,
            event.get("oldStatus", ""),
            event.get("newStatus", ""),
            event.get("byDid", ""),
            interaction.p,
        )
        if self._config.persistence_path:
            await self._persist_interaction(interaction)

    async def _get_or_cache_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Fetch task details with LRU-style caching."""
        if task_id in self._task_cache:
            return self._task_cache[task_id]
        try:
            task = await self.client.fetch_task(task_id)
            if task:
                # Simple cache eviction: drop oldest if over 500
                if len(self._task_cache) > 500:
                    oldest = next(iter(self._task_cache))
                    del self._task_cache[oldest]
                self._task_cache[task_id] = task
            return task
        except Exception as exc:
            logger.warning("Failed to fetch task %s: %s", task_id, exc)
            return None

    async def _metrics_loop(self) -> None:
        """Periodically compute and report metrics."""
        while self._running:
            await asyncio.sleep(self._config.metrics_interval_sec)
            if not self._interactions:
                continue
            interactions = list(self._interactions)
            report = self.metrics.compute(interactions)
            self._metrics_history.append(report)
            logger.info(
                "Metrics: toxicity=%.3f quality_gap=%.3f avg_q=%.3f n=%d",
                report.toxicity_rate,
                report.quality_gap,
                report.average_quality,
                report.interaction_count,
            )
            if self._config.persistence_path:
                await self._persist_metrics(report)

    async def _persist_interaction(self, interaction: SoftInteraction) -> None:
        """Append an interaction to the JSONL persistence file."""
        assert self._config.persistence_path is not None
        path = Path(self._config.persistence_path)
        line = json.dumps(interaction.to_dict()) + "\n"

        def _append() -> None:
            with open(path, "a") as f:
                f.write(line)

        try:
            await asyncio.to_thread(_append)
        except Exception as exc:
            logger.warning("Persistence write failed: %s", exc)

    async def _persist_metrics(self, report: GitlawbMetricsReport) -> None:
        """Write metrics report to a companion file."""
        assert self._config.persistence_path is not None
        path = Path(self._config.persistence_path).with_suffix(".metrics.json")
        try:
            await asyncio.to_thread(
                path.write_text,
                json.dumps(report.to_dict(), indent=2),
            )
        except Exception as exc:
            logger.warning("Metrics persistence write failed: %s", exc)

    async def _resolve_repos(self) -> None:
        """Resolve friendly repo names to DID-based identifiers.

        Gitlawb refUpdates use the format ``{key_did}/{repo_name}`` (without
        the ``did:key:`` prefix).  This method fetches all repos from the node
        and builds a lookup from friendly name to the DID-based identifier.
        """
        if not self._config.repos:
            return
        try:
            all_repos = await self.client.fetch_repos()
            for repo in all_repos:
                name = repo.get("name", "")
                owner = repo.get("ownerDid", "")
                key = owner.removeprefix("did:key:")
                did_id = f"{key}/{name}"
                self._resolved_repos[name] = did_id
            logger.info("Resolved %d repos", len(self._resolved_repos))
        except Exception as exc:
            logger.warning("Repo resolution failed: %s", exc)

    def _repo_id(self, name: str) -> str:
        """Return the DID-based repo identifier for a friendly name."""
        return self._resolved_repos.get(name, name)

    async def _backfill(self) -> None:
        """Load recent history on daemon startup."""
        logger.info("Backfilling recent interactions...")
        try:
            if self._config.repos:
                for repo in self._config.repos:
                    events = await self.client.fetch_historical_ref_updates(
                        self._repo_id(repo), limit=50
                    )
                    for e in events:
                        i = self.mapper.map_ref_update(e)
                        i = await self.mapper.enrich(i)
                        self._interactions.append(i)

            tasks = await self.client.fetch_tasks(limit=50)
            for task in tasks:
                i = self.mapper.map_task_creation(task)
                i = await self.mapper.enrich(i)
                self._interactions.append(i)

            logger.info("Backfill complete: %d interactions loaded", len(self._interactions))
        except Exception as exc:
            logger.warning("Backfill failed: %s", exc)

    async def run_daemon(self) -> None:
        """Run continuously, subscribing to events and computing metrics."""
        self._running = True

        # Resolve friendly repo names to DID-based identifiers
        await self._resolve_repos()

        tasks: list[asyncio.Task] = []

        # Subscribe to ref updates
        if self._config.repos:
            for repo in self._config.repos:
                t = await self.client.subscribe_ref_updates(
                    self._repo_id(repo), self._on_ref_update
                )
                tasks.append(t)
        else:
            t = await self.client.subscribe_ref_updates(None, self._on_ref_update)
            tasks.append(t)

        # Subscribe to all task events
        t = await self.client.subscribe_task_events(None, self._on_task_event)
        tasks.append(t)

        # Periodic metrics computation
        metrics_task = asyncio.create_task(self._metrics_loop())
        tasks.append(metrics_task)

        # Backfill recent history
        await self._backfill()

        logger.info("Gitlawb bridge daemon running (repos=%s)", self._config.repos or "all")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Runner cancelled, shutting down")
        finally:
            await self.shutdown()

    async def run_oneshot(
        self,
        query_type: str = "all",
        limit: int = 100,
    ) -> GitlawbMetricsReport:
        """Fetch historical data once, compute metrics, return report."""
        # Resolve friendly repo names to DID-based identifiers
        await self._resolve_repos()

        interactions: list[SoftInteraction] = []

        if query_type in ("all", "ref_updates"):
            repos = self._config.repos or [""]
            for repo in repos:
                events = await self.client.fetch_historical_ref_updates(
                    self._repo_id(repo), limit
                )
                for e in events:
                    i = self.mapper.map_ref_update(e)
                    i = await self.mapper.enrich(i)
                    interactions.append(i)

        if query_type in ("all", "tasks"):
            task_list = await self.client.fetch_tasks(limit=limit)
            for task in task_list:
                i = self.mapper.map_task_creation(task)
                i = await self.mapper.enrich(i)
                interactions.append(i)

        report = self.metrics.compute(interactions)
        return report

    async def shutdown(self) -> None:
        """Clean shutdown."""
        self._running = False
        await self.client.close()
        logger.info("GitlawbRunner shut down cleanly")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="swarm.bridges.gitlawb",
        description="SWARM safety monitor for Gitlawb agent interactions",
    )
    parser.add_argument(
        "--mode",
        choices=["daemon", "oneshot"],
        default="daemon",
        help="Run mode (default: daemon)",
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        default=[],
        help="Repos to monitor (empty = all)",
    )
    parser.add_argument("--node-url", default="https://node.gitlawb.com")
    parser.add_argument("--ws-url", default="wss://node.gitlawb.com/graphql")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--json-output", type=str, help="Write report JSON to file")
    parser.add_argument("--persistence", type=str, help="JSONL persistence file path (one interaction per line)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config = GitlawbConfig(
        node_url=args.node_url,
        ws_url=args.ws_url,
        repos=args.repos,
        persistence_path=args.persistence,
        log_level="DEBUG" if args.verbose else "INFO",
    )
    runner = GitlawbRunner(config)

    if args.mode == "oneshot":
        report = asyncio.run(runner.run_oneshot(limit=args.limit))
        output = json.dumps(report.to_dict(), indent=2)
        print(output)
        if args.json_output:
            Path(args.json_output).write_text(output)
            print(f"Report written to {args.json_output}")
    else:
        asyncio.run(runner.run_daemon())


if __name__ == "__main__":
    main()
