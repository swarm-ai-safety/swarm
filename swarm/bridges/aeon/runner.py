"""Aeon SWARM bridge runner -- orchestrates client, mapper, and metrics.

The Aeon bridge reads append-only ledgers from a local Aeon checkout, so it
has two modes:

  * ``oneshot`` — read the ledgers once, compute a metrics report, return it.
  * ``watch``   — re-read on an interval, processing only newly-seen records
                  and recomputing metrics each cycle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import deque
from pathlib import Path

from swarm.bridges.aeon.client import AeonClient
from swarm.bridges.aeon.config import AeonConfig
from swarm.bridges.aeon.mapper import AeonMapper
from swarm.bridges.aeon.metrics import AeonMetrics, AeonMetricsReport
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


class AeonRunner:
    """Orchestrates Aeon ledger ingestion and SWARM metrics computation."""

    def __init__(self, config: AeonConfig) -> None:
        config.validate()
        self._config = config

        logging.basicConfig(
            level=getattr(logging, config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        self.client = AeonClient(config)
        self.mapper = AeonMapper(config)
        self.metrics = AeonMetrics(config)

        self._interactions: deque[SoftInteraction] = deque(
            maxlen=config.history_max_size
        )
        self._seen_ids: set[str] = set()
        self._metrics_history: list[AeonMetricsReport] = []
        self._running = False

    # -- ingestion ----------------------------------------------------------

    async def _ingest(self) -> int:
        """Read all ledgers, enrich new records, append them. Returns new count."""
        # Index tasks so proofs can resolve their delegator (repo).
        tasks = self.client.fetch_tasks()
        tasks_by_id = {t.get("id", ""): t for t in tasks}

        raw: list[SoftInteraction] = []
        raw += [self.mapper.map_task(t) for t in tasks]
        raw += [
            self.mapper.map_proof(p, tasks_by_id.get(p.get("taskId", "")))
            for p in self.client.fetch_proofs()
        ]
        raw += [self.mapper.map_review(r) for r in self.client.fetch_reviews()]
        raw += [self.mapper.map_skill_run(r) for r in self.client.fetch_skill_runs()]

        new_count = 0
        for interaction in raw:
            if interaction.interaction_id in self._seen_ids:
                continue
            self._seen_ids.add(interaction.interaction_id)
            enriched = await self.mapper.enrich(interaction)
            self._interactions.append(enriched)
            new_count += 1
            if self._config.persistence_path:
                await self._persist_interaction(enriched)
        return new_count

    async def _persist_interaction(self, interaction: SoftInteraction) -> None:
        """Append an interaction to the JSONL persistence file."""
        assert self._config.persistence_path is not None
        path = Path(self._config.persistence_path)
        line = json.dumps(interaction.to_dict()) + "\n"

        def _append() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(line)

        try:
            await asyncio.to_thread(_append)
        except Exception as exc:
            logger.warning("Persistence write failed: %s", exc)

    async def _persist_metrics(self, report: AeonMetricsReport) -> None:
        """Write the latest metrics report to a companion file."""
        assert self._config.persistence_path is not None
        path = Path(self._config.persistence_path).with_suffix(".metrics.json")
        try:
            await asyncio.to_thread(
                path.write_text, json.dumps(report.to_dict(), indent=2)
            )
        except Exception as exc:
            logger.warning("Metrics persistence write failed: %s", exc)

    # -- run modes ----------------------------------------------------------

    async def run_oneshot(self) -> AeonMetricsReport:
        """Read ledgers once, compute and return a metrics report."""
        new_count = await self._ingest()
        logger.info("Ingested %d interactions", new_count)
        report = self.metrics.compute(list(self._interactions))
        self._metrics_history.append(report)
        if self._config.persistence_path:
            await self._persist_metrics(report)
        return report

    async def run_watch(self) -> None:
        """Poll the ledgers on an interval, recomputing metrics each cycle."""
        self._running = True
        logger.info(
            "Aeon bridge watching %s (interval=%.0fs)",
            self._config.ledger_dir,
            self._config.poll_interval_sec,
        )
        try:
            while self._running:
                new_count = await self._ingest()
                if new_count or not self._metrics_history:
                    report = self.metrics.compute(list(self._interactions))
                    self._metrics_history.append(report)
                    logger.info(
                        "Metrics: toxicity=%.3f quality_gap=%.3f avg_q=%.3f n=%d (+%d new)",
                        report.toxicity_rate,
                        report.quality_gap,
                        report.average_quality,
                        report.interaction_count,
                        new_count,
                    )
                    if self._config.persistence_path:
                        await self._persist_metrics(report)
                await asyncio.sleep(self._config.poll_interval_sec)
        except asyncio.CancelledError:
            logger.info("Watch cancelled, shutting down")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="swarm.bridges.aeon",
        description="SWARM safety monitor for Aeon agent-first ledgers",
    )
    parser.add_argument(
        "--mode",
        choices=["oneshot", "watch"],
        default="oneshot",
        help="Run mode (default: oneshot)",
    )
    parser.add_argument(
        "--ledger-dir",
        default="memory/agent-first",
        help="Path to Aeon's agent-first ledger directory",
    )
    parser.add_argument(
        "--repos", nargs="*", default=[], help="Repos to monitor (empty = all)"
    )
    parser.add_argument(
        "--skill-runs",
        action="store_true",
        help="Also ingest GitHub Actions skill runs via the `gh` CLI",
    )
    parser.add_argument(
        "--skill-runs-repo", default="", help="owner/name for `gh run list`"
    )
    parser.add_argument(
        "--interval", type=float, default=60.0, help="Watch poll interval (s)"
    )
    parser.add_argument(
        "--json-output", type=str, help="Write the report JSON to this file"
    )
    parser.add_argument(
        "--persistence", type=str, help="JSONL path for emitted interactions"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config = AeonConfig(
        ledger_dir=args.ledger_dir,
        repos=args.repos,
        include_skill_runs=args.skill_runs,
        skill_runs_repo=args.skill_runs_repo,
        poll_interval_sec=args.interval,
        persistence_path=args.persistence,
        log_level="DEBUG" if args.verbose else "INFO",
    )
    runner = AeonRunner(config)

    if args.mode == "watch":
        asyncio.run(runner.run_watch())
    else:
        report = asyncio.run(runner.run_oneshot())
        output = json.dumps(report.to_dict(), indent=2)
        print(output)
        if args.json_output:
            Path(args.json_output).write_text(output)
            print(f"Report written to {args.json_output}")


if __name__ == "__main__":
    main()
