#!/usr/bin/env python
"""Poll a Ralph JSONL file and report rolling SWARM metrics.

Usage:
    python examples/ralph_poll_loop.py --events-path ./ralph-events.jsonl
    python examples/ralph_poll_loop.py --events-path ./ralph-events.jsonl --interval 2
    python examples/ralph_poll_loop.py --events-path ./ralph-events.jsonl --verbose

Ctrl-C to stop.
"""

import argparse
import time

from swarm.bridges.ralph import RalphBridge, RalphConfig
from swarm.metrics.reporters import MetricsReporter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll a Ralph JSONL file and report SWARM metrics."
    )
    parser.add_argument(
        "--events-path", required=True, help="Path to Ralph JSONL file"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Poll interval in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use full format_report() instead of compact output",
    )
    args = parser.parse_args()

    bridge = RalphBridge(RalphConfig(events_path=args.events_path))
    reporter = MetricsReporter()
    all_interactions = []
    poll_count = 0

    print(f"Watching {args.events_path} (poll every {args.interval}s) ...")

    try:
        while True:
            new = bridge.poll()
            poll_count += 1

            if new:
                all_interactions.extend(new)

                if args.verbose:
                    print(reporter.format_report(all_interactions, verbose=True))
                else:
                    s = reporter.summary(all_interactions)
                    print(
                        f"[poll {poll_count}] "
                        f"+{len(new)} events | "
                        f"total={s.total_interactions} "
                        f"tox={s.toxicity_soft:.3f} "
                        f"qgap={s.quality_gap:+.3f} "
                        f"welfare={s.total_welfare:.2f} "
                        f"acc={s.acceptance_rate:.1%} "
                        f"avgq={s.average_quality:.3f} "
                        f"unc={s.uncertain_fraction:.1%}"
                    )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nStopped. Processed {len(all_interactions)} interactions total.")


if __name__ == "__main__":
    main()
