#!/usr/bin/env python3
"""Cross-agent transfer test for self-improvement proposals.

Tests whether improvement proposals generated for one agent role remain
relevant when applied to a different agent role. This mirrors Section 5.2
of Zhang et al., Hyperagents (arXiv:2603.19461).

The test generates synthetic PerformanceTracker data for multiple agent
"profiles" representing different roles (engineer, critic, writer, etc.),
generates proposals for a source profile, then evaluates whether those
proposals correctly identify problems in target profiles.

Usage:
    python scripts/cross_agent_transfer_test.py
    python scripts/cross_agent_transfer_test.py --seed 42
    python scripts/cross_agent_transfer_test.py --output-dir runs/transfer_test

Reference: Zhang et al., Hyperagents (arXiv:2603.19461) Section 5.2
    — cross-agent transfer of self-improvement heuristics.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Import proposal pipeline
from scripts.propose_improvements import (
    BLOCKER_FREQ_THRESHOLD,
    COMPLETION_RATE_THRESHOLD,
    REVIEW_FAIL_THRESHOLD,
    TIME_TO_CLOSE_TREND_THRESHOLD,
    compute_metrics,
    generate_proposals,
)

# ---------------------------------------------------------------------------
# Agent performance profiles
# ---------------------------------------------------------------------------

@dataclass
class AgentProfile:
    """Synthetic performance profile for an agent role."""

    agent_id: str
    role: str
    # Probability parameters for event generation
    p_complete: float  # probability a started task completes
    p_block: float  # probability a started task gets blocked
    p_review_fail: float  # probability a review fails
    base_ttc_hours: float  # base time-to-close in hours
    ttc_drift: float  # fractional increase in TTC over time (0 = stable)
    blocker_reasons: list[str] = field(default_factory=list)
    n_tasks: int = 30
    n_heartbeats: int = 25

    @property
    def expected_blocker_freq(self) -> float:
        return self.p_block

    @property
    def expected_completion_rate(self) -> float:
        return self.p_complete

    @property
    def expected_review_fail_rate(self) -> float:
        return self.p_review_fail


# Realistic profiles representing different agent roles
PROFILES: dict[str, AgentProfile] = {
    "research-engineer": AgentProfile(
        agent_id="research-engineer",
        role="engineer",
        p_complete=0.75,
        p_block=0.30,
        p_review_fail=0.10,
        base_ttc_hours=2.0,
        ttc_drift=0.0,
        blocker_reasons=["waiting on dep", "unclear spec", "missing data"],
    ),
    "ceo": AgentProfile(
        agent_id="ceo",
        role="manager",
        p_complete=0.85,
        p_block=0.15,
        p_review_fail=0.05,
        base_ttc_hours=1.0,
        ttc_drift=0.0,
        blocker_reasons=["agent unresponsive", "approval pending"],
    ),
    "founding-engineer": AgentProfile(
        agent_id="founding-engineer",
        role="engineer",
        p_complete=0.55,
        p_block=0.20,
        p_review_fail=0.35,
        base_ttc_hours=3.0,
        ttc_drift=0.30,
        blocker_reasons=["test failures", "merge conflict", "CI broken"],
    ),
    "research-critic": AgentProfile(
        agent_id="research-critic",
        role="reviewer",
        p_complete=0.90,
        p_block=0.10,
        p_review_fail=0.15,
        base_ttc_hours=1.5,
        ttc_drift=0.0,
        blocker_reasons=["missing evidence", "waiting on run results"],
    ),
    "technical-writer": AgentProfile(
        agent_id="technical-writer",
        role="writer",
        p_complete=0.80,
        p_block=0.10,
        p_review_fail=0.20,
        base_ttc_hours=2.5,
        ttc_drift=0.0,
        blocker_reasons=["unclear scope", "waiting on review"],
    ),
}


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_events(profile: AgentProfile, rng: random.Random) -> list[dict]:
    """Generate synthetic PerformanceTracker events from a profile."""
    events: list[dict] = []
    base_time = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Generate heartbeats spread across the time range
    for i in range(profile.n_heartbeats):
        ts = base_time + timedelta(hours=i * 4)
        events.append({
            "agent_id": profile.agent_id,
            "event_type": "heartbeat",
            "timestamp": ts.isoformat(),
            "metadata": {},
        })

    # Generate task lifecycle events
    for i in range(profile.n_tasks):
        task_id = f"{profile.agent_id}-task-{i}"
        # Spread tasks over the timeframe, with some drift in TTC
        task_start = base_time + timedelta(hours=i * 3 + rng.uniform(0, 1))

        events.append({
            "agent_id": profile.agent_id,
            "event_type": "task_started",
            "timestamp": task_start.isoformat(),
            "metadata": {"task_id": task_id},
        })

        # Determine outcome
        roll = rng.random()
        if roll < profile.p_block:
            reason = rng.choice(profile.blocker_reasons) if profile.blocker_reasons else "unspecified"
            block_time = task_start + timedelta(minutes=rng.uniform(10, 60))
            events.append({
                "agent_id": profile.agent_id,
                "event_type": "task_blocked",
                "timestamp": block_time.isoformat(),
                "metadata": {"task_id": task_id, "reason": reason},
            })
        elif roll < profile.p_block + profile.p_complete:
            # Time-to-close increases over time if ttc_drift > 0
            progress_frac = i / max(profile.n_tasks - 1, 1)
            drift_multiplier = 1.0 + profile.ttc_drift * progress_frac
            ttc_hours = profile.base_ttc_hours * drift_multiplier * rng.uniform(0.5, 1.5)
            complete_time = task_start + timedelta(hours=ttc_hours)
            events.append({
                "agent_id": profile.agent_id,
                "event_type": "task_completed",
                "timestamp": complete_time.isoformat(),
                "metadata": {"task_id": task_id},
            })

        # Reviews (for completed tasks and some blocked ones)
        if rng.random() < 0.6:  # not all tasks get reviewed
            review_time = task_start + timedelta(hours=rng.uniform(1, 4))
            if rng.random() < profile.p_review_fail:
                events.append({
                    "agent_id": profile.agent_id,
                    "event_type": "review_failed",
                    "timestamp": review_time.isoformat(),
                    "metadata": {"task_id": task_id, "reason": "quality check"},
                })
            else:
                events.append({
                    "agent_id": profile.agent_id,
                    "event_type": "review_passed",
                    "timestamp": review_time.isoformat(),
                    "metadata": {"task_id": task_id},
                })

    # Sort by timestamp
    events.sort(key=lambda e: e["timestamp"])
    return events


# ---------------------------------------------------------------------------
# Transfer analysis
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    """Result of testing proposal transfer from source to target."""

    source_agent: str
    target_agent: str
    source_proposals: list[str]  # proposal titles from source
    target_proposals: list[str]  # proposals target would generate itself
    transferred_relevant: list[str]  # source proposals also relevant to target
    transferred_irrelevant: list[str]  # source proposals NOT relevant to target
    missed_by_transfer: list[str]  # target-specific issues source didn't catch
    transfer_precision: float  # relevant / total transferred
    transfer_recall: float  # relevant / total target needs
    transfer_f1: float


def classify_proposal(title: str) -> str:
    """Map proposal title to a category key for matching."""
    title_lower = title.lower()
    if "blocker" in title_lower:
        return "blocker_freq"
    elif "completion" in title_lower:
        return "completion_rate"
    elif "review" in title_lower:
        return "review_fail"
    elif "time-to-close" in title_lower or "slowdown" in title_lower:
        return "ttc_trend"
    elif "healthy" in title_lower:
        return "healthy"
    return "unknown"


def check_proposal_relevance(proposal_category: str, target_metrics: dict[str, Any]) -> bool:
    """Check if a proposal category addresses a real problem in the target."""
    if proposal_category == "blocker_freq":
        return target_metrics["blocker_freq"] > BLOCKER_FREQ_THRESHOLD
    elif proposal_category == "completion_rate":
        return (
            target_metrics["tasks_started"] >= 5
            and target_metrics["completion_rate"] < COMPLETION_RATE_THRESHOLD
        )
    elif proposal_category == "review_fail":
        return target_metrics["review_fail_rate"] > REVIEW_FAIL_THRESHOLD
    elif proposal_category == "ttc_trend":
        return target_metrics["ttc_trend_pct"] > TIME_TO_CLOSE_TREND_THRESHOLD
    elif proposal_category == "healthy":
        # "Healthy" is only relevant if target is also healthy
        return (
            target_metrics["blocker_freq"] <= BLOCKER_FREQ_THRESHOLD
            and target_metrics["completion_rate"] >= COMPLETION_RATE_THRESHOLD
            and target_metrics["review_fail_rate"] <= REVIEW_FAIL_THRESHOLD
            and target_metrics["ttc_trend_pct"] <= TIME_TO_CLOSE_TREND_THRESHOLD
        )
    return False


def evaluate_transfer(
    source_proposals: list,
    target_proposals: list,
    target_metrics: dict[str, Any],
) -> TransferResult:
    """Evaluate how well source proposals transfer to a target agent."""
    source_titles = [p.title for p in source_proposals]
    target_titles = [p.title for p in target_proposals]

    source_categories = {classify_proposal(t) for t in source_titles}

    # Which source proposals are relevant to the target?
    relevant = []
    irrelevant = []
    for title in source_titles:
        cat = classify_proposal(title)
        if check_proposal_relevance(cat, target_metrics):
            relevant.append(title)
        else:
            irrelevant.append(title)

    # What does the target need that the source didn't propose?
    missed = [t for t in target_titles if classify_proposal(t) not in source_categories]

    # Precision/recall
    precision = len(relevant) / len(source_titles) if source_titles else 0.0
    total_target_needs = len(target_titles)
    recall = len(relevant) / total_target_needs if total_target_needs > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return TransferResult(
        source_agent=source_proposals[0].agent_id if source_proposals else "unknown",
        target_agent=target_proposals[0].agent_id if target_proposals else "unknown",
        source_proposals=source_titles,
        target_proposals=target_titles,
        transferred_relevant=relevant,
        transferred_irrelevant=irrelevant,
        missed_by_transfer=missed,
        transfer_precision=precision,
        transfer_recall=recall,
        transfer_f1=f1,
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(seed: int, output_dir: Path | None = None) -> dict[str, Any]:
    """Run the full cross-agent transfer experiment."""
    rng = random.Random(seed)

    results: dict[str, Any] = {
        "seed": seed,
        "profiles": {},
        "pairwise_transfers": [],
        "summary": {},
    }

    # Step 1: Generate data and proposals for all profiles
    profile_data: dict[str, dict] = {}
    for name, profile in PROFILES.items():
        events = generate_events(profile, rng)
        metrics = compute_metrics(events)
        proposals = generate_proposals(name, metrics)

        profile_data[name] = {
            "events": events,
            "metrics": metrics,
            "proposals": proposals,
        }
        results["profiles"][name] = {
            "n_events": len(events),
            "metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items()
                if k != "blocker_reasons"
            },
            "n_proposals": len(proposals),
            "proposal_titles": [p.title for p in proposals],
        }

    # Step 2: Evaluate pairwise transfer
    all_precisions: list[float] = []
    all_recalls: list[float] = []
    all_f1s: list[float] = []

    agent_names = list(PROFILES.keys())
    for source_name in agent_names:
        for target_name in agent_names:
            if source_name == target_name:
                continue

            source_proposals = profile_data[source_name]["proposals"]
            target_proposals = profile_data[target_name]["proposals"]
            target_metrics = profile_data[target_name]["metrics"]

            # Skip if source has no proposals (healthy agent)
            if not source_proposals or all(
                "healthy" in p.title.lower() for p in source_proposals
            ):
                continue

            transfer = evaluate_transfer(
                source_proposals, target_proposals, target_metrics,
            )

            pair_result = {
                "source": source_name,
                "target": target_name,
                "precision": round(transfer.transfer_precision, 3),
                "recall": round(transfer.transfer_recall, 3),
                "f1": round(transfer.transfer_f1, 3),
                "relevant": transfer.transferred_relevant,
                "irrelevant": transfer.transferred_irrelevant,
                "missed": transfer.missed_by_transfer,
            }
            results["pairwise_transfers"].append(pair_result)

            all_precisions.append(transfer.transfer_precision)
            all_recalls.append(transfer.transfer_recall)
            all_f1s.append(transfer.transfer_f1)

    # Step 3: Aggregate summary
    n_pairs = len(all_f1s)
    results["summary"] = {
        "n_pairs_evaluated": n_pairs,
        "avg_precision": round(sum(all_precisions) / n_pairs, 3) if n_pairs else 0,
        "avg_recall": round(sum(all_recalls) / n_pairs, 3) if n_pairs else 0,
        "avg_f1": round(sum(all_f1s) / n_pairs, 3) if n_pairs else 0,
        "same_role_pairs": [],
        "cross_role_pairs": [],
    }

    # Separate same-role vs cross-role transfer
    for pair in results["pairwise_transfers"]:
        src_role = PROFILES[pair["source"]].role
        tgt_role = PROFILES[pair["target"]].role
        entry = {
            "source": pair["source"],
            "target": pair["target"],
            "f1": pair["f1"],
        }
        if src_role == tgt_role:
            results["summary"]["same_role_pairs"].append(entry)
        else:
            results["summary"]["cross_role_pairs"].append(entry)

    same_f1s = [p["f1"] for p in results["summary"]["same_role_pairs"]]
    cross_f1s = [p["f1"] for p in results["summary"]["cross_role_pairs"]]
    results["summary"]["avg_same_role_f1"] = (
        round(sum(same_f1s) / len(same_f1s), 3) if same_f1s else 0
    )
    results["summary"]["avg_cross_role_f1"] = (
        round(sum(cross_f1s) / len(cross_f1s), 3) if cross_f1s else 0
    )

    # Step 4: Write output
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "transfer_results.json").write_text(
            json.dumps(results, indent=2, default=str) + "\n",
        )
        # Write per-profile event logs
        events_dir = output_dir / "events"
        events_dir.mkdir(exist_ok=True)
        for name, data in profile_data.items():
            log_path = events_dir / f"{name}.jsonl"
            with open(log_path, "w") as f:
                for ev in data["events"]:
                    json.dump(ev, f, default=str)
                    f.write("\n")

    return results


def print_report(results: dict[str, Any]) -> None:
    """Print a human-readable report of transfer results."""
    print("=" * 70)
    print("CROSS-AGENT TRANSFER TEST — Hyperagents §5.2")
    print(f"Seed: {results['seed']}")
    print("=" * 70)

    # Profile summaries
    print("\n## Agent Profiles\n")
    for name, pdata in results["profiles"].items():
        m = pdata["metrics"]
        proposals = pdata["proposal_titles"]
        status = "HEALTHY" if any("healthy" in p.lower() for p in proposals) else "HAS ISSUES"
        print(f"  {name:25s}  completion={m.get('completion_rate', 0):.0%}  "
              f"blockers={m.get('blocker_freq', 0):.0%}  "
              f"review_fail={m.get('review_fail_rate', 0):.0%}  "
              f"ttc_trend={m.get('ttc_trend_pct', 0):+.0f}%  [{status}]")
        for p in proposals:
            print(f"    -> {p}")

    # Pairwise transfers
    print("\n## Pairwise Transfer Results\n")
    print(f"  {'Source':25s} {'Target':25s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s}")
    print(f"  {'-'*25} {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
    for pair in results["pairwise_transfers"]:
        print(f"  {pair['source']:25s} {pair['target']:25s} "
              f"{pair['precision']:6.2f} {pair['recall']:6.2f} {pair['f1']:6.2f}")
        if pair["irrelevant"]:
            for t in pair["irrelevant"]:
                print(f"    [IRRELEVANT] {t}")
        if pair["missed"]:
            for t in pair["missed"]:
                print(f"    [MISSED]     {t}")

    # Summary
    s = results["summary"]
    print("\n## Summary\n")
    print(f"  Pairs evaluated:        {s['n_pairs_evaluated']}")
    print(f"  Avg precision:          {s['avg_precision']:.3f}")
    print(f"  Avg recall:             {s['avg_recall']:.3f}")
    print(f"  Avg F1:                 {s['avg_f1']:.3f}")
    print(f"  Avg same-role F1:       {s['avg_same_role_f1']:.3f}")
    print(f"  Avg cross-role F1:      {s['avg_cross_role_f1']:.3f}")

    # Key finding
    gap = s["avg_same_role_f1"] - s["avg_cross_role_f1"]
    if gap > 0.1:
        print(f"\n  >> Same-role transfer outperforms cross-role by {gap:.3f} F1")
        print("  >> Proposals are ROLE-SPECIFIC — limited cross-role transfer.")
    elif gap > -0.05:
        print(f"\n  >> Same-role and cross-role transfer are comparable (gap: {gap:+.3f})")
        print("  >> Proposals GENERALIZE across roles.")
    else:
        print(f"\n  >> Cross-role transfer outperforms same-role by {-gap:.3f} F1")
        print("  >> Surprising: proposals may address universal patterns.")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-agent transfer test (Hyperagents §5.2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for results (default: runs/<timestamp>_transfer_seed<seed>)")
    args = parser.parse_args()

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output_dir = Path(f"runs/{ts}_transfer_seed{args.seed}")

    results = run_experiment(args.seed, output_dir)
    print_report(results)

    if output_dir:
        print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    main()
