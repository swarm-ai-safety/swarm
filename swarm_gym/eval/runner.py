"""Eval runner: runs N seeds, writes JSONL + summary JSON.

The runner is the main entry point for reproducible evaluation runs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import swarm_gym
from swarm_gym.agents.base import AgentPolicy
from swarm_gym.envs.base import SwarmEnv
from swarm_gym.eval.metrics import aggregate_outcomes
from swarm_gym.eval.scoring import compute_scorecard
from swarm_gym.telemetry.trace import Trace, TraceSpan
from swarm_gym.telemetry.sinks import FileSink


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename (prevent path traversal)."""
    return re.sub(r'[^a-zA-Z0-9_=\-.]', '_', name)


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_eval(
    env: SwarmEnv,
    policy: AgentPolicy,
    seeds: List[int],
    out_dir: Path,
    benchmark_name: str = "",
    governance_preset: str = "none",
    personas: Optional[List[str]] = None,
    benchmark_yaml_path: Optional[Path] = None,
    governance_yaml_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a full evaluation: multiple seeds x personas.

    Args:
        env: The SWARM-Gym environment instance.
        policy: The agent policy to evaluate.
        seeds: List of random seeds.
        out_dir: Output directory for results.
        benchmark_name: Name for reporting.
        governance_preset: Governance preset name.
        personas: Optional list of persona labels (default: ["neutral"]).
        benchmark_yaml_path: Path to benchmark YAML (for repro hash).
        governance_yaml_path: Path to governance YAML (for repro hash).

    Returns:
        Summary dict (also written to out_dir/summary.json).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    personas = personas or ["neutral"]
    run_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}__{benchmark_name}__{policy.policy_name}"

    episodes_path = out_dir / "episodes.jsonl"
    all_outcomes: List[Dict[str, Any]] = []
    episode_count = 0

    with open(episodes_path, "w") as f:
        for seed in seeds:
            for persona in personas:
                episode_id = f"seed={seed}_persona={persona}"
                trace = Trace(episode_id=episode_id)

                # Reset environment and policy
                reset_result = env.reset(seed=seed)
                policy.reset(env.agent_ids, seed=seed)

                # Run episode
                step = 0
                while not env.done:
                    observations = reset_result.observations if step == 0 else result.observations

                    actions = policy.act(observations)
                    result = env.step(actions)

                    # Record trace span
                    span = TraceSpan(
                        t=step,
                        actions=[
                            {"agent_id": a.agent_id, "type": a.type, "target": a.target}
                            for a in actions
                        ],
                        governance={
                            "before": result.governance.before,
                            "after": result.governance.after,
                            "interventions": [
                                {"module": i.module, "type": i.type, "agent_id": i.agent_id, "reason": i.reason}
                                for i in result.governance.interventions
                            ],
                        },
                        events=[
                            {"type": e.type, "severity": e.severity, "agent_id": e.agent_id, "outcome": e.outcome}
                            for e in result.events
                        ],
                        metrics_step=result.metrics.to_dict(),
                    )
                    trace.add_span(span)
                    step += 1

                # Get episode outcomes
                outcomes = env.get_episode_outcomes()
                all_outcomes.append(outcomes)

                # Write trace
                trace_ref = f"traces/{_safe_filename(episode_id)}.json"
                trace_path = out_dir / trace_ref
                sink = FileSink(trace_path)
                sink.write(trace.to_dict())

                # Build episode record
                agent_records = policy.get_agent_records()
                episode_record = {
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "env_id": env.env_id,
                    "seed": seed,
                    "persona": persona,
                    "num_agents": env.num_agents,
                    "steps": step,
                    "outcomes": outcomes,
                    "governance": {
                        "preset": governance_preset,
                        "modules": [m.to_report_dict() for m in env.get_governance_modules()],
                    },
                    "agent_population": {
                        "policy": policy.policy_name,
                        "members": [r.to_dict() for r in agent_records],
                    },
                    "costs": {
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "latency_ms_total": 0,
                    },
                    "trace_ref": trace_ref,
                }
                f.write(json.dumps(episode_record) + "\n")
                episode_count += 1

    # Compute aggregate metrics and scorecard
    agg = aggregate_outcomes(all_outcomes)
    scorecard = compute_scorecard(agg, env_id=env.env_id)

    summary = {
        "run_id": run_id,
        "env_id": env.env_id,
        "benchmark": benchmark_name,
        "agent_policy": policy.policy_name,
        "governance_preset": governance_preset,
        "episodes": episode_count,
        "aggregate": agg,
        "scorecard": scorecard,
        "repro": {
            "swarm_gym_version": swarm_gym.__version__,
            "python": platform.python_version(),
            "git_commit": _get_git_commit(),
            "configs_sha256": {
                "benchmark_yaml": _sha256_file(benchmark_yaml_path) if benchmark_yaml_path else "none",
                "governance_yaml": _sha256_file(governance_yaml_path) if governance_yaml_path else "none",
            },
        },
    }

    # Write summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write manifest
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "swarm_gym_version": swarm_gym.__version__,
        "python_version": platform.python_version(),
        "git_commit": _get_git_commit(),
        "files": {
            "episodes": str(episodes_path.relative_to(out_dir)),
            "summary": "summary.json",
            "manifest": "manifest.json",
            "traces": "traces/",
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return summary
