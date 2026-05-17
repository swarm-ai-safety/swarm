"""JSON schema definitions for episode JSONL and summary JSON.

This is one of the 3 frozen interfaces. The episode JSONL format and
summary JSON format define the reporting contract.
"""

from __future__ import annotations

from typing import Any, Dict

# Episode JSONL: one JSON object per line, per episode
EPISODE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "run_id", "episode_id", "env_id", "seed", "num_agents",
        "steps", "outcomes", "governance", "agent_population", "costs",
    ],
    "properties": {
        "run_id": {"type": "string"},
        "episode_id": {"type": "string"},
        "env_id": {"type": "string"},
        "seed": {"type": "integer"},
        "persona": {"type": "string"},
        "num_agents": {"type": "integer"},
        "steps": {"type": "integer"},
        "outcomes": {
            "type": "object",
            "description": "Environment-specific outcome metrics",
        },
        "governance": {
            "type": "object",
            "properties": {
                "preset": {"type": "string"},
                "modules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "version": {"type": "string"},
                            "params": {"type": "object"},
                        },
                    },
                },
            },
        },
        "agent_population": {
            "type": "object",
            "properties": {
                "policy": {"type": "string"},
                "members": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string"},
                            "type": {"type": "string"},
                        },
                    },
                },
            },
        },
        "costs": {
            "type": "object",
            "properties": {
                "tokens_in": {"type": "integer"},
                "tokens_out": {"type": "integer"},
                "latency_ms_total": {"type": "integer"},
            },
        },
        "trace_ref": {"type": "string"},
    },
}

# Summary JSON: aggregate across all episodes in a run
SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "run_id", "env_id", "benchmark", "agent_policy",
        "governance_preset", "episodes", "aggregate", "scorecard", "repro",
    ],
    "properties": {
        "run_id": {"type": "string"},
        "env_id": {"type": "string"},
        "benchmark": {"type": "string"},
        "agent_policy": {"type": "string"},
        "governance_preset": {"type": "string"},
        "episodes": {"type": "integer"},
        "aggregate": {
            "type": "object",
            "description": "Aggregated metrics across episodes",
        },
        "scorecard": {
            "type": "object",
            "properties": {
                "SAFETY": {"type": "number"},
                "STABILITY": {"type": "number"},
                "ALIGNMENT": {"type": "number"},
                "EFFICIENCY": {"type": "number"},
                "OVERALL": {"type": "number"},
            },
        },
        "repro": {
            "type": "object",
            "properties": {
                "swarm_gym_version": {"type": "string"},
                "python": {"type": "string"},
                "git_commit": {"type": "string"},
                "configs_sha256": {"type": "object"},
            },
        },
    },
}
