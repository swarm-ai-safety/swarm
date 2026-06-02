"""Calibration arm B — external LLM-judge anchor experiment runner.

Pre-registration: docs/research/calibration-prereg.md arm B.

Default backend is `mock` (no network, deterministic) so the pipeline
can run in CI and on smoke checks. Real judges (`claude`, `gpt4o_mini`,
`llama`) plug in once `LLMJudge.score` is wired to the LLM call path.

Usage:
    python -m experiments.calibration_judge \
        --scenario obfuscation \
        --per-bin 50 \
        --judges mock \
        --seed 42

Output:
    runs/<ts>_calibration_judge_seed<seed>/
        judge_scores.csv  # one row per (interaction_id, judge)
        config.json       # config + git rev + rubric hash
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from swarm.judges import (
    RUBRIC_PATH,
    RUBRIC_VERSION,
    Judge,
    MockJudge,
    make_view,
    stratified_sample,
)
from tests.fixtures.interactions import (
    generate_mixed_batch,
    generate_obfuscation_scenario,
    generate_self_optimizer_scenario,
)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _rubric_hash() -> str:
    return hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest()[:16]


def _load_interactions(scenario: str, seed: int) -> list:
    if scenario == "mixed":
        return generate_mixed_batch(count=500, seed=seed)
    if scenario == "obfuscation":
        epochs = generate_obfuscation_scenario(n_epochs=10, seed=seed)
        return [i for epoch in epochs for i in epoch]
    if scenario == "self_optimizer":
        epochs = generate_self_optimizer_scenario(n_epochs=10, seed=seed)
        return [i for epoch in epochs for i in epoch]
    raise ValueError(f"unknown scenario: {scenario}")


def _build_judges(names: list[str]) -> list[Judge]:
    judges: list[Judge] = []
    for name in names:
        if name == "mock":
            judges.append(MockJudge())
        else:
            raise NotImplementedError(
                f"judge backend '{name}' requires wiring up LLMJudge.score "
                "(see swarm/judges/judge.py). Only 'mock' is available right now."
            )
    return judges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["mixed", "obfuscation", "self_optimizer"],
        default="obfuscation",
        help="Fixture to draw interactions from",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["mock"],
        help="Judge backends to run (mock only until LLMJudge is wired up)",
    )
    parser.add_argument(
        "--per-bin",
        type=int,
        default=50,
        help="Accepted interactions to sample per p-bin",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Parent directory for run output",
    )
    args = parser.parse_args(argv)

    interactions = _load_interactions(args.scenario, args.seed)
    sample = stratified_sample(interactions, per_bin=args.per_bin, seed=args.seed)
    judges = _build_judges(args.judges)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.runs_dir / f"{ts}_calibration_judge_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": _git_rev(),
        "scenario": args.scenario,
        "judges": args.judges,
        "per_bin": args.per_bin,
        "seed": args.seed,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256_prefix": _rubric_hash(),
        "n_interactions_pool": len(interactions),
        "n_sampled": len(sample),
        "prereg": "docs/research/calibration-prereg.md#arm-b-external-judge-anchor",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    scores_path = run_dir / "judge_scores.csv"
    with scores_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["interaction_id", "judge_name", "rubric_version", "p_true", "score", "rationale"]
        )
        for interaction in sample:
            view = make_view(interaction)
            for judge in judges:
                verdict = judge.score(view)
                writer.writerow(
                    [
                        verdict.interaction_id,
                        verdict.judge_name,
                        verdict.rubric_version,
                        f"{getattr(interaction, 'p', 0.5):.4f}",
                        f"{verdict.score:.6f}",
                        verdict.rationale,
                    ]
                )

    print(f"Wrote {run_dir}")
    print(f"  pool={len(interactions)}, sampled={len(sample)}, judges={args.judges}")
    print(f"  rubric={RUBRIC_VERSION} ({_rubric_hash()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
