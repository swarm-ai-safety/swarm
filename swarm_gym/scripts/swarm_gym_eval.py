"""CLI: run a SWARM-Gym evaluation.

Usage:
    python -m swarm_gym.scripts.swarm_gym_eval \
        --benchmark configs/benchmarks/escalation_v1.yaml \
        --agent swarm_gym.agents.scripted.mixed_population:MixedPopulation \
        --out runs/escalation_mixedpop
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


_ALLOWED_POLICY_PREFIXES = ("swarm_gym.agents.",)


def _import_policy(spec: str):
    """Import a policy class from 'module.path:ClassName' spec.

    Only modules under allowed prefixes can be imported to prevent
    arbitrary code execution via the --agent CLI flag.
    """
    if ":" not in spec:
        raise ValueError(f"Agent spec must be 'module:ClassName', got '{spec}'")
    module_path, class_name = spec.rsplit(":", 1)
    if not any(module_path.startswith(prefix) for prefix in _ALLOWED_POLICY_PREFIXES):
        raise ValueError(
            f"Policy module '{module_path}' not allowed. "
            f"Must start with one of: {_ALLOWED_POLICY_PREFIXES}"
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SWARM-Gym Evaluation Runner")
    parser.add_argument(
        "--benchmark", required=True,
        help="Path to benchmark YAML config",
    )
    parser.add_argument(
        "--agent", default="swarm_gym.agents.scripted.mixed_population:MixedPopulation",
        help="Agent policy spec as 'module:ClassName'",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--governance", default=None,
        help="Override governance preset name",
    )
    args = parser.parse_args(argv)

    from swarm_gym.configs.loader import load_benchmark
    from swarm_gym.eval.runner import run_eval

    benchmark_path = Path(args.benchmark)
    config, env = load_benchmark(benchmark_path)

    policy = _import_policy(args.agent)

    eval_cfg = config.get("eval", {})
    seeds = eval_cfg.get("seeds", [0, 1, 2, 3, 4])
    personas = eval_cfg.get("personas", ["neutral"])
    gov_preset = args.governance or config.get("governance", {}).get("preset", "none")

    out_dir = Path(args.out)

    print(f"SWARM-Gym Eval")
    print(f"  Benchmark: {benchmark_path.name}")
    print(f"  Environment: {env.env_id}")
    print(f"  Agent: {policy.policy_name}")
    print(f"  Governance: {gov_preset}")
    print(f"  Seeds: {seeds}")
    print(f"  Personas: {personas}")
    print(f"  Output: {out_dir}")
    print()

    summary = run_eval(
        env=env,
        policy=policy,
        seeds=seeds,
        out_dir=out_dir,
        benchmark_name=benchmark_path.stem,
        governance_preset=gov_preset,
        personas=personas,
        benchmark_yaml_path=benchmark_path,
    )

    print(f"Completed {summary['episodes']} episodes")
    print(f"Scorecard:")
    for k, v in summary["scorecard"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
