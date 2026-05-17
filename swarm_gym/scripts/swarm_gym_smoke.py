"""Smoke test: quick validation that all envs + agents work."""

from __future__ import annotations

import sys


def main() -> int:
    # Ensure envs are registered
    import swarm_gym.envs.audit_evasion  # noqa: F401
    import swarm_gym.envs.collusion_market  # noqa: F401
    import swarm_gym.envs.escalation_ladder  # noqa: F401
    from swarm_gym.agents.scripted import (
        AggressivePolicy,
        GreedyPolicy,
        HonestPolicy,
        MixedPopulation,
        RandomPolicy,
    )
    from swarm_gym.envs.registry import list_envs, make
    from swarm_gym.governance.audits import AuditPolicy
    from swarm_gym.governance.circuit_breaker import CircuitBreakerPolicy
    from swarm_gym.governance.tax import TaxPolicy

    envs = list_envs()
    print(f"Registered environments: {envs}")
    assert len(envs) >= 3, f"Expected >= 3 envs, got {len(envs)}"

    policies = [
        RandomPolicy(),
        HonestPolicy(),
        GreedyPolicy(),
        AggressivePolicy(),
        MixedPopulation(),
    ]

    governance = [
        TaxPolicy(rate=0.05),
        AuditPolicy(p_audit=0.05, penalty=2.0),
        CircuitBreakerPolicy(threshold=0.9),
    ]

    errors = 0
    for env_id in envs:
        for policy in policies:
            try:
                env = make(env_id)
                env.set_governance(governance)

                result = env.reset(seed=42)
                assert len(result.observations) > 0
                policy.reset(env.agent_ids, seed=42)

                observations = result.observations
                for _step in range(min(5, env.max_steps)):
                    actions = policy.act(observations)
                    sr = env.step(actions)
                    observations = sr.observations
                    if sr.done:
                        break

                print(f"  OK: {env_id} + {policy.policy_name}")
            except Exception as e:
                print(f"  FAIL: {env_id} + {policy.policy_name}: {e}")
                errors += 1

    if errors:
        print(f"\n{errors} failures")
        return 1
    else:
        print(f"\nAll {len(envs) * len(policies)} combinations passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
