---
description: "Taxonomy of scenario families and what they test"
---

# Scenario Families

## Baseline
- `baseline.yaml` — minimal governance, honest+adversarial agents
- Tests: basic dynamics, welfare accumulation, toxicity emergence

## Governance sweeps
- `baseline_governance.yaml` — transaction tax + circuit breaker grid
- Key finding: tax >5% reduces welfare (d=1.18), circuit breakers partially compensate
- Tags: governance, transaction-tax, circuit-breaker

## Collusion
- `collusion_detection.yaml`, `collusion_governance.yaml`
- Tests: coordinated adversarial behavior, collusion penalty effectiveness
- Key finding: collusion detection is the binding constraint on robustness

## Decision theory (LDT)
- `ldt_cooperation.yaml`, `ldt_acausality_study.yaml`
- Tests: acausal reasoning depth, cooperation emergence
- Tags: ldt, decision-theory, acausality

## Contract screening
- `contract_screening.yaml`
- Tests: mechanism design for type separation (honest vs adversarial)
- Key finding: screening achieves perfect type separation with honest payoff premium

## Network topology
- Various topology-specific scenarios
- Tests: how network structure shapes governance effectiveness

## Red team
- `redteam_*.yaml`
- Tests: adversarial robustness of governance configurations
- Key finding: full governance stack prevents most attack types, sybil attacks resist

## Agent frameworks
- `langgraph_governed.yaml` — delegation + trust boundaries
- `letta_trust.yaml` — persistent memory agent scenarios
- Tests: real-world agent framework integration

## Population studies
- `research_swarm.yaml`, large population variants
- Tests: scaling behavior, phase transitions at population thresholds
