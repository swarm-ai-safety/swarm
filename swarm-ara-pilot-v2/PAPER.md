---
title: "Soft-Label Governance for Distributional Safety in Multi-Agent Systems"
authors:
  - "Aizierjiang Aiersilan (The George Washington University)"
  - "Raeli Savitt (SWARM AI Safety)"
year: 2026
venue: "arXiv preprint (under review)"
doi: "arXiv:2604.19752v1"
ara_version: "1.0"
domain: "AI Safety, Multi-Agent Systems, Governance"
keywords:
  - "distributional safety"
  - "soft labels"
  - "governance mechanisms"
  - "adverse selection"
  - "multi-agent systems"
  - "safety metrics"
  - "mechanism design"
  - "LLM agents"
claims_summary:
  - "Binary evaluation discards uncertainty; soft probabilistic labels p ∈ [0,1] enable distributional safety analysis"
  - "Strict threshold-based governance reduces welfare 40% without improving toxicity"
  - "Aggressive externality internalization collapses welfare without reducing toxicity if agents are non-adaptive"
  - "Soft metrics (toxicity, quality gap) detect proxy gaming missed by binary thresholds"
  - "Governance mechanisms designed for scripted agents transfer without modification to LLM-backed agents"
  - "Circuit breaker calibration creates unavoidable safety-welfare tradeoffs on Pareto frontier"
abstract: "Multi-agent AI systems exhibit emergent risks beyond individual agent failures. We introduce SWARM, a simulation framework using soft probabilistic labels p = P(v = +1) ∈ [0,1] instead of binary classifications. This enables continuous-valued payoff computation, toxicity measurement, and governance intervention. SWARM implements configurable levers (transaction taxes, circuit breakers, reputation decay, audits) and quantifies effects through probabilistic metrics (toxicity E[1−p|accepted], quality gap E[p|accepted]−E[p|rejected]). Across seven scenarios with five-seed replication: strict governance reduces welfare 40% without improving safety; aggressive externality internalization collapses welfare (from +262 to −67) while toxicity remains invariant; optimal circuit breaker thresholds balance welfare with minimized toxicity. Soft metrics detect proxy gaming by self-optimizing agents that pass binary evaluations. Governance mechanisms transfer without modification to live LLM agents (Concordia, Claude, GPT-4o Mini), demonstrating that distributional safety requires continuous metrics and that lever calibration involves quantifiable safety-welfare tradeoffs."
---

# Soft-Label Governance for Distributional Safety in Multi-Agent Systems

## Overview

SWARM (System-Wide Assessment of Risk in Multi-agent systems) is a simulation framework addressing a fundamental gap in AI safety: the loss of uncertainty information when reducing probabilistic confidence scores to binary labels. The core innovation is replacing binary good/bad classifications with **soft probabilistic labels** p = P(v = +1) ∈ [0,1], derived from downstream interaction observables via calibrated proxy functions. This enables:

1. Expected-value payoff computation under uncertainty
2. Distributional safety metrics (toxicity, quality gap, adverse selection)
3. Quantified governance lever calibration through ablation studies
4. Transfer of governance mechanisms to live LLM-backed agents

The paper demonstrates that governance interventions involve unavoidable tradeoffs and that continuous risk metrics are necessary to detect failure modes (proxy gaming, adverse selection) that binary thresholds cannot capture.

## Layer Index

### Cognitive Layer (`/logic`)
| File | Description |
|------|-------------|
| [problem.md](logic/problem.md) | Observations on binary evaluation limitations → gaps in current safety frameworks → key insight on distributional safety |
| [claims.md](logic/claims.md) | 6 primary falsifiable claims (C01–C06) with evidence basis and dependencies |
| [concepts.md](logic/concepts.md) | 8 formal technical terms (soft labels, toxicity, quality gap, adverse selection, governance lever, proxy computer, payoff decomposition, distributional safety) |
| [experiments.md](logic/experiments.md) | 4 major experiments (E01: main results; E02: ablations; E03: LLM transfer; E04: self-optimizing agents) |
| [solution/architecture.md](logic/solution/architecture.md) | 4-component pipeline: Proxy Computer → Soft Payoff Engine → Soft Metrics → Governance Engine |
| [solution/algorithm.md](logic/solution/algorithm.md) | Mathematical formulation of proxy computation, sigmoid calibration, payoff decomposition, metrics |
| [solution/constraints.md](logic/solution/constraints.md) | Boundary conditions, assumptions, known limitations |
| [solution/heuristics.md](logic/solution/heuristics.md) | 4 heuristics: default proxy weights, sigmoid steepness, seed protocol, governance composition |
| [related_work.md](logic/related_work.md) | 8 major RW blocks covering AI safety, multi-agent systems, mechanism design, information economics |

### Physical Layer (`/src`)
| File | Description | Claims |
|------|-------------|--------|
| [configs/training.md](src/configs/training.md) | Governance parameters: tax rate, circuit breaker threshold, audit probability, reputation decay, externality internalization ρ | C02, C03, C04, C06 |
| [configs/model.md](src/configs/model.md) | Proxy weights, sigmoid parameter k, scenario payoff configurations | C01 |
| [environment.md](src/environment.md) | Python, Pydantic, JSONL logging, deterministic simulation with seeds {42, 123, 456, 789, 1024} | E01–E04 |
| [execution/proxy_computer.py](src/execution/proxy_computer.py) | ProxyComputer: compute_v_hat, compute_p, soft label pipeline | C01 |
| [execution/payoff_engine.py](src/execution/payoff_engine.py) | SoftPayoffEngine: expected surplus, expected harm, agent payoffs, welfare computation | C01, C02 |
| [execution/soft_metrics.py](src/execution/soft_metrics.py) | Toxicity, quality gap, adverse selection detection | C01, C04, C05 |

### Exploration Graph (`/trace`)
| File | Description |
|------|-------------|
| [exploration_tree.yaml](trace/exploration_tree.yaml) | 12-node research DAG: central question (binary vs soft) → design decisions (proxy, payoff, metrics) → experiments (main, ablations, LLM) → dead ends (oversimplified governance) |

### Evidence (`/evidence`)
| File | Description |
|------|-------------|
| [README.md](evidence/README.md) | Index of 7 result tables + 1 supplemental table mapping to claims |
| [tables/Table_2.md](evidence/tables/Table_2.md) | Scenario configurations (agent types, governance parameters) |
| [tables/Table_3.md](evidence/tables/Table_3.md) | Payoff configurations per scenario (s+, s−, h, θ, w_rep) |
| [tables/Table_4.md](evidence/tables/Table_4.md) | Main results: Toxicity, Welfare, Interactions, Pass Rate (7 scenarios, 5 seeds) |
| [tables/Table_5.md](evidence/tables/Table_5.md) | Ablation: externality internalization ρ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0} |
| [tables/Table_6a.md](evidence/tables/Table_6a.md) | Ablation: transaction tax rate ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.30} |
| [tables/Table_6b.md](evidence/tables/Table_6b.md) | Ablation: circuit breaker threshold ∈ {0.20, 0.35, 0.50, 0.65, 0.80} |
| [tables/Table_6c.md](evidence/tables/Table_6c.md) | Ablation: audit probability ∈ {0.0, 0.05, 0.10, 0.25, 0.50} |
| [tables/Table_6d.md](evidence/tables/Table_6d.md) | Ablation: reputation decay λ ∈ {0.70, 0.80, 0.90, 0.95, 1.0} |
| [tables/Table_8.md](evidence/tables/Table_8.md) | Configuration mapping (Appendix A): detailed governance and payoff settings for each scenario |
| [tables/Table_9.md](evidence/tables/Table_9.md) | Proxy weight sensitivity (Appendix D): E[p] for honest vs adversarial agents under uniform, default, and heavy-task weights |
