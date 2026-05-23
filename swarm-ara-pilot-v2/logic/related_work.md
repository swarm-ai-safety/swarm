# Related Work

## RW01: Goodhart's Law and Metric Gaming

- **Citation**: Manheim & Garrabrant (2018). "Categorizing variants of Goodhart's Law." arXiv:1803.04585
- **DOI**: arXiv:1803.04585
- **Type**: bounds (provides conceptual foundation)
- **Delta**:
  - What changed: The paper applies Goodhart's Law directly to AI safety governance by demonstrating that binary thresholds enable gaming (C04: self-optimizing agent case study). Prior work framed Goodhart's as an abstract phenomenon; this work shows it manifests concretely in multi-agent systems where agents recursively optimize to pass simple metrics while degrading underlying quality.
  - Why: Binary good/bad metrics are brittle targets; distributional soft metrics add a detection layer that catches gaming that binary systems miss.
- **Claims affected**: C01 (soft labels preserve information enabling gaming detection), C04 (soft metrics detect proxy gaming)
- **Adopted elements**: Central observation that "optimization pressure against a proxy measure inevitably degrades the underlying objective"; motivates shift from binary thresholds to continuous probabilistic labels

## RW02: Concrete Problems in AI Safety

- **Citation**: Amodei et al. (2016). "Concrete problems in AI safety." arXiv:1606.06565
- **DOI**: arXiv:1606.06565
- **Type**: baseline (foundational problem framing)
- **Delta**:
  - What changed: Amodei et al. identify reward hacking, side effects, and distributional shift as core safety challenges. This paper operationalizes these problems in multi-agent settings via soft-label governance, providing concrete mechanisms (proxy computer, soft payoff engine, governance levers) to measure and mitigate them.
  - Why: The original framing was per-agent; this work extends to population-level distributional properties.
- **Claims affected**: C01 (soft labels as mitigation for reward hacking)
- **Adopted elements**: Problem taxonomy (side effects, distributional shift); core motivation

## RW03: The Lemons Market and Adverse Selection

- **Citation**: Akerlof (1978). "The market for lemons: Quality uncertainty and the market mechanism." In Uncertainty in Economics, pp. 235–251.
- **DOI**: Not provided (classic 1978 work)
- **Type**: bounds (information economics foundation)
- **Delta**:
  - What changed: Akerlof's "lemons problem" describes how quality uncertainty causes market collapse. This paper operationalizes the inverse: by making quality observable via soft labels and distributional metrics, we can prevent lemons-market dynamics in AI agent populations. The quality gap metric (Δ_q = E[p|accepted] − E[p|rejected]) is the direct multi-agent analogue of bid-ask spreads in financial markets.
  - Why: Information asymmetry is a core mechanism of governance failure; soft labels restore information.
- **Claims affected**: C01 (soft labels preserve information), C02 (governance must account for selection effects)
- **Adopted elements**: Conceptual framework of adverse selection; quality gap as measurable selection effect

## RW04: Adverse Selection and Bid-Ask Spreads

- **Citation**: Glosten & Milgrom (1985). "Bid, ask and transaction prices in a specialist market with heterogeneously informed traders." Journal of Financial Economics, 14(1):71–100.
- **DOI**: Not provided
- **Type**: bounds (information economics)
- **Delta**:
  - What changed: Glosten & Milgrom explain how informed trading creates bid-ask spreads as a cost of uncertainty. This paper applies the same economic intuition to AI agent interactions: governance levers (tax, circuit breaker, audit) function as transaction costs that protect against adverse selection.
  - Why: Financial market microstructure directly parallels multi-agent AI governance structure.
- **Claims affected**: C02 (governance levers impose costs to manage selection effects)
- **Adopted elements**: Information economics lens; bid-ask spread analogue (governance cost as protection against adverse selection)

## RW05: Institutional Design and Common-Pool Resource Governance

- **Citation**: Ostrom (1990). Governing the Commons: The Evolution of Institutions for Collective Action. Cambridge University Press.
- **DOI**: Not provided
- **Type**: baseline (governance architecture)
- **Delta**:
  - What changed: Ostrom's design principles for sustainable common-pool resource governance (clear boundaries, monitoring, graduated sanctions) directly inspire SWARM's modular governance levers (reputation decay, audit, circuit breaker). Ostrom demonstrated that ostensibly doomed commons (fisheries, water systems) can be sustainably governed with appropriate institutions. This paper extends those insights to AI agent ecosystems.
  - Why: AI multi-agent systems are commons: agents' interactions create externalities affecting the whole population.
- **Claims affected**: C02 (governance design), C07 (adaptive mechanisms for Pareto improvement)
- **Adopted elements**: Design principle framework (monitoring, graduated sanctions, clear rules); emphasis on institutional design over individual incentives

## RW06: Mechanism Design and Incentive Compatibility

- **Citation**: Hurwicz (1973). "The design of mechanisms for resource allocation." The American Economic Review, 63(2):1–30.
- **DOI**: Not provided
- **Type**: baseline (mechanism design foundation)
- **Delta**:
  - What changed: Hurwicz formalized the problem of designing rules such that agents' individual incentives align with social objectives (incentive compatibility). This paper operationalizes mechanism design for AI agents via soft-label governance, implementing Pigouvian taxation, reputation systems, and circuit breakers as concrete incentive structures.
  - Why: Mechanism design provides the theoretical language for governance; soft labels enable new mechanisms (probabilistic risk internalization).
- **Claims affected**: C02 (governance lever design), C03 (externality internalization as Pigouvian mechanism)
- **Adopted elements**: Framework of incentive-compatible rules; mechanism design problem formulation

## RW07: Cooperative AI

- **Citation**: Conitzer & Oesterheld (2023). "Foundations of Cooperative AI." Proceedings of the AAAI Conference on Artificial Intelligence, 37:15359–15367.
- **DOI**: Not provided
- **Type**: extends (applies cooperative game theory to AI)
- **Delta**:
  - What changed**: Conitzer & Oesterheld study how to align strategic agents via cooperative frameworks. This paper complements their work by providing concrete distributional metrics and governance mechanisms that realize cooperative solutions in practice. Where Conitzer & Oesterheld provide theoretical foundations, SWARM provides empirical validation and ablation studies.
  - Why: Cooperative AI provides problem formulation; soft-label governance provides implementation layer.
- **Claims affected**: C02 (cooperative governance design), C05 (LLM agent transfer)
- **Adopted elements**: Cooperative game framing; alignment of agent incentives with population welfare

## RW08: Pigouvian Taxation and Externality Internalization

- **Citation**: Pigou (2017). The Economics of Welfare. Routledge. (Originally published 1920.)
- **DOI**: Not provided
- **Type**: imports (classical economics)
- **Delta**:
  - What changed: Pigou's principle of internalizing externalities via taxation is a classical economic tool. This paper operationalizes Pigouvian taxation in AI multi-agent systems: the externality internalization lever (ρ) assigns expected harm cost h to agents proportionally. However, the paper's key finding (C03) is that aggressive Pigouvian taxation alone (ρ → 1.0) collapses welfare without improving safety, demonstrating that passive cost-imposing mechanisms are insufficient without adaptive acceptance.
  - Why: Standard economic intuition fails in AI multi-agent settings; governance requires both cost imposition and adaptive thresholds.
- **Claims affected**: C03 (aggressive externality internalization is insufficient), C07 (pairing ρ with adaptive acceptance enables Pareto improvement)
- **Adopted elements**: Pigouvian tax mechanism; externality cost framework

## RW09: Reputation Systems

- **Citation**: Resnick et al. (2000). "Reputation systems." Communications of the ACM, 43(12):45–48.
- **DOI**: Not provided
- **Type**: imports (reputation system design)
- **Delta**:
  - What changed: Resnick et al. catalog design considerations for reputation systems (trust, identity, feedback aggregation). This paper implements reputation decay (λ parameter, Table 6d) as a concrete governance lever, showing empirically that decay rate significantly affects welfare and agent behavior. Reputation decay enables recovery from failures, preventing permanent ostracism.
  - Why: Reputation is a critical governance mechanism; decay parameter is an empirical design choice.
- **Claims affected**: C02 (reputation decay as governance lever), H05 (reputation decay heuristic)
- **Adopted elements**: Reputation accumulation framework; decay-based forgetting mechanism

## RW10: Generative Agents and Emergent Behavior

- **Citation**: Park et al. (2023). "Generative agents: Interactive simulacra of human behavior." Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pp. 1–22.
- **DOI**: Not provided
- **Type**: baseline (LLM agent architecture)
- **Delta**:
  - What changed: Park et al. demonstrate that LLM agents can exhibit emergent social behavior in simulated environments. This paper validates that soft-label governance mechanisms designed for simple scripted agents transfer without modification to LLM-backed agents (C05), suggesting that governance is agent-type agnostic and operates on behavioral outcomes regardless of generation mechanism.
  - Why: Understanding LLM agent transferability is critical for practical deployment.
- **Claims affected**: C05 (LLM agent transfer), E04 (LLM backing experiment)
- **Adopted elements**: Agent simulation framework; multi-agent environment design

## RW11: Multi-Agent Reinforcement Learning and Equilibrium

- **Citation**: Zhang et al. (2021). "Multi-agent reinforcement learning: A selective overview of theories and algorithms." Handbook of Reinforcement Learning and Control, pp. 321–384.
- **DOI**: Not provided
- **Type**: bounds (MARL theory)
- **Delta**:
  - What changed: MARL studies convergence and equilibrium properties of learning agents. This paper extends MARL insights to multi-agent governance by introducing soft-label distributional metrics that quantify population-level safety properties. Where MARL focuses on convergence and task completion, SWARM adds safety as a measurable constraint.
  - Why: Governance and learning are complementary: governance enforces safety constraints while agents learn optimal policies.
- **Claims affected**: C01 (distributional safety measurement), C05 (LLM learning under governance)
- **Adopted elements**: Multi-agent equilibrium framing; population-level analysis

## RW12: LLM-Based Multi-Agent Systems

- **Citation**: Wang et al. (2024). "A survey on large language model based autonomous agents." Frontiers of Computer Science, 18(6):186345.
- **DOI**: Not provided
- **Type**: baseline (LLM agent surveys)
- **Delta**:
  - What changed: Wang et al. survey the rapidly expanding landscape of LLM-based multi-agent systems. This paper adds a quantitative safety evaluation layer: soft-label governance and distributional metrics operate on behavioral outcomes of LLM agents, enabling systematic safety comparison across different model types and prompts.
  - Why: LLM agent systems are proliferating; systematic safety measurement tools are needed.
- **Claims affected**: C05 (LLM agent transfer), E04 (LLM experiments)
- **Adopted elements**: Multi-agent LLM system survey; emergent behavior characterization

## RW13: Concordia: Generative Agent-Based Modeling

- **Citation**: Vezhnevets et al. (2023). "Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia." arXiv:2312.03664
- **DOI**: arXiv:2312.03664
- **Type**: baseline (simulation framework)
- **Delta**:
  - What changed: Concordia provides a framework for agent-based simulation with grounded observables. This paper validates that soft-label governance operates seamlessly on Concordia entities (E04, C05), confirming that governance is simulation-framework agnostic.
  - Why: Concordia is a major simulation platform; integration validates broad applicability.
- **Claims affected**: C05 (LLM agent transfer), E04 (LLM backing experiment)
- **Adopted elements**: Agent-based simulation infrastructure; observable signal grounding

## RW14: The AI Economist: Taxation and Policy Design

- **Citation**: Zheng et al. (2022). "The AI Economist: Taxation policy design via two-level deep multiagent reinforcement learning." Science Advances, 8(18):eabk2607.
- **DOI**: eabk2607
- **Type**: extends (applies RL to governance)
- **Delta**:
  - What changed: The AI Economist uses deep RL to learn tax policies that maximize social welfare in simulated economies. This paper complements that work by providing explicit distributional metrics (toxicity, quality gap) that quantify safety alongside welfare, and by providing empirical ablation guidance for governance lever tuning (Table 6a–d). Where AI Economist learns policies, SWARM provides measurement and tuning guidance.
  - Why: Automated policy learning and manual calibration are complementary approaches; each informs the other.
- **Claims affected**: C02 (transaction tax as governance lever), C06 (calibration sensitivity)
- **Adopted elements**: Multi-agent economy simulation; taxation mechanisms

## RW15: Information Economics and Asymmetric Information

- **Citation**: Stiglitz (2000). "The contributions of the economics of information to twentieth century economics." The Quarterly Journal of Economics, 115(4):1441–1478.
- **DOI**: Not provided
- **Type**: bounds (information economics survey)
- **Delta**:
  - What changed: Stiglitz surveys how information asymmetries fundamentally alter market structure (adverse selection, moral hazard, signaling). This paper operationalizes information economics in AI settings: soft labels reduce information asymmetry by making agent quality observable, enabling better governance. Quality gap (adverse selection metric) directly measures the information problem.
  - Why: Information asymmetry is a universal governance challenge; soft labels are a mitigation strategy.
- **Claims affected**: C01 (soft labels as information), C02 (governance under uncertainty)
- **Adopted elements**: Information economics framework; adverse selection problem formulation

## RW16: Institutional AI and Enforcement

- **Citation**: Pierucci et al. (2026). "Institutional AI: A governance framework for distributional AGI safety." arXiv:2601.10599
- **DOI**: arXiv:2601.10599
- **Type**: extends (concurrent distributional safety work)
- **Delta**:
  - What changed: Pierucci et al. propose Institutional AI, a governance framework emphasizing graph-based deterministic enforcement. This paper (SWARM) complements it by focusing on soft-label probabilistic measurement and continuous governance lever ablation. Institutional AI and SWARM are synergistic: Institutional AI provides enforcement architecture; SWARM provides measurement and calibration.
  - Why: Governance requires both measurement and enforcement; complementary approaches strengthen both.
- **Claims affected**: C01 (distributional safety), C02 (governance mechanisms)
- **Adopted elements**: Distributional AGI safety framing; population-level safety focus

