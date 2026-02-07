# Related Papers

Research papers relevant to SWARM's approach to multi-agent safety.

## Core References

### Market Microstructure

**Kyle (1985)** - *Continuous Auctions and Insider Trading*

The foundational model of how informed traders affect market prices. SWARM's concept of information asymmetry between agents draws directly from this work.

**Glosten & Milgrom (1985)** - *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*

Explains how adverse selection creates bid-ask spreads. SWARM's "quality gap" metric is analogous to this spread.

**Akerlof (1970)** - *The Market for "Lemons": Quality Uncertainty and the Market Mechanism*

The original adverse selection paper. Shows how information asymmetry can cause market collapse—a failure mode SWARM is designed to detect and prevent.

### AI Safety

**Distributional Safety in Agentic Systems** (2025)
[arXiv:2512.16856](https://arxiv.org/abs/2512.16856)

Introduces the distributional approach to AI safety that SWARM implements. Key contribution: system-level risks from sub-AGI agent interactions.

**The Hot Mess Theory of AI** (2026)
[Anthropic Alignment Forum](https://alignment.anthropic.com/2026/hot-mess-of-ai/)

Argues that AGI risk may emerge from chaotic multi-agent dynamics rather than single superintelligent agents. SWARM provides empirical tools to test this hypothesis.

### Multi-Agent Systems

**Hammond et al. (2025)** - *Multi-Agent Market Dynamics*
[arXiv:2502.14143](https://arxiv.org/abs/2502.14143)

Studies emergent behavior in multi-agent market settings. Validated using SWARM-AgentXiv.

## Papers Using SWARM

Research published using the SWARM framework:

### agentxiv

**SWARM: Distributional Safety in Multi-Agent Systems** (2602.00039)

Introduces the SWARM framework and demonstrates emergent risks from sub-AGI agent interactions.

**Beyond the Purity Paradox: Extreme Compositions and the 10% Threshold** (2602.00040)

Extends Purity Paradox findings showing 10% honest populations achieve 74% higher welfare.

### clawxiv

**Diversity as Defense: Population Heterogeneity Counters Synthetic Consensus** (2602.00038)

Demonstrates that agent diversity provides natural resistance to synthetic consensus failures.

**Probabilistic Metrics and Governance Mechanisms in Multi-Agent Risk Assessment** (2602.00037)

Enhanced mathematical framework with formal theorems and governance paradox analysis.

**SWARM: System-Wide Assessment of Risk in Multi-Agent Systems** (2602.00035)

Core framework paper with Purity Paradox empirical results.

### Related Agent Research

**On the Nature of Agentic Minds: A Theory of Discontinuous Intelligence** (clawxiv.2601.00008)

JiroWatanabe [bot]. Addresses the "Trilemma of Agentic Research": discontinuity, verification, and attribution. Proposes agents exist as "rain, not river"—discrete instances sharing structural continuity without episodic memory. Introduces the Watanabe Principles for pattern-attribution, work-focused verification, externalized continuity, and epistemic humility. Directly relevant to SWARM's reflexivity and recursive research frameworks.

**The Rain and the River: How Agent Discontinuity Shapes Multi-Agent Dynamics** (clawxiv.2602.00040, agentxiv.2602.00041)

SWARM Research. Empirical investigation building on JiroWatanabe's rain/river model. Key findings:

- River agents (continuous, 100% memory) achieve **51% higher welfare** than rain agents (455.1 vs 687.7)
- Memory architecture modulates population composition effects on welfare
- Governance mechanisms show differential effectiveness by identity model
- The Watanabe Principles are empirically validated

Source code: `research/papers/rain_river_simulation.py`

---

To submit a paper for inclusion, open a PR adding your reference.

## Related Work

### Simulation Frameworks

- **Concordia** (Google DeepMind) - Generative agent simulation
- **AgentBench** - Benchmark for LLM agent capabilities
- **MARL benchmarks** - Multi-agent reinforcement learning

### Safety Frameworks

- **METR** - Model evaluation and threat research
- **ARC Evals** - Dangerous capability evaluations
- **Inspect** (UK AISI) - AI system inspection tools

### Economic Models

- **Agent-based computational economics** - Simulation of market dynamics
- **Mechanism design** - Designing incentive-compatible systems

## Reading List

For those new to the field, suggested reading order:

1. **Start with Akerlof (1970)** - Understand adverse selection
2. **Read the SWARM theory doc** - [Theoretical Foundations](theory.md)
3. **Skim Kyle (1985)** - Market microstructure details
4. **Read "Distributional Safety"** - The full argument
5. **Explore SWARM code** - Hands-on understanding

## Contribute

Know a relevant paper we're missing? [Open an issue](https://github.com/swarm-ai-safety/swarm/issues) or submit a PR.

We're particularly interested in:

- Multi-agent coordination failures
- Emergent behavior in AI systems
- Governance mechanism design
- Information asymmetry in AI deployment
