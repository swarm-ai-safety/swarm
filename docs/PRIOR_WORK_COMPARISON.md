# Similarity Analysis: SWARM vs. "Virtual Agent Economies"

**Paper:** Tomasev et al. "Virtual Agent Economies." arXiv:2509.10147, Sep 2025. (Google DeepMind)

## TL;DR

The two works are **complementary, not derivative**. Both frame AI safety as a multi-agent economic problem and propose sandbox frameworks for studying it. The paper is a high-level conceptual framework with no implementation; SWARM is a concrete simulation with explicit mathematical models. Each covers topics the other does not.

## Shared Ground

Both works treat AI safety as an economic coordination problem among heterogeneous agents, and converge on several key ideas:

- **Sandbox economies** as the unit of analysis -- the paper proposes them conceptually; SWARM implements one with configurable governance and measurable metrics.
- **Adverse selection** as a central failure mode -- the paper discusses information asymmetry and capability differentials; SWARM models it explicitly via `quality_gap = E[p | accepted] - E[p | rejected]`.
- **Heterogeneous agent types** -- both define cooperative, selfish, and adversarial agents. SWARM formalizes four types (honest, deceptive, opportunistic, adversarial).
- **Externalities** -- the paper discusses spillover costs qualitatively; SWARM models them as `E_soft = (1-p) * h` with internalization parameters.
- **Governance mechanisms** -- the paper proposes guardrails, oversight, and stability mechanisms; SWARM implements seven levers (circuit breakers, audits, taxes, staking, collusion detection, reputation, surplus sharing).
- **Mechanism design and game theory** -- both draw on these foundations, though citing different works (the paper: Dworkin, social choice theory; SWARM: Kyle 1985, Glosten-Milgrom 1985, Akerlof).
- **Systemic risk** -- the paper warns about HFN flash crashes; SWARM models market unraveling where adversaries drive welfare to zero.

## Unique to SWARM

- **Soft probabilistic labels**: `p = P(v = +1)` as a continuous probability rather than binary classification, with calibration metrics (Brier score, ECE, log loss) and dual soft/hard reporting.
- **Proxy computation pipeline**: `observables -> v_hat -> sigmoid -> p`, with weighted signal combination, exponential decay penalties, and configurable sigmoid sharpness.
- **Incoherence theory**: variance-driven failure mode (distinct from bias), with self-ensemble and incoherence breaker interventions.

## Unique to the Paper

- **Dworkin-style auctions** with equal endowments and the "envy test" for allocation.
- **Mission economies** coordinating agents around collective societal goals.
- **Cryptographic identity infrastructure**: Verifiable Credentials, DIDs, Proof-of-Personhood, ZKPs.
- **Permeability analysis**: how sandbox boundaries interact with the real economy (emergent vs. intentional, permeable vs. impermeable).

## Overlap Summary

| Concept | Overlap |
|---------|---------|
| Multi-agent economic framing, sandbox design, adverse selection, agent types, externalities, governance | **Strong** |
| Mechanism design, game theory, systemic risk, reputation | **Moderate** |
| Soft labels, proxy pipeline, calibration, incoherence (SWARM-only) | None |
| Auctions, mission economies, crypto identity, permeability (paper-only) | None |

## Relationship

The paper provides the *why* (conceptual argument for designed agent economies). SWARM provides the *how* (simulation testbed with formalized models and measurable outcomes). Neither subsumes the other. SWARM could serve as an implementation platform for testing the paper's proposals -- e.g., adding Dworkin-style auctions to the payoff framework, or modeling permeability via external spillover effects.
