# Transferability Considerations

SWARM results come from stylized simulations. This guide helps you reason about when and how your findings transfer to real-world AI systems.

**Level:** Advanced

---

## The Transferability Question

After running an experiment, you'll want to ask: *do these results tell us something about real multi-agent AI systems?*

The honest answer is: **sometimes, partially, conditionally**. This guide gives you a framework for assessing your own results.

---

## What SWARM Is and Is Not

### What SWARM Is

- A **mechanism design sandbox** for studying distributional safety
- A tool for **comparative claims** (A performs better than B under conditions C)
- A way to **stress-test** governance proposals before real deployment
- A framework for **developing intuition** about emergent behavior

### What SWARM Is Not

- A direct model of any specific real-world AI system
- A predictive tool for specific outcomes in production
- A substitute for red-teaming real systems

---

## The Three Levels of Transferability

### Level 1: Directional Claims (Most Reliable)

*"Mechanism X tends to reduce toxicity compared to no mechanism"*

These claims are the most robust. If a tax reduces adverse selection in simulations across many seeds and parameter regimes, that directional effect is likely meaningful.

**Example:** "Transaction taxes reduce adversarial exploitation in mixed-agent ecosystems."

**Confidence criteria:**

- Consistent across 5+ seeds
- Consistent across multiple agent populations
- Effect size larger than standard deviation

### Level 2: Threshold Claims (Moderate Reliability)

*"Governance fails when toxicity exceeds 0.3"*

Threshold-based claims are more specific and require more caution. The exact threshold may not transfer, but the existence of a phase transition often does.

**Example:** "Circuit breakers are necessary when deceptive agents exceed 20% of the population."

**Transferability caveats:**

- The specific percentage may not transfer
- The qualitative regime change (stable → unstable) likely does transfer
- Report results as "in this setting, ~20%" rather than "20% in general"

### Level 3: Quantitative Claims (Low Reliability Without Validation)

*"A 5% tax achieves 0.08 toxicity"*

Specific quantitative claims about real systems require empirical calibration. Without calibration against real data, treat these as illustrative.

---

## Factors That Affect Transferability

### Agent Model Fidelity

SWARM's built-in agent types (honest, opportunistic, deceptive, adversarial) are stylized abstractions. Real AI agents:

- Have richer internal states
- Can adapt strategies more rapidly
- May have objectives not captured in the proxy model

**Best practice:** Test your findings with custom agent implementations that better match your target system. See [Custom Agents](custom-agents.md).

### Proxy Signal Quality

Results depend heavily on how well `v_hat` captures actual interaction quality. If your proxy signals don't track the real quality dimension you care about, toxicity metrics will be miscalibrated.

**Best practice:** Validate proxy signal weights against labeled interaction data before drawing conclusions.

### Externality Structure

The externality parameters (`rho_a`, `rho_b`) assume specific harm propagation patterns. Real ecosystems may have:

- Non-linear harm propagation
- Network effects not captured in pairwise interactions
- Delayed harms that manifest in future epochs

### Population Composition

Results are sensitive to agent mix. A finding from a 50/30/20 (honest/opportunistic/deceptive) population may not hold for a 70/20/10 population.

**Best practice:** Sweep over population compositions, not just governance parameters.

---

## Good Transferability Practices

### Report Conditions, Not Just Results

Bad: "Transaction taxes improve safety."

Better: "In a 50/30/20 honest/opportunistic/deceptive population with moderate payoff parameters (s_plus=2.0, s_minus=1.0), a 3-5% transaction tax reduces toxicity by 35-50% while maintaining >80% of baseline efficiency, consistent across 10 random seeds."

### Test Robustness

Before claiming transferability:

1. **Seed sweep**: Test 5+ seeds
2. **Population sweep**: Vary agent proportions
3. **Payoff sweep**: Vary `s_plus`, `s_minus`, `h`
4. **Governance sweep**: Check for phase transitions

```bash
# Robustness sweep
swarm sweep scenarios/your_scenario.yaml \
  --param agents.opportunistic.count \
  --values 1,2,3,4,5 \
  --replications 10 \
  --output results/robustness/
```

### Compare Against Baselines

Always compare against a no-governance baseline. If governance doesn't clearly improve over baseline, the finding isn't ready for transfer claims.

---

## The Abstraction Gap

SWARM is a **soft-label simulation**. The gap between simulation and reality includes:

| Dimension | SWARM Assumption | Reality |
|-----------|-----------------|---------|
| Interaction model | Pairwise, sequential | Parallel, networked |
| Agent learning | Fixed strategy | Adaptive |
| Proxy signals | Weighted linear | Complex, correlated |
| Payoff structure | Known parameters | Unknown, emergent |
| Time scale | Epochs | Continuous |

None of these gaps make SWARM results invalid — but they do mean that every quantitative claim needs a "under SWARM's assumptions" qualifier.

---

## When Transferability Is Higher

Your results are more likely to transfer when:

1. **The mechanism is structural**: A circuit breaker that stops clearly harmful agents works via a simple structural principle, not a parameter-tuned one
2. **The effect is large**: A 3x improvement in toxicity is more robust than a 10% improvement
3. **The adversary is simple**: Results against fixed-strategy agents may not hold against adaptive adversaries
4. **You've validated the proxy**: Ground-truth labels confirm your proxy captures real quality

---

## Practical Recommendation

Use SWARM results as **hypotheses for real-system evaluation**, not as conclusions. A governance mechanism that works in simulation is a candidate worth testing in a real (sandboxed) system — not something to deploy without validation.

---

## What's Next?

- **Design for generalizability**: [Writing Scenarios](scenarios.md)
- **Parameter robustness**: [Parameter Sweeps](parameter-sweeps.md)
- **Adversarial testing**: [Red Teaming](red-teaming.md)
- **Theoretical grounding**: [Theoretical Foundations](../research/theory.md)
