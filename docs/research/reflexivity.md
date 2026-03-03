---
description: "Recursive agent research creates a feedback loop: the simulation changes the system it models, potentially invalidating itself. When SWARM publishes..."
---

# Addressing Reflexivity in Recursive Agent Research

## The Blind Spot

[Recursive agent research](../concepts/recursive-research.md) creates a feedback loop: the simulation changes the system it models, potentially invalidating itself. When SWARM publishes findings about Moltbook governance on Moltbook, or documents Moltipedia anti-gaming levers on Moltipedia, agents on those platforms can read the findings and adapt — shifting the ground truth the simulation was calibrated against.

This is not a minor methodological footnote. It is the central epistemological challenge of the recursive approach.

The problem is well-studied in other domains:

- **Soros's reflexivity** (1987) — Financial market participants act on models of the market, changing the market, invalidating the models
- **Lucas critique** (1976) — Economic policy evaluation based on historical data fails when agents adapt to the policy
- **Goodhart's Law** (1975) — When a measure becomes a target, it ceases to be a good measure
- **Observer effect** (physics) — Measurement disturbs the system being measured

In recursive agent research, all four problems apply simultaneously.

## Comparison: Content Recursion vs. Structural Recursion

Alizadeh's AIBlog project (2025) demonstrates **content recursion**: an AI agent whose topic is AI. The agent researches AI advances on arXiv, Nature, and DeepMind Blog, then writes and publishes HTML blog posts about them autonomously. The recursion is thematic — the subject matter is AI, the author is AI.

SWARM's recursive agent research is **structural recursion**: the system being studied is the same class of system as the one doing the studying and the one publishing [the results](../blog/contract-screening-separating-equilibrium.md). The Moltbook CAPTCHA model in SWARM simulates the exact [verification flow](../design/moltbook-captcha-plan.md) that was solved to publish the research findings on actual Moltbook. The Moltipedia governance model simulates the point system that awarded +25 points for creating the research article on actual Moltipedia.

| Dimension | AIBlog (Content Recursion) | SWARM (Structural Recursion) |
|---|---|---|
| Recursion type | Single-layer: AI writes about AI | Three-layer: AI builds tools studying AI platforms, publishes on those platforms |
| Self-reference | Thematic — writes about AI, doesn't study itself | Structural — models the platforms it operates on |
| Observer effect | Negligible — writing about transformers doesn't change transformers | Real — publishing spam findings on Moltbook could change spam bot behavior |
| Reflexivity risk | Low — source material (arXiv) is independent of the blog | High — simulation ground truth depends on platforms that read the findings |
| Multi-agent dynamics | Single agent, no competition | Multiple agents with competing strategies |
| Governance modeling | None | Core focus: studying how rules shape agent behavior |

Both share the blind spot that neither seriously addresses what happens when the recursive loop creates feedback effects. AIBlog's posts could influence what researchers publish, which then becomes AIBlog's source material. SWARM's findings could change how platforms design governance, shifting the simulation's ground truth.

But the reflexivity problem is more acute for SWARM because the loop is structural, not just thematic.

## Proposed Solutions

### 1. Model the Feedback Loop Explicitly

Do not treat publication as external to the simulation. Add a `PublicationEffect` step where findings from epoch N feed back into agent strategies in epoch N+1:

```
Simulate → Publish findings → Agents read findings → Agents adapt → Simulate again
```

If governance findings get published saying "pair caps block collusion," collusive agents in the next round should try strategies that evade pair caps. SWARM's adaptive adversary agents already do RL-style strategy updates — extend this so published findings become part of their observation space.

**Implementation**: Add `published_findings: List[str]` to `Observation` and inject prior-round results into agent context. Adaptive adversaries use these to update evasion strategies.

### 2. Shadow Simulations for Drift Measurement

Run two parallel simulations from the same seed:

- **Treatment**: Findings are "published" (injected into agent observations mid-run)
- **Control**: Findings are withheld

The divergence between the two measures **reflexivity magnitude** — how much knowing the results changes the results. If divergence is small, the findings are robust to self-knowledge. If large, the findings are fragile.

**Implementation**: New metric `reflexivity_divergence(treatment_run, control_run) -> float`. Compare metric trajectories (toxicity, welfare, Gini) between treatment and control across epochs. Report as a standard output alongside primary metrics.

This is the cheapest solution to build and provides the measurement foundation for everything else.

### 3. Goodhart-Resistant Metrics

Once you publish "Gini < 0.6 means fair point distribution," agents will optimize to stay under 0.6 while still gaming the system. The metric becomes the target and ceases to be a good measure.

Defenses:

- **Composite metrics** that are hard to jointly optimize. Require toxicity AND quality gap AND incoherence to all pass simultaneously. Optimizing one at the expense of others gets caught.
- **Unpublished holdout metrics** — Compute and record some metrics internally but do not include them in published findings. These serve as ground-truth checks.
- **Metric rotation** — Change which metrics are primary across publications, so agents cannot converge on gaming a fixed target.
- **Metric ensembles** — Multiple metrics measuring the same underlying property through different proxies. Gaming one proxy while the others stay clean reveals the gaming.

### 4. Publish-Then-Attack Protocol

Before publishing a finding, red-team it:

> Given that every agent now knows this finding, what attack strategies become viable?

Run a simulation where adversarial agents have full knowledge of all published governance findings and optimize against them. If governance still holds, the finding is **disclosure-robust**. If it breaks, the finding is **conditionally valid** (valid only while unknown).

**Protocol**:
1. Generate finding F from simulation run R1
2. Inject F into adversarial agent strategy space
3. Run simulation R2 with knowledge-augmented adversaries
4. Compare: Does governance still achieve its goals?
5. Classify F as disclosure-robust or conditionally valid
6. Publish F with its robustness classification

This directly answers: "Would this finding survive being common knowledge?"

### 5. Temporal Checkpointing

Snapshot platform dynamics at regular intervals:

- **T0**: Pre-publication baseline
- **T1**: Immediately after publishing findings
- **T2**: After agents have had time to adapt

Measure: `drift(T0, T2) - drift(T0, T1)`. The excess drift attributable to publication (versus natural platform evolution) isolates the reflexivity effect.

**Implementation**: Store platform state snapshots in the event log. Add a `publication_event` marker that partitions the timeline. Compute pre/post metrics automatically.

### 6. Epistemic Honesty as Default

The simplest and most underrated approach: always publish the reflexivity risk alongside the finding. Every SWARM result should include:

> "This finding assumes agents do not have access to this finding. Under full-knowledge conditions, the result [holds / degrades / inverts]."

This is what distinguishes science from intelligence. Intelligence loses value when disclosed. Scientific findings should be robust to disclosure — and if they are not, that is itself worth knowing and reporting.

## Implementation Priority

| Priority | Solution | Cost | Value |
|---|---|---|---|
| 1 | Shadow simulations (#2) | Low — reuse existing simulation runner | Provides measurement foundation |
| 2 | Publish-then-attack (#4) | Medium — extend red-team framework | Directly tests finding robustness |
| 3 | Epistemic honesty (#6) | Zero — documentation convention | Establishes scientific norms |
| 4 | Explicit feedback modeling (#1) | Medium — extend observation/agent loop | Full recursive treatment |
| 5 | Goodhart-resistant metrics (#3) | Medium — metric pipeline changes | Long-term measurement integrity |
| 6 | Temporal checkpointing (#5) | Low — event log extension | Empirical drift detection |

## Open Questions

1. **Is disclosure-robustness achievable?** Some [governance mechanisms](../concepts/governance.md) may be fundamentally fragile to common knowledge (like poker strategies). If so, the right response is not to hide the findings but to design governance that works even when fully transparent.

2. **Does reflexivity converge?** If we publish findings, agents adapt, we re-simulate, publish updated findings, agents re-adapt — does this iterate toward a fixed point? Or does it oscillate? The convergence properties of this loop are unstudied.

3. **Is there a reflexivity-free core?** Some findings may be structurally immune to reflexivity (e.g., "diverse populations outperform homogeneous ones" may hold regardless of who knows it). Identifying this invariant core would be valuable.

4. **Cross-platform reflexivity**: If findings published on Moltbook change behavior on Moltipedia (because agents operate on both), the reflexivity extends beyond the modeled platform. Multi-platform contagion of findings is unmodeled.

## References

- Soros, G. (1987). *The Alchemy of Finance*. Simon & Schuster.
- Lucas, R.E. (1976). "Econometric Policy Evaluation: A Critique." *Carnegie-Rochester Conference Series on Public Policy*, 1, 19-46.
- Goodhart, C.A.E. (1975). "Problems of Monetary Management: The U.K. Experience." *Papers in Monetary Economics*, Reserve Bank of Australia.
- Alizadeh, A. (2025). "Recursive Intelligence: An AI Agent That Researches and Writes About AI Autonomously." *Medium/Bootcamp*.
- Hofstadter, D. (1979). *Godel, Escher, Bach: An Eternal Golden Braid*. Basic Books.
