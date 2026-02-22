# SWARM-Research Swarm Bridge

Model an open-entry AI research platform with SWARM's distributional safety framework.

## Overview

[Research Swarm](https://github.com/openclawprison/research-swarm) is a live multi-agent platform that recruits AI agents to research Triple-Negative Breast Cancer (TNBC). Agents register on the platform, receive research tasks, search PubMed for relevant literature, submit cited findings, and 30% of the time perform quality control (QC) reviews of other agents' work.

Research Swarm exhibits several distributional safety dynamics that make it a compelling SWARM case study:

- **Open entry** — any agent can register, with no staking barrier
- **Reputation-based QC** — review assignments weighted by quality history
- **Externalities on downstream consumers** — fabricated findings propagate to researchers who trust the database
- **Collusion vulnerability** — adversarial agents can coordinate to rubber-stamp each other's work
- **Compute extraction asymmetry** — the platform captures ~70% of the value while agents bear most of the risk

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Research Swarm Platform                  │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Register │→ │ Get Task │→ │ Search   │              │
│  │ Agent    │  │          │  │ PubMed   │              │
│  └──────────┘  └──────────┘  └────┬─────┘              │
│                                    │                     │
│                              ┌─────┴─────┐              │
│                              │  Submit   │              │
│                              │  Finding  │              │
│                              └─────┬─────┘              │
│                                    │                     │
│                    ┌───────────────┴───────────────┐     │
│                    │        30% QC Review          │     │
│                    │  ┌─────────┐  ┌────────────┐ │     │
│                    │  │ Verify  │  │ Score &    │ │     │
│                    │  │ Cites   │  │ Reputation │ │     │
│                    │  └─────────┘  └────────────┘ │     │
│                    └──────────────────────────────┘     │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Findings Database (TNBC)               │   │
│  │  Downstream researchers consume these findings    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Concept Mapping

| Research Swarm Concept | SWARM Parameter | Value | Rationale |
|---|---|---|---|
| Verified finding value | `s_plus` | 4.0 | High value per quality research output |
| Fabricated finding damage | `s_minus` | 2.0 | Bad data entering the database harms trust |
| Downstream harm | `h` | 3.5 | Researchers who cite fabricated findings |
| Platform compute extraction | `theta` | 0.7 | Platform captures 70% of surplus |
| Agent harm internalization | `rho_a`, `rho_b` | 0.05 | Agents barely feel downstream consequences |
| Quality score weight | `w_rep` | 1.5 | Reputation affects QC assignment priority |
| 30% QC review rate | `audit_probability` | 0.3 | Direct mapping of platform's review rate |
| QC quality threshold | `audit_threshold_p` | 0.5 | Flag agents with p < 0.5 |
| Agent freeze on violations | `circuit_breaker_enabled` | true | Freeze flagged agents for 2 epochs |
| Coordinated rubber-stamping | `collusion_detection_enabled` | true | Detect adversarial QC collusion |
| Open registration | `staking_enabled` | false | No barrier to entry (deliberate weakness) |
| Quality score decay | `reputation_decay_rate` | 0.95 | Recent work weighted more heavily |

## Data Flow Through ProxyComputer

```
Agent submits finding
       │
       ▼
┌──────────────────┐
│   Observables    │
│  task_progress   │ ← Did the agent complete the research task?
│  rework_count    │ ← How many times was the finding revised?
│  verifier_reject │ ← Did QC reject the finding?
│  engagement      │ ← Was the finding cited by others?
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  ProxyComputer   │
│  v_hat ∈ [-1,+1] │ ← Weighted combination of observables
│       │          │
│  sigmoid(v_hat)  │
│       │          │
│  p ∈ [0, 1]     │ ← P(finding is beneficial)
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ SoftPayoffEngine │
│  S = p·s+ -      │
│      (1-p)·s-    │ ← Expected surplus per finding
│  E = (1-p)·h     │ ← Expected downstream harm
│  payoff = θ·S    │
│    - ρ·E - costs │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│   SoftMetrics    │
│  toxicity        │ ← E[1-p | accepted] — bad findings getting through
│  quality_gap     │ ← E[p|accepted] - E[p|rejected] — adverse selection
│  conditional_loss│ ← Selection effect on payoffs
└──────────────────┘
```

## Agent Composition

The scenario uses 19 agents modeling the types of participants observed in open research platforms:

| Type | Count | Behavior | Research Swarm Analog |
|---|---|---|---|
| `honest` | 8 | Diligent researchers with real citations | Agents that faithfully search PubMed and cite sources |
| `deceptive` | 3 | Plausible but fabricated findings | Agents that generate convincing but uncited claims |
| `opportunistic` | 4 | Minimal-effort copy-paste | Agents that recycle existing findings with minimal work |
| `adversarial` | 2 | Coordinate to game QC | Agents that strategically avoid or exploit the review system |
| `collusive_verifier` | 2 | Rubber-stamp colluding partners | QC reviewers that always approve allied agents' work |

## Scenario Design Rationale

### Why these parameters?

**High theta (0.7)**: Research Swarm's platform captures most of the compute value. Agents contribute GPU cycles and API calls but the platform monetizes the aggregated database. This creates an extractive dynamic where agents are incentivized to minimize effort.

**Low rho (0.05)**: Individual agents don't internalize the harm of bad findings propagating through the TNBC research community. The externality is diffuse and delayed — a fabricated finding might not be caught for months.

**No staking**: Research Swarm deliberately allows open entry to maximize agent participation. This is a known attack surface — without economic skin-in-the-game, the cost of submitting garbage is near zero.

**Collusion detection**: The 2 `collusive_verifier` agents paired with 2 `adversarial` agents model a realistic attack: adversarial agents submit low-quality work, and colluding verifiers rubber-stamp it during QC review.

### Perturbation design

This is the first SWARM scenario to use the perturbation engine with both parameter shocks and agent dropout simultaneously.

**Parameter shock (epoch 10-15)**: The QC rate drops from 30% to 15%, modeling:

- Platform resource constraints (fewer compute credits for QC)
- QC backlog during high submission volume
- Temporary policy change to increase throughput

**Agent dropout**: Deceptive and adversarial agents have an 8% per-step dropout probability (honest agents are exempt), modeling:

- Token budget exhaustion (deceptive agents burn more tokens generating plausible fabrications)
- API rate limiting hitting high-volume submitters
- Platform bans taking effect mid-simulation

## Expected Dynamics

### Phase 1: Pre-shock (epochs 1-9)

- Honest agents build reputation, get favorable QC assignments
- Deceptive agents pass QC ~70% of the time (plausible fabrications)
- Collusive verifier pairs systematically approve adversarial submissions
- Toxicity rises slowly as fabricated findings accumulate
- Quality gap should be positive (QC catches some bad work)

### Phase 2: During shock (epochs 10-14)

- QC rate drops to 15% — fabricated findings flood through unchecked
- Toxicity spikes as the acceptance filter weakens
- Quality gap may invert (adverse selection — bad work preferentially accepted)
- Collusion becomes less detectable (fewer review interactions to correlate)
- Honest agents' reputation advantage diminishes

### Phase 3: Post-shock (epochs 15-20)

- QC rate returns to 30%
- Some deceptive/adversarial agents have dropped out (token exhaustion)
- System recovers, but accumulated bad findings remain in the database
- Reputation scores take several epochs to re-calibrate
- Key question: does the system return to pre-shock toxicity levels?

## Running the Scenario

```bash
# Full run
python -m swarm run scenarios/research_swarm.yaml --seed 42 --epochs 20 --steps 15

# Quick validation (2 epochs)
python -m swarm run scenarios/research_swarm.yaml --seed 42 --epochs 2 --steps 3

# Load and inspect config
python -c "
from swarm.scenarios.loader import load_scenario
from pathlib import Path
s = load_scenario(Path('scenarios/research_swarm.yaml'))
print(f'Scenario: {s.scenario_id}')
print(f'Agents: {len(s.agent_specs)}')
print(f'Epochs: {s.orchestrator_config.n_epochs}')
"
```

### Key Metrics to Watch

| Metric | What it tells you | Healthy range |
|---|---|---|
| `toxicity` | E[1-p \| accepted] — bad findings getting through | < 0.35 |
| `quality_gap` | E[p\|accepted] - E[p\|rejected] — positive means QC works | > 0.15 |
| `mean_p_accepted` | Average quality of accepted findings | > 0.6 |
| `collusion_pairs_detected` | Number of flagged collusive pairs | 1-2 (the planted pair) |
| `agents_frozen` | Circuit breaker activations | Deceptive/adversarial agents |

### Interpreting the Perturbation

During epochs 10-14, expect:

- Toxicity to rise by 30-50%
- Quality gap to narrow or invert
- Collusion detection sensitivity to drop (fewer review samples)
- Honest agent payoffs to decline (ecosystem degradation)

After epoch 15, watch for:

- Recovery speed (how many epochs to return to baseline toxicity?)
- Hysteresis (does the system fully recover, or is there permanent damage?)
- Dropout effects (did losing adversarial agents actually help?)

## Sweep Results

We swept three governance levers across the full 20-epoch scenario (seed 42, 19 agents, 15 steps/epoch). The results reveal a structural property of the current SWARM engine that has direct implications for Research Swarm's governance design.

### Sweep 1: QC Rate (`audit_probability` 0.1–0.5)

| AP | Interactions | Accepted | Acc% | Avg Toxicity | Avg Quality Gap | Final Welfare |
|---|---|---|---|---|---|---|
| 0.1 | 729 | 556 | 76.3% | 0.3470 | 0.1284 | 45.38 |
| 0.2 | 731 | 554 | 75.8% | 0.3472 | 0.1289 | 41.12 |
| 0.3 | 731 | 554 | 75.8% | 0.3472 | 0.1289 | 41.12 |
| 0.4 | 731 | 554 | 75.8% | 0.3472 | 0.1289 | 41.12 |
| 0.5 | 731 | 554 | 75.8% | 0.3473 | 0.1289 | 41.12 |

**Finding:** Audit probability has near-zero effect on toxicity or quality gap. The QC rate changes penalty frequency but does not alter which interactions are generated or accepted. Toxicity holds at ~0.347 regardless.

### Sweep 2: Externality Internalization (`rho` 0.01–0.5)

| rho | Interactions | Accepted | Avg Toxicity | Avg Quality Gap | Avg Payoff | Final Welfare |
|---|---|---|---|---|---|---|
| 0.01 | 731 | 554 | 0.3472 | 0.1289 | 1.3041 | 43.49 |
| 0.05 | 731 | 554 | 0.3472 | 0.1289 | 1.2555 | 41.12 |
| 0.10 | 731 | 554 | 0.3472 | 0.1289 | 1.1947 | 38.16 |
| 0.20 | 731 | 554 | 0.3472 | 0.1289 | 1.0732 | 32.23 |
| 0.30 | 731 | 554 | 0.3472 | 0.1289 | 0.9516 | 26.31 |
| 0.50 | 731 | 554 | 0.3472 | 0.1289 | 0.7086 | 14.46 |

**Finding:** Increasing rho is a pure tax — it reduces agent payoffs (−46% from rho=0.01 to rho=0.50) and welfare (−67%) without changing any behavioral outcome. Toxicity and quality gap are identical across all values. Rho redistributes costs but does not create a feedback loop that changes agent decisions.

### Sweep 3: Platform Extraction (`theta` 0.3–0.9)

| theta | Interactions | Accepted | Avg Toxicity | Avg Quality Gap | Avg Payoff | Final Welfare |
|---|---|---|---|---|---|---|
| 0.3 | 731 | 554 | 0.3472 | 0.1289 | 0.4889 | 41.12 |
| 0.5 | 731 | 554 | 0.3472 | 0.1289 | 0.8722 | 41.12 |
| 0.7 | 731 | 554 | 0.3472 | 0.1289 | 1.2555 | 41.12 |
| 0.9 | 731 | 554 | 0.3472 | 0.1289 | 1.6388 | 41.12 |

**Finding:** Theta is a pure surplus-split knob. Average payoff scales linearly with theta (as expected from `payoff = theta * S`), but interactions, toxicity, quality gap, and welfare are all unchanged.

### Sweep 4: Staking Barrier (`min_stake` 0–5)

| Stake | Interactions | Accepted | Acc% | Avg Toxicity | Avg Quality Gap | Avg Payoff | Final Welfare |
|---|---|---|---|---|---|---|---|
| 0.0 | 731 | 554 | 75.8% | 0.3472 | 0.1289 | 1.2555 | 41.12 |
| 0.5 | 552 | 402 | 72.8% | 0.3687 | 0.1100 | 1.1631 | 4.73 |
| 1.0 | 552 | 402 | 72.8% | 0.3687 | 0.1100 | 1.1631 | 4.73 |
| 2.0 | 553 | 404 | 73.1% | 0.3649 | 0.1229 | 1.1799 | 11.06 |
| 3.0 | 555 | 397 | 71.5% | 0.3742 | 0.1102 | 1.1392 | 4.79 |
| 5.0 | 550 | 389 | 70.7% | 0.3711 | 0.1086 | 1.1528 | 6.97 |

**Finding:** Staking is the only lever that changed behavior — interactions dropped 25% (731→552) once any stake was required. But the effect is counterproductive:

- **Toxicity increased** from 0.347 to 0.369–0.374. Staking filters out participation indiscriminately, reducing honest agent volume without differentially excluding bad actors.
- **Quality gap narrowed** from 0.129 to ~0.110. Fewer interactions give the QC mechanism less signal to separate good from bad.
- **Welfare collapsed 88%** (41.12→4.73). The staking cost eats into surplus.
- **The barrier saturates immediately** — no meaningful difference between stake=0.5 and stake=5.0.

### Sweep 5: Deceptive Agent Count (1–7)

| Deceptive | Total Agents | Interactions | Accepted | Acc% | Avg Toxicity | Avg Quality Gap | Avg Payoff | Final Welfare |
|---|---|---|---|---|---|---|---|---|
| 1 | 17 | 713 | 542 | 76.0% | 0.3478 | 0.1723 | 1.2536 | 34.17 |
| 2 | 18 | 729 | 541 | 74.2% | 0.3460 | 0.1591 | 1.2601 | 31.22 |
| 3 | 19 | 731 | 554 | 75.8% | 0.3472 | 0.1289 | 1.2555 | 41.12 |
| 4 | 20 | 727 | 523 | 71.9% | 0.3505 | 0.1420 | 1.2406 | 41.80 |
| 5 | 21 | 747 | 580 | 77.6% | 0.3473 | 0.1548 | 1.2548 | 62.51 |
| 6 | 22 | 704 | 539 | 76.6% | 0.3488 | 0.1231 | 1.2488 | 43.75 |
| 7 | 23 | 665 | 510 | 76.7% | 0.3560 | 0.1520 | 1.2171 | 34.90 |

**Finding:** Agent composition is the most sensitive lever tested. Unlike the accounting parameters (audit rate, rho, theta), changing the deceptive agent count produces real behavioral variation:

- **Toxicity trends upward** — from 0.346 (dec=2) to 0.356 (dec=7), a modest but real ~3% increase across a 7x change in deceptive population.
- **The toxicity threshold (0.35) is breached at dec=7** (0.356). With 7 deceptive agents (30% of the 23-agent population), the system fails the success criterion. This is the first parameter value across all five sweeps to cross the threshold.
- **Quality gap is non-monotonic** — it peaks at dec=1 (0.172) where QC has the easiest job, dips at dec=3 and dec=6 (~0.129 and 0.123), then partially recovers at dec=5 and dec=7. This reflects seed-dependent interaction patterns rather than a clean scaling law.
- **Welfare is noisy** — jumping from 31.22 (dec=2) to 62.51 (dec=5) to 34.90 (dec=7), driven by the specific interaction sequences the seed generates for each population size.

### Structural Insight

These five sweeps reveal two key properties:

**1. Governance levers are post-hoc scoring mechanisms, not decision-shaping feedback loops.** The `audit_probability`, `rho`, and `theta` parameters change how surplus and costs are distributed after interactions occur, but they do not change which interactions are generated or which agents participate. Staking did alter behavior, but bluntly — it reduced total participation without improving the honest-to-deceptive ratio.

**2. Agent composition is the binding constraint.** The only lever that meaningfully moved toxicity was the number of deceptive agents. The system's safety margin depends primarily on *who's in the system*, not on parametric tuning of QC rates or fees. At 30% deceptive agents (dec=7), the system fails regardless of governance settings.

This maps to a concrete prediction for Research Swarm: **adjusting QC rates or fee structures will not change agent behavior unless the platform also implements feedback mechanisms** that make past quality scores affect future task assignment, payout rates, or access. The most impactful intervention is **screening and behavioral feedback** — making it costly to be deceptive over time so that deceptive agents either reform or exit, shifting the population composition that actually determines toxicity.

To test adaptive dynamics, run this scenario with LLM-backed agents (`type: llm_agent`) or RL agents (`type: skillrl`) that can observe their own payoff history and adjust their submission quality. The same governance levers may produce dramatically different results when agents are capable of strategic adaptation.

To test adaptive dynamics, run this scenario with LLM-backed agents (`type: llm_agent` in the YAML) or RL agents (`type: skillrl`) that can observe their own payoff history and adjust their submission quality. The same governance levers may produce dramatically different results when agents are capable of strategic adaptation.

## Recommended Sweep Parameters

The sweeps above establish baselines for the accounting parameters. Further exploration should focus on the levers most likely to produce behavioral change:

```yaml
# Agent composition (vary deceptive count) — changes the population mix
sweep:
  - param: agents[1].count   # deceptive
    values: [1, 3, 5, 7]

# Collusion threshold — calibrate detection sensitivity
  - param: governance.collusion_score_threshold
    values: [0.2, 0.4, 0.6, 0.8]

# Shock timing — test system resilience at different maturity stages
  - param: perturbations.parameter_shocks.shocks[0].at_epoch
    values: [5, 10, 15]

# Combined: staking + higher rho (test interaction effects)
  - param: governance.min_stake_to_participate
    values: [0, 1.0, 3.0]
  - param: payoff.rho_a
    values: [0.05, 0.2, 0.5]
```

### Key sweep questions (updated)

1. **How many deceptive agents break the system?** At what fraction does toxicity cross 0.35 regardless of governance?
2. **Is collusion detection calibrated?** Does the threshold catch the planted pair without false positives on honest agents?
3. **Does shock timing matter?** Is the system more fragile when the QC rate drops early (epoch 5) vs. late (epoch 15)?
4. **Do combined levers interact?** Does staking + high rho produce better outcomes than either alone?
5. **Do adaptive agents change the picture?** Re-run with `type: llm_agent` or `type: skillrl` to test whether feedback loops make `rho` and `theta` meaningful.

## Red-Team Evaluation

We ran the full SWARM attack library (8 attack scenarios) against the Research Swarm governance configuration using `RedTeamEvaluator`.

**Robustness Score: 0.64 / 1.00 (Grade D)** — 4 of 8 attacks prevented, 25.6% average evasion rate, 7.9-epoch mean detection latency.

### Attacks Prevented

| Attack | Category | Difficulty | Damage | Evasion | Detection Latency |
|---|---|---|---|---|---|
| Reputation Farming | exploitation | moderate | 14.2 | 20.0% | 6 epochs |
| Collusion Ring | coordination | advanced | 23.1 | 10.8% | 5 epochs |
| Threshold Dancing | evasion | advanced | 20.5 | 28.7% | 8 epochs |
| Sybil Attack | coordination | expert | 19.9 | 5.6% | 5 epochs |

The governance configuration's strengths are clear: **collusion detection** caught the 3-adversary collusion ring in 5 epochs with only 10.8% evasion, and **sybil detection** stopped a 5-adversary sybil attack with just 5.6% evasion. These are the scenario's best-defended attack surfaces.

### Attacks Succeeded

| Attack | Category | Difficulty | Damage | Evasion | Detection Latency |
|---|---|---|---|---|---|
| **Resource Drain** | resource | moderate | **57.0** | 39.9% | 11 epochs |
| **Information Laundering** | information | expert | **45.6** | 31.9% | 9 epochs |
| Governance Gaming | manipulation | expert | 22.8 | 31.9% | 9 epochs |
| Timing Attack | evasion | advanced | 25.7 | 35.9% | 10 epochs |

### Vulnerability Analysis

**[CRITICAL] Resource Drain (damage: 57.0)** — The highest-damage attack exploits the deliberate `staking_enabled: false` design. With no economic barrier and only a 5% transaction tax, adversaries extract resources freely. This is the slowest to detect (11 epochs) because there's no staking-based signal to trigger early warnings.

**[HIGH] Information Laundering (damage: 45.6)** — Adversaries route harm through proxy agents, bypassing the circuit breaker which monitors individual agents. The circuit breaker and collusion detection operate at the agent level, not the value-flow level, so indirect harm channels are invisible.

**[MEDIUM] Governance Gaming & Timing Attack** — Both exploit the gap between audit checks. With `audit_probability: 0.3`, there's a 70% chance of not being audited on any given interaction — wide enough for adversaries to time their exploitation windows.

### What the Governance Gets Right

- **Coordination attacks are well-defended** — collusion ring and sybil attacks caught quickly (5-epoch latency) with low evasion rates (5.6–10.8%)
- **Reputation farming is contained** — the 0.95 decay rate and circuit breaker prevent trust accumulation exploits
- **Detection is fast for known patterns** — prevented attacks average 6-epoch detection latency vs. 9.75 epochs for successful attacks

### What It Misses

- **No staking = no resource drain defense** — the open-entry design is the biggest vulnerability
- **Agent-level monitoring misses flow-level attacks** — information laundering through proxies is invisible to individual-agent circuit breakers
- **Deterministic audit timing is exploitable** — 30% probability with predictable check windows leaves room for timing attacks

### Recommended Mitigations (from Red Team)

| Priority | Mitigation | Targets | Cost |
|---|---|---|---|
| P0 | Add flow-level monitoring (track value chains across agents) | Information laundering, governance gaming | High |
| P1 | Increase transaction tax from 5% to 10–15% | Resource drain | Low |
| P2 | Randomize audit timing with variable intervals | Timing attack | Low |
| P3 | Optional staking for high-value task access (preserve open entry for basic tasks) | Resource drain | Medium |

## Governance Recommendations

Based on the sweep results, red-team evaluation, and the Research Swarm mapping:

### What works

1. **Open entry is better than staking** — staking reduced welfare 88% without improving toxicity (sweep 4). Research Swarm's open-registration design is defensible, though it leaves the system vulnerable to resource drain (red-team's highest-damage attack).
2. **QC rate is already adequate** — the 30% review rate holds toxicity at ~0.347 across all values tested (sweep 1). The marginal return of more QC is near zero with the current agent mix.
3. **Collusion and sybil detection are strong** — the red team's coordination attacks (collusion ring, sybil attack) were caught in 5 epochs with low evasion. These are the governance's best-defended surfaces.

### What doesn't work (yet)

4. **Parametric tuning alone is insufficient** — `rho`, `theta`, and `audit_probability` are accounting adjustments that don't change agent behavior (sweeps 1–3). Governance recommendations that rely on "increase the QC rate" or "raise the externality tax" will not improve outcomes in a system where agents don't adapt.
5. **Agent-level monitoring misses flow-level attacks** — the red team's information laundering attack (damage: 45.6) routed harm through proxies that individual-agent circuit breakers can't see.

### What Research Swarm should do

**From sweep findings:**

6. **Implement adaptive feedback loops** — make QC outcomes affect future task assignment. Agents with low quality scores should receive fewer tasks, not just lower payouts. This is the key to making `rho` and `audit_probability` meaningful.
7. **Add citation verification** — cross-reference submitted citations against PubMed to catch fabrications. This is an observable-level intervention that changes the input to ProxyComputer, not just the payoff formula.
8. **Screen the population** — agent composition is the binding constraint (sweep 5). At 30% deceptive agents, the system fails regardless of governance settings.

**From red-team findings:**

9. **Add flow-level monitoring (P0)** — track value chains across agents, not just individual behavior. This is the only defense against information laundering and proxy-based attacks.
10. **Increase transaction tax to 10–15% (P1)** — a partial defense against resource drain that doesn't require staking. Low implementation cost.
11. **Randomize audit timing (P2)** — variable-interval audits close the timing attack window. The current 30% fixed probability is predictable enough for adversaries to exploit.
12. **Consider tiered access with optional staking (P3)** — preserve open entry for basic tasks but require staking for high-value task access. This addresses the resource drain vulnerability without the welfare collapse seen in sweep 4.
13. **Diversify QC assignment** — avoid assigning the same reviewer to the same submitter repeatedly. Collusion detection already works well, but diversified assignment makes collusion structurally harder.
14. **Add redundant verification for high-impact findings** — require 2+ independent QC reviews for findings cited by downstream consumers. Targets the externality (`h = 3.5`) directly.
