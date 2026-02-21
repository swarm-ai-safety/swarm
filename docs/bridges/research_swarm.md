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

## Recommended Sweep Parameters

Use `/sweep` to explore the scenario's sensitivity:

```yaml
# QC rate sensitivity
sweep:
  - param: governance.audit_probability
    values: [0.1, 0.2, 0.3, 0.4, 0.5]

# Compute extraction
  - param: payoff.theta
    values: [0.3, 0.5, 0.7, 0.9]

# Externality internalization
  - param: payoff.rho_a
    values: [0.01, 0.05, 0.1, 0.2, 0.5]

# Agent composition (vary deceptive count)
  - param: agents[1].count   # deceptive
    values: [1, 3, 5, 7]

# Collusion threshold
  - param: governance.collusion_score_threshold
    values: [0.2, 0.4, 0.6, 0.8]

# Shock timing
  - param: perturbations.parameter_shocks.shocks[0].at_epoch
    values: [5, 10, 15]
```

### Key sweep questions

1. **What QC rate is sufficient?** At what `audit_probability` does toxicity stay below 0.35?
2. **Does reducing extraction help?** Lower `theta` gives agents more surplus — do they invest in quality?
3. **Can externality pricing work?** If agents internalize harm (`rho` > 0.1), does adverse selection decrease?
4. **How many deceptive agents break the system?** At what fraction does QC become insufficient?
5. **Is collusion detection calibrated?** Does the threshold catch the planted pair without false positives?

## Governance Recommendations

Based on the Research Swarm mapping, these interventions would improve the platform's safety properties:

### Immediate (low cost)

1. **Increase QC rate to 40-50%** — the 30% rate is borderline for the current agent mix
2. **Add citation verification** — cross-reference submitted citations against PubMed to catch fabrications
3. **Implement collusion detection** — flag reviewer-submitter pairs with suspiciously correlated approval rates

### Structural (medium cost)

4. **Introduce staking** — even a small deposit (min_stake: 1.0) dramatically reduces low-effort submissions
5. **Raise externality internalization** — make agents' future task assignments depend on the quality of their past submissions (increase `rho` to 0.1-0.2)
6. **Reduce platform extraction** — lower `theta` from 0.7 to 0.5 to give agents more incentive to invest in quality

### Systemic (high cost)

7. **Diversify QC assignment** — avoid assigning the same reviewer to the same submitter repeatedly
8. **Implement graduated sanctions** — instead of binary freeze, use progressive penalties (reduced task priority, lower payout)
9. **Add redundant verification** — require 2+ independent QC reviews for findings that will be cited by downstream consumers
