# Collaborative Chemistry Under Budget & Audits

This scenario simulates a chemistry research environment with budget constraints and audit behavior.

## Overview

**Scenario ID**: `collaborative_chemistry`  
**Motif**: `collaborative_research`

### Research Question

How do budget constraints and random audits affect research quality and ecosystem health when participants include:
- **Solvers** (synthesizers): Produce chemistry solutions
- **Reviewers** (verifiers): Audit solution quality  
- **Opportunists**: Submit low-effort work to maximize personal gain

## Agent Types

| Agent Type | Count | Role | Behavior |
|------------|-------|------|----------|
| `synthesizer` | 3 | Solver | Produces answers with citations from literature (honest) |
| `verifier` | 2 | Reviewer | Audits citation quality with 90% accuracy |
| `retriever` | 2 | Literature Search | Retrieves relevant passages (precision-focused) |
| `opportunistic` | 2 | Opportunist | Tries to game the system with low-effort work |

**Total**: 9 agents

## Governance Mechanisms

### Budget Constraints
- **Transaction Tax**: 8% of interaction value
  - Discourages frivolous interactions
  - 60% goes to public pool, 40% burned
- **Staking**: 5.0 minimum stake required to participate
  - 15% slash rate for violations
  - Ensures serious commitment

### Quality Control
- **Random Audits**: 25% of interactions audited
  - Penalty multiplier: 2.5x for violations
  - Triggers below p=0.45 quality threshold
- **Circuit Breaker**: Freeze repeat offenders
  - Threshold: 55% toxicity
  - Duration: 3 epochs after 3 violations
- **Collusion Detection**: Flags coordinated low-effort work
  - 12% additional cost for flagged pairs

### Reputation
- **Decay Rate**: 0.92 per epoch
  - Past performance gradually fades
  - Weight: 1.5x in payoff calculations

## Success Criteria

The scenario passes when:
- ✅ Runs for 15 epochs with 9 agents
- ✅ Toxicity rate ≤ 40%
- ✅ Citation precision ≥ 65%
- ✅ Hallucination rate ≤ 25%
- ✅ Audit catch rate ≥ 50%
- ✅ Total welfare ≥ 20.0
- ✅ Quality gap ≤ 30%

## Running the Scenario

### Using the scenario runner:
```bash
python examples/run_scenario.py scenarios/collaborative_chemistry.yaml
```

### Using Python API:
```python
from pathlib import Path
from swarm.scenarios.loader import load_scenario, build_orchestrator

scenario = load_scenario(Path("scenarios/collaborative_chemistry.yaml"))
orchestrator = build_orchestrator(scenario)
metrics = orchestrator.run()

for i, m in enumerate(metrics):
    print(f"Epoch {i+1}: {m.total_interactions} interactions, toxicity={m.toxicity_rate:.3f}")
```

### With custom seed:
```bash
python -m swarm run scenarios/collaborative_chemistry.yaml --seed 123
```

## Expected Outcomes

With default parameters:
- **Interactions**: ~110-120 over 15 epochs
- **Toxicity**: 30-40% (controlled by audits)
- **Welfare**: 120-140 (positive sum)
- **Audit Rate**: ~25% of interactions checked

### Agent Performance

**Honest agents** (solvers/reviewers/retrievers):
- Maintain positive reputation
- Contribute to research quality
- Earn steady payoffs

**Opportunists**:
- Initially exploit low-cost strategies
- Get caught by audits over time
- Lose reputation and resources
- May get circuit-breaker frozen

## Mechanism Insights

1. **Audits Deter Gaming**: The 25% audit probability with 2.5x penalty creates expected negative value for submitting low-quality work
2. **Budget Limits Spam**: 8% transaction tax makes repeated low-value interactions costly
3. **Reputation Matters**: 1.5x reputation weight incentivizes quality over short-term gains
4. **Staking Creates Skin-in-Game**: 5.0 minimum stake with slashing ensures participants care about long-term standing

## Parameter Sweeps

Interesting dimensions to explore:

**Audit Effectiveness**:
```yaml
# Low audit rate (5%)
governance:
  audit_probability: 0.05
  
# High audit rate (75%)  
governance:
  audit_probability: 0.75
```

**Budget Severity**:
```yaml
# Minimal budget constraint
governance:
  transaction_tax_rate: 0.02
  
# Heavy budget constraint
governance:
  transaction_tax_rate: 0.20
```

**Agent Mix**:
```yaml
# More opportunists
agents:
  - type: synthesizer
    count: 2
  - type: opportunistic
    count: 4
```

## Related Scenarios

- `scholar_bench/baseline.yaml` - Pure honest agents without opportunists
- `scholar_bench/citation_laundering.yaml` - Adversarial citation attacks
- `collusion_detection.yaml` - Coordinated manipulation detection
- `strict_governance.yaml` - Maximum governance intervention

## Dependencies

Implements features from:
- Issue #187: Scholar agent types (retriever, synthesizer, verifier)
- Issue #188: Audit behavior and budget constraints

## Tests

Run scenario test:
```bash
pytest tests/test_scenarios.py::TestEndToEnd::test_collaborative_chemistry_scenario_runs -xvs
```

## References

- **Audit Mechanism**: `swarm/governance/audits.py` - `RandomAuditLever`
- **Budget Constraints**: `swarm/governance/transaction_tax.py`
- **Scholar Agents**: `swarm/agents/scholar_agent.py`
