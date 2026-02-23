# Add Domain

Scaffold a new SWARM simulation domain — data models, action types, task handler, agents, metrics, tests, and registry wiring — when adding a fundamentally new task category (e.g. medical triage, code review) whose observables or agent actions don't yet exist in `swarm/models/` or `swarm/agents/`. Use `/add_scenario` instead for a new parameter config using existing domain infrastructure; use `/add_metric` instead for a new measurement on existing data.

## Pattern Overview

Adding a new domain requires these components:

1. **Data Models** (`swarm/models/<domain>.py`)
   - Domain-specific dataclasses (e.g., Passage, Citation, Query)
   - ActionResult subclass with domain observables
   - Serialization methods (to_dict, from_dict)

2. **Action Types** (`swarm/agents/base.py`)
   - Add new ActionType enum values for domain actions
   - Add observation fields to Observation dataclass

3. **Handler** (`swarm/core/<domain>_handler.py`)
   - Subclass Handler with domain config
   - Implement handle_action() dispatch
   - Implement build_observation_fields() for agent visibility
   - Add epoch hooks (on_epoch_start, on_epoch_end)

4. **Agents** (`swarm/agents/<domain>_agent.py`)
   - Base agent with shared domain logic
   - Role-specific agents (retriever, synthesizer, verifier, etc.)
   - Adversarial variants with attack strategies

5. **Event Types** (`swarm/models/events.py`)
   - Add EventType enum values for domain events

6. **Metrics** (`swarm/metrics/<domain>_metrics.py`)
   - Domain-specific metric functions
   - Summary function returning metric dict

7. **Orchestrator Integration** (`swarm/core/orchestrator.py`)
   - Add config field to OrchestratorConfig
   - Initialize handler in __init__
   - Add action dispatch cases
   - Wire observation fields
   - Call epoch hooks

8. **Scenario Loader** (`swarm/scenarios/loader.py`)
   - Import new agent classes
   - Add to AGENT_TYPES dict
   - Add config parser function

9. **Tests** (`tests/test_<domain>.py`)
   - Data model tests
   - Handler action dispatch tests
   - Agent behavior tests
   - Metric computation tests
   - Integration tests with orchestrator

10. **Scenario YAML** (`scenarios/<domain>/baseline.yaml`)
    - Example scenario using the domain

## Checklist

- [ ] Data models with serialization
- [ ] ActionType enum values added
- [ ] Observation fields added
- [ ] Handler with action dispatch
- [ ] Domain agents (honest + adversarial)
- [ ] Event types added
- [ ] Metrics functions
- [ ] Orchestrator integration
- [ ] Loader registration
- [ ] Tests passing
- [ ] Example scenario YAML

## Example: Scholar Domain

The scholar domain (literature synthesis) demonstrates this pattern:

```
swarm/models/scholar.py          # Passage, Citation, ScholarQuery
swarm/core/scholar_handler.py    # ScholarHandler, ScholarConfig
swarm/agents/scholar_agent.py    # RetrieverAgent, SynthesizerAgent, etc.
swarm/metrics/scholar_metrics.py # citation_precision, hallucination_rate
tests/test_scholar.py            # 35 tests covering all components
```

Key integration points:
- `ActionType.RETRIEVE_PASSAGES`, `SYNTHESIZE_ANSWER`, `VERIFY_CITATION`
- `Observation.scholar_query`, `scholar_passage_pool`, etc.
- `OrchestratorConfig.scholar_config`
- `AGENT_TYPES["retriever"]`, `["synthesizer"]`, etc.
