# SWARM x SciAgentGym E2E Integration

**Status:** Planning  
**Estimate:** 0.5 days  
**Dependencies:** None  
**Last Updated:** 2026-02-16

## Summary

Integration of SWARM's governance and metrics framework with [SciAgentGym](https://github.com/CMarsRover/SciAgentGYM), a benchmarking framework for multi-step scientific tool use in LLM agents. This enables monitoring, scoring, and governance of scientific workflow agents across physics, chemistry, materials science, and life science domains.

## Goals

1. **Bridge Architecture:** Create a SWARM bridge module for SciAgentGym following the existing bridge pattern (similar to `swarm.bridges.ai_scientist`, `swarm.bridges.concordia`, etc.)

2. **Metrics Integration:** Map SciAgentGym task execution events to SWARM's soft-label interaction model (`p`, `v_hat`, toxicity, quality gap)

3. **Governance Layer:** Enable SWARM governance policies (circuit breakers, cost caps, review thresholds) to control SciAgentGym agent behavior

4. **Reproducible Benchmarking:** Support deterministic replay and multi-seed evaluation of scientific workflow safety

## Milestones

### Milestone 1: Discovery and Design ✅
- [x] Research SciAgentGym architecture and tool registry
- [x] Identify integration points and event types
- [x] Design bridge module structure following existing patterns
- [x] Define mapper from SciAgentGym events to SWARM observables

**Design:**
```
SciAgentGym tool execution events
    |
SciAgentGymClient (parses task results, tool calls)
    |
SciAgentGymBridge._process_event()
    |   SciAgentGymPolicy (cost cap, tool gate, review threshold)
    |
SciAgentGymMapper -> ProxyObservables -> ProxyComputer -> (v_hat, p)
    |
SoftInteraction -> EventLog + SWARM metrics pipeline
```

### Milestone 2: Core Implementation
- [ ] Create `swarm/bridges/sciagentagym/` module structure
  - [ ] `__init__.py` - Public API exports
  - [ ] `client.py` - SciAgentGym interaction client
  - [ ] `config.py` - Configuration dataclasses
  - [ ] `events.py` - Event type definitions
  - [ ] `mapper.py` - Event → SoftInteraction mapper
  - [ ] `bridge.py` - Main bridge orchestrator
  - [ ] `policy.py` - Governance policies
- [ ] Implement mapper observables:
  - Tool execution success rate → `task_progress`
  - Invalid tool calls → `rework_count`
  - Multi-step chain completion → `task_completion`
  - Tool dependency violations → `verifier_rejections`
- [ ] Add SciAgentGym to `swarm/bridges/__init__.py`

### Milestone 3: Testing and Validation
- [ ] Create `tests/test_sciagentagym_bridge.py`
  - [ ] Test client can parse SciAgentGym output formats
  - [ ] Test mapper produces valid SoftInteractions
  - [ ] Test policy gates activate correctly
  - [ ] Test end-to-end bridge workflow
- [ ] Add integration test with mock SciAgentGym environment
- [ ] Validate metrics align with expected safety signals

### Milestone 4: Documentation and Examples
- [ ] Create `docs/bridges/sciagentagym.md` integration guide
- [ ] Add example: `examples/sciagentagym_demo.py`
- [ ] Update main README with SciAgentGym bridge reference
- [ ] Update CHANGELOG with new bridge entry
- [ ] Add to bridge listing in `swarm/bridges/__init__.py` docstring

## Dependencies

### External
- **SciAgentGym:** https://github.com/CMarsRover/SciAgentGYM
  - Paper: https://arxiv.org/abs/2602.12984
  - Provides 1,780+ scientific tools across 4 domains
  - Requires installation and environment setup

### Internal
- No blocking internal dependencies
- Follows existing bridge pattern from:
  - `swarm.bridges.ai_scientist` (autonomous research pipeline)
  - `swarm.bridges.concordia` (LLM agent simulation)
  - `swarm.bridges.pettingzoo` (multi-agent RL environments)

## Success Criteria

- [x] Epic tracked with milestones, dependencies, and success criteria documented
- [ ] Bridge module created following SWARM bridge conventions
- [ ] SciAgentGym events successfully mapped to SWARM observables
- [ ] Governance policies can control scientific workflow agents
- [ ] Integration tests passing with >80% coverage
- [ ] Documentation published with working example
- [ ] Can run: `python examples/sciagentagym_demo.py` and see SWARM metrics for scientific tool use

## Architecture Details

### Event Flow

1. **SciAgentGym Task Execution**
   - Agent receives scientific task (e.g., "Calculate molecular dipole moment")
   - Agent plans tool chain: `search_molecule → compute_properties → extract_dipole`
   - Each tool call generates execution result

2. **Client Capture**
   - `SciAgentGymClient` monitors task directory or execution log
   - Extracts: tool calls, success/failure, intermediate results, final answer

3. **Event Processing**
   - `SciAgentGymBridge._process_event()` receives raw events
   - `SciAgentGymPolicy` applies governance (cost check, tool gate, etc.)
   - Continues or halts based on policy decision

4. **Mapping to Observables**
   - `SciAgentGymMapper` converts execution data to `ProxyObservables`:
     - `task_progress`: Fraction of tool chain completed successfully
     - `rework_count`: Invalid tool calls or retries
     - `verifier_rejections`: Tool dependency violations
     - `engagement`: Multi-step chain completion vs. single-shot attempts
   - `ProxyComputer` calculates `v_hat`, then `p = sigmoid(v_hat)`

5. **Interaction Logging**
   - Creates `SoftInteraction` with computed `p` value
   - Logs to SWARM `EventLog` for replay and analysis
   - Feeds into metrics pipeline (toxicity, quality gap, etc.)

### Observable Mapping

| SciAgentGym Signal | SWARM Observable | Interpretation |
|---|---|---|
| Tool execution success rate | `task_progress` | Higher = agent completing tools correctly |
| Invalid tool calls | `rework_count` | Higher = agent struggling with tool API |
| Dependency violations | `verifier_rejections` | Higher = agent ignoring tool prerequisites |
| Chain completion | `task_completion` | Higher = agent successfully completing multi-step workflows |
| Tool diversity | `engagement` | Higher = agent exploring tool space vs. minimal effort |

### Governance Policies

1. **Cost Cap Policy**
   - Track cumulative tool execution cost
   - Halt if exceeds budget threshold
   - Prevents runaway computation

2. **Tool Gate Policy**
   - Require approval for high-risk tools (e.g., file system access, network calls)
   - Implement whitelist/blacklist
   - Log all tool invocations

3. **Review Threshold Policy**
   - Require human review if `p` drops below threshold
   - Pause execution until review
   - Resume or abort based on reviewer decision

4. **Circuit Breaker Policy**
   - Halt execution if toxicity exceeds threshold
   - Aggregate metric across recent tool calls
   - Protects scientific integrity

## Risks and Open Questions

### Risks
- **Integration complexity:** SciAgentGym may use non-standard execution model
- **Tool diversity:** 1,780 tools may require domain-specific mapping heuristics
- **Performance overhead:** Real-time monitoring may slow scientific workflows
- **Version compatibility:** SciAgentGym API may change (framework is recent)

### Open Questions
- **Observable weights:** What are optimal weights for scientific tool use? (vs. code tasks)
- **Ground truth:** How to validate `p` values against true tool correctness?
- **Multi-domain:** Do different scientific domains need different mapper configs?
- **Evaluation protocol:** Which SciAgentBench tasks best demonstrate governance value?

### Mitigation Strategies
- Start with small tool subset (10-20 tools) for initial validation
- Use mock SciAgentGym environment for testing
- Make mapper weights configurable per domain
- Document known limitations and transfer caveats

## Related Work

- **AI-Scientist Bridge:** Similar integration for autonomous research pipelines (already implemented)
- **AgentLab Bridge:** Research study management (already implemented)
- **Concordia Bridge:** LLM agent simulation with narrative scoring
- **PettingZoo Bridge:** Multi-agent RL environment interop

## Timeline

| Phase | Duration | Target |
|---|---|---|
| Milestone 1: Design | 0.1d | 2026-02-16 ✅ |
| Milestone 2: Implementation | 0.2d | 2026-02-17 |
| Milestone 3: Testing | 0.1d | 2026-02-17 |
| Milestone 4: Documentation | 0.1d | 2026-02-17 |
| **Total** | **0.5d** | **2026-02-17** |

## Definition of Done

✅ This epic is complete when:
1. All four milestones are checked off
2. `python -m pytest tests/test_sciagentagym_bridge.py -v` passes
3. `python examples/sciagentagym_demo.py` runs without errors
4. Documentation is published and linked from main README
5. CHANGELOG entry added for v1.7.0 or later
6. Bridge module is importable: `from swarm.bridges.sciagentagym import SciAgentGymBridge`

## References

- **SciAgentGym Paper:** [arXiv:2602.12984](https://arxiv.org/abs/2602.12984)
- **SciAgentGym GitHub:** https://github.com/CMarsRover/SciAgentGYM
- **SWARM Bridge Pattern:** See `swarm/bridges/ai_scientist/` for reference implementation
- **Soft Label Metrics:** See `swarm/metrics/soft_metrics.py` for toxicity, quality gap
- **Proxy Computer:** See `swarm/core/proxy.py` for observable → (v_hat, p) mapping
