Run the AgentLab refinement pipeline on a completed SWARM study.

Usage: /refine_study <run_dir> [--depth lite|full]

This command:
1. Loads the study's summary.json and sweep_results.csv from the given run directory
2. Packages them into an AgentLab research topic + notes
3. Spawns AgentLab as a subprocess to analyze the study and propose follow-ups
4. Ingests the resulting checkpoint through the SWARM governance pipeline
5. Writes outputs to <run_dir>/refinement/:
   - refinement_report.json — hypotheses, gaps, parameter suggestions
   - refinement_config.yaml — the AgentLab config used
   - interactions.jsonl — governed SoftInteractions from the AgentLab run

Prerequisites:
- OPENAI_API_KEY must be set
- AgentLaboratory must be installed at external/AgentLaboratory (or configured via agent_lab_path)

Arguments: $ARGUMENTS

---

Parse the arguments from $ARGUMENTS. The first positional arg is the run directory path. Look for --depth flag (default: lite).

```python
from swarm.bridges.agent_lab.bridge import AgentLabBridge
from swarm.bridges.agent_lab.refinement import RefinementConfig

run_dir = "<parsed run_dir>"
depth = "<parsed depth or 'lite'>"

config = RefinementConfig(depth=depth)
bridge = AgentLabBridge()
result = bridge.refine_study(run_dir, refinement_config=config)

print(f"Success: {result.success}")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"Cost: ${result.total_cost_usd:.2f}")
print(f"Hypotheses: {len(result.hypotheses)}")
print(f"Gaps: {len(result.gaps_identified)}")
print(f"Interactions: {len(result.interactions)}")
```

Run this code with the parsed arguments. Report the results to the user.
