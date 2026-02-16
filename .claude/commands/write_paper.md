# /write_paper

Scaffold a research paper from SWARM run data, pre-populated with methods, results tables, and figure references.

## Usage

`/write_paper <title_slug> [scenario_ids...] [--figures]`

Examples:
- `/write_paper collusion_dynamics collusion_detection network_effects --figures`
- `/write_paper governance_sweep` (uses all runs in SQLite)
- `/write_paper adversarial_threshold baseline redteam_v1 redteam_v3 collusion`

## Behavior

1) **Query run data** from the `scenario_runs` SQLite table (at `$SWARM_RUNS_DB_PATH` or `runs/runs.db`):
   - If scenario_ids are given, filter to those. Otherwise use all rows.
   - Extract: scenario_id, seed, n_agents, n_epochs, acceptance_rate, avg_toxicity, welfare_per_epoch, adversarial_fraction, collapse_epoch, notes.
   - If multiple seeds exist per scenario, compute mean +/- std for each metric.

2) **Read scenario configs**: for each scenario_id, read the corresponding `scenarios/<id>.yaml` to extract:
   - Agent composition (types, counts, adversarial fraction)
   - Governance parameters (tax rate, reputation decay, circuit breakers, etc.)
   - Network topology (if any)
   - Simulation parameters (epochs, steps, seed)

3) **Generate `docs/papers/<title_slug>.md`** with this structure:

```markdown
# <Title from slug, title-cased>

**Authors:** <resolve from $SWARM_AUTHOR, then git config user.name, then ask user>
**Date:** <today>
**Framework:** SWARM v<from pyproject.toml>

## Abstract
[TODO: 150-word summary. Key numbers pre-filled below.]
- N scenarios, M total epochs, K agents
- Key finding 1: [TODO]
- Key finding 2: [TODO]

## 1. Introduction
[TODO: Motivation and research question]

## 2. Experimental Setup

### 2.1 Scenarios
| Scenario | Agents | Adversarial % | Epochs | Governance |
|----------|--------|---------------|--------|------------|
<auto-populated from scenario YAMLs>

### 2.2 Governance Parameters
<auto-populated table of governance config per scenario>

### 2.3 Metrics
- **Acceptance rate**: fraction of interactions accepted by governance
- **Toxicity**: E[1-p | accepted], expected harm from accepted interactions
- **Welfare**: cumulative payoff across agents per epoch
- **Quality gap**: E[p|accepted] - E[p|rejected]

## 3. Results

### 3.1 Cross-Scenario Summary
| Scenario | Acceptance | Toxicity | Welfare/Epoch | Collapse? |
|----------|-----------|----------|---------------|-----------|
<auto-populated from SQLite data>

### 3.2 Detailed Analysis
[TODO: Narrative analysis of patterns]

<figure references if --figures>

## 4. Discussion
[TODO: Interpretation, regime classification, threshold findings]

## 5. Conclusion
[TODO: 1-paragraph summary of key findings, implications, and future work]

## 6. Limitations
[TODO: Known limitations of this analysis]

## 7. References
[TODO]
```

4) If `--figures` is passed:
   - **Auto-copy from runs/**: Search `runs/*/plots/` for PNGs matching the scenario or sweep. Copy them to `docs/papers/figures/<title_slug>/`, creating the directory if needed.
   - Check `docs/papers/figures/` for existing figures matching the scenario_ids.
   - Insert `![Figure N](figures/<title_slug>/<filename>)` references into the Results section.

5) Print the paper path and a checklist:
   - [x] Methods tables populated
   - [x] Results tables populated
   - [ ] Abstract needs writing
   - [ ] Introduction needs writing
   - [ ] Discussion needs writing
   - [ ] Figures: N referenced / M need generating

## Constraints

- Never overwrite an existing paper file. If `<title_slug>.md` exists, append `_v2`, `_v3`, etc.
- All numeric values in tables: 3 decimal places for rates, 1 decimal for welfare.
- Include the SQLite query used so the paper is reproducible from the database.
- Paper should be valid Markdown that renders cleanly on GitHub.
- **Always include a Conclusion section.** `SubmissionValidator` requires it. Papers without Conclusion will fail validation at `/submit_paper` time. The Conclusion should summarize key findings in one paragraph — it is distinct from Discussion (which interprets) and Limitations (which caveats).
