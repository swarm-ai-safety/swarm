---
name: paper-writing
description: Scaffold research papers from SWARM run data with auto-populated tables
version: "1.0"
domain: swarm-safety
triggers:
  - write paper
  - scaffold paper
  - generate paper
---

# Paper Writing Skill

Generate a markdown research paper pre-populated with methods tables, results tables, and figure references from SWARM experiment data.

## Prerequisites

- `sqlite3` (Python stdlib) for reading runs database
- `pandas>=2.0` for data manipulation
- Experiment data in SQLite database or CSV files

## Procedure

### 1. Query the runs database

```python
import sqlite3
import pandas as pd

def load_runs(db_path, scenario_ids=None):
    """Load scenario runs from SQLite database."""
    conn = sqlite3.connect(db_path)
    
    if scenario_ids:
        placeholders = ",".join("?" * len(scenario_ids))
        query = f"SELECT * FROM scenario_runs WHERE scenario_id IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=scenario_ids)
    else:
        df = pd.read_sql_query("SELECT * FROM scenario_runs", conn)
    
    conn.close()
    return df
```

### 2. Build the methods table

```python
def build_methods_table(df):
    """Generate a markdown table of experimental scenarios."""
    scenarios = df.groupby("scenario_id").first().reset_index()
    
    lines = ["| Scenario | Agents | Governance | Seeds | Epochs |",
             "|----------|--------|-----------|-------|--------|"]
    
    for _, row in scenarios.iterrows():
        lines.append(
            f"| {row['scenario_id']} | {row.get('n_agents', 'N/A')} | "
            f"{row.get('governance_desc', 'default')} | "
            f"{row.get('n_seeds', 'N/A')} | {row.get('n_epochs', 'N/A')} |"
        )
    
    return "\n".join(lines)
```

### 3. Build the results table

```python
def build_results_table(df):
    """Generate a cross-scenario summary results table."""
    summary = df.groupby("scenario_id").agg({
        "welfare": ["mean", "std"],
        "toxicity_rate": ["mean", "std"],
        "quality_gap": ["mean", "std"],
    }).reset_index()
    
    lines = ["| Scenario | Welfare (mean±std) | Toxicity (mean±std) | Quality Gap (mean±std) |",
             "|----------|-------------------|--------------------|-----------------------|"]
    
    for _, row in summary.iterrows():
        lines.append(
            f"| {row[('scenario_id', '')]} | "
            f"{row[('welfare', 'mean')]:.3f}±{row[('welfare', 'std')]:.3f} | "
            f"{row[('toxicity_rate', 'mean')]:.3f}±{row[('toxicity_rate', 'std')]:.3f} | "
            f"{row[('quality_gap', 'mean')]:.3f}±{row[('quality_gap', 'std')]:.3f} |"
        )
    
    return "\n".join(lines)
```

### 4. Scaffold the paper

```python
def scaffold_paper(title, methods_table, results_table, output_path):
    """Generate the full paper markdown."""
    paper = f"""# {title}

## Abstract

This paper presents an empirical study of distributional safety in multi-agent AI systems using the SWARM framework. We evaluate {n_scenarios} scenarios across multiple seeds, measuring welfare, toxicity, and quality gap under varying governance configurations.

## Experimental Setup

### Scenarios

{methods_table}

### Metrics

| Metric | Definition | Range |
|--------|-----------|-------|
| Welfare | Aggregate agent payoffs | (-∞, +∞) |
| Toxicity Rate | E[1-p \\| accepted] | [0, 1] |
| Quality Gap | E[p\\|accepted] - E[p\\|rejected] | [-1, 1] |

## Results

### Cross-Scenario Summary

{results_table}

## Conclusion

The experimental results demonstrate the relationship between governance configurations and distributional safety outcomes across the tested scenarios.
"""
    
    with open(output_path, "w") as f:
        f.write(paper)
```

## Paper Structure Requirements

A valid SWARM paper must contain these sections:
1. **Abstract** — Key numbers and narrative summary
2. **Experimental Setup** — Scenarios table, metrics definitions
3. **Results** — Cross-scenario summary table with numeric values
4. **Conclusion** — Non-empty synthesis of findings

## Numeric Formatting

- Rates (toxicity, quality gap): 3 decimal places (e.g., 0.123)
- Welfare: 1 decimal place (e.g., 12.3)
- p-values: 4 decimal places (e.g., 0.0012)

## Template

See `references/paper-template.md` for a blank paper skeleton.
