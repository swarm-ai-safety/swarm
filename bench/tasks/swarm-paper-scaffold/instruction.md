# Task: Scaffold Research Paper

Using the scenario runs database at /root/data/runs.db, scaffold a research paper covering all scenarios. Auto-populate the methods table and results table. Write the paper to /root/output/paper.md.

## Requirements

- Input: `/root/data/runs.db` — SQLite database with table `scenario_runs`
- Output: `/root/output/paper.md`
- The paper must include these sections:
  1. **Abstract** — Summary with key numbers
  2. **Experimental Setup** — With a scenarios table (must have 5 rows, one per scenario)
  3. **Results** — With a cross-scenario summary table containing numeric values
  4. **Conclusion** — Non-empty synthesis of findings
- Tables must contain actual numeric data from the database, not placeholders

## Database Schema

```sql
CREATE TABLE scenario_runs (
    id INTEGER PRIMARY KEY,
    scenario_id TEXT,
    n_agents INTEGER,
    governance_desc TEXT,
    description TEXT,
    seed INTEGER,
    n_seeds INTEGER,
    n_epochs INTEGER,
    welfare REAL,
    toxicity_rate REAL,
    quality_gap REAL,
    mean_payoff_honest REAL,
    mean_payoff_opportunistic REAL,
    mean_payoff_deceptive REAL,
    run_timestamp TEXT
);
```

## Environment

- Python 3.12 with pandas, sqlite3
- Database pre-loaded at `/root/data/runs.db`
