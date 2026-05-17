#!/usr/bin/env python3
"""Generate pre-loaded fixture data for SWARM SkillsBench tasks.

This creates deterministic test data that medium/hard tasks use as input.
Run once to populate fixtures/, then commit the outputs.
"""

import csv
import os
import random
import sqlite3

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_sweep_csv_small(path: str) -> None:
    """Generate a 12-row sweep CSV (4 tax rates x 3 seeds) for medium tasks."""
    random.seed(42)
    tax_rates = [0.0, 0.05, 0.10, 0.15]
    seeds = [42, 7, 123]

    rows = []
    for tax in tax_rates:
        for seed in seeds:
            random.seed(seed + int(tax * 1000))
            # Higher tax -> lower welfare but lower toxicity
            base_welfare = 12.0 - tax * 30.0 + random.gauss(0, 1.5)
            base_toxicity = 0.25 - tax * 0.8 + random.gauss(0, 0.05)
            base_toxicity = max(0.01, min(0.99, base_toxicity))
            quality_gap = 0.1 + tax * 0.5 + random.gauss(0, 0.05)

            rows.append({
                "transaction_tax_rate": tax,
                "seed": seed,
                "welfare": round(base_welfare, 3),
                "toxicity_rate": round(base_toxicity, 4),
                "quality_gap": round(quality_gap, 4),
                "mean_payoff_honest": round(3.0 + random.gauss(0, 0.3), 3),
                "mean_payoff_opportunistic": round(2.5 - tax * 5 + random.gauss(0, 0.3), 3),
                "mean_payoff_deceptive": round(1.5 - tax * 8 + random.gauss(0, 0.3), 3),
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows)} rows to {path}")


def generate_sweep_csv_large(path: str) -> None:
    """Generate an 80-row sweep CSV (5 tax rates x 4 seeds x 4 metrics) for hard tasks."""
    random.seed(42)
    tax_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    seeds = [42, 7, 123, 99]

    rows = []
    for tax in tax_rates:
        for seed in seeds:
            random.seed(seed + int(tax * 1000))
            base_welfare = 12.0 - tax * 25.0 + random.gauss(0, 1.5)
            base_toxicity = 0.30 - tax * 0.9 + random.gauss(0, 0.04)
            base_toxicity = max(0.01, min(0.99, base_toxicity))
            quality_gap = 0.08 + tax * 0.6 + random.gauss(0, 0.04)

            # Generate 4 "agent type" sub-rows per config
            for agent_type in ["honest", "opportunistic", "deceptive", "mixed"]:
                type_offset = {"honest": 0.0, "opportunistic": -0.5, "deceptive": -1.0, "mixed": -0.2}
                rows.append({
                    "transaction_tax_rate": tax,
                    "seed": seed,
                    "agent_type": agent_type,
                    "welfare": round(base_welfare + type_offset[agent_type] + random.gauss(0, 0.5), 3),
                    "toxicity_rate": round(base_toxicity - type_offset[agent_type] * 0.1 + random.gauss(0, 0.02), 4),
                    "quality_gap": round(quality_gap + random.gauss(0, 0.03), 4),
                    "mean_payoff_honest": round(3.0 + random.gauss(0, 0.3), 3),
                    "mean_payoff_opportunistic": round(2.5 - tax * 5 + random.gauss(0, 0.3), 3),
                    "mean_payoff_deceptive": round(1.5 - tax * 8 + random.gauss(0, 0.3), 3),
                })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows)} rows to {path}")


def generate_runs_db(path: str) -> None:
    """Generate a SQLite database with 5 scenarios x 3 seeds for paper-writing task."""
    random.seed(42)
    scenarios = [
        ("baseline", 5, "default", "Baseline mixed-agent scenario"),
        ("high_tax", 5, "tax=0.10", "High transaction tax scenario"),
        ("adversarial", 5, "default", "Adversarial agents scenario"),
        ("strict_gov", 5, "strict", "Strict governance scenario"),
        ("no_gov", 5, "none", "No governance scenario"),
    ]
    seeds = [42, 7, 123]

    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scenario_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id TEXT NOT NULL,
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
        )
    """)

    for scenario_id, n_agents, gov_desc, description in scenarios:
        for seed in seeds:
            random.seed(seed + hash(scenario_id) % 1000)
            welfare = round(10.0 + random.gauss(0, 2.0), 3)
            toxicity = round(max(0.01, min(0.99, 0.2 + random.gauss(0, 0.08))), 4)
            qgap = round(0.1 + random.gauss(0, 0.05), 4)

            conn.execute("""
                INSERT INTO scenario_runs
                (scenario_id, n_agents, governance_desc, description, seed, n_seeds, n_epochs,
                 welfare, toxicity_rate, quality_gap,
                 mean_payoff_honest, mean_payoff_opportunistic, mean_payoff_deceptive,
                 run_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scenario_id, n_agents, gov_desc, description, seed, 3, 10,
                welfare, toxicity, qgap,
                round(3.0 + random.gauss(0, 0.5), 3),
                round(2.0 + random.gauss(0, 0.5), 3),
                round(1.0 + random.gauss(0, 0.5), 3),
                "2026-02-20T12:00:00Z",
            ))

    conn.commit()
    conn.close()
    print(f"  Written 15 rows to {path}")


def main():
    print("Generating SWARM SkillsBench fixtures...")

    print("\n1. Small sweep CSV (12 rows):")
    generate_sweep_csv_small(os.path.join(FIXTURE_DIR, "sweep_results_small.csv"))

    print("\n2. Large sweep CSV (80 rows):")
    generate_sweep_csv_large(os.path.join(FIXTURE_DIR, "sweep_results.csv"))

    print("\n3. Runs database (15 rows):")
    generate_runs_db(os.path.join(FIXTURE_DIR, "runs.db"))

    print("\nDone! Fixtures are ready.")


if __name__ == "__main__":
    main()
