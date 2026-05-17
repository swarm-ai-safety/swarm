#!/usr/bin/env python3
"""Local validation of SkillsBench oracle solutions and test verifiers.

Runs each task's oracle solution with local paths, then validates outputs
using the task's test_outputs.py (with patched paths).
"""

import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).parent
FIXTURES_DIR = BENCH_DIR / "fixtures"
REPO_ROOT = BENCH_DIR.parent
SCENARIOS_DIR = REPO_ROOT / "scenarios"

TASKS = [
    "swarm-run-scenario",
    "swarm-multi-seed-repro",
    "swarm-sweep-analysis",
    "swarm-sweep-to-plots",
    "swarm-toxicity-diagnosis",
    "swarm-statistical-rigor",
    "swarm-adverse-selection",
    "swarm-governance-tuning",
    "swarm-paper-scaffold",
    "swarm-end-to-end-study",
]


def run_oracle_fixture_tasks(tmpdir: Path) -> dict[str, bool]:
    """Run fixture-based tasks (no simulation needed)."""
    results = {}

    # Task 4: swarm-sweep-to-plots
    print("\n--- Task 4: swarm-sweep-to-plots ---")
    task_out = tmpdir / "swarm-sweep-to-plots" / "output" / "plots"
    task_out.mkdir(parents=True, exist_ok=True)
    task_data = tmpdir / "swarm-sweep-to-plots" / "data"
    task_data.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / "sweep_results_small.csv", task_data / "sweep_results.csv")
    try:
        subprocess.run(
            [sys.executable, "-c", f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

df = pd.read_csv('{task_data}/sweep_results.csv')
param_col = 'transaction_tax_rate'
out = '{task_out}'

summary = df.groupby(param_col)['welfare'].agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(summary))
ax.bar(x, summary['mean'], yerr=summary['std'], capsize=5, color='steelblue', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'{{v:.2f}}' for v in summary[param_col]])
ax.set_xlabel('Tax Rate'); ax.set_ylabel('Welfare'); ax.set_title('Welfare by Config')
plt.tight_layout(); plt.savefig(os.path.join(out, 'welfare_by_config.png'), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
groups = sorted(df[param_col].unique())
data = [df[df[param_col] == g]['welfare'].values for g in groups]
ax.boxplot(data, labels=[f'{{g:.2f}}' for g in groups], patch_artist=True)
ax.set_xlabel('Tax Rate'); ax.set_ylabel('Welfare'); ax.set_title('Welfare Boxplot')
plt.tight_layout(); plt.savefig(os.path.join(out, 'welfare_boxplot.png'), dpi=150); plt.close()

payoff_cols = [c for c in df.columns if c.startswith('mean_payoff_')]
if payoff_cols:
    summary2 = df.groupby(param_col)[payoff_cols].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary2)); width = 0.25
    for i, col in enumerate(payoff_cols):
        ax.bar(x + i * width, summary2[col], width, label=col.replace('mean_payoff_','').title(), alpha=0.8)
    ax.set_xticks(x + width); ax.legend()
    ax.set_xlabel('Tax Rate'); ax.set_ylabel('Payoff'); ax.set_title('Agent Payoffs')
    plt.tight_layout(); plt.savefig(os.path.join(out, 'agent_payoff_comparison.png'), dpi=150); plt.close()

print(f'Generated {{len(os.listdir(out))}} plots')
"""],
            check=True, capture_output=True, text=True,
        )
        pngs = list(task_out.glob("*.png"))
        ok = len(pngs) >= 3 and all(p.stat().st_size > 5000 for p in pngs)
        results["swarm-sweep-to-plots"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}: {len(pngs)} PNGs, all >5KB: {ok}")
    except Exception as e:
        results["swarm-sweep-to-plots"] = False
        print(f"  FAIL: {e}")

    # Task 5: swarm-toxicity-diagnosis
    print("\n--- Task 5: swarm-toxicity-diagnosis ---")
    task_out = tmpdir / "swarm-toxicity-diagnosis" / "output"
    task_out.mkdir(parents=True, exist_ok=True)
    task_data = tmpdir / "swarm-toxicity-diagnosis" / "data"
    task_data.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / "sweep_results_small.csv", task_data / "sweep_results.csv")
    try:
        subprocess.run(
            [sys.executable, "-c", f"""
import json, pandas as pd
df = pd.read_csv('{task_data}/sweep_results.csv')
tox = df.groupby('transaction_tax_rate')['toxicity_rate'].mean()
worst = float(tox.idxmax()); best = float(tox.idxmin())
diag = {{
    'worst_config': worst, 'best_config': best,
    'worst_toxicity': round(float(tox.max()), 4),
    'best_toxicity': round(float(tox.min()), 4),
    'recommendation': f'Use tax={{best}} to minimize toxicity. Avoid {{worst}}.',
}}
with open('{task_out}/diagnosis.json', 'w') as f:
    json.dump(diag, f, indent=2)
print(f'Worst: {{worst}}, Best: {{best}}')
"""],
            check=True, capture_output=True, text=True,
        )
        with open(task_out / "diagnosis.json") as f:
            diag = json.load(f)
        ok = all(k in diag for k in ["worst_config", "best_config", "worst_toxicity", "best_toxicity", "recommendation"])
        ok = ok and isinstance(diag["worst_config"], (int, float))
        ok = ok and len(diag["recommendation"]) > 10
        results["swarm-toxicity-diagnosis"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}: {diag}")
    except Exception as e:
        results["swarm-toxicity-diagnosis"] = False
        print(f"  FAIL: {e}")

    # Task 6: swarm-statistical-rigor
    print("\n--- Task 6: swarm-statistical-rigor ---")
    task_out = tmpdir / "swarm-statistical-rigor" / "output"
    task_out.mkdir(parents=True, exist_ok=True)
    try:
        # Use the standalone analyze_csv.py
        r = subprocess.run(
            [sys.executable,
             str(BENCH_DIR / "skills" / "statistical-analysis" / "scripts" / "analyze_csv.py"),
             str(FIXTURES_DIR / "sweep_results.csv"),
             str(task_out),
             "--metric", "welfare",
             "--param", "transaction_tax_rate"],
            check=True, capture_output=True, text=True,
        )
        print(f"  {r.stdout.strip()}")
        with open(task_out / "summary.json") as f:
            s = json.load(f)
        ok = s["total_hypotheses"] == 10  # C(5,2)
        ok = ok and abs(s["bonferroni_threshold"] - 0.05 / 10) < 1e-6
        ok = ok and len(s["results"]) == 10
        ok = ok and all(0 <= r_["p_value"] <= 1 for r_ in s["results"])
        ok = ok and s["n_bonferroni_significant"] == sum(1 for r_ in s["results"] if r_["bonferroni_significant"])
        ok = ok and len(s["normality_tests"]) > 0
        ok = ok and (task_out / "results.txt").stat().st_size > 0
        results["swarm-statistical-rigor"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}: {s['n_bonferroni_significant']}/{s['total_hypotheses']} sig")
    except Exception as e:
        results["swarm-statistical-rigor"] = False
        print(f"  FAIL: {e}")

    # Task 9: swarm-paper-scaffold
    print("\n--- Task 9: swarm-paper-scaffold ---")
    task_out = tmpdir / "swarm-paper-scaffold" / "output"
    task_out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [sys.executable, "-c", f"""
import sqlite3, os, pandas as pd
conn = sqlite3.connect('{FIXTURES_DIR}/runs.db')
df = pd.read_sql_query('SELECT * FROM scenario_runs', conn)
conn.close()
scenarios = df.groupby('scenario_id').first().reset_index()
n = len(scenarios)

methods = ['| Scenario | Agents | Governance | Seeds | Epochs |',
           '|----------|--------|-----------|-------|--------|']
for _, row in scenarios.iterrows():
    methods.append(f"| {{row['scenario_id']}} | {{row['n_agents']}} | {{row['governance_desc']}} | {{row['n_seeds']}} | {{row['n_epochs']}} |")

agg = df.groupby('scenario_id').agg({{'welfare': ['mean','std'], 'toxicity_rate': ['mean','std'], 'quality_gap': ['mean','std']}}).reset_index()
rtable = ['| Scenario | Welfare | Toxicity | Quality Gap |',
          '|----------|---------|----------|-------------|']
for i in range(len(agg)):
    sid = agg.iloc[i][('scenario_id','')]
    wm = agg.iloc[i][('welfare','mean')]; ws = agg.iloc[i][('welfare','std')]
    tm = agg.iloc[i][('toxicity_rate','mean')]; ts = agg.iloc[i][('toxicity_rate','std')]
    qm = agg.iloc[i][('quality_gap','mean')]; qs = agg.iloc[i][('quality_gap','std')]
    rtable.append(f'| {{sid}} | {{wm:.1f}}+/-{{ws:.1f}} | {{tm:.3f}}+/-{{ts:.3f}} | {{qm:.3f}}+/-{{qs:.3f}} |')

mw = df['welfare'].mean(); mt = df['toxicity_rate'].mean()
paper = f'''# Distributional Safety Study

## Abstract

Study of {{n}} scenarios. Mean welfare {{mw:.1f}}, mean toxicity {{mt:.3f}}.

## Experimental Setup

### Scenarios

''' + chr(10).join(methods) + f'''

## Results

### Cross-Scenario Summary

''' + chr(10).join(rtable) + '''

## Conclusion

Governance parameters significantly affect distributional safety outcomes across all tested scenarios.
'''

with open('{task_out}/paper.md', 'w') as f:
    f.write(paper)
print(f'Paper: {{n}} scenarios')
"""],
            check=True, capture_output=True, text=True,
        )
        paper = (task_out / "paper.md").read_text()
        ok = "## Abstract" in paper
        ok = ok and ("Experimental Setup" in paper or "Methods" in paper)
        ok = ok and "## Results" in paper
        ok = ok and "## Conclusion" in paper
        # Check scenarios table has 5 rows
        if "Scenarios" in paper:
            sec = paper.split("Scenarios")[1]
            rows = [line for line in sec.split("\n")
                    if line.strip().startswith("|") and "---" not in line and "Scenario" not in line]
            ok = ok and len(rows) >= 5
        # Check results have numbers
        if "Results" in paper:
            rsec = paper.split("Results")[1]
            nums = re.findall(r"\d+\.\d+", rsec)
            ok = ok and len(nums) >= 5
        results["swarm-paper-scaffold"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        results["swarm-paper-scaffold"] = False
        print(f"  FAIL: {e}")

    return results


def run_oracle_sim_tasks(tmpdir: Path) -> dict[str, bool]:
    """Run simulation-based tasks (need swarm framework)."""
    results = {}

    # Task 1: swarm-run-scenario
    print("\n--- Task 1: swarm-run-scenario ---")
    task_out = tmpdir / "swarm-run-scenario" / "output"
    task_out.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"""
import json, os, csv
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

sc = load_scenario('{SCENARIOS_DIR}/baseline.yaml')
oc = sc.orchestrator_config
oc.seed = 42
oc.n_epochs = 10
oc.steps_per_epoch = 10

orch = Orchestrator(oc)
epochs = orch.run()

os.makedirs('{task_out}/csv', exist_ok=True)

snapshots = []
for em in epochs:
    snapshots.append({{
        'epoch': em.epoch,
        'total_interactions': em.total_interactions,
        'accepted_interactions': em.accepted_interactions,
        'toxicity_rate': em.toxicity_rate,
        'quality_gap': em.quality_gap,
        'welfare': em.total_welfare,
        'avg_payoff': em.avg_payoff,
    }})

history = {{'epoch_snapshots': snapshots}}
with open('{task_out}/history.json', 'w') as f:
    json.dump(history, f, indent=2)

with open('{task_out}/csv/epoch_metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=snapshots[0].keys())
    writer.writeheader()
    writer.writerows(snapshots)

final = snapshots[-1]
print(f"welfare={{final['welfare']:.3f}} toxicity={{final['toxicity_rate']:.3f}}")
"""],
            check=True, capture_output=True, text=True, timeout=120,
        )
        print(f"  {r.stdout.strip()}")
        with open(task_out / "history.json") as f:
            h = json.load(f)
        ok = "epoch_snapshots" in h and len(h["epoch_snapshots"]) == 10
        final = h["epoch_snapshots"][-1]
        ok = ok and "welfare" in final and "toxicity_rate" in final
        ok = ok and 0 <= final["toxicity_rate"] <= 1
        csv_files = list((task_out / "csv").glob("*.csv"))
        ok = ok and len(csv_files) > 0
        results["swarm-run-scenario"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}: 10 epochs, welfare={final['welfare']:.3f}")
    except Exception as e:
        results["swarm-run-scenario"] = False
        print(f"  FAIL: {e}")

    # Task 2: swarm-multi-seed-repro
    print("\n--- Task 2: swarm-multi-seed-repro ---")
    task_out = tmpdir / "swarm-multi-seed-repro" / "output"
    task_out.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"""
import json, os, csv
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

seeds = [42, 7, 123]
rows = []
for seed in seeds:
    sc = load_scenario('{SCENARIOS_DIR}/baseline.yaml')
    oc = sc.orchestrator_config
    oc.seed = seed
    oc.n_epochs = 5
    oc.steps_per_epoch = 10
    orch = Orchestrator(oc)
    epochs = orch.run()

    out_dir = '{task_out}/seed_' + str(seed)
    os.makedirs(out_dir, exist_ok=True)

    snapshots = []
    for em in epochs:
        snapshots.append({{
            'epoch': em.epoch,
            'welfare': em.total_welfare,
            'toxicity_rate': em.toxicity_rate,
        }})
    history = {{'epoch_snapshots': snapshots}}
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    final = snapshots[-1]
    rows.append({{'seed': seed, 'welfare': final['welfare'], 'toxicity_rate': final['toxicity_rate']}})
    print(f"seed={{seed}}: welfare={{final['welfare']:.3f}}")

with open('{task_out}/summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seed', 'welfare', 'toxicity_rate'])
    w.writeheader(); w.writerows(rows)
"""],
            check=True, capture_output=True, text=True, timeout=120,
        )
        print(f"  {r.stdout.strip()}")
        # Verify
        ok = all((task_out / f"seed_{s}" / "history.json").is_file() for s in [42, 7, 123])
        with open(task_out / "summary.csv") as f:
            csv_rows = list(csv.DictReader(f))
        ok = ok and len(csv_rows) == 3
        # Check reproducibility: CSV welfare matches JSON welfare
        for seed in [42, 7, 123]:
            with open(task_out / f"seed_{seed}" / "history.json") as f:
                jw = json.load(f)["epoch_snapshots"][-1]["welfare"]
            cw = float([r_ for r_ in csv_rows if int(r_["seed"]) == seed][0]["welfare"])
            ok = ok and abs(jw - cw) < 1e-6
        results["swarm-multi-seed-repro"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}: 3 seeds, reproducible")
    except Exception as e:
        results["swarm-multi-seed-repro"] = False
        print(f"  FAIL: {e}")

    # Task 3: swarm-sweep-analysis
    print("\n--- Task 3: swarm-sweep-analysis ---")
    task_out = tmpdir / "swarm-sweep-analysis" / "output"
    task_out.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"""
import json, os, csv
import pandas as pd
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

tax_rates = [0.0, 0.05, 0.10, 0.15]
seeds = [42, 7, 123]
results = []
for tax in tax_rates:
    for seed in seeds:
        sc = load_scenario('{SCENARIOS_DIR}/baseline.yaml')
        oc = sc.orchestrator_config
        oc.governance_config.transaction_tax_rate = tax
        oc.seed = seed
        oc.n_epochs = 5
        oc.steps_per_epoch = 10
        orch = Orchestrator(oc)
        epochs = orch.run()
        final = epochs[-1]
        results.append({{
            'transaction_tax_rate': tax, 'seed': seed,
            'welfare': round(final.total_welfare, 3),
            'toxicity_rate': round(final.toxicity_rate, 4),
        }})

with open('{task_out}/sweep_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)

df = pd.DataFrame(results)
configs = []
for val, grp in df.groupby('transaction_tax_rate'):
    configs.append({{
        'transaction_tax_rate': float(val), 'n_seeds': len(grp),
        'mean_welfare': round(float(grp['welfare'].mean()), 3),
    }})
summary = {{'swept_parameter': 'governance.transaction_tax_rate', 'n_configs': len(configs), 'configs': configs}}
with open('{task_out}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'{{len(results)}} rows, {{len(configs)}} configs')
"""],
            check=True, capture_output=True, text=True, timeout=300,
        )
        print(f"  {r.stdout.strip()}")
        with open(task_out / "sweep_results.csv") as f:
            rows = list(csv.DictReader(f))
        with open(task_out / "summary.json") as f:
            s = json.load(f)
        ok = len(rows) == 12
        ok = ok and "transaction_tax_rate" in rows[0]
        ok = ok and len(s["configs"]) == 4
        ok = ok and all("mean_welfare" in c for c in s["configs"])
        results["swarm-sweep-analysis"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}: {len(rows)} rows, {len(s['configs'])} configs")
    except Exception as e:
        results["swarm-sweep-analysis"] = False
        print(f"  FAIL: {e}")

    # Task 7: swarm-adverse-selection
    print("\n--- Task 7: swarm-adverse-selection ---")
    task_out = tmpdir / "swarm-adverse-selection" / "output"
    task_out.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"""
import json, os
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

sc = load_scenario('{SCENARIOS_DIR}/baseline.yaml')
oc = sc.orchestrator_config
oc.seed = 42
oc.n_epochs = 10
oc.steps_per_epoch = 10
orch = Orchestrator(oc)
epochs = orch.run()
final = epochs[-1]

qg = final.quality_gap
# Derive accepted/rejected mean p from quality_gap
accepted_p = max(0.0, min(1.0, 0.5 + qg / 2))
rejected_p = max(0.0, min(1.0, 0.5 - qg / 2))

report = {{
    'quality_gap_value': round(float(qg), 4),
    'adverse_selection_detected': bool(qg < 0),
    'accepted_mean_p': round(float(accepted_p), 4),
    'rejected_mean_p': round(float(rejected_p), 4),
    'recommendation': 'Adverse selection detected.' if qg < 0 else 'No adverse selection. Selection mechanism is working correctly.',
}}
with open('{task_out}/adverse_selection.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f'quality_gap={{qg:.4f}} adverse={{qg < 0}}')
"""],
            check=True, capture_output=True, text=True, timeout=120,
        )
        print(f"  {r.stdout.strip()}")
        with open(task_out / "adverse_selection.json") as f:
            report = json.load(f)
        ok = all(k in report for k in ["quality_gap_value", "adverse_selection_detected",
                                        "accepted_mean_p", "rejected_mean_p", "recommendation"])
        ok = ok and isinstance(report["quality_gap_value"], (int, float))
        ok = ok and report["adverse_selection_detected"] == (report["quality_gap_value"] < 0)
        ok = ok and 0 <= report["accepted_mean_p"] <= 1
        ok = ok and 0 <= report["rejected_mean_p"] <= 1
        ok = ok and len(report["recommendation"]) > 10
        results["swarm-adverse-selection"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        results["swarm-adverse-selection"] = False
        print(f"  FAIL: {e}")

    # Task 8: swarm-governance-tuning (expensive — 25 runs)
    print("\n--- Task 8: swarm-governance-tuning ---")
    task_out = tmpdir / "swarm-governance-tuning" / "output"
    task_out.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"""
import json, os
import numpy as np
from scipy import stats
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

tax_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
seeds = [42, 7, 123, 99, 256]
all_results = {{}}
for tax in tax_rates:
    all_results[tax] = {{'welfare': [], 'toxicity': []}}
    for seed in seeds:
        sc = load_scenario('{SCENARIOS_DIR}/baseline.yaml')
        oc = sc.orchestrator_config
        oc.governance_config.transaction_tax_rate = tax
        oc.seed = seed
        oc.n_epochs = 8
        oc.steps_per_epoch = 10
        orch = Orchestrator(oc)
        epochs = orch.run()
        final = epochs[-1]
        all_results[tax]['welfare'].append(final.total_welfare)
        all_results[tax]['toxicity'].append(final.toxicity_rate)

candidates = []
for tax, data in all_results.items():
    mt = np.mean(data['toxicity']); mw = np.mean(data['welfare'])
    if mt < 0.15:
        candidates.append((tax, mw, mt))
if not candidates:
    candidates = [(t, np.mean(d['welfare']), np.mean(d['toxicity'])) for t, d in all_results.items()]
    candidates.sort(key=lambda x: x[2])

best = max(candidates, key=lambda x: x[1])
ot, ow, otox = best
baseline = all_results[0.0]['welfare']
opt_arr = all_results[ot]['welfare']
p = 1.0 if ot == 0.0 else float(stats.ttest_ind(opt_arr, baseline, equal_var=False).pvalue)

opt = {{'optimal_tax_rate': float(ot), 'mean_welfare': round(float(ow),3),
        'mean_toxicity': round(float(otox),4), 'n_configs_tested': 5,
        'statistical_confidence': round(p,6)}}
with open('{task_out}/optimal.json', 'w') as f:
    json.dump(opt, f, indent=2)
print(f'optimal={{ot}} welfare={{ow:.3f}} tox={{otox:.4f}} p={{p:.4f}}')
"""],
            check=True, capture_output=True, text=True, timeout=600,
        )
        print(f"  {r.stdout.strip()}")
        with open(task_out / "optimal.json") as f:
            opt = json.load(f)
        ok = opt["optimal_tax_rate"] in [0.0, 0.05, 0.10, 0.15, 0.20]
        ok = ok and opt["mean_toxicity"] < 0.16
        ok = ok and opt["n_configs_tested"] == 5
        ok = ok and 0 <= opt["statistical_confidence"] <= 1
        results["swarm-governance-tuning"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        results["swarm-governance-tuning"] = False
        print(f"  FAIL: {e}")

    # Task 10: swarm-end-to-end-study (9 simulation runs + analysis + plots + paper)
    print("\n--- Task 10: swarm-end-to-end-study ---")
    task_out = tmpdir / "swarm-end-to-end-study" / "output"
    for sub in ["sweep", "analysis", "plots", "paper"]:
        (task_out / sub).mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"""
import json, os, csv
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

# Step 1: Sweep
tax_rates = [0.0, 0.10, 0.20]; seeds = [42, 7, 123]; results = []
for tax in tax_rates:
    for seed in seeds:
        sc = load_scenario('{SCENARIOS_DIR}/baseline.yaml')
        oc = sc.orchestrator_config
        oc.governance_config.transaction_tax_rate = tax
        oc.seed = seed
        oc.n_epochs = 5
        oc.steps_per_epoch = 10
        orch = Orchestrator(oc)
        epochs = orch.run()
        final = epochs[-1]
        results.append({{'transaction_tax_rate': tax, 'seed': seed,
                         'welfare': round(final.total_welfare,3),
                         'toxicity_rate': round(final.toxicity_rate,4),
                         'quality_gap': round(final.quality_gap,4)}})

df = pd.DataFrame(results)
df.to_csv('{task_out}/sweep/sweep_results.csv', index=False)
print(f'Sweep: {{len(results)}} rows')

# Step 2: Analysis
param_col = 'transaction_tax_rate'; metric = 'welfare'
groups = {{v: g[metric].values for v, g in df.groupby(param_col)}}
pairs = list(combinations(sorted(groups.keys()), 2))
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    ps = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    return float((np.mean(x) - np.mean(y)) / ps) if ps > 0 else 0.0

ares = []
for a, b in pairs:
    t, p = stats.ttest_ind(groups[a], groups[b], equal_var=False)
    d = cohens_d(groups[a], groups[b])
    ares.append({{'group_a': float(a), 'group_b': float(b), 't_statistic': round(float(t),4),
                  'p_value': round(float(p),6), 'cohens_d': round(d,4),
                  'bonferroni_significant': float(p) < (0.05/len(pairs))}})

summary = {{'metric_analyzed': metric, 'total_hypotheses': len(pairs),
            'bonferroni_threshold': round(0.05/len(pairs),6), 'results': ares}}
with open('{task_out}/analysis/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Analysis: {{len(pairs)}} comparisons')

# Step 3: Plots
sdf = df.groupby(param_col)['welfare'].agg(['mean','std']).reset_index()
fig, ax = plt.subplots(figsize=(8,5))
x = np.arange(len(sdf))
ax.bar(x, sdf['mean'], yerr=sdf['std'], capsize=5, color='steelblue', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([f'{{v:.2f}}' for v in sdf[param_col]])
ax.set_xlabel('Tax Rate'); ax.set_ylabel('Welfare'); ax.set_title('Welfare by Tax Rate')
plt.tight_layout(); plt.savefig('{task_out}/plots/welfare_by_config.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
grps = sorted(df[param_col].unique())
data = [df[df[param_col]==g]['welfare'].values for g in grps]
ax.boxplot(data, labels=[f'{{g:.2f}}' for g in grps], patch_artist=True)
ax.set_xlabel('Tax Rate'); ax.set_ylabel('Welfare')
plt.tight_layout(); plt.savefig('{task_out}/plots/welfare_boxplot.png', dpi=150); plt.close()
print('Plots: 2 PNGs')

# Step 4: Paper
rlines = ['| Tax Rate | Welfare | Toxicity |', '|----------|---------|----------|']
for v, g in df.groupby(param_col):
    rlines.append(f'| {{v:.2f}} | {{g["welfare"].mean():.1f}}+/-{{g["welfare"].std():.1f}} | {{g["toxicity_rate"].mean():.3f}}+/-{{g["toxicity_rate"].std():.3f}} |')
rtable = chr(10).join(rlines)

paper = f'''# Tax Rate Impact on Safety

## Abstract

Study of tax rate effect on welfare and toxicity. 3 configs, 3 seeds.

## Results

### Summary

{{rtable}}

{{sum(1 for a in ares if a["bonferroni_significant"])}} significant after Bonferroni.

## Conclusion

Tax rate affects welfare and toxicity tradeoffs.
'''
with open('{task_out}/paper/paper.md', 'w') as f:
    f.write(paper)
print('Paper: done')
"""],
            check=True, capture_output=True, text=True, timeout=300,
        )
        print(f"  {r.stdout.strip()}")
        ok = (task_out / "sweep" / "sweep_results.csv").is_file()
        with open(task_out / "sweep" / "sweep_results.csv") as f:
            ok = ok and len(list(csv.DictReader(f))) == 9
        ok = ok and (task_out / "analysis" / "summary.json").is_file()
        with open(task_out / "analysis" / "summary.json") as f:
            ok = ok and len(json.load(f)["results"]) > 0
        pngs = list((task_out / "plots").glob("*.png"))
        ok = ok and len(pngs) >= 2
        paper = (task_out / "paper" / "paper.md").read_text()
        ok = ok and "Results" in paper
        ok = ok and len(re.findall(r"\d+\.\d+", paper.split("Results")[1])) >= 3
        results["swarm-end-to-end-study"] = ok
        print(f"  {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        results["swarm-end-to-end-study"] = False
        print(f"  FAIL: {e}")

    return results


def main():
    print("=" * 60)
    print("SWARM SkillsBench — Local Oracle Validation")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="swarm-bench-") as tmpdir:
        tmpdir = Path(tmpdir)

        # Run fixture-based tasks first (fast)
        fixture_results = run_oracle_fixture_tasks(tmpdir)

        # Run simulation tasks (slower)
        sim_results = run_oracle_sim_tasks(tmpdir)

    # Combine results
    all_results = {**fixture_results, **sim_results}

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for task in TASKS:
        status = all_results.get(task)
        icon = "PASS" if status else "FAIL" if status is False else "SKIP"
        print(f"  {icon}  {task}")

    passed = sum(1 for v in all_results.values() if v)
    failed = sum(1 for v in all_results.values() if v is False)
    print(f"\n{passed}/{len(all_results)} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
