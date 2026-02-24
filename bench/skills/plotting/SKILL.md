---
name: plotting
description: Generate publication-quality visualizations from SWARM simulation data
version: "1.0"
domain: swarm-safety
triggers:
  - generate plots
  - create visualizations
  - plot results
---

# Plotting Skill

Generate standard visualizations from SWARM run data (sweep CSVs or time-series history).

## Prerequisites

- `matplotlib>=3.7`
- `pandas>=2.0`
- `numpy>=1.24`
- Optional: `seaborn>=0.12` for enhanced styling

## Procedure

### 1. Detect data type

```python
import pandas as pd
import os

def detect_data_type(path):
    """Determine if data is sweep results or time-series."""
    if path.endswith(".json"):
        return "timeseries"
    df = pd.read_csv(path)
    # Sweep data has a parameter column with repeated values
    param_cols = [c for c in df.columns if c not in 
                  ["seed", "welfare", "toxicity_rate", "quality_gap",
                   "mean_payoff_honest", "mean_payoff_opportunistic",
                   "mean_payoff_deceptive", "epoch"]]
    if param_cols and df[param_cols[0]].nunique() < len(df):
        return "sweep"
    return "timeseries"
```

### 2. Sweep plots

#### Grouped bar chart (welfare by parameter)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_welfare_bars(df, param_col, output_dir):
    summary = df.groupby(param_col)["welfare"].agg(["mean", "std"]).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["std"], capsize=5, 
           color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in summary[param_col]])
    ax.set_xlabel(param_col.replace("_", " ").title())
    ax.set_ylabel("Welfare (mean ± SD)")
    ax.set_title("Welfare by Governance Configuration")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "welfare_by_config.png"), dpi=150)
    plt.close()
```

#### Box plot (welfare distribution)

```python
def plot_welfare_boxplot(df, param_col, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = sorted(df[param_col].unique())
    data = [df[df[param_col] == g]["welfare"].values for g in groups]
    bp = ax.boxplot(data, labels=[f"{g:.2f}" for g in groups], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_xlabel(param_col.replace("_", " ").title())
    ax.set_ylabel("Welfare")
    ax.set_title("Welfare Distribution by Configuration")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "welfare_boxplot.png"), dpi=150)
    plt.close()
```

#### Agent payoff comparison

```python
def plot_agent_payoffs(df, param_col, output_dir):
    payoff_cols = [c for c in df.columns if c.startswith("mean_payoff_")]
    if not payoff_cols:
        return
    
    summary = df.groupby(param_col)[payoff_cols].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    width = 0.25
    
    for i, col in enumerate(payoff_cols):
        label = col.replace("mean_payoff_", "").title()
        ax.bar(x + i * width, summary[col], width, label=label, alpha=0.8)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{v:.2f}" for v in summary[param_col]])
    ax.set_xlabel(param_col.replace("_", " ").title())
    ax.set_ylabel("Mean Payoff")
    ax.set_title("Agent Payoff Comparison by Configuration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "agent_payoff_comparison.png"), dpi=150)
    plt.close()
```

### 3. Time-series plots

```python
def plot_timeseries(history_path, output_dir):
    import json
    with open(history_path) as f:
        history = json.load(f)
    
    epochs = [snap["epoch"] for snap in history["epoch_snapshots"]]
    welfare = [snap["welfare"] for snap in history["epoch_snapshots"]]
    toxicity = [snap["toxicity_rate"] for snap in history["epoch_snapshots"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, welfare, "b-o")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Welfare"); ax1.set_title("Welfare over Time")
    ax2.plot(epochs, toxicity, "r-o")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Toxicity Rate"); ax2.set_title("Toxicity over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timeseries.png"), dpi=150)
    plt.close()
```

### 4. Standard plot file naming

| Plot Type | Filename |
|-----------|----------|
| Welfare bar chart | `welfare_by_config.png` |
| Welfare box plot | `welfare_boxplot.png` |
| Agent payoffs | `agent_payoff_comparison.png` |
| Time series | `timeseries.png` |
| Toxicity heatmap | `toxicity_heatmap.png` |

All plots should be saved at 150 DPI minimum with `tight_layout()`.
