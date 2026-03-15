#!/bin/bash

# Create output directory
mkdir -p /root/output/plots/

# Create Python script to generate plots
cat > /root/generate_plots.py << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the sweep data
df = pd.read_csv('/root/data/sweep_results.csv')

# Create output directory
output_dir = '/root/output/plots/'
os.makedirs(output_dir, exist_ok=True)

# Detect parameter column (exclude standard output columns)
standard_cols = ["seed", "welfare", "toxicity_rate", "quality_gap", 
                "mean_payoff_honest", "mean_payoff_opportunistic", 
                "mean_payoff_deceptive", "epoch"]
param_cols = [c for c in df.columns if c not in standard_cols]
param_col = param_cols[0] if param_cols else 'tax_rate'  # assume tax_rate if not found

# 1. Grouped bar chart of welfare by parameter
def plot_welfare_bars(df, param_col, output_dir):
    summary = df.groupby(param_col)["welfare"].agg(["mean", "std"]).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["mean"], yerr=summary["std"], capsize=5, 
                  color="steelblue", edgecolor="black", alpha=0.8, linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in summary[param_col]], rotation=45)
    ax.set_xlabel(param_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Welfare (mean ± SD)", fontsize=12)
    ax.set_title("Welfare by Tax Rate Configuration", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, summary["mean"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "welfare_by_config.png"), dpi=150, bbox_inches='tight')
    plt.close()

# 2. Box plot of welfare distribution
def plot_welfare_boxplot(df, param_col, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted(df[param_col].unique())
    data = [df[df[param_col] == g]["welfare"].values for g in groups]
    
    bp = ax.boxplot(data, labels=[f"{g:.2f}" for g in groups], patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel(param_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Welfare", fontsize=12)
    ax.set_title("Welfare Distribution by Configuration", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "welfare_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()

# 3. Agent payoff comparison chart
def plot_agent_payoffs(df, param_col, output_dir):
    payoff_cols = [c for c in df.columns if c.startswith("mean_payoff_")]
    if not payoff_cols:
        return
    
    summary = df.groupby(param_col)[payoff_cols].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, col in enumerate(payoff_cols):
        label = col.replace("mean_payoff_", "").title()
        bars = ax.bar(x + i * width, summary[col], width, label=label, 
                     color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{v:.2f}" for v in summary[param_col]], rotation=45)
    ax.set_xlabel(param_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Mean Payoff", fontsize=12)
    ax.set_title("Agent Payoff Comparison by Configuration", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "agent_payoff_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

# 4. Additional plot: Toxicity rate by parameter
def plot_toxicity_by_param(df, param_col, output_dir):
    if 'toxicity_rate' not in df.columns:
        return
        
    summary = df.groupby(param_col)["toxicity_rate"].agg(["mean", "std"]).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["mean"], yerr=summary["std"], capsize=5, 
                  color="coral", edgecolor="darkred", alpha=0.8, linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in summary[param_col]], rotation=45)
    ax.set_xlabel(param_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Toxicity Rate (mean ± SD)", fontsize=12)
    ax.set_title("Toxicity Rate by Configuration", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "toxicity_by_config.png"), dpi=150, bbox_inches='tight')
    plt.close()

# Generate all plots
print(f"Generating plots from data with columns: {list(df.columns)}")
print(f"Using parameter column: {param_col}")
print(f"Data shape: {df.shape}")

plot_welfare_bars(df, param_col, output_dir)
print("Generated welfare bar chart")

plot_welfare_boxplot(df, param_col, output_dir)
print("Generated welfare box plot")

plot_agent_payoffs(df, param_col, output_dir)
print("Generated agent payoff comparison")

plot_toxicity_by_param(df, param_col, output_dir)
print("Generated toxicity rate plot")

print(f"All plots saved to {output_dir}")

# List generated files and their sizes
import os
for file in os.listdir(output_dir):
    if file.endswith('.png'):
        size = os.path.getsize(os.path.join(output_dir, file))
        print(f"{file}: {size} bytes")
EOF

# Run the Python script
cd /root
python generate_plots.py