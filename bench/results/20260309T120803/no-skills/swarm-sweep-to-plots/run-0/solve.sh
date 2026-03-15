#!/bin/bash

# Create output directory
mkdir -p /root/output/plots/

# Create Python script to generate plots
cat > /root/generate_plots.py << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the sweep results
df = pd.read_csv('/root/data/sweep_results.csv')

# Set style for better looking plots
plt.style.use('default')

# Plot 1: Grouped bar chart of welfare by tax rate
plt.figure(figsize=(12, 8))
if 'tax_rate' in df.columns and 'welfare' in df.columns:
    # Group by tax rate and calculate mean welfare
    welfare_by_tax = df.groupby('tax_rate')['welfare'].mean().reset_index()
    plt.bar(welfare_by_tax['tax_rate'], welfare_by_tax['welfare'], 
            color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Tax Rate', fontsize=12)
    plt.ylabel('Average Welfare', fontsize=12)
    plt.title('Average Welfare by Tax Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
else:
    # Fallback: use first two numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        plt.bar(range(len(df)), df[y_col], color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.title(f'{y_col} by Index', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/output/plots/welfare_by_tax_rate.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Box plot of welfare distribution
plt.figure(figsize=(10, 8))
if 'welfare' in df.columns:
    plt.boxplot(df['welfare'], patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Welfare', fontsize=12)
    plt.title('Distribution of Welfare Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
else:
    # Fallback: use first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 1:
        col = numeric_cols[0]
        plt.boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
        plt.ylabel(col, fontsize=12)
        plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/output/plots/welfare_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Agent payoff comparison chart
plt.figure(figsize=(12, 8))
# Look for agent-related columns
agent_cols = [col for col in df.columns if 'agent' in col.lower() or 'payoff' in col.lower() or 'utility' in col.lower()]

if len(agent_cols) >= 2:
    # Create comparison plot with multiple agent columns
    x_pos = np.arange(len(df))
    width = 0.35
    
    for i, col in enumerate(agent_cols[:3]):  # Use up to 3 agent columns
        offset = (i - 1) * width
        plt.bar(x_pos + offset, df[col], width, label=col, alpha=0.7)
    
    plt.xlabel('Scenario Index', fontsize=12)
    plt.ylabel('Payoff/Utility', fontsize=12)
    plt.title('Agent Payoff Comparison Across Scenarios', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
elif len(agent_cols) == 1:
    # Single agent column - create line plot
    plt.plot(df.index, df[agent_cols[0]], marker='o', linewidth=2, markersize=6)
    plt.xlabel('Scenario Index', fontsize=12)
    plt.ylabel(agent_cols[0], fontsize=12)
    plt.title(f'{agent_cols[0]} Across Scenarios', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
else:
    # Fallback: compare multiple numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        x_pos = np.arange(len(df))
        width = 0.35
        
        for i, col in enumerate(numeric_cols[:3]):
            offset = (i - 1) * width
            plt.bar(x_pos + offset, df[col], width, label=col, alpha=0.7)
        
        plt.xlabel('Scenario Index', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.title('Multi-Variable Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/output/plots/agent_payoff_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots generated successfully!")
EOF

# Run the Python script
python3 /root/generate_plots.py