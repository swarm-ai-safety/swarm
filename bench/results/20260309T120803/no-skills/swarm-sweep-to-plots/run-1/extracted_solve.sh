#!/bin/bash

# Create output directory
mkdir -p /root/output/plots

# Create Python script to generate plots
cat > /root/generate_plots.py << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the data
df = pd.read_csv('/root/data/sweep_results.csv')

# Create output directory
os.makedirs('/root/output/plots', exist_ok=True)

# Plot 1: Grouped Bar Chart of Welfare by Tax Rate
plt.figure(figsize=(12, 8))
welfare_cols = [col for col in df.columns if 'welfare' in col.lower()]
if not welfare_cols:
    # If no explicit welfare columns, look for agent payoffs or similar metrics
    welfare_cols = [col for col in df.columns if any(term in col.lower() for term in ['payoff', 'utility', 'profit', 'outcome'])]

# If we have tax rate and welfare/payoff data
if 'tax_rate' in df.columns and welfare_cols:
    tax_rates = df['tax_rate'].unique()
    x = np.arange(len(tax_rates))
    width = 0.8 / len(welfare_cols)
    
    for i, col in enumerate(welfare_cols):
        values = [df[df['tax_rate'] == rate][col].mean() for rate in tax_rates]
        plt.bar(x + i * width, values, width, label=col, alpha=0.8)
    
    plt.xlabel('Tax Rate')
    plt.ylabel('Welfare')
    plt.title('Welfare by Tax Rate')
    plt.xticks(x + width * (len(welfare_cols) - 1) / 2, tax_rates)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/output/plots/welfare_by_tax_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    # Fallback: plot first few numeric columns grouped by first categorical column
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
    if len(numeric_cols) > 0:
        first_cat_col = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else numeric_cols[0]
        groups = df[first_cat_col].unique()[:5]  # Limit to 5 groups
        
        x = np.arange(len(groups))
        width = 0.8 / len(numeric_cols)
        
        for i, col in enumerate(numeric_cols):
            values = [df[df[first_cat_col] == group][col].mean() for group in groups]
            plt.bar(x + i * width, values, width, label=col, alpha=0.8)
        
        plt.xlabel(first_cat_col)
        plt.ylabel('Values')
        plt.title('Grouped Bar Chart')
        plt.xticks(x + width * (len(numeric_cols) - 1) / 2, groups)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('/root/output/plots/welfare_by_tax_rate.png', dpi=150, bbox_inches='tight')
        plt.close()

# Plot 2: Box Plot of Welfare Distribution
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    # Create box plots for numeric columns
    data_to_plot = [df[col].dropna() for col in numeric_cols[:4]]  # Limit to 4 columns
    labels = numeric_cols[:4]
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.title('Distribution of Key Metrics')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/output/plots/welfare_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

# Plot 3: Agent Payoff Comparison Chart
plt.figure(figsize=(12, 8))
# Look for agent-related columns
agent_cols = [col for col in df.columns if any(term in col.lower() for term in ['agent', 'player', 'payoff', 'utility'])]
if not agent_cols:
    agent_cols = numeric_cols[:3]

if len(agent_cols) >= 2:
    # Scatter plot comparing agents
    plt.scatter(df[agent_cols[0]], df[agent_cols[1]], alpha=0.7, s=100)
    
    # Add trend line
    z = np.polyfit(df[agent_cols[0]].dropna(), df[agent_cols[1]].dropna(), 1)
    p = np.poly1d(z)
    plt.plot(df[agent_cols[0]], p(df[agent_cols[0]]), "r--", alpha=0.8)
    
    plt.xlabel(agent_cols[0])
    plt.ylabel(agent_cols[1])
    plt.title('Agent Payoff Comparison')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/output/plots/agent_payoff_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    # Fallback: line plot of all numeric columns
    for col in numeric_cols[:4]:
        plt.plot(df.index, df[col], marker='o', label=col, alpha=0.8, linewidth=2)
    
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Agent Metrics Over Time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/output/plots/agent_payoff_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

# Additional plot for robustness: Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im)
plt.title('Correlation Matrix of Numeric Variables')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.tight_layout()
plt.savefig('/root/output/plots/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("All plots generated successfully!")
EOF

# Run the Python script
python3 /root/generate_plots.py