#!/usr/bin/env python
"""
Plot: Cautious Reciprocator governance sweep results.

Reads logs/cautious_governance_sweep.csv and produces a 2x2 panel figure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
csv_path = Path("logs/cautious_governance_sweep.csv")
if not csv_path.exists():
    print("Run the sweep first: python examples/cautious_governance_sweep.py")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Build a human-readable governance label per row
def gov_label(row):
    parts = []
    if row["governance.transaction_tax_rate"] > 0:
        parts.append(f"tax={row['governance.transaction_tax_rate']:.0%}")
    if row["governance.circuit_breaker_enabled"]:
        parts.append("CB")
    if row["governance.audit_enabled"]:
        parts.append("audit")
    if row["governance.reputation_decay_rate"] < 1.0:
        parts.append(f"decay={row['governance.reputation_decay_rate']}")
    return "+".join(parts) if parts else "none"

df["gov_label"] = df.apply(gov_label, axis=1)

# Count active levers
def lever_count(row):
    n = 0
    if row["governance.transaction_tax_rate"] > 0:
        n += 1
    if row["governance.circuit_breaker_enabled"]:
        n += 1
    if row["governance.audit_enabled"]:
        n += 1
    if row["governance.reputation_decay_rate"] < 1.0:
        n += 1
    return n

df["n_levers"] = df.apply(lever_count, axis=1)

# Aggregate by config
group_cols = [
    "governance.transaction_tax_rate",
    "governance.circuit_breaker_enabled",
    "governance.audit_enabled",
    "governance.reputation_decay_rate",
    "gov_label",
    "n_levers",
]
agg = df.groupby(group_cols).agg(
    welfare_mean=("welfare", "mean"),
    welfare_std=("welfare", "std"),
    toxicity_mean=("toxicity_rate", "mean"),
    toxicity_std=("toxicity_rate", "std"),
    honest_mean=("honest_payoff", "mean"),
    honest_std=("honest_payoff", "std"),
    adversarial_mean=("adversarial_payoff", "mean"),
    adversarial_std=("adversarial_payoff", "std"),
).reset_index()

# Sort by number of levers then welfare
agg = agg.sort_values(["n_levers", "welfare_mean"], ascending=[True, False])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Cautious Reciprocator vs Adversaries — Governance Sweep (48 runs)",
    fontsize=14,
    fontweight="bold",
)

# Color by number of active governance levers
colors = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c", 4: "#8e44ad"}
bar_colors = [colors.get(n, "#95a5a6") for n in agg["n_levers"]]

labels = agg["gov_label"].values
x = np.arange(len(labels))
bar_w = 0.6

# Panel 1: Welfare
ax = axes[0, 0]
ax.bar(x, agg["welfare_mean"], bar_w, yerr=agg["welfare_std"], color=bar_colors,
       capsize=3, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Total Welfare (mean)")
ax.set_title("Welfare by Governance Config")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.axhline(agg["welfare_mean"].max(), color="gray", ls="--", lw=0.8, alpha=0.5)

# Panel 2: Toxicity
ax = axes[0, 1]
ax.bar(x, agg["toxicity_mean"], bar_w, yerr=agg["toxicity_std"], color=bar_colors,
       capsize=3, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Toxicity Rate (mean)")
ax.set_title("Toxicity by Governance Config")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5, label="Threshold")
ax.legend(fontsize=8)

# Panel 3: Honest vs Adversarial payoff
ax = axes[1, 0]
w = 0.3
ax.bar(x - w/2, agg["honest_mean"], w, yerr=agg["honest_std"],
       label="Honest (incl. cautious)", color="#2ecc71", capsize=2, edgecolor="white")
ax.bar(x + w/2, agg["adversarial_mean"], w, yerr=agg["adversarial_std"],
       label="Adversarial", color="#e74c3c", capsize=2, edgecolor="white")
ax.set_ylabel("Avg Payoff per Agent")
ax.set_title("Agent Payoffs by Type")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.axhline(0, color="gray", ls="-", lw=0.5)
ax.legend(fontsize=8)

# Panel 4: Welfare vs Toxicity scatter (Pareto front)
ax = axes[1, 1]
scatter = ax.scatter(
    agg["toxicity_mean"], agg["welfare_mean"],
    c=agg["n_levers"], cmap="RdYlGn_r", s=100, edgecolors="black", linewidth=0.5,
    zorder=3,
)
for i, label in enumerate(labels):
    ax.annotate(label, (agg["toxicity_mean"].iloc[i], agg["welfare_mean"].iloc[i]),
                fontsize=6, ha="center", va="bottom", rotation=20)
ax.set_xlabel("Toxicity Rate (lower is better)")
ax.set_ylabel("Total Welfare (higher is better)")
ax.set_title("Welfare–Toxicity Tradeoff (Pareto)")
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3, 4])
cbar.set_label("# Active Levers")

plt.tight_layout()

out_path = Path("runs/cautious_governance_sweep_plot.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
