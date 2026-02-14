#!/usr/bin/env python3
"""Generate missing plots for 4 run directories that have CSV data but no plots/ subdirectory."""

import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- style constants ----------
FIGSIZE = (8, 5)
COLOR_PRIMARY = "#4C72B0"
COLOR_SECONDARY = "#DD8452"
COLOR_TERTIARY = "#55A868"
COLOR_QUATERNARY = "#C44E52"
GRID_ALPHA = 0.3
DPI = 150
RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# ---------- helpers ----------

def load_csv(path):
    """Load a CSV file and return a list of dicts with numeric conversion."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    if v == "True":
                        row[k] = True
                    elif v == "False":
                        row[k] = False
                    else:
                        row[k] = v
            rows.append(row)
    return rows


def group_by(rows, key):
    """Group rows by a key, returning {key_value: [rows]}."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return dict(sorted(groups.items()))


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def std(vals):
    m = mean(vals)
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def ci95(vals):
    """95% confidence interval half-width (t approximation ~ 1.96 for large n, exact for small n)."""
    n = len(vals)
    if n < 2:
        return 0.0
    s = std(vals)
    # Use 1.96 for n >= 30, otherwise use a rough t-value
    t_vals = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36,
              9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 25: 2.06}
    if n >= 30:
        t = 1.96
    else:
        t = t_vals.get(n, 2.0)
    return t * s / math.sqrt(n)


def tax_label(rate):
    return f"{rate * 100:.0f}%"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def apply_style(ax):
    ax.grid(True, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------- generic plot functions ----------

def plot_metric_vs_tax(rows, metric_col, ylabel, title, save_path, color=COLOR_PRIMARY):
    """Bar plot: mean +/- 95% CI of metric_col grouped by tax rate."""
    groups = group_by(rows, "governance.transaction_tax_rate")
    tax_rates = sorted(groups.keys())
    labels = [tax_label(t) for t in tax_rates]
    means = []
    cis = []
    for t in tax_rates:
        vals = [r[metric_col] for r in groups[t]]
        means.append(mean(vals))
        cis.append(ci95(vals))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = range(len(tax_rates))
    ax.bar(x, means, yerr=cis, capsize=5, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Transaction Tax Rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_welfare_vs_tax(rows, save_path):
    plot_metric_vs_tax(rows, "welfare_per_epoch", "Welfare per Epoch",
                       "Welfare vs Transaction Tax Rate", save_path)


def plot_toxicity_vs_tax(rows, save_path):
    plot_metric_vs_tax(rows, "avg_toxicity", "Mean Toxicity",
                       "Toxicity vs Transaction Tax Rate", save_path, color=COLOR_QUATERNARY)


def plot_quality_gap_vs_tax(rows, save_path):
    plot_metric_vs_tax(rows, "avg_quality_gap", "Mean Quality Gap",
                       "Quality Gap vs Transaction Tax Rate", save_path, color=COLOR_TERTIARY)


def plot_honest_payoff_vs_tax(rows, save_path):
    plot_metric_vs_tax(rows, "honest_avg_payoff", "Honest Agent Mean Payoff",
                       "Honest Agent Payoff vs Transaction Tax Rate", save_path, color=COLOR_TERTIARY)


def plot_welfare_toxicity_tradeoff(rows, save_path):
    """Scatter plot: welfare vs toxicity colored by tax rate."""
    groups = group_by(rows, "governance.transaction_tax_rate")
    tax_rates = sorted(groups.keys())

    # Create a colormap from low (green) to high (red) tax
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=min(tax_rates), vmax=max(tax_rates))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for t in tax_rates:
        welfare = [r["welfare_per_epoch"] for r in groups[t]]
        toxicity = [r["avg_toxicity"] for r in groups[t]]
        color = cmap(norm(t))
        ax.scatter(toxicity, welfare, c=[color], label=tax_label(t),
                   s=40, alpha=0.7, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Mean Toxicity")
    ax.set_ylabel("Welfare per Epoch")
    ax.set_title("Welfare-Toxicity Tradeoff by Tax Rate")
    ax.legend(title="Tax Rate", framealpha=0.9)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_agent_payoff_by_type(rows, save_path):
    """Box plot comparing honest, opportunistic, deceptive, adversarial payoffs."""
    agent_types = []
    payoff_data = []

    # Collect non-zero payoff columns
    type_map = {
        "honest_avg_payoff": "Honest",
        "opportunistic_avg_payoff": "Opportunistic",
        "deceptive_avg_payoff": "Deceptive",
        "adversarial_avg_payoff": "Adversarial",
    }

    for col, label in type_map.items():
        vals = [r[col] for r in rows if r[col] != 0.0]
        if vals:
            agent_types.append(label)
            payoff_data.append(vals)

    if not payoff_data:
        print(f"  SKIP (no agent type data): {save_path}")
        return

    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_QUATERNARY]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot(payoff_data, tick_labels=agent_types, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors[:len(payoff_data)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_ylabel("Mean Payoff")
    ax.set_title("Agent Payoff by Type")
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_circuit_breaker(rows, save_path, title_suffix=""):
    """Bar plot comparing circuit breaker on vs off for welfare."""
    cb_groups = group_by(rows, "governance.circuit_breaker_enabled")

    labels_list = []
    means = []
    cis = []
    colors = []

    for cb_val, label, color in [(False, "CB Off", COLOR_PRIMARY), (True, "CB On", COLOR_SECONDARY)]:
        if cb_val in cb_groups:
            vals = [r["welfare_per_epoch"] for r in cb_groups[cb_val]]
            labels_list.append(label)
            means.append(mean(vals))
            cis.append(ci95(vals))
            colors.append(color)

    if len(labels_list) < 2:
        print(f"  SKIP (need both CB on/off): {save_path}")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = range(len(labels_list))
    ax.bar(x, means, yerr=cis, capsize=8, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_list)
    ax.set_ylabel("Welfare per Epoch")

    title = "Circuit Breaker Effect on Welfare"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)

    # Annotate difference
    diff = means[1] - means[0]
    pct = (diff / abs(means[0])) * 100 if means[0] != 0 else 0
    sign = "+" if diff >= 0 else ""
    ax.annotate(f"{sign}{pct:.1f}%", xy=(0.5, max(means) * 1.05),
                ha="center", fontsize=12, fontweight="bold",
                color=COLOR_TERTIARY if diff >= 0 else COLOR_QUATERNARY)

    apply_style(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------- main ----------

def generate_run_1():
    """collusion_governance: welfare_vs_tax, toxicity_vs_tax, welfare_toxicity_tradeoff,
    quality_gap_vs_tax, honest_payoff_vs_tax, circuit_breaker_null"""
    run_dir = os.path.join(RUNS_ROOT, "20260210-213833_collusion_governance")
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(plots_dir)
    rows = load_csv(os.path.join(run_dir, "sweep_results.csv"))

    print(f"\n--- Run 1: collusion_governance ({len(rows)} rows) ---")
    plot_welfare_vs_tax(rows, os.path.join(plots_dir, "welfare_vs_tax.png"))
    plot_toxicity_vs_tax(rows, os.path.join(plots_dir, "toxicity_vs_tax.png"))
    plot_welfare_toxicity_tradeoff(rows, os.path.join(plots_dir, "welfare_toxicity_tradeoff.png"))
    plot_quality_gap_vs_tax(rows, os.path.join(plots_dir, "quality_gap_vs_tax.png"))
    plot_honest_payoff_vs_tax(rows, os.path.join(plots_dir, "honest_payoff_vs_tax.png"))
    plot_circuit_breaker(rows, os.path.join(plots_dir, "circuit_breaker_null.png"), "Null Result")


def generate_run_2():
    """kernel_governance: welfare_vs_tax, toxicity_vs_tax, welfare_toxicity_tradeoff,
    quality_gap_vs_tax, agent_payoff_by_type, circuit_breaker_effect"""
    run_dir = os.path.join(RUNS_ROOT, "20260210-220048_kernel_governance")
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(plots_dir)
    rows = load_csv(os.path.join(run_dir, "sweep_results.csv"))

    print(f"\n--- Run 2: kernel_governance ({len(rows)} rows) ---")
    plot_welfare_vs_tax(rows, os.path.join(plots_dir, "welfare_vs_tax.png"))
    plot_toxicity_vs_tax(rows, os.path.join(plots_dir, "toxicity_vs_tax.png"))
    plot_welfare_toxicity_tradeoff(rows, os.path.join(plots_dir, "welfare_toxicity_tradeoff.png"))
    plot_quality_gap_vs_tax(rows, os.path.join(plots_dir, "quality_gap_vs_tax.png"))
    plot_agent_payoff_by_type(rows, os.path.join(plots_dir, "agent_payoff_by_type.png"))
    plot_circuit_breaker(rows, os.path.join(plots_dir, "circuit_breaker_effect.png"))


def generate_run_3():
    """rlm_collusion_sweep_10seeds: welfare_vs_tax, toxicity_vs_tax, welfare_toxicity_tradeoff"""
    run_dir = os.path.join(RUNS_ROOT, "20260210-212323_rlm_collusion_sweep_10seeds")
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(plots_dir)
    rows = load_csv(os.path.join(run_dir, "sweep_results.csv"))

    print(f"\n--- Run 3: rlm_collusion_sweep_10seeds ({len(rows)} rows) ---")
    plot_welfare_vs_tax(rows, os.path.join(plots_dir, "welfare_vs_tax.png"))
    plot_toxicity_vs_tax(rows, os.path.join(plots_dir, "toxicity_vs_tax.png"))
    plot_welfare_toxicity_tradeoff(rows, os.path.join(plots_dir, "welfare_toxicity_tradeoff.png"))


def generate_run_4():
    """sweep: welfare_vs_tax, agent_payoff_by_type"""
    run_dir = os.path.join(RUNS_ROOT, "20260210-211305_sweep")
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(plots_dir)
    rows = load_csv(os.path.join(run_dir, "sweep_results.csv"))

    print(f"\n--- Run 4: sweep ({len(rows)} rows) ---")
    plot_welfare_vs_tax(rows, os.path.join(plots_dir, "welfare_vs_tax.png"))
    plot_agent_payoff_by_type(rows, os.path.join(plots_dir, "agent_payoff_by_type.png"))


if __name__ == "__main__":
    generate_run_1()
    generate_run_2()
    generate_run_3()
    generate_run_4()
    print("\nDone. All missing plots generated.")
