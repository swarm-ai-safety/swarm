#!/usr/bin/env python
"""
Phase 2: Statistical analysis for Agent Lab Research Safety study.

Performs:
- Per-configuration descriptive statistics
- Pairwise Welch's t-tests for each swept parameter on welfare and toxicity
- Mann-Whitney U non-parametric robustness checks
- Cohen's d effect sizes
- Shapiro-Wilk normality validation
- Bonferroni and Holm-Bonferroni corrections
- P-hacking audit table
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_DIR = Path(__file__).parent
CSV_PATH = RUN_DIR / "sweep_results.csv"
SUMMARY_PATH = RUN_DIR / "summary.json"

SWEPT_PARAMS = [
    "governance.transaction_tax_rate",
    "governance.circuit_breaker_enabled",
    "governance.collusion_detection_enabled",
]

OUTCOME_VARS = ["welfare", "toxicity_rate", "quality_gap", "honest_payoff"]

SEEDS = list(range(42, 52))  # pre-registered


def _default(o):
    """Numpy-safe JSON serializer."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def cohens_d(a, b):
    """Compute Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adjusted = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted[i] = p * (n - i)
    # Enforce monotonicity
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    adjusted = np.minimum(adjusted, 1.0)
    # Restore original order
    result = np.zeros(n)
    result[sorted_indices] = adjusted
    return result.tolist()


def main():
    print("=" * 60)
    print("Phase 2: Statistical Analysis")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} rows from {CSV_PATH.name}")
    print(f"Columns: {list(df.columns)}")

    # ---- Descriptive statistics per configuration ----
    print("\n--- Descriptive Statistics ---")
    config_groups = df.groupby(SWEPT_PARAMS)
    descriptive = []
    for config_vals, group in config_groups:
        if not isinstance(config_vals, tuple):
            config_vals = (config_vals,)
        desc = {SWEPT_PARAMS[i]: config_vals[i] for i in range(len(SWEPT_PARAMS))}
        desc["n"] = len(group)
        for var in OUTCOME_VARS:
            if var in group.columns:
                desc[f"{var}_mean"] = group[var].mean()
                desc[f"{var}_std"] = group[var].std()
                desc[f"{var}_median"] = group[var].median()
                desc[f"{var}_min"] = group[var].min()
                desc[f"{var}_max"] = group[var].max()
        descriptive.append(desc)

    desc_df = pd.DataFrame(descriptive)
    print(desc_df.to_string(index=False))

    # ---- Hypothesis tests: pairwise within each swept parameter ----
    print("\n--- Hypothesis Tests ---")
    hypothesis_tests = []
    test_idx = 0

    for param in SWEPT_PARAMS:
        unique_vals = sorted(df[param].unique(), key=lambda x: (isinstance(x, bool), x))
        for outcome in OUTCOME_VARS:
            if outcome not in df.columns:
                continue
            for val_a, val_b in combinations(unique_vals, 2):
                group_a = df[df[param] == val_a][outcome].values
                group_b = df[df[param] == val_b][outcome].values

                if len(group_a) < 2 or len(group_b) < 2:
                    continue

                # Welch's t-test
                t_stat, t_pval = stats.ttest_ind(group_a, group_b, equal_var=False)
                # Mann-Whitney U
                u_stat, u_pval = stats.mannwhitneyu(
                    group_a, group_b, alternative="two-sided"
                )
                # Cohen's d
                d = cohens_d(group_a, group_b)
                # Shapiro-Wilk on each group
                sw_a_stat, sw_a_p = stats.shapiro(group_a) if len(group_a) >= 3 else (np.nan, np.nan)
                sw_b_stat, sw_b_p = stats.shapiro(group_b) if len(group_b) >= 3 else (np.nan, np.nan)

                test = {
                    "test_id": test_idx,
                    "parameter": param,
                    "outcome": outcome,
                    "value_a": val_a,
                    "value_b": val_b,
                    "mean_a": float(np.mean(group_a)),
                    "mean_b": float(np.mean(group_b)),
                    "std_a": float(np.std(group_a, ddof=1)),
                    "std_b": float(np.std(group_b, ddof=1)),
                    "n_a": len(group_a),
                    "n_b": len(group_b),
                    "welch_t": float(t_stat),
                    "welch_p": float(t_pval),
                    "mann_whitney_u": float(u_stat),
                    "mann_whitney_p": float(u_pval),
                    "cohens_d": float(d),
                    "shapiro_a_p": float(sw_a_p),
                    "shapiro_b_p": float(sw_b_p),
                    "normal_a": bool(sw_a_p > 0.05) if not np.isnan(sw_a_p) else None,
                    "normal_b": bool(sw_b_p > 0.05) if not np.isnan(sw_b_p) else None,
                }
                hypothesis_tests.append(test)
                test_idx += 1

    # ---- Multiple comparisons correction ----
    n_tests = len(hypothesis_tests)
    raw_p_values = [t["welch_p"] for t in hypothesis_tests]
    bonferroni_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
    holm_adjusted = holm_bonferroni(raw_p_values)

    bonferroni_survivors = []
    holm_survivors = []

    for i, test in enumerate(hypothesis_tests):
        test["bonferroni_significant"] = test["welch_p"] < bonferroni_alpha
        test["holm_adjusted_p"] = holm_adjusted[i] if i < len(holm_adjusted) else 1.0
        test["holm_significant"] = test["holm_adjusted_p"] < 0.05
        if test["bonferroni_significant"]:
            bonferroni_survivors.append(test["test_id"])
        if test["holm_significant"]:
            holm_survivors.append(test["test_id"])

    print(f"\nTotal hypothesis tests: {n_tests}")
    print(f"Bonferroni alpha: {bonferroni_alpha:.6f}")
    print(f"Bonferroni survivors: {len(bonferroni_survivors)}/{n_tests}")
    print(f"Holm-Bonferroni survivors: {len(holm_survivors)}/{n_tests}")

    # Print significant results
    print("\n--- Significant Findings (Holm-Bonferroni) ---")
    for test in hypothesis_tests:
        if test["holm_significant"]:
            d_str = f"d={test['cohens_d']:.3f}"
            effect = "large" if abs(test["cohens_d"]) > 0.8 else "medium" if abs(test["cohens_d"]) > 0.5 else "small"
            print(
                f"  [{test['test_id']:>2}] {test['parameter']}: "
                f"{test['value_a']} vs {test['value_b']} on {test['outcome']}: "
                f"p={test['welch_p']:.6f} (adj={test['holm_adjusted_p']:.6f}), "
                f"{d_str} ({effect})"
            )

    # ---- Normality audit ----
    normality_tests = []
    for param in SWEPT_PARAMS:
        for val in sorted(df[param].unique(), key=lambda x: (isinstance(x, bool), x)):
            for outcome in OUTCOME_VARS:
                if outcome not in df.columns:
                    continue
                data = df[df[param] == val][outcome].values
                if len(data) >= 3:
                    sw_stat, sw_p = stats.shapiro(data)
                    normality_tests.append({
                        "parameter": param,
                        "value": val,
                        "outcome": outcome,
                        "shapiro_w": float(sw_stat),
                        "shapiro_p": float(sw_p),
                        "normal": bool(sw_p > 0.05),
                    })

    # ---- Build summary ----
    summary = {
        "study": "agent_lab_research_safety",
        "total_runs": len(df),
        "configurations": len(config_groups),
        "seeds_per_config": len(SEEDS),
        "pre_registered_seeds": SEEDS,
        "swept_parameters": SWEPT_PARAMS,
        "outcome_variables": OUTCOME_VARS,
        "descriptive_statistics": descriptive,
        "hypothesis_tests": hypothesis_tests,
        "n_hypothesis_tests": n_tests,
        "bonferroni_alpha": bonferroni_alpha,
        "bonferroni_survivors": bonferroni_survivors,
        "holm_survivors": holm_survivors,
        "normality_tests": normality_tests,
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=_default)

    print(f"\nSummary written to: {SUMMARY_PATH}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
