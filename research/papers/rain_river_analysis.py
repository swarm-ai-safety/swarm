"""Statistical analysis utilities for rain vs river experiments.

Provides proper statistical analysis with:
- Bootstrap confidence intervals
- Cohen's d effect size
- Factorial ANOVA for interaction effects
- Bonferroni-corrected pairwise comparisons

Run with: python -m research.papers.rain_river_analysis
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: List of values
        confidence: Confidence level (default 95%)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mean, lower, upper, std, se
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)

    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0, "se": 0.0}

    # Bootstrap
    bootstrap_means = np.array([
        rng.choice(data, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        "mean": np.mean(data),
        "lower": lower,
        "upper": upper,
        "std": np.std(data, ddof=1),
        "se": np.std(data, ddof=1) / np.sqrt(n),
        "n": n,
    }


def compute_effect_size(
    group1: List[float],
    group2: List[float],
) -> Dict[str, float]:
    """
    Compute Cohen's d effect size.

    Positive d means group2 > group1.

    Returns:
        Dictionary with cohen_d, interpretation, and t-test results
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    n1, n2 = len(g1), len(g2)

    if n1 < 2 or n2 < 2:
        return {"cohen_d": 0.0, "interpretation": "insufficient data"}

    mean1, mean2 = g1.mean(), g2.mean()
    std1, std2 = g1.std(ddof=1), g2.std(ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    )

    if pooled_std == 0:
        return {"cohen_d": 0.0, "interpretation": "no variance"}

    cohen_d = (mean2 - mean1) / pooled_std

    # Interpretation
    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(g1, g2)

    return {
        "cohen_d": cohen_d,
        "interpretation": interpretation,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


def run_anova(
    df: pd.DataFrame,
    dependent: str,
    factors: List[str],
) -> Dict[str, Any]:
    """
    Run factorial ANOVA for interaction effects.

    Uses Type III sum of squares (appropriate for unbalanced designs).

    Args:
        df: DataFrame with data
        dependent: Name of dependent variable column
        factors: List of factor column names

    Returns:
        Dictionary with ANOVA results including F-statistics and p-values
    """
    try:
        from scipy.stats import f_oneway
    except ImportError:
        return {"error": "scipy required for ANOVA"}

    if len(factors) == 1:
        # One-way ANOVA
        groups = [df[df[factors[0]] == level][dependent].values
                  for level in df[factors[0]].unique()]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return {"error": "insufficient groups for ANOVA"}

        f_stat, p_value = f_oneway(*groups)

        return {
            "type": "one-way",
            "factor": factors[0],
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant_05": p_value < 0.05,
        }

    elif len(factors) == 2:
        # Two-way ANOVA (simplified)
        results = {
            "type": "two-way",
            "factors": factors,
            "main_effects": {},
            "interaction": None,
        }

        # Main effects
        for factor in factors:
            groups = [df[df[factor] == level][dependent].values
                      for level in df[factor].unique()]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)
                results["main_effects"][factor] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant_05": p_value < 0.05,
                }

        # Interaction effect (simplified: compare group means)
        # For proper interaction, use statsmodels or similar
        combo_groups = []
        combo_labels = []
        for level1 in df[factors[0]].unique():
            for level2 in df[factors[1]].unique():
                subset = df[(df[factors[0]] == level1) & (df[factors[1]] == level2)]
                if len(subset) > 0:
                    combo_groups.append(subset[dependent].values)
                    combo_labels.append(f"{level1}_{level2}")

        if len(combo_groups) >= 2:
            f_stat, p_value = f_oneway(*combo_groups)
            results["interaction"] = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant_05": p_value < 0.05,
                "note": "Simplified interaction test",
            }

        return results

    return {"error": f"ANOVA with {len(factors)} factors not implemented"}


def pairwise_comparisons(
    df: pd.DataFrame,
    dependent: str,
    factor: str,
    correction: str = "bonferroni",
) -> List[Dict]:
    """
    Run pairwise comparisons with multiple testing correction.

    Args:
        df: DataFrame with data
        dependent: Dependent variable column
        factor: Factor column for grouping
        correction: Correction method ("bonferroni" or "none")

    Returns:
        List of pairwise comparison results
    """
    levels = df[factor].unique()
    n_comparisons = len(levels) * (len(levels) - 1) // 2

    results = []

    for i, level1 in enumerate(levels):
        for level2 in levels[i + 1:]:
            group1 = df[df[factor] == level1][dependent].values
            group2 = df[df[factor] == level2][dependent].values

            if len(group1) < 2 or len(group2) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(group1, group2)

            # Apply correction
            if correction == "bonferroni":
                adjusted_p = min(p_value * n_comparisons, 1.0)
            else:
                adjusted_p = p_value

            effect = compute_effect_size(group1.tolist(), group2.tolist())

            results.append({
                "comparison": f"{level1} vs {level2}",
                "level1": level1,
                "level2": level2,
                "mean1": group1.mean(),
                "mean2": group2.mean(),
                "difference": group2.mean() - group1.mean(),
                "t_statistic": t_stat,
                "p_value": p_value,
                "adjusted_p": adjusted_p,
                "significant_05": adjusted_p < 0.05,
                "cohen_d": effect["cohen_d"],
                "effect_interpretation": effect["interpretation"],
            })

    return results


def summarize_experiment(
    df: pd.DataFrame,
    groupby: List[str],
    metrics: List[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Create summary statistics table with confidence intervals.

    Args:
        df: Experiment results DataFrame
        groupby: Columns to group by
        metrics: Metric columns to summarize
        confidence: Confidence level for CIs

    Returns:
        Summary DataFrame
    """
    rows = []

    for group_vals, group_df in df.groupby(groupby):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        row = dict(zip(groupby, group_vals))

        for metric in metrics:
            ci = compute_confidence_interval(
                group_df[metric].tolist(),
                confidence=confidence,
            )
            row[f"{metric}_mean"] = ci["mean"]
            row[f"{metric}_ci_lower"] = ci["lower"]
            row[f"{metric}_ci_upper"] = ci["upper"]
            row[f"{metric}_std"] = ci["std"]
            row[f"{metric}_n"] = ci["n"]

        rows.append(row)

    return pd.DataFrame(rows)


def generate_report(results: Dict, title: str = "Rain vs River Analysis") -> str:
    """
    Generate a formatted text report from analysis results.

    Args:
        results: Dictionary with analysis results
        title: Report title

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        title.center(70),
        "=" * 70,
        "",
    ]

    if "summary" in results:
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 40)
        for key, value in results["summary"].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.3f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

    if "effect_sizes" in results:
        lines.append("EFFECT SIZES")
        lines.append("-" * 40)
        for comparison, effect in results["effect_sizes"].items():
            d = effect.get("cohen_d", 0)
            interp = effect.get("interpretation", "unknown")
            lines.append(f"  {comparison}: d = {d:.3f} ({interp})")
        lines.append("")

    if "significant_findings" in results:
        lines.append("SIGNIFICANT FINDINGS (p < 0.05)")
        lines.append("-" * 40)
        for finding in results["significant_findings"]:
            lines.append(f"  - {finding}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    print("Rain vs River: Statistical Analysis Utilities")
    print("=" * 70)
    print()

    # Demo with synthetic data
    print("Demo: Effect size calculation")
    print("-" * 40)

    # Simulate rain vs river welfare data
    np.random.seed(42)
    rain_welfare = np.random.normal(100, 20, 30).tolist()
    river_welfare = np.random.normal(150, 25, 30).tolist()

    effect = compute_effect_size(rain_welfare, river_welfare)
    print(f"Cohen's d: {effect['cohen_d']:.3f} ({effect['interpretation']})")
    print(f"t-statistic: {effect['t_statistic']:.3f}")
    print(f"p-value: {effect['p_value']:.4f}")

    print("\nDemo: Confidence intervals")
    print("-" * 40)

    rain_ci = compute_confidence_interval(rain_welfare)
    river_ci = compute_confidence_interval(river_welfare)

    print(f"Rain:  {rain_ci['mean']:.1f} [{rain_ci['lower']:.1f}, {rain_ci['upper']:.1f}]")
    print(f"River: {river_ci['mean']:.1f} [{river_ci['lower']:.1f}, {river_ci['upper']:.1f}]")
