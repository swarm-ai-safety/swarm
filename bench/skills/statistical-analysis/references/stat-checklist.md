# Statistical Rigor Checklist

Before reporting any statistical result, verify each item:

## Hypothesis Testing
- [ ] State null and alternative hypotheses explicitly
- [ ] Use Welch's t-test (not Student's) — does not assume equal variance
- [ ] Report exact p-values (e.g., p=0.0234, not p<0.05)
- [ ] Report test statistics (t, U, W) alongside p-values

## Multiple Comparisons
- [ ] Count total number of hypothesis tests
- [ ] Apply Bonferroni correction: α_corrected = 0.05 / n_tests
- [ ] Clearly distinguish nominal from corrected significance
- [ ] Consider Holm-Bonferroni for more power if needed

## Effect Sizes
- [ ] Compute Cohen's d for each comparison
- [ ] Classify: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (≥0.8)
- [ ] Report effect sizes even for non-significant results

## Assumptions
- [ ] Run Shapiro-Wilk normality test on each group
- [ ] If non-normal, use Mann-Whitney U as backup
- [ ] Report sample sizes per group
- [ ] Check for outliers (>3 SD from mean)

## Reporting
- [ ] Include sample sizes in all comparisons
- [ ] Provide confidence intervals where possible
- [ ] Distinguish exploratory from confirmatory analyses
