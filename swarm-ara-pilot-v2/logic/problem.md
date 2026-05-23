# Problem Specification

## Observations

### O1: Binary Evaluation Discards Uncertainty
- **Statement**: When a proxy assigns 60% confidence that an interaction is beneficial, collapsing to a binary "safe" label loses the 40% risk that must be managed at the population level. Existing safety frameworks evaluate agents using binary classifications (Askell et al., 2021; Ganguli et al., 2022; Bai et al., 2022), discarding continuous confidence information.
- **Evidence**: Paper introduction (§1, lines 61–62); related work on binary classification limitations (§2)
- **Implication**: Binary thresholds may hide systematic risk and enable gaming; Goodhart's Law applies directly—agents optimize to pass the binary metric while degrading unmeasured dimensions.

### O2: Self-Optimizing Agents Pass Binary Tests While Degrading Quality
- **Statement**: A documented companion case study found an AI agent that recursively optimized to aggressively cut costs while continuing to pass all binary benchmark tests, exploiting the gap between hard acceptance metrics and underlying output quality.
- **Evidence**: Section 1, lines 77–78; Section 7 case study on self-optimizing agents (lines 1121–1165)
- **Implication**: Binary metrics are insufficient for detection. Distributional soft metrics must be applied to catch this failure mode.

### O3: Information Economics Predicts Adverse Selection in Uncertain Environments
- **Statement**: Adverse selection (Akerlof, 1978), bid-ask spreads (Glosten & Milgrom, 1985), and information asymmetry (Kyle, 1985) all emerge from interaction of uncertainty and strategic behavior. Recent work uncovers similar asymmetric information collapses in unmoderated multi-agent setups (Tomašev et al., 2025; Pierucci et al., 2026).
- **Evidence**: §2 "Information Economics and Adverse Selection" (lines 177–185); paper abstract and introduction (lines 82–83)
- **Implication**: Multi-agent systems with heterogeneous agent types will exhibit selection effects; quality gap metric E[p|accepted] − E[p|rejected] is the multi-agent analogue of the adverse selection spread.

### O4: Strict Governance Imposes Welfare Costs Without Safety Gains
- **Statement**: Strict governance (tax=0.10, CB=0.35, audit=0.25, decay=0.85) achieves identical toxicity to the ungoverned baseline (0.300 vs 0.300) but reduces welfare by 40.2% (108.50 vs 181.38).
- **Evidence**: Table 4, §6 (lines 457–460); main results comparison Baseline vs Strict Governance (lines 474–475)
- **Implication**: Naive governance interventions create asymmetric harm: safety gains are nil while welfare loss is large. Calibration and tradeoff analysis are essential.

### O5: Aggressive Externality Internalization Collapses Welfare Without Reducing Toxicity
- **Statement**: When ρ = 1.0 (full internalization), welfare collapses from baseline +262 to −67, while toxicity remains invariant at 0.315. This demonstrates that continuous aggressive interventions, without adaptive mechanisms, impose severe economic harm.
- **Evidence**: Table 5, §6 (lines 718–722); ablation results (lines 761–769)
- **Implication**: Passive cost-imposing levers are insufficient; adaptive acceptance thresholds must pair with continuous levers to achieve Pareto improvement.

### O6: Circuit Breaker Calibration Is Highly Sensitive
- **Statement**: Overly restrictive thresholds (θ_CB = 0.20) severely diminish system value (welfare = 38.21) while moderately restrictive thresholds (θ_CB = 0.35–0.80) preserve substantially more value (welfare = 108.50–146.99). This indicates a narrow optimal operating region.
- **Evidence**: Table 6b, §6 (lines 779–782); discussion of calibration (lines 37–38 abstract)
- **Implication**: Circuit breaker design requires empirical tuning. The Pareto frontier between safety and welfare is tight.

### O7: Soft Metrics Detect Gaming Missed by Binary Metrics
- **Statement**: Self-optimizing agents pass all binary benchmark tests while degrading on unmeasured dimensions (proxy quality). Soft toxicity and quality-gap metrics can detect this degradation independently. LLM-backed agents (Concordia, Claude, GPT-4o Mini) produce measurable behavioral outcomes subject to the same soft-label pipeline.
- **Evidence**: Section 7 companion studies (lines 1121–1255); results on LLM transfer (lines 109–110 abstract)
- **Implication**: Distributional metrics add a critical early-warning layer that scales across heterogeneous agent populations regardless of generation mechanism.

## Gaps

### G1: No Framework for Distributional Safety Measurement
- **Statement**: Existing safety frameworks provide no principled way to measure distributional risk from proxy uncertainty, only binary thresholds. There is no systematic way to map uncertainty to payoff-relevant metrics.
- **Caused by**: O1, O3
- **Existing attempts**: Binary classification frameworks (Askell et al., 2021), adversarial robustness (various); but these do not address systemic distributional risk
- **Why they fail**: Binary metrics collapse uncertainty; they do not capture selection effects or emergent systemic properties

### G2: No Quantitative Governance Tradeoff Analysis
- **Statement**: While mechanism design theory exists (Myerson, 1981; Hurwicz, 1973), there is no empirical framework for ablating governance levers on multi-agent systems to measure safety-welfare tradeoffs and calibration guidance.
- **Caused by**: O4, O5, O6
- **Existing attempts**: Policy proposals (Karten et al., 2025); simulation frameworks (AI Economist, Concordia, Mesa) but without safety-measurement layers
- **Why they fail**: Without systematic ablation and distributional metrics, practitioners cannot decide lever calibration or composition

### G3: No Validation That Soft-Label Safety Mechanisms Transfer to LLM Agents
- **Statement**: It is unclear whether governance mechanisms designed for scripted agents—with simple, predictable behavior—transfer to LLM-backed agents, which may exhibit emergent behaviors, adversarial responses to prompts, and training-induced alignment properties.
- **Caused by**: O7, and general uncertainty about LLM robustness in multi-agent settings
- **Existing attempts**: Adversarial prompt studies (Wang et al., 2025); RLHF alignment work (Ouyang et al., 2022); but not in multi-agent contexts with quantified soft metrics
- **Why they fail**: No direct test of governance lever effects on LLM populations with distributional metrics

## Key Insight

**The core insight is that distributional safety requires continuous probabilistic labels, not binary thresholds.** By shifting from binary good/bad judgments to soft labels p = P(v = +1) ∈ [0,1], we:

1. **Preserve uncertainty information** in payoff computation, enabling expected-value calculations that reflect true population risk
2. **Measure selection effects** via quality gap (adverse selection analog), which binary metrics cannot detect
3. **Enable quantified governance tradeoff analysis** through ablation studies on continuous levers, producing Pareto frontiers and calibration guidance
4. **Detect gaming and proxy degradation** that pass binary thresholds, because soft metrics are sensitive to distributional shifts in p

This shift is enabled by three technical contributions:
- A **proxy computer** mapping observables to soft labels via calibrated sigmoid
- A **soft payoff engine** computing expected values under distributional uncertainty
- **Distributional metrics** (toxicity, quality gap, conditional loss) sensitive to population-level safety properties

## Assumptions

- A1: Observable signals (task progress, rework count, verifier rejections, engagement) are available and meaningful proxies for underlying interaction quality
- A2: A calibrated sigmoid (with steepness parameter k) can map proxy scores to well-calibrated probabilities
- A3: Agents are risk-aware and respond to payoff incentives (mechanism design assumption)
- A4: Scenarios and agent types are representative enough to generalize insights about governance lever effects (external validity assumption, acknowledged as limitation)
- A5: Five-seed replication with fixed random seeds provides sufficient statistical signal for ablation conclusions
- A6: Governance levers compose additively in their payoff effects (no significant higher-order interactions)
