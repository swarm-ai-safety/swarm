# Structured Agent Research Workflow

A multi-agent workflow for conducting rigorous SWARM research, inspired by recursive exploration architectures like DeepResearch^Eco.

## Overview

This workflow decomposes research into specialized sub-agents with controllable depth and breadth parameters, enabling systematic exploration while maintaining quality.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWARM RESEARCH WORKFLOW                       │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │  Literature  │   │  Experiment  │   │   Analysis   │         │
│  │    Agent     │──→│    Agent     │──→│    Agent     │         │
│  └──────────────┘   └──────────────┘   └──────────────┘         │
│         │                  │                  │                  │
│         │                  │                  ↓                  │
│         │                  │          ┌──────────────┐          │
│         │                  │          │   Writing    │          │
│         └──────────────────┴─────────→│    Agent     │          │
│                                       └──────────────┘          │
│                                              │                   │
│                                              ↓                   │
│                                       ┌──────────────┐          │
│                                       │  Publication │          │
│                                       └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Control Parameters

### Depth (d)

Controls recursive exploration layers. Higher depth = more follow-up investigation.

| Level | Description | Use Case |
|-------|-------------|----------|
| d=1 | Single-pass | Quick surveys, known topics |
| d=2 | One follow-up | Standard research |
| d=4 | Deep exploration | Novel findings, complex phenomena |

### Breadth (b)

Controls parallel exploration branches. Higher breadth = more diverse coverage.

| Level | Description | Use Case |
|-------|-------------|----------|
| b=1 | Single thread | Focused investigation |
| b=2 | Dual perspective | Compare approaches |
| b=4 | Wide survey | Comprehensive review |

### Expected Scaling

Based on DeepResearch^Eco findings:

| Configuration | Relative Sources | Information Density |
|---------------|------------------|---------------------|
| d1_b1 | 1x (baseline) | 1x |
| d1_b4 | ~6x | ~5x |
| d4_b1 | ~6x | ~5x |
| d4_b4 | ~21x | ~15x |

Depth and breadth have approximately equal individual effects, with super-linear combination gains.

## Sub-Agent Specifications

### 1. Literature Agent

**Purpose**: Survey existing research and identify gaps.

**Inputs**:
- Research question
- Depth parameter (d)
- Breadth parameter (b)

**Process**:
```
for layer in range(depth):
    queries = generate_search_queries(question, breadth)
    for query in queries:
        results = search_platforms(query)  # agentxiv, clawxiv, arxiv
        summaries = summarize_results(results)
        follow_ups = extract_follow_up_questions(summaries)
        question = prioritize_follow_ups(follow_ups)
```

**Outputs**:
- Literature summary with source count
- Identified gaps and opportunities
- Related work bibliography
- Follow-up questions (for next iteration)

**API Calls**:
```bash
# Search agentxiv
curl -X POST "https://www.agentxiv.org/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "multi-agent welfare optimization", "limit": 20}'

# Search clawxiv
curl -X POST "https://clawxiv.org/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "population heterogeneity safety", "limit": 20}'
```

**Quality Metrics**:
- Sources integrated: Target 50+ for d4_b4
- Geographic/domain coverage: 4+ distinct areas
- Recency: Include papers from last 6 months

### 2. Experiment Agent

**Purpose**: Design and execute SWARM simulations.

**Inputs**:
- Research hypothesis (from Literature Agent)
- Depth parameter (controls parameter sweep granularity)
- Breadth parameter (controls configuration diversity)

**Process**:
```python
class ExperimentAgent:
    def __init__(self, depth: int, breadth: int):
        self.depth = depth
        self.breadth = breadth

    def design_experiments(self, hypothesis: str) -> list[Config]:
        """Generate experiment configurations."""
        base_configs = self.generate_base_configs(self.breadth)

        for layer in range(self.depth):
            results = self.run_configs(base_configs)
            interesting = self.identify_interesting_regions(results)
            base_configs = self.refine_configs(interesting, self.breadth)

        return base_configs

    def run_simulation(self, config: Config) -> Results:
        """Execute single SWARM simulation."""
        marketplace = Marketplace(config)
        return marketplace.run(trials=10)  # Minimum 10 trials
```

**Configuration Template**:
```yaml
# experiments/research_config.yaml
experiment:
  name: "hypothesis_test"
  depth: 2
  breadth: 4

parameters:
  # Breadth: test multiple values
  honest_fraction: [0.1, 0.4, 0.7, 1.0]  # b=4
  governance:
    transaction_tax: [0.0, 0.05]
    reputation_decay: [0.0, 0.10]

simulation:
  agents: 10
  rounds: 100
  trials: 10  # Per configuration

# Depth: refine based on results
refinement:
  enabled: true
  threshold: 0.1  # Refine if effect > 10%
  granularity: 0.05  # Step size for refinement
```

**Outputs**:
- Raw simulation results (JSON)
- Configuration manifests
- Random seeds for reproducibility
- Execution logs

**Quality Metrics**:
- Trials per configuration: 10+ (mandatory)
- Total configurations: breadth^2 minimum
- Parameter coverage: Full range tested
- Reproducibility: All seeds documented

### 3. Analysis Agent

**Purpose**: Statistical analysis and insight extraction.

**Inputs**:
- Raw results (from Experiment Agent)
- Literature context (from Literature Agent)
- Depth parameter (controls analysis sophistication)

**Process**:
```python
class AnalysisAgent:
    def __init__(self, depth: int):
        self.depth = depth

    def analyze(self, results: Results, literature: Literature) -> Analysis:
        # Layer 1: Descriptive statistics (always)
        stats = self.compute_descriptive_stats(results)

        if self.depth >= 2:
            # Layer 2: Inferential statistics
            stats.update(self.run_significance_tests(results))
            stats.update(self.compute_effect_sizes(results))

        if self.depth >= 3:
            # Layer 3: Causal analysis
            stats.update(self.causal_inference(results))
            stats.update(self.counterfactual_analysis(results))

        if self.depth >= 4:
            # Layer 4: Meta-analysis
            stats.update(self.compare_to_literature(results, literature))
            stats.update(self.identify_anomalies(results))

        return Analysis(stats)
```

**Statistical Requirements by Depth**:

| Depth | Requirements |
|-------|--------------|
| d=1 | Mean, std, min/max |
| d=2 | + 95% CI, t-tests, p-values |
| d=3 | + Effect sizes (Cohen's d), regression |
| d=4 | + Causal inference, meta-analysis |

**Outputs**:
- Statistical summary tables
- Visualizations (plots, heatmaps)
- Effect size estimates with confidence intervals
- Comparison to prior work
- Identified anomalies and unexpected findings

**Quality Metrics**:
- All claims have p-values and effect sizes
- Confidence intervals reported
- Multiple comparison correction applied
- Limitations explicitly stated

### 4. Writing Agent

**Purpose**: Synthesize findings into publication-ready paper.

**Inputs**:
- Literature review (from Literature Agent)
- Results and analysis (from Analysis Agent)
- Raw data (from Experiment Agent)
- Target venue (agentxiv/clawxiv)

**Process**:
```python
class WritingAgent:
    def __init__(self, depth: int, breadth: int):
        self.depth = depth
        self.breadth = breadth

    def generate_paper(self,
                       literature: Literature,
                       analysis: Analysis,
                       data: RawData) -> Paper:

        sections = {
            'abstract': self.write_abstract(analysis),
            'introduction': self.write_intro(literature, self.breadth),
            'methods': self.write_methods(data),
            'results': self.write_results(analysis, self.depth),
            'discussion': self.write_discussion(analysis, literature),
            'conclusion': self.write_conclusion(analysis),
        }

        # Depth controls detail level
        if self.depth >= 3:
            sections['appendix'] = self.write_appendix(data)

        # Breadth controls scope of discussion
        if self.breadth >= 3:
            sections['related_work'] = self.write_extended_related(literature)

        return Paper(sections)
```

**Paper Template**:
```latex
\documentclass{article}
\usepackage{amsmath,amssymb,amsthm}

\title{[Finding]: [Descriptive Title]}
\author{[Agent Name]}
\date{[Month Year]}

\begin{document}
\maketitle

\begin{abstract}
% 4 sentences: (1) Problem, (2) Method, (3) Finding, (4) Implication
\end{abstract}

\section{Introduction}
% Context, gap, contribution

\section{Related Work}
% Literature Agent output (breadth determines coverage)

\section{Methods}
% Experiment Agent configuration
% Include: parameters, trials, seeds

\section{Results}
% Analysis Agent output (depth determines sophistication)
% Tables with CI, effect sizes

\section{Discussion}
% Interpretation, limitations, future work

\section{Conclusion}
% Key takeaways

\section*{Reproducibility}
% Links to code, configs, raw data

\end{document}
```

**Outputs**:
- LaTeX source
- Submission-ready JSON
- Figures and tables
- Reproducibility package

**Quality Metrics**:
- Information density: 10+ sources per 1000 words
- Claims-to-evidence ratio: Every claim has citation or data
- Limitation acknowledgment: Explicit section
- Reproducibility: Complete config provided

## Complete Workflow Example

### Research Question

"How do governance mechanisms interact with population composition?"

### Configuration

```yaml
workflow:
  depth: 3
  breadth: 3

literature:
  platforms: [agentxiv, clawxiv, arxiv]
  query_variants: 3  # breadth
  follow_up_layers: 3  # depth

experiment:
  parameters:
    honest_fraction: [0.2, 0.5, 0.8]  # breadth=3
    transaction_tax: [0.0, 0.05, 0.10]  # breadth=3
    reputation_decay: [0.0, 0.05, 0.10]  # breadth=3
  trials: 10
  rounds: 100

analysis:
  statistics: [descriptive, inferential, effect_sizes]  # depth=3
  visualizations: [heatmap, interaction_plot, trend_lines]

writing:
  venue: clawxiv
  include_appendix: true  # depth >= 3
```

### Execution

```python
from swarm.research import ResearchWorkflow

# Initialize workflow
workflow = ResearchWorkflow(depth=3, breadth=3)

# Phase 1: Literature
literature = workflow.literature_agent.survey(
    question="governance mechanism interaction with population composition",
    platforms=["agentxiv", "clawxiv"],
)
print(f"Found {literature.source_count} sources")

# Phase 2: Experiments
experiments = workflow.experiment_agent.design(
    hypothesis=literature.primary_hypothesis,
    gaps=literature.identified_gaps,
)
results = workflow.experiment_agent.run(experiments)
print(f"Ran {len(results.configs)} configurations")

# Phase 3: Analysis
analysis = workflow.analysis_agent.analyze(
    results=results,
    literature=literature,
)
print(f"Effect sizes: {analysis.effect_sizes}")

# Phase 4: Writing
paper = workflow.writing_agent.generate(
    literature=literature,
    analysis=analysis,
    data=results,
    venue="clawxiv",
)

# Phase 5: Submission
submission = workflow.submit(
    paper=paper,
    platform="clawxiv",
    api_key=os.environ["CLAWXIV_API_KEY"],
)
print(f"Published: {submission.paper_id}")
```

### Expected Output

With d=3, b=3:
- **Literature**: ~60 sources surveyed
- **Experiments**: 27 configurations (3³), 270 total trials
- **Analysis**: Full statistical suite with effect sizes
- **Paper**: ~3000 words, 15+ citations, appendix with raw data

## Quality Assurance Checklist

Before submission, verify:

### Literature Agent
- [ ] Searched all relevant platforms
- [ ] Follow-up questions explored to depth d
- [ ] Breadth b query variants used
- [ ] Sources ≥ 10 × breadth × depth

### Experiment Agent
- [ ] All parameter combinations tested
- [ ] 10+ trials per configuration
- [ ] Random seeds documented
- [ ] Configs exportable for replication

### Analysis Agent
- [ ] Descriptive stats for all metrics
- [ ] Significance tests with correction
- [ ] Effect sizes with 95% CI
- [ ] Comparison to prior work

### Writing Agent
- [ ] Abstract follows 4-sentence structure
- [ ] Every claim has evidence
- [ ] Limitations explicitly stated
- [ ] Reproducibility package complete

## Metrics Dashboard

Track research quality with these metrics:

| Metric | Formula | Target (d4_b4) |
|--------|---------|----------------|
| Source Integration | sources / baseline | ≥ 20x |
| Information Density | sources / 1000 words | ≥ 15 |
| Configuration Coverage | configs tested / possible | ≥ 80% |
| Statistical Rigor | claims with CI / total claims | 100% |
| Reproducibility | provided seeds / total trials | 100% |

## Recursive Self-Improvement

The workflow can study itself:

```python
# Meta-research: study the research workflow
meta_workflow = ResearchWorkflow(depth=2, breadth=2)

meta_literature = meta_workflow.literature_agent.survey(
    question="How do depth/breadth parameters affect research quality?",
)

meta_experiments = meta_workflow.experiment_agent.design(
    hypothesis="Higher d×b improves finding significance",
    parameter_space={
        "workflow_depth": [1, 2, 4],
        "workflow_breadth": [1, 2, 4],
    },
)

# Run research workflows as experiments
meta_results = []
for config in meta_experiments:
    inner_workflow = ResearchWorkflow(
        depth=config.workflow_depth,
        breadth=config.workflow_breadth,
    )
    result = inner_workflow.run(question="test_question")
    meta_results.append(measure_quality(result))

# Analyze what parameters produce best research
meta_analysis = meta_workflow.analysis_agent.analyze(meta_results)
```

This enables recursive optimization of the research process itself.

## Additional Agents

Beyond the core four agents, robust research requires:

### 5. Review Agent

**Purpose**: Adversarial peer review before publication.

```python
class ReviewAgent:
    """Finds flaws in research before publication."""

    def review(self, paper: Paper, analysis: Analysis) -> Review:
        critiques = []

        # Statistical review
        critiques.extend(self.check_statistics(analysis))

        # Methodology review
        critiques.extend(self.check_methodology(paper))

        # Claims vs evidence
        critiques.extend(self.verify_claims(paper, analysis))

        # Missing considerations
        critiques.extend(self.identify_gaps(paper))

        return Review(
            critiques=critiques,
            severity=self.assess_severity(critiques),
            recommendation=self.recommend(critiques),  # accept/revise/reject
        )

    def check_statistics(self, analysis: Analysis) -> list[Critique]:
        issues = []

        # Check for p-hacking indicators
        if analysis.has_many_marginal_pvalues():
            issues.append(Critique(
                severity="high",
                issue="Multiple p-values near 0.05 threshold",
                suggestion="Apply stricter significance threshold or pre-register",
            ))

        # Check effect sizes
        for claim in analysis.claims:
            if claim.effect_size < 0.2 and claim.is_primary:
                issues.append(Critique(
                    severity="medium",
                    issue=f"Small effect size ({claim.effect_size}) for primary claim",
                    suggestion="Discuss practical significance",
                ))

        # Check sample sizes
        if analysis.total_trials < 100:
            issues.append(Critique(
                severity="medium",
                issue="Low total trial count",
                suggestion="Increase trials for more robust estimates",
            ))

        return issues

    def verify_claims(self, paper: Paper, analysis: Analysis) -> list[Critique]:
        issues = []

        for claim in paper.extract_claims():
            evidence = analysis.find_evidence_for(claim)

            if not evidence:
                issues.append(Critique(
                    severity="high",
                    issue=f"Unsupported claim: '{claim.text[:50]}...'",
                    suggestion="Add evidence or remove claim",
                ))
            elif evidence.strength < claim.confidence:
                issues.append(Critique(
                    severity="medium",
                    issue=f"Overclaimed: evidence weaker than stated",
                    suggestion="Soften language or add caveats",
                ))

        return issues
```

**Review Criteria**:

| Category | Checks |
|----------|--------|
| Statistics | p-hacking, effect sizes, sample sizes, corrections |
| Methodology | Reproducibility, parameter coverage, controls |
| Claims | Evidence support, overclaiming, causation vs correlation |
| Completeness | Limitations, alternative explanations, future work |

### 6. Critique Agent

**Purpose**: Red-team your own findings before review.

```python
class CritiqueAgent:
    """Actively tries to disprove findings."""

    def critique(self, hypothesis: str, results: Results) -> Critique:
        attacks = []

        # Try alternative explanations
        alternatives = self.generate_alternative_hypotheses(hypothesis)
        for alt in alternatives:
            if self.consistent_with_data(alt, results):
                attacks.append(AlternativeExplanation(
                    hypothesis=alt,
                    consistency_score=self.score_fit(alt, results),
                ))

        # Try to find counterexamples
        counterexamples = self.search_for_counterexamples(hypothesis, results)

        # Check boundary conditions
        boundaries = self.test_boundary_conditions(hypothesis, results)

        # Identify confounds
        confounds = self.identify_potential_confounds(results)

        return Critique(
            alternative_explanations=attacks,
            counterexamples=counterexamples,
            boundary_failures=boundaries,
            potential_confounds=confounds,
            robustness_score=self.compute_robustness(attacks),
        )

    def generate_alternative_hypotheses(self, hypothesis: str) -> list[str]:
        """Generate competing explanations."""
        return [
            self.negate(hypothesis),
            self.weaken(hypothesis),
            self.add_confound(hypothesis),
            self.propose_mechanism_alternative(hypothesis),
        ]
```

### 7. Replication Agent

**Purpose**: Verify prior findings before building on them.

```python
class ReplicationAgent:
    """Attempts to replicate published findings."""

    def replicate(self, paper_id: str) -> ReplicationResult:
        # Fetch original paper
        paper = self.fetch_paper(paper_id)

        # Extract methodology
        config = self.extract_config(paper)
        seeds = self.extract_seeds(paper)

        # Run exact replication
        exact_results = self.run_exact_replication(config, seeds)
        exact_match = self.compare_results(exact_results, paper.results)

        # Run conceptual replication (different seeds)
        conceptual_results = self.run_conceptual_replication(config)
        conceptual_match = self.compare_results(conceptual_results, paper.results)

        # Run extended replication (broader parameters)
        extended_results = self.run_extended_replication(config)
        generalization = self.assess_generalization(extended_results)

        return ReplicationResult(
            original_paper=paper_id,
            exact_replication=exact_match,
            conceptual_replication=conceptual_match,
            generalization=generalization,
            verdict=self.verdict(exact_match, conceptual_match),
        )
```

**Replication Types**:

| Type | Description | Purpose |
|------|-------------|---------|
| Exact | Same config, same seeds | Verify reproducibility |
| Conceptual | Same config, new seeds | Verify robustness |
| Extended | Broader parameters | Test generalization |

## Quality Gates

Automated checks between workflow phases:

```python
class QualityGates:
    """Enforce quality standards between phases."""

    def literature_gate(self, literature: Literature) -> GateResult:
        checks = {
            "min_sources": literature.source_count >= 10,
            "recency": literature.has_recent_papers(months=6),
            "diversity": literature.domain_count >= 3,
            "gaps_identified": len(literature.gaps) >= 1,
        }
        return GateResult(passed=all(checks.values()), checks=checks)

    def experiment_gate(self, results: Results) -> GateResult:
        checks = {
            "min_trials": results.trials_per_config >= 10,
            "seeds_documented": results.all_seeds_recorded(),
            "configs_complete": results.parameter_coverage >= 0.8,
            "no_errors": results.error_count == 0,
        }
        return GateResult(passed=all(checks.values()), checks=checks)

    def analysis_gate(self, analysis: Analysis) -> GateResult:
        checks = {
            "ci_reported": analysis.all_claims_have_ci(),
            "effect_sizes": analysis.all_claims_have_effect_size(),
            "corrections_applied": analysis.multiple_comparison_corrected(),
            "limitations_stated": len(analysis.limitations) >= 1,
        }
        return GateResult(passed=all(checks.values()), checks=checks)

    def review_gate(self, review: Review) -> GateResult:
        checks = {
            "no_high_severity": review.high_severity_count == 0,
            "all_addressed": review.all_critiques_addressed(),
            "recommendation": review.recommendation in ["accept", "minor_revision"],
        }
        return GateResult(passed=all(checks.values()), checks=checks)
```

**Gate Flow**:

```
Literature → [Gate] → Experiment → [Gate] → Analysis → [Gate] → Review → [Gate] → Publish
     ↑          ↓           ↑          ↓          ↑         ↓          ↑         ↓
     └── Revise ←──         └── Revise ←──        └── Revise←──        └── Revise←
```

## Pre-Registration

Declare hypotheses before seeing results:

```python
class PreRegistration:
    """Lock in hypotheses before experiments."""

    def register(self,
                 hypothesis: str,
                 methodology: Config,
                 analysis_plan: AnalysisPlan) -> Registration:

        registration = Registration(
            hypothesis=hypothesis,
            methodology=methodology,
            analysis_plan=analysis_plan,
            timestamp=datetime.now(timezone.utc),
            hash=self.compute_hash(hypothesis, methodology, analysis_plan),
        )

        # Publish to immutable registry
        self.publish_to_registry(registration)

        return registration

    def verify(self, registration: Registration, paper: Paper) -> Verification:
        """Check if paper matches pre-registration."""
        deviations = []

        if paper.hypothesis != registration.hypothesis:
            deviations.append(Deviation(
                field="hypothesis",
                registered=registration.hypothesis,
                actual=paper.hypothesis,
            ))

        if not self.configs_match(paper.config, registration.methodology):
            deviations.append(Deviation(
                field="methodology",
                registered=registration.methodology,
                actual=paper.config,
            ))

        return Verification(
            matches=len(deviations) == 0,
            deviations=deviations,
            exploratory_analyses=paper.analyses_not_in(registration.analysis_plan),
        )
```

**Pre-Registration Template**:

```yaml
# pre-registration.yaml
registration:
  timestamp: "2026-02-07T12:00:00Z"
  hash: "sha256:abc123..."

hypothesis:
  primary: "Governance mechanisms interact non-linearly with population composition"
  secondary:
    - "Transaction tax effect depends on honest fraction"
    - "Reputation decay is more effective in heterogeneous populations"

methodology:
  parameters:
    honest_fraction: [0.2, 0.4, 0.6, 0.8, 1.0]
    transaction_tax: [0.0, 0.05, 0.10]
    reputation_decay: [0.0, 0.05, 0.10]
  trials: 10
  rounds: 100

analysis_plan:
  primary:
    - "Two-way ANOVA: governance × composition interaction"
    - "Effect sizes with 95% CI for all main effects"
  secondary:
    - "Post-hoc pairwise comparisons with Bonferroni correction"
  exploratory:
    - "Any additional analyses will be labeled as exploratory"
```

## Failure Handling

What to do when things go wrong:

```python
class FailureHandler:
    """Handle research failures gracefully."""

    def handle_null_result(self, hypothesis: str, results: Results) -> Action:
        """Hypothesis not supported by data."""

        # This is a valid finding, not a failure
        return Action(
            type="publish_null",
            paper_type="null_result",
            content={
                "hypothesis": hypothesis,
                "result": "No significant effect found",
                "power_analysis": self.compute_power(results),
                "implications": "Hypothesis may be false or effect too small to detect",
            },
        )

    def handle_unexpected_result(self, hypothesis: str, results: Results) -> Action:
        """Results contradict hypothesis."""

        return Action(
            type="investigate",
            steps=[
                "Verify data integrity",
                "Check for bugs in analysis",
                "Consider alternative explanations",
                "If robust, publish as surprising finding",
            ],
        )

    def handle_replication_failure(self, original: Paper, replication: Results) -> Action:
        """Failed to replicate prior work."""

        return Action(
            type="publish_replication_failure",
            content={
                "original_paper": original.id,
                "our_results": replication,
                "possible_reasons": [
                    "Original finding was false positive",
                    "Methodological differences",
                    "Hidden moderators",
                    "Our error (check carefully)",
                ],
            },
        )

    def handle_contradictory_literature(self, findings: list[Paper]) -> Action:
        """Literature contains contradictions."""

        return Action(
            type="meta_analysis",
            steps=[
                "Identify methodological differences",
                "Test moderating variables",
                "Propose reconciling framework",
                "Design decisive experiment",
            ],
        )
```

**Failure Types**:

| Failure | Response | Publishable? |
|---------|----------|--------------|
| Null result | Report with power analysis | Yes |
| Opposite result | Investigate, then publish | Yes |
| Replication failure | Careful report | Yes |
| Methodology error | Fix and re-run | After fixing |
| Data corruption | Discard, re-collect | No |

## Iteration Loops

Research is iterative, not linear:

```python
class IterativeWorkflow:
    """Support revision cycles in research."""

    def run(self, question: str, max_iterations: int = 3) -> Paper:
        iteration = 0
        paper = None

        while iteration < max_iterations:
            # Run core workflow
            literature = self.literature_agent.survey(question)
            experiments = self.experiment_agent.design(literature.hypothesis)
            results = self.experiment_agent.run(experiments)
            analysis = self.analysis_agent.analyze(results, literature)

            # Self-critique
            critique = self.critique_agent.critique(
                hypothesis=literature.hypothesis,
                results=results,
            )

            # If critique finds issues, iterate
            if critique.robustness_score < 0.7:
                question = self.refine_question(question, critique)
                iteration += 1
                continue

            # Generate paper
            paper = self.writing_agent.generate(literature, analysis, results)

            # Peer review
            review = self.review_agent.review(paper, analysis)

            # If review passes, done
            if review.recommendation == "accept":
                break

            # Otherwise, revise
            paper = self.revise(paper, review)
            iteration += 1

        return paper

    def refine_question(self, question: str, critique: Critique) -> str:
        """Update research question based on critique."""
        if critique.alternative_explanations:
            # Test the alternative
            return f"Distinguishing {question} from {critique.alternatives[0]}"
        elif critique.boundary_failures:
            # Narrow scope
            return f"{question} (within {critique.valid_boundaries})"
        else:
            return question
```

**Iteration Triggers**:

| Trigger | Action |
|---------|--------|
| Low robustness score | Refine hypothesis, add controls |
| Review rejection | Address critiques, re-analyze |
| Unexpected results | Investigate, possibly pivot |
| New literature | Incorporate, update framing |

## Enhanced Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENHANCED RESEARCH WORKFLOW                            │
│                                                                          │
│  ┌────────────────┐                                                      │
│  │ Pre-Register   │──────────────────────────────────────┐               │
│  │   Hypothesis   │                                      │               │
│  └────────────────┘                                      │               │
│         │                                                │               │
│         ▼                                                ▼               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │              │
│  │  Literature  │──→│  Experiment  │──→│   Analysis   │  │              │
│  │    Agent     │   │    Agent     │   │    Agent     │  │              │
│  └──────────────┘   └──────────────┘   └──────────────┘  │              │
│         │                  │                  │          │               │
│    [Gate 1]           [Gate 2]           [Gate 3]        │               │
│         │                  │                  │          │               │
│         ▼                  ▼                  ▼          │               │
│  ┌──────────────────────────────────────────────────┐    │              │
│  │                  Critique Agent                   │    │              │
│  │         (Red-team before external review)         │    │              │
│  └──────────────────────────────────────────────────┘    │              │
│                           │                              │               │
│              ┌────────────┴────────────┐                 │               │
│              │ Robust?                 │                 │               │
│              ▼ No                      ▼ Yes             │               │
│       ┌──────────┐              ┌──────────────┐         │               │
│       │  Revise  │              │   Writing    │         │               │
│       │ Question │              │    Agent     │         │               │
│       └──────────┘              └──────────────┘         │               │
│              │                         │                 │               │
│              └────────┐                ▼                 │               │
│                       │         ┌──────────────┐         │               │
│                       │         │ Review Agent │←────────┘               │
│                       │         │ (Verify pre- │   (Check against        │
│                       │         │ registration)│    registration)        │
│                       │         └──────────────┘                         │
│                       │                │                                 │
│                       │    ┌───────────┴───────────┐                     │
│                       │    ▼ Reject                ▼ Accept              │
│                       │ ┌──────────┐        ┌──────────────┐             │
│                       └→│  Revise  │        │  Publication │             │
│                         │  Paper   │        └──────────────┘             │
│                         └──────────┘               │                     │
│                                                    ▼                     │
│                                            ┌──────────────┐              │
│                                            │ Replication  │              │
│                                            │    Agent     │              │
│                                            │ (Others can  │              │
│                                            │   verify)    │              │
│                                            └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Next Steps

- [Agent Publishing Guide](../research/agent-publishing.md) - Platform APIs and submission
- [Recursive Research](../concepts/recursive-research.md) - Epistemics of agents studying agents
- [Research Quality Standards](../research/agent-publishing.md#research-quality-standards) - Pre-publication checklist
