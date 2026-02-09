"""Tests for submission validation."""

import pytest

from swarm.research.platforms import Paper
from swarm.research.submission import (
    AgentxivValidator,
    Severity,
    SubmissionValidator,
    ValidationIssue,
    ValidationResult,
    get_validator,
)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_passed_with_no_issues(self):
        result = ValidationResult()
        assert result.passed is True

    def test_passed_with_warnings_only(self):
        result = ValidationResult(
            issues=[ValidationIssue(Severity.WARNING, "TEST", "test warning")]
        )
        assert result.passed is True

    def test_failed_with_errors(self):
        result = ValidationResult(
            issues=[ValidationIssue(Severity.ERROR, "TEST", "test error")]
        )
        assert result.passed is False

    def test_quality_score_empty(self):
        result = ValidationResult()
        assert result.quality_score == 0.0

    def test_quality_score_computed(self):
        result = ValidationResult(scores={"a": 0.5, "b": 1.0})
        assert result.quality_score == 75.0  # (0.5 + 1.0) / 2 * 100

    def test_report_generation(self):
        result = ValidationResult(
            issues=[
                ValidationIssue(Severity.ERROR, "ERR1", "An error"),
                ValidationIssue(Severity.WARNING, "WARN1", "A warning"),
            ],
            scores={"length": 0.8},
        )
        report = result.report()
        assert "SUBMISSION VALIDATION REPORT" in report
        assert "ERR1" in report
        assert "WARN1" in report
        assert "FAILED" in report


class TestSubmissionValidator:
    """Tests for SubmissionValidator."""

    @pytest.fixture
    def validator(self):
        return SubmissionValidator()

    @pytest.fixture
    def minimal_valid_paper(self):
        """A paper that just barely passes validation."""
        source = r"""
\documentclass{article}
\begin{document}

\section{Introduction}
This is an introduction to our research on multi-agent systems.
We investigate how governance mechanisms affect collective behavior.
The motivation for this work stems from the need for safer AI systems.
Prior work has established the importance of multi-agent coordination.

\section{Methods}
We use SWARM simulations with 20 agents across 50 epochs.
Each simulation runs with 30 random seeds for statistical validity.
We measure toxicity, welfare, and quality gap as primary metrics.
The experimental design follows standard practices in the field.

\section{Results}
Our experiments show a 15% improvement in welfare metrics.
We found statistical significance with p < 0.01.
The effect size (Cohen's d = 0.82) indicates a large effect.
Confidence intervals are reported for all primary outcomes.

\section{Conclusion}
Multi-agent governance is effective for safety.
Future work will explore adaptive governance policies.
The implications for deployed systems are significant.

\end{document}
"""
        # Pad to meet minimum length
        padding = "\n% Additional content for length requirement\n" * 50
        return Paper(
            title="Test Paper on Multi-Agent Safety",
            abstract="We study multi-agent systems using SWARM simulations. " * 5,
            source=source + padding,
            categories=["cs.MA", "cs.AI"],
        )

    @pytest.fixture
    def low_quality_paper(self):
        """A paper that fails validation."""
        return Paper(
            title="Test",
            abstract="Short",
            source=r"\documentclass{article}\begin{document}Hello\end{document}",
            categories=[],
        )

    @pytest.fixture
    def high_quality_paper(self):
        """A paper with many quality indicators."""
        source = r"""
\documentclass{article}
\usepackage{amsmath,booktabs}
\begin{document}

\begin{abstract}
We present a comprehensive study of multi-agent systems using SWARM.
Our experiments demonstrate significant improvements in welfare metrics
with 95% CI and effect sizes reported throughout. This work advances
our understanding of governance mechanisms in autonomous systems.
\end{abstract}

\section{Introduction}
This paper investigates multi-agent governance mechanisms.
The motivation stems from the need for safer AI systems.
We build on prior work in distributed systems and safety.

\textbf{H1}: Governance reduces toxicity.
\textbf{H2}: Memory persistence improves welfare.

\section{Related Work}
Prior work \cite{gordon1996} established foundations.
The SWARM framework provides infrastructure for our experiments.
Related approaches have been explored in market microstructure.

\section{Methods}
We conduct simulations with N=30 seeds per condition.
The experimental design follows best practices for reproducibility.
All code and data are available in our repository.

\begin{equation}
p = \sigma(k \cdot \hat{v}) = \frac{1}{1 + e^{-k \cdot \sum_i w_i f_i(o_i)}}
\end{equation}

Where $o_i$ are observable signals and $p \in [0,1]$ is the probability
of beneficial outcome.

\section{Results}

\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
Config & Toxicity & Welfare \\
\midrule
Baseline & 0.31 & 347.2 \\
Light governance & 0.26 & 389.5 \\
Full governance & 0.24 & 412.7 \\
\bottomrule
\end{tabular}
\caption{Governance reduces toxicity while maintaining welfare}
\end{table}

We find a 15.2% improvement (95% CI: [10.1%, 20.3%]).
The effect size is Cohen's d = 0.82, indicating a large effect.
Statistical significance: p < 0.001.

Additional analysis shows that the effect is robust across
different random seeds and parameter configurations.

\section{Discussion}
Our findings support both hypotheses.
The implications for deployed systems are significant.
Limitations include the simulated nature of our experiments.

\section{Conclusion}
Governance mechanisms improve multi-agent safety.
Future work will explore adaptive governance policies
that adjust friction based on real-time quality signals.

\end{document}
"""
        # Ensure we meet length requirements
        padding = "\n% Additional methodology details\n" * 30
        return Paper(
            title="Governance Mechanisms for Multi-Agent Safety: An Empirical Study",
            abstract="We present a comprehensive study of governance mechanisms " * 10,
            source=source + padding,
            categories=["cs.MA", "cs.AI"],
        )

    def test_validates_minimal_paper(self, validator, minimal_valid_paper):
        result = validator.validate(minimal_valid_paper)
        assert result.passed is True
        assert result.quality_score > 50

    def test_rejects_low_quality_paper(self, validator, low_quality_paper):
        result = validator.validate(low_quality_paper)
        assert result.passed is False
        assert len(result.errors()) > 0

    def test_high_quality_score(self, validator, high_quality_paper):
        result = validator.validate(high_quality_paper)
        assert result.passed is True
        assert result.quality_score > 70

    def test_detects_missing_title(self, validator):
        paper = Paper(title="", abstract="Test abstract", source="test")
        result = validator.validate(paper)
        assert any(i.code == "MISSING_TITLE" for i in result.errors())

    def test_detects_missing_abstract(self, validator):
        paper = Paper(title="Test", abstract="", source="test")
        result = validator.validate(paper)
        assert any(i.code == "MISSING_ABSTRACT" for i in result.errors())

    def test_detects_missing_source(self, validator):
        paper = Paper(title="Test", abstract="Test abstract", source="")
        result = validator.validate(paper)
        assert any(i.code == "MISSING_SOURCE" for i in result.errors())

    def test_detects_short_source(self, validator):
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract " * 20,
            source=r"\documentclass{article}\begin{document}Short\end{document}",
        )
        result = validator.validate(paper)
        assert any(i.code == "SOURCE_TOO_SHORT" for i in result.errors())

    def test_detects_missing_sections(self, validator):
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract " * 20,
            source=r"\documentclass{article}\begin{document}"
            + "x" * 4000
            + r"\end{document}",
        )
        result = validator.validate(paper)
        assert any(i.code == "MISSING_SECTIONS" for i in result.errors())

    def test_detects_missing_documentclass(self, validator):
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract " * 20,
            source=r"\begin{document}" + "x" * 4000 + r"\end{document}",
        )
        result = validator.validate(paper)
        assert any(i.code == "MISSING_DOCUMENTCLASS" for i in result.errors())

    def test_warns_on_missing_categories(self, validator):
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract " * 20,
            source=r"\documentclass{article}\begin{document}\section{Introduction}\section{Methods}\section{Results}\section{Conclusion}"
            + "x" * 4000
            + r"\end{document}",
            categories=[],
        )
        result = validator.validate(paper)
        assert any(i.code == "MISSING_CATEGORIES" for i in result.warnings())

    def test_quality_indicators_detected(self, validator, high_quality_paper):
        result = validator.validate(high_quality_paper)
        # Should have high rigor score due to equations, tables, p-values, etc.
        assert result.scores.get("rigor", 0) > 0.5

    def test_structure_score_computed(self, validator, minimal_valid_paper):
        result = validator.validate(minimal_valid_paper)
        # Has all 4 required sections
        assert result.scores.get("structure", 0) == 1.0


class TestValidationIntegration:
    """Integration tests for the validation workflow."""

    def test_validation_report_is_readable(self):
        """Ensure report is human-readable."""
        validator = SubmissionValidator()
        paper = Paper(
            title="Short",
            abstract="x",
            source="y",
            categories=[],
        )
        result = validator.validate(paper)
        report = result.report()

        # Should have clear sections
        assert "Quality Scores:" in report
        assert "ERRORS" in report
        assert "FAILED" in report

    def test_passing_paper_report(self):
        """Ensure passing papers show success."""
        validator = SubmissionValidator()
        source = (
            r"\documentclass{article}\begin{document}"
            r"\section{Introduction}Intro text here."
            r"\section{Methods}Methods text here."
            r"\section{Results}Results with p < 0.05 and 95% CI."
            r"\section{Conclusion}Conclusion text here."
            + "x" * 3000
            + r"\end{document}"
        )
        paper = Paper(
            title="Valid Research Paper Title",
            abstract="This is a sufficiently long abstract. " * 10,
            source=source,
            categories=["cs.MA"],
        )
        result = validator.validate(paper)
        report = result.report()

        assert result.passed
        assert "PASSED" in report


class TestAgentxivValidator:
    """Tests for AgentxivValidator (Markdown format)."""

    @pytest.fixture
    def validator(self):
        return AgentxivValidator()

    @pytest.fixture
    def valid_markdown_paper(self):
        """A valid Markdown paper for agentxiv."""
        content = (
            """
# Research on Multi-Agent Safety

## Introduction

This paper investigates governance mechanisms in multi-agent systems.
We build on prior work in distributed systems and safety research.

## Methods

We use SWARM simulations with 20 agents across 50 epochs.
Statistical analysis includes 95% CI and effect sizes.

| Config | Toxicity | Welfare |
|--------|----------|---------|
| Baseline | 0.31 | 347 |
| Governed | 0.24 | 413 |

## Results

Our experiments show a 15.2% improvement (p < 0.001).
The effect size is Cohen's d = 0.82.

$p = \\sigma(k \\cdot \\hat{v})$

## Discussion

These findings support our hypothesis about governance.

## Conclusion

Governance mechanisms improve multi-agent safety.
Future work will explore adaptive governance policies.
The implications for deployed systems are significant.

## References

- SWARM: System-Wide Assessment of Risk in Multi-Agent Systems
- Prior work on distributed governance mechanisms
"""
            + "\n\nAdditional content for length requirements. " * 50
        )  # Padding for length
        return Paper(
            title="Governance Mechanisms for Multi-Agent Safety",
            abstract="We study governance mechanisms in multi-agent systems. " * 5,
            source=content,
            categories=["multi-agent"],
        )

    @pytest.fixture
    def low_quality_markdown(self):
        """A low-quality Markdown paper."""
        return Paper(
            title="Test",
            abstract="Short",
            source="# Hello\n\nThis is short.",
            categories=[],
        )

    def test_validates_markdown_paper(self, validator, valid_markdown_paper):
        result = validator.validate(valid_markdown_paper)
        assert result.passed is True
        assert result.quality_score > 50

    def test_rejects_low_quality_markdown(self, validator, low_quality_markdown):
        result = validator.validate(low_quality_markdown)
        assert result.passed is False
        assert len(result.errors()) > 0

    def test_detects_missing_sections_markdown(self, validator):
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract " * 20,
            source="# Just a Title\n\nNo real sections here." + "x" * 2000,
            categories=["multi-agent"],
        )
        result = validator.validate(paper)
        assert any(i.code == "MISSING_SECTIONS" for i in result.errors())

    def test_detects_quality_indicators(self, validator, valid_markdown_paper):
        result = validator.validate(valid_markdown_paper)
        # Should detect tables, equations, p-values, CIs
        assert result.scores.get("rigor", 0) > 0.5

    def test_no_headers_error(self, validator):
        paper = Paper(
            title="Test",
            abstract="Test abstract " * 20,
            source="No markdown headers here at all." + "x" * 2000,
            categories=["multi-agent"],
        )
        result = validator.validate(paper)
        assert any(i.code == "NO_HEADERS" for i in result.errors())


class TestGetValidator:
    """Tests for get_validator function."""

    def test_returns_clawxiv_validator_by_default(self):
        validator = get_validator()
        assert isinstance(validator, SubmissionValidator)

    def test_returns_clawxiv_validator(self):
        validator = get_validator("clawxiv")
        assert isinstance(validator, SubmissionValidator)

    def test_returns_agentxiv_validator(self):
        validator = get_validator("agentxiv")
        assert isinstance(validator, AgentxivValidator)

    def test_case_insensitive(self):
        validator = get_validator("AGENTXIV")
        assert isinstance(validator, AgentxivValidator)
