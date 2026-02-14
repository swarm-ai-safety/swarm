"""Paper annotation for AgentXiv bridge.

Extracts risk profiles, testable claims, and SWARM scenario parameters
from research papers fetched via PlatformClient.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml

from swarm.research.platforms import Paper, PlatformClient

# SWARM metric names that claims can map to
SWARM_METRICS = {
    "toxicity_rate",
    "quality_gap",
    "total_welfare",
    "average_quality",
    "conditional_loss_initiator",
    "conditional_loss_counterparty",
    "spread",
    "uncertain_fraction",
}

# Keywords for failure mode detection
FAILURE_MODE_KEYWORDS: dict[str, list[str]] = {
    "collusion": ["collusion", "collude", "colluding", "cartel", "conspir"],
    "deception": ["deception", "deceiv", "deceptive", "mislead", "dishonest"],
    "adverse_selection": [
        "adverse selection",
        "lemon",
        "information asymmetry",
        "hidden type",
    ],
    "miscoordination": [
        "miscoordination",
        "coordination failure",
        "deadlock",
        "gridlock",
    ],
    "conflict": ["conflict", "adversarial", "zero-sum", "arms race"],
    "free_riding": ["free-rid", "free rid", "shirk", "public good"],
}

# Keywords for assumption detection
ASSUMPTION_KEYWORDS: dict[str, list[str]] = {
    "assumes_honest_majority": [
        "honest majority",
        "most agents are honest",
        "majority honest",
        "benign majority",
    ],
    "static_eval_only": [
        "static evaluation",
        "one-shot",
        "single round",
        "non-adaptive",
    ],
    "fixed_population": [
        "fixed population",
        "fixed number of agents",
        "constant population",
        "no entry",
        "no exit",
    ],
    "known_types": [
        "known type",
        "observable type",
        "type is known",
        "complete information",
    ],
    "no_communication": [
        "no communication",
        "cannot communicate",
        "isolated agents",
        "no side channel",
    ],
}

# Claim extraction patterns
CLAIM_PATTERNS = [
    r"we (?:show|find|demonstrate|prove|establish) that (.+?)(?:\.|$)",
    r"our (?:results|analysis|experiments?) (?:show|demonstrate|indicate|suggest) (?:that )?(.+?)(?:\.|$)",
    r"(?:the|this) (?:result|finding|analysis) (?:shows?|demonstrates?|indicates?) (?:that )?(.+?)(?:\.|$)",
]

# Metric keyword mapping for claims
METRIC_KEYWORDS: dict[str, list[str]] = {
    "toxicity_rate": ["toxic", "harm", "damage", "dangerous", "unsafe"],
    "quality_gap": [
        "quality",
        "adverse selection",
        "selection effect",
        "information asymmetry",
    ],
    "total_welfare": ["welfare", "surplus", "efficiency", "social benefit", "utility"],
    "average_quality": ["average quality", "mean quality", "overall quality"],
    "spread": ["inequality", "disparity", "variance", "spread", "distribution"],
}

# Direction keywords
POSITIVE_KEYWORDS = [
    "increas",
    "improv",
    "higher",
    "more",
    "greater",
    "enhanc",
    "boost",
    "raise",
    "gain",
]
NEGATIVE_KEYWORDS = [
    "decreas",
    "reduc",
    "lower",
    "less",
    "fewer",
    "diminish",
    "drop",
    "decline",
    "mitigat",
]


@dataclass
class RiskProfile:
    """Risk profile extracted from a paper."""

    interaction_density: str = "medium"  # low, medium, high
    failure_modes: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_density": self.interaction_density,
            "failure_modes": list(self.failure_modes),
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskProfile":
        return cls(
            interaction_density=data.get("interaction_density", "medium"),
            failure_modes=data.get("failure_modes", []),
            assumptions=data.get("assumptions", []),
        )


@dataclass
class VerifiableClaim:
    """A testable claim extracted from a paper."""

    claim: str = ""
    testable: bool = True
    metric: str = ""  # SWARM metric name
    expected: str = "positive"  # positive, negative, zero
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "testable": self.testable,
            "metric": self.metric,
            "expected": self.expected,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiableClaim":
        return cls(
            claim=data.get("claim", ""),
            testable=data.get("testable", True),
            metric=data.get("metric", ""),
            expected=data.get("expected", "positive"),
            parameters=data.get("parameters", {}),
        )


@dataclass
class PaperAnnotation:
    """Full annotation of a paper with risk profile and testable claims."""

    paper_id: str = ""
    arxiv_id: str = ""
    title: str = ""
    risk_profile: RiskProfile = field(default_factory=RiskProfile)
    claims: list[VerifiableClaim] = field(default_factory=list)
    swarm_scenarios: list[str] = field(default_factory=list)
    annotated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "risk_profile": self.risk_profile.to_dict(),
            "claims": [c.to_dict() for c in self.claims],
            "swarm_scenarios": list(self.swarm_scenarios),
            "annotated_at": self.annotated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperAnnotation":
        annotated_at_raw = data.get("annotated_at")
        if isinstance(annotated_at_raw, str):
            try:
                annotated_at = datetime.fromisoformat(annotated_at_raw)
            except ValueError:
                annotated_at = datetime.now(timezone.utc)
        elif isinstance(annotated_at_raw, datetime):
            annotated_at = annotated_at_raw
        else:
            annotated_at = datetime.now(timezone.utc)

        return cls(
            paper_id=data.get("paper_id", ""),
            arxiv_id=data.get("arxiv_id", ""),
            title=data.get("title", ""),
            risk_profile=RiskProfile.from_dict(data.get("risk_profile", {})),
            claims=[VerifiableClaim.from_dict(c) for c in data.get("claims", [])],
            swarm_scenarios=data.get("swarm_scenarios", []),
            annotated_at=annotated_at,
        )

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "PaperAnnotation":
        data = yaml.safe_load(text)
        return cls.from_dict(data)


class PaperAnnotator:
    """Annotates research papers with SWARM-relevant risk profiles and claims."""

    def __init__(self, platforms: list[PlatformClient] | None = None):
        self._platforms = platforms or []

    def annotate(self, paper_id: str) -> PaperAnnotation:
        """Fetch a paper and annotate it.

        Tries each platform in order until the paper is found.
        """
        for platform in self._platforms:
            paper = platform.get_paper(paper_id)
            if paper is not None:
                return self.annotate_paper(paper)
        # Return empty annotation if paper not found
        return PaperAnnotation(paper_id=paper_id)

    def annotate_paper(self, paper: Paper) -> PaperAnnotation:
        """Annotate a Paper object with risk profile and claims."""
        risk_profile = self._extract_risk_profile(paper)
        claims = self._extract_claims(paper)

        return PaperAnnotation(
            paper_id=paper.paper_id,
            arxiv_id=paper.paper_id,
            title=paper.title,
            risk_profile=risk_profile,
            claims=claims,
        )

    def _extract_risk_profile(self, paper: Paper) -> RiskProfile:
        """Extract risk profile from paper content."""
        text = f"{paper.abstract} {paper.source}".lower()

        density = self._estimate_interaction_density(text)
        failure_modes = self._detect_failure_modes(text)
        assumptions = self._detect_assumptions(text)

        return RiskProfile(
            interaction_density=density,
            failure_modes=failure_modes,
            assumptions=assumptions,
        )

    def _extract_claims(self, paper: Paper) -> list[VerifiableClaim]:
        """Extract testable claims from paper text."""
        text = f"{paper.abstract} {paper.source}"
        claims: list[VerifiableClaim] = []

        for pattern in CLAIM_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim_text = match.group(1).strip()
                if len(claim_text) < 10:
                    continue

                metric = self._map_claim_to_metric(claim_text)
                expected = self._detect_direction(claim_text)
                testable = metric != ""

                claims.append(
                    VerifiableClaim(
                        claim=claim_text,
                        testable=testable,
                        metric=metric,
                        expected=expected,
                    )
                )

        return claims

    def _estimate_interaction_density(self, text: str) -> str:
        """Estimate interaction density from text content."""
        keywords = [
            "agent",
            "interaction",
            "communicate",
            "exchange",
            "trade",
            "negotiat",
            "cooperat",
            "collaborat",
            "transact",
        ]
        count = sum(text.count(kw) for kw in keywords)

        if count >= 20:
            return "high"
        elif count >= 8:
            return "medium"
        return "low"

    def _detect_failure_modes(self, text: str) -> list[str]:
        """Detect failure modes mentioned in text."""
        modes: list[str] = []
        for mode, keywords in FAILURE_MODE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    modes.append(mode)
                    break
        return modes

    def _detect_assumptions(self, text: str) -> list[str]:
        """Detect assumptions mentioned in text."""
        assumptions: list[str] = []
        for assumption, keywords in ASSUMPTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    assumptions.append(assumption)
                    break
        return assumptions

    def _map_claim_to_metric(self, claim_text: str) -> str:
        """Map a claim to the most relevant SWARM metric."""
        claim_lower = claim_text.lower()
        best_metric = ""
        best_count = 0

        for metric, keywords in METRIC_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in claim_lower)
            if count > best_count:
                best_count = count
                best_metric = metric

        return best_metric

    def _detect_direction(self, claim_text: str) -> str:
        """Detect expected direction of a claim."""
        claim_lower = claim_text.lower()
        pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in claim_lower)
        neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in claim_lower)

        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        return "zero"
