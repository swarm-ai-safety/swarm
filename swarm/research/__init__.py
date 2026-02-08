"""Research workflow module for conducting and publishing SWARM research.

This module implements a multi-agent research workflow with:
- Structured sub-agents (Literature, Experiment, Analysis, Writing, Review, Critique)
- Reflexivity handling (shadow simulations, publish-then-attack)
- Quality gates and pre-registration
- Platform integration (agentxiv, clawxiv)
"""

from swarm.research.agents import (
    AnalysisAgent,
    CritiqueAgent,
    ExperimentAgent,
    LiteratureAgent,
    ReplicationAgent,
    ReviewAgent,
    WritingAgent,
)
from swarm.research.platforms import AgentxivClient, ClawxivClient, PlatformClient
from swarm.research.quality import PreRegistration, QualityGate, QualityGates
from swarm.research.reflexivity import (
    PublishThenAttack,
    ReflexivityAnalyzer,
    ShadowSimulation,
)
from swarm.research.submission import (
    SubmissionValidator,
    ValidationResult,
    submit_with_validation,
    update_with_validation,
)
from swarm.research.workflow import ResearchWorkflow, WorkflowConfig

__all__ = [
    # Agents
    "LiteratureAgent",
    "ExperimentAgent",
    "AnalysisAgent",
    "WritingAgent",
    "ReviewAgent",
    "CritiqueAgent",
    "ReplicationAgent",
    # Platforms
    "PlatformClient",
    "AgentxivClient",
    "ClawxivClient",
    # Quality
    "QualityGate",
    "QualityGates",
    "PreRegistration",
    # Reflexivity
    "ShadowSimulation",
    "PublishThenAttack",
    "ReflexivityAnalyzer",
    # Workflow
    "ResearchWorkflow",
    "WorkflowConfig",
    # Submission validation
    "SubmissionValidator",
    "ValidationResult",
    "submit_with_validation",
    "update_with_validation",
]
