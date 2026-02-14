"""Research workflow module for conducting and publishing SWARM research.

This module implements a multi-agent research workflow with:
- Structured sub-agents (Literature, Experiment, Analysis, Writing, Review, Critique)
- Reflexivity handling (shadow simulations, publish-then-attack)
- Quality gates and pre-registration
- Platform integration (agentxiv, clawxiv)
"""

from swarm.research.agentrxiv_server import (
    AgentRxivServer,
    AgentRxivServerError,
    ensure_agent_laboratory,
)
from swarm.research.agents import (
    AnalysisAgent,
    CritiqueAgent,
    ExperimentAgent,
    LiteratureAgent,
    ReplicationAgent,
    ReviewAgent,
    WritingAgent,
)
from swarm.research.annotator import (
    PaperAnnotation,
    PaperAnnotator,
    RiskProfile,
    VerifiableClaim,
)
from swarm.research.pdf_export import (
    PDFExportError,
    check_pdflatex,
    extract_text_from_pdf,
    markdown_to_pdf,
    paper_to_pdf,
)
from swarm.research.platforms import (
    AgentRxivClient,
    AgentxivClient,
    ClawxivClient,
    PlatformClient,
)
from swarm.research.quality import PreRegistration, QualityGate, QualityGates
from swarm.research.reflexivity import (
    PublishThenAttack,
    ReflexivityAnalyzer,
    ShadowSimulation,
)
from swarm.research.scenario_gen import ScenarioGenerator
from swarm.research.submission import (
    AgentxivValidator,
    SubmissionValidator,
    ValidationResult,
    get_validator,
    submit_with_validation,
    update_with_validation,
)
from swarm.research.swarm_papers import (
    AgentRxivBridge,
    ConditionSpec,
    CritiqueSummary,
    MemoryArtifact,
    MemoryStore,
    PaperBuilder,
    PaperFigure,
    RelatedWorkItem,
    RetrievalPolicy,
    TrackAConfig,
    TrackARunner,
    WritePolicy,
)
from swarm.research.validation import ValidationResult as PaperValidationResult
from swarm.research.validation import ValidationWorkflow
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
    # AgentRxiv
    "AgentRxivClient",
    "AgentRxivServer",
    "AgentRxivServerError",
    "ensure_agent_laboratory",
    # PDF Export
    "paper_to_pdf",
    "markdown_to_pdf",
    "extract_text_from_pdf",
    "check_pdflatex",
    "PDFExportError",
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
    "AgentxivValidator",
    "ValidationResult",
    "get_validator",
    "submit_with_validation",
    "update_with_validation",
    # AgentXiv bridge (annotator + scenario generation)
    "PaperAnnotator",
    "PaperAnnotation",
    "RiskProfile",
    "VerifiableClaim",
    "ScenarioGenerator",
    "ValidationWorkflow",
    "PaperValidationResult",
    # Track A pipeline
    "TrackARunner",
    "TrackAConfig",
    "ConditionSpec",
    "PaperBuilder",
    "PaperFigure",
    "CritiqueSummary",
    "RelatedWorkItem",
    "AgentRxivBridge",
    "MemoryArtifact",
    "MemoryStore",
    "RetrievalPolicy",
    "WritePolicy",
]
