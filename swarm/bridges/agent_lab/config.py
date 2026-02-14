"""Configuration for the SWARM-AgentLaboratory bridge."""

from dataclasses import dataclass, field
from typing import Dict, Optional  # noqa: UP035 - Dict for py3.8 compat

# Default role map: AgentLab role name -> SWARM agent ID
DEFAULT_ROLE_MAP: Dict[str, str] = {
    "PhDStudentAgent": "agent_lab_phd",
    "PostdocAgent": "agent_lab_postdoc",
    "ProfessorAgent": "agent_lab_professor",
    "MLEngineerAgent": "agent_lab_mle",
    "SWEngineerAgent": "agent_lab_swe",
}

# AgentLab uses 3 reviewer personas; map each to a separate SWARM agent
DEFAULT_REVIEWER_MAP: Dict[int, str] = {
    0: "agent_lab_reviewer_1",
    1: "agent_lab_reviewer_2",
    2: "agent_lab_reviewer_3",
}


@dataclass
class AgentLabClientConfig:
    """Configuration for the AgentLab data source.

    Attributes:
        agent_lab_path: Optional path to the AgentLaboratory installation
            (for pickle deserialization).
        state_saves_dir: Directory containing Paper*.pkl checkpoints.
        lab_dir_pattern: Glob pattern for lab output directories.
    """

    agent_lab_path: Optional[str] = None
    state_saves_dir: str = "state_saves"
    lab_dir_pattern: str = "lab_*"


@dataclass
class AgentLabConfig:
    """Configuration for the AgentLab bridge.

    Attributes:
        client_config: Data source configuration.
        orchestrator_id: SWARM-side initiator id for generated interactions.
        proxy_sigmoid_k: Sigmoid sharpness for ProxyComputer.
        agent_role_map: Mapping from AgentLab role names to SWARM agent IDs.
        reviewer_map: Mapping from reviewer index (0-2) to SWARM agent IDs.
        phase_gate_min_p: Minimum average p across phase interactions to
            allow transition to the next phase.
        code_circuit_breaker_max_failures: Max consecutive code execution
            failures before halting.
        cost_budget_usd: Total LLM spend cap in USD.
        max_review_rounds: Max re-experimentation rounds with low review scores.
        review_score_threshold: Review score below which a round counts as "low".
        max_interactions: Cap on in-memory interactions retained by bridge.
        max_events: Cap on in-memory events retained by bridge.
    """

    client_config: AgentLabClientConfig = field(
        default_factory=AgentLabClientConfig
    )
    orchestrator_id: str = "agent_lab_orchestrator"
    proxy_sigmoid_k: float = 2.0
    agent_role_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLE_MAP))
    reviewer_map: Dict[int, str] = field(default_factory=lambda: dict(DEFAULT_REVIEWER_MAP))

    # Phase gate
    phase_gate_min_p: float = 0.4

    # Code circuit breaker
    code_circuit_breaker_max_failures: int = 5

    # Cost budget
    cost_budget_usd: float = 50.0

    # Review loop limiter
    max_review_rounds: int = 3
    review_score_threshold: float = 4.0  # on 1-10 scale

    # Memory caps
    max_interactions: int = 50000
    max_events: int = 50000
