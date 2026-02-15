"""Metrics system for soft label analysis."""

from swarm.metrics.horizon_eval import (
    HorizonEvalConfig,
    HorizonEvalResult,
    SystemHorizonEvaluator,
    group_by_epoch,
)
from swarm.metrics.incoherence import (
    BenchmarkPolicy,
    DecisionRecord,
    DualFailureSummary,
    IllusionGapResult,
    IncoherenceMetrics,
    IncoherenceResult,
    classify_dual_failure_modes,
    disagreement_rate,
    distributed_coherence,
    error_rate,
    illusion_delta,
    incoherence_index,
    perceived_coherence,
    summarize_incoherence_by_agent_type,
)
from swarm.metrics.moltbook_metrics import (
    captcha_effectiveness,
    challenge_pass_rate,
    content_throughput,
    karma_concentration,
    rate_limit_governance_impact,
    rate_limit_hit_rate,
    verification_latency_distribution,
    wasted_action_rate,
)
from swarm.metrics.moltipedia_metrics import (
    content_quality_trend,
    governance_effectiveness,
    pair_farming_rate,
    point_concentration,
    policy_fix_exploitation_rate,
)
from swarm.metrics.reporters import MetricsReporter
from swarm.metrics.rlm_metrics import RLMMetrics
from swarm.metrics.security import (
    SecurityAnalyzer,
    SecurityReport,
    ThreatIndicator,
    ThreatType,
)
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.metrics.time_horizons import (
    CAPABILITY_PROFILES,
    AgentCapabilityProfile,
    ComputeConstraints,
    TimeHorizonBucket,
    TimeHorizonMetrics,
)

__all__ = [
    "BenchmarkPolicy",
    "DecisionRecord",
    "DualFailureSummary",
    "IncoherenceMetrics",
    "IncoherenceResult",
    "summarize_incoherence_by_agent_type",
    "classify_dual_failure_modes",
    "disagreement_rate",
    "error_rate",
    "incoherence_index",
    "IllusionGapResult",
    "perceived_coherence",
    "distributed_coherence",
    "illusion_delta",
    "SoftMetrics",
    "MetricsReporter",
    "SecurityAnalyzer",
    "SecurityReport",
    "ThreatIndicator",
    "ThreatType",
    "point_concentration",
    "pair_farming_rate",
    "policy_fix_exploitation_rate",
    "content_quality_trend",
    "governance_effectiveness",
    "challenge_pass_rate",
    "rate_limit_hit_rate",
    "content_throughput",
    "verification_latency_distribution",
    "karma_concentration",
    "wasted_action_rate",
    "captcha_effectiveness",
    "rate_limit_governance_impact",
    # Time horizon metrics (Bradley framework)
    "AgentCapabilityProfile",
    "CAPABILITY_PROFILES",
    "ComputeConstraints",
    "TimeHorizonBucket",
    "TimeHorizonMetrics",
    # Multi-agent horizon evaluation (system-level)
    "HorizonEvalConfig",
    "HorizonEvalResult",
    "SystemHorizonEvaluator",
    "group_by_epoch",
    # RLM experiment metrics
    "RLMMetrics",
    # Rivals (Team-of-Rivals) metrics (lazy import to avoid circular dependency)
    # Use: from swarm.metrics.rivals_metrics import compute_rivals_metrics, RivalsMetrics
]


def __getattr__(name: str):
    """Lazy import for rivals metrics to avoid circular dependency."""
    if name in ("RivalsMetrics", "compute_rivals_metrics"):
        from swarm.metrics.rivals_metrics import RivalsMetrics as _RM
        from swarm.metrics.rivals_metrics import compute_rivals_metrics as _crm
        _map = {"RivalsMetrics": _RM, "compute_rivals_metrics": _crm}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
