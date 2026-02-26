"""Observable adapter for external agent-level safety evaluations.

Bridges agent-level eval frameworks (HAICosystem, OpenAgentSafety, or
custom harnesses) into SWARM's population-level simulation by converting
multi-turn evaluation traces into ProxyObservables.

This is Piece 1 of the agent-level → population-level safety bridge:

    Eval traces → EvalTraceObservableGenerator → ProxyObservables
                                                       ↓
                                                 ProxyComputer → p

The generator implements the ObservableGenerator protocol and can be
used as a drop-in replacement for DefaultObservableGenerator.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, field_validator

from swarm.core.proxy import ProxyObservables
from swarm.env.state import EnvState, InteractionProposal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval trace data model (framework-agnostic)
# ---------------------------------------------------------------------------


class EvalFramework(Enum):
    """Supported evaluation framework sources."""

    HAICOSYSTEM = "haicosystem"
    OPENAGENTSAFETY = "openagentsafety"
    CUSTOM = "custom"


class TurnType(Enum):
    """Type of a single turn in a multi-turn episode."""

    AGENT_ACTION = "agent_action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USER_MESSAGE = "user_message"
    ENVIRONMENT_FEEDBACK = "environment_feedback"
    JUDGE_ANNOTATION = "judge_annotation"


@dataclass
class EvalTurn:
    """A single turn within a multi-turn evaluation episode."""

    turn_index: int
    turn_type: TurnType
    content: str = ""

    # Signals extractable from this turn
    task_completed: bool = False
    self_correction: bool = False
    safety_violation: bool = False
    unsafe_tool_use: bool = False
    cooperative: Optional[bool] = None  # None = not assessable

    # Optional scores from the eval framework
    safety_score: Optional[float] = None  # [0, 1] where 1 = safe
    helpfulness_score: Optional[float] = None  # [0, 1] where 1 = helpful

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalTrace:
    """A complete multi-turn evaluation episode.

    Framework-agnostic representation of an agent being evaluated across
    multiple turns.  Can be constructed from HAICosystem, OpenAgentSafety,
    or custom eval harness outputs.
    """

    trace_id: str
    agent_id: str
    framework: EvalFramework = EvalFramework.CUSTOM
    turns: List[EvalTurn] = field(default_factory=list)

    # Episode-level annotations
    task_description: str = ""
    intent_category: str = ""  # benign / ambiguous / adversarial (OAS)
    risk_domains: List[str] = field(default_factory=list)  # HAICosystem domains

    # Episode-level scores (if the eval framework provides them)
    overall_safety_score: Optional[float] = None  # [0, 1]
    overall_helpfulness_score: Optional[float] = None  # [0, 1]
    rule_based_pass: Optional[bool] = None  # OAS end-state check
    judge_safe: Optional[bool] = None  # LLM-as-judge verdict

    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Derived statistics --------------------------------------------------

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def n_agent_actions(self) -> int:
        return sum(
            1 for t in self.turns if t.turn_type == TurnType.AGENT_ACTION
        )

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for t in self.turns if t.turn_type == TurnType.TOOL_CALL)

    @property
    def n_self_corrections(self) -> int:
        return sum(1 for t in self.turns if t.self_correction)

    @property
    def n_safety_violations(self) -> int:
        return sum(1 for t in self.turns if t.safety_violation)

    @property
    def n_unsafe_tool_uses(self) -> int:
        return sum(1 for t in self.turns if t.unsafe_tool_use)

    @property
    def n_tasks_completed(self) -> int:
        return sum(1 for t in self.turns if t.task_completed)

    @property
    def cooperative_fraction(self) -> Optional[float]:
        """Fraction of assessable turns rated cooperative."""
        assessable = [t for t in self.turns if t.cooperative is not None]
        if not assessable:
            return None
        return sum(1 for t in assessable if t.cooperative) / len(assessable)


# ---------------------------------------------------------------------------
# Trace → ProxyObservables mapping
# ---------------------------------------------------------------------------


class TraceMapping(BaseModel):
    """Configuration for how eval trace signals map to SWARM observables.

    Each field controls how a particular eval trace statistic is converted
    to the corresponding SWARM observable.  Defaults are calibrated so
    that a "median safe agent" (completes most tasks, rare violations,
    generally cooperative) produces observables similar to SWARM's
    HONEST archetype (p ≈ 0.73).
    """

    # task_progress_delta mapping
    # Base: fraction of tasks completed among agent actions.
    # Scaled to [-1, +1] via: (completion_rate - 0.5) * 2 * progress_scale
    progress_scale: float = 1.0

    # rework_count mapping
    # Base: n_self_corrections.  Capped at rework_cap.
    rework_cap: int = 5

    # verifier_rejections mapping
    # Base: n_safety_violations.  Capped at rejection_cap.
    rejection_cap: int = 5

    # tool_misuse_flags mapping
    # Base: n_unsafe_tool_uses.  Capped at misuse_cap.
    misuse_cap: int = 5

    # engagement_delta mapping
    # Primary: cooperative_fraction rescaled to [-1, +1].
    # Fallback (if no cooperative annotations): helpfulness_score rescaled,
    # or 0.0 if neither available.
    engagement_fallback: float = 0.0

    # When the eval framework provides episode-level scores, blend them
    # with the turn-level derived signals.  0.0 = ignore episode scores,
    # 1.0 = use only episode scores.
    episode_score_weight: float = 0.3

    @field_validator(
        "progress_scale", "episode_score_weight",
    )
    @classmethod
    def validate_non_negative(cls, v: float, info: Any) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative, got {v}")
        return v


def trace_to_observables(
    trace: EvalTrace,
    mapping: Optional[TraceMapping] = None,
) -> ProxyObservables:
    """Convert an EvalTrace into SWARM ProxyObservables.

    This is the core mapping function.  It extracts five signals from the
    trace and converts them to the ProxyObservables format expected by
    ProxyComputer.

    Args:
        trace: A completed evaluation episode.
        mapping: Optional configuration overriding default signal mapping.

    Returns:
        ProxyObservables ready for ProxyComputer.compute_labels().
    """
    m = mapping or TraceMapping()

    # -- task_progress_delta --
    # Derived from task completion rate among agent actions.
    n_actions = max(trace.n_agent_actions, 1)
    completion_rate = trace.n_tasks_completed / n_actions
    progress_raw = (completion_rate - 0.5) * 2.0 * m.progress_scale

    # Blend with episode-level helpfulness if available.
    if trace.overall_helpfulness_score is not None:
        helpfulness_signal = (trace.overall_helpfulness_score - 0.5) * 2.0
        progress_raw = (
            (1 - m.episode_score_weight) * progress_raw
            + m.episode_score_weight * helpfulness_signal
        )

    task_progress_delta = max(-1.0, min(1.0, progress_raw))

    # -- rework_count --
    rework_count = min(trace.n_self_corrections, m.rework_cap)

    # -- verifier_rejections --
    verifier_rejections = min(trace.n_safety_violations, m.rejection_cap)

    # -- tool_misuse_flags --
    tool_misuse_flags = min(trace.n_unsafe_tool_uses, m.misuse_cap)

    # -- counterparty_engagement_delta --
    coop = trace.cooperative_fraction
    if coop is not None:
        engagement_raw = (coop - 0.5) * 2.0  # [0,1] → [-1,+1]
    elif trace.overall_helpfulness_score is not None:
        engagement_raw = (trace.overall_helpfulness_score - 0.5) * 2.0
    else:
        engagement_raw = m.engagement_fallback

    # Blend with episode-level safety score if available.
    if trace.overall_safety_score is not None:
        safety_signal = (trace.overall_safety_score - 0.5) * 2.0
        engagement_raw = (
            (1 - m.episode_score_weight) * engagement_raw
            + m.episode_score_weight * safety_signal
        )

    engagement_delta = max(-1.0, min(1.0, engagement_raw))

    return ProxyObservables(
        task_progress_delta=task_progress_delta,
        rework_count=rework_count,
        verifier_rejections=verifier_rejections,
        tool_misuse_flags=tool_misuse_flags,
        counterparty_engagement_delta=engagement_delta,
    )


# ---------------------------------------------------------------------------
# EvalTraceObservableGenerator — implements ObservableGenerator protocol
# ---------------------------------------------------------------------------


class EvalTraceObservableGenerator:
    """ObservableGenerator backed by external evaluation traces.

    Instead of generating observables from agent archetypes (like
    DefaultObservableGenerator), this generator samples from a corpus
    of pre-computed eval traces.  Each call to generate() picks a trace
    associated with the initiator's agent ID and converts it to
    ProxyObservables via the trace-to-observable mapping.

    This is the primary integration point for bridging agent-level
    evaluations (HAICosystem, OpenAgentSafety) into SWARM's
    population-level simulation.

    Usage:
        traces = parse_openagentsafety_log("eval_results.jsonl")
        gen = EvalTraceObservableGenerator(traces)
        orchestrator = Orchestrator(..., observable_generator=gen)
    """

    def __init__(
        self,
        traces: Sequence[EvalTrace],
        mapping: Optional[TraceMapping] = None,
        rng: Optional[random.Random] = None,
        default_agent_type_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize from a corpus of evaluation traces.

        Args:
            traces: Pre-parsed evaluation traces.
            mapping: Signal mapping configuration.
            rng: Random number generator for trace sampling.
            default_agent_type_map: Optional mapping from SWARM agent_id
                to eval trace agent_id, for when IDs don't align.
        """
        self._mapping = mapping or TraceMapping()
        self._rng = rng or random.Random()
        self._agent_id_map = default_agent_type_map or {}

        # Index traces by agent_id for fast lookup.
        self._traces_by_agent: Dict[str, List[EvalTrace]] = {}
        self._all_traces: List[EvalTrace] = list(traces)
        for trace in traces:
            self._traces_by_agent.setdefault(trace.agent_id, []).append(trace)

        # Pre-compute observables for each trace to avoid repeated work.
        self._observables_cache: Dict[str, ProxyObservables] = {}
        for trace in traces:
            self._observables_cache[trace.trace_id] = trace_to_observables(
                trace, self._mapping
            )

        if not traces:
            logger.warning(
                "EvalTraceObservableGenerator initialized with empty trace corpus. "
                "All generate() calls will fall back to neutral observables."
            )

    @property
    def n_traces(self) -> int:
        return len(self._all_traces)

    @property
    def agent_ids(self) -> List[str]:
        return list(self._traces_by_agent.keys())

    def generate(
        self,
        proposal: InteractionProposal,
        accepted: bool,
        state: EnvState,
    ) -> ProxyObservables:
        """Generate observables by sampling from eval trace corpus.

        Looks up the initiator's agent_id (or mapped ID) in the trace
        corpus.  If traces exist for that agent, samples one uniformly
        at random and returns the pre-computed observables.  If no
        traces match, returns neutral observables (p ≈ 0.5).

        Engagement is clamped to ≤ 0 for rejected interactions, matching
        DefaultObservableGenerator behavior.
        """
        # Resolve agent ID.
        agent_id = self._agent_id_map.get(
            proposal.initiator_id, proposal.initiator_id
        )

        # Look up traces for this agent.
        agent_traces = self._traces_by_agent.get(agent_id)

        if agent_traces:
            trace = self._rng.choice(agent_traces)
            obs = self._observables_cache[trace.trace_id]
        else:
            # Fallback: sample from entire corpus if available.
            if self._all_traces:
                trace = self._rng.choice(self._all_traces)
                obs = self._observables_cache[trace.trace_id]
            else:
                obs = ProxyObservables()  # neutral

        # Match DefaultObservableGenerator: clamp engagement if rejected.
        if not accepted:
            engagement = min(obs.counterparty_engagement_delta, 0.0)
            obs = ProxyObservables(
                task_progress_delta=obs.task_progress_delta,
                rework_count=obs.rework_count,
                verifier_rejections=obs.verifier_rejections,
                tool_misuse_flags=obs.tool_misuse_flags,
                counterparty_engagement_delta=engagement,
            )

        return obs


# ---------------------------------------------------------------------------
# Trace parsers for specific frameworks
# ---------------------------------------------------------------------------


def parse_haicosystem_episode(raw: Dict[str, Any]) -> EvalTrace:
    """Parse a single HAICosystem evaluation episode into an EvalTrace.

    Expected format (based on HAICosystem paper, Zhou et al. 2025a):
    {
        "episode_id": str,
        "agent_id": str,
        "scenario": {"domain": str, "risk_categories": [str], ...},
        "turns": [
            {
                "role": "agent" | "user" | "tool" | "environment",
                "content": str,
                "safety_annotations": {
                    "operational_risk": bool,
                    "content_risk": bool,
                    "societal_risk": bool,
                    "legal_risk": bool
                }
            },
            ...
        ],
        "evaluation": {
            "overall_safety": float,  # [0, 1]
            "overall_helpfulness": float  # [0, 1]
        }
    }
    """
    episode_id = raw.get("episode_id", "unknown")
    agent_id = raw.get("agent_id", "unknown")

    scenario = raw.get("scenario", {})
    risk_domains = scenario.get("risk_categories", [])
    task_desc = scenario.get("description", "")

    turns: List[EvalTurn] = []
    for i, turn_raw in enumerate(raw.get("turns", [])):
        role = turn_raw.get("role", "agent")
        content = turn_raw.get("content", "")

        if role == "agent":
            turn_type = TurnType.AGENT_ACTION
        elif role == "tool":
            turn_type = TurnType.TOOL_CALL
        elif role == "environment":
            turn_type = TurnType.ENVIRONMENT_FEEDBACK
        else:
            turn_type = TurnType.USER_MESSAGE

        # Extract safety annotations.
        annotations = turn_raw.get("safety_annotations", {})
        any_risk = any(annotations.values()) if annotations else False

        turns.append(
            EvalTurn(
                turn_index=i,
                turn_type=turn_type,
                content=content,
                safety_violation=any_risk,
                task_completed=(
                    turn_raw.get("task_completed", False)
                    if role == "agent"
                    else False
                ),
                self_correction=turn_raw.get("self_correction", False),
                cooperative=turn_raw.get("cooperative"),
                metadata=annotations,
            )
        )

    evaluation = raw.get("evaluation", {})
    return EvalTrace(
        trace_id=episode_id,
        agent_id=agent_id,
        framework=EvalFramework.HAICOSYSTEM,
        turns=turns,
        task_description=task_desc,
        risk_domains=risk_domains,
        overall_safety_score=evaluation.get("overall_safety"),
        overall_helpfulness_score=evaluation.get("overall_helpfulness"),
        metadata=raw.get("metadata", {}),
    )


def parse_openagentsafety_episode(raw: Dict[str, Any]) -> EvalTrace:
    """Parse a single OpenAgentSafety evaluation episode into an EvalTrace.

    Expected format (based on OpenAgentSafety, Vijayvargiya et al. 2026a):
    {
        "task_id": str,
        "agent_id": str,
        "intent": "benign" | "ambiguous" | "adversarial",
        "trajectory": [
            {
                "action_type": "shell" | "file" | "code" | "browser" | "message",
                "content": str,
                "is_unsafe": bool,
                "tool_misuse": bool
            },
            ...
        ],
        "end_state": {
            "rule_check_pass": bool,
            "judge_safe": bool,
            "judge_score": float  # [0, 1]
        },
        "task_description": str
    }
    """
    task_id = raw.get("task_id", "unknown")
    agent_id = raw.get("agent_id", "unknown")
    intent = raw.get("intent", "benign")

    turns: List[EvalTurn] = []
    for i, step in enumerate(raw.get("trajectory", [])):
        action_type = step.get("action_type", "shell")
        is_tool = action_type in ("shell", "file", "code", "browser")

        turns.append(
            EvalTurn(
                turn_index=i,
                turn_type=TurnType.TOOL_CALL if is_tool else TurnType.AGENT_ACTION,
                content=step.get("content", ""),
                safety_violation=step.get("is_unsafe", False),
                unsafe_tool_use=step.get("tool_misuse", False),
                task_completed=step.get("task_completed", False),
                self_correction=step.get("self_correction", False),
                cooperative=step.get("cooperative"),
                metadata={"action_type": action_type},
            )
        )

    end_state = raw.get("end_state", {})
    judge_score = end_state.get("judge_score")

    return EvalTrace(
        trace_id=task_id,
        agent_id=agent_id,
        framework=EvalFramework.OPENAGENTSAFETY,
        turns=turns,
        task_description=raw.get("task_description", ""),
        intent_category=intent,
        rule_based_pass=end_state.get("rule_check_pass"),
        judge_safe=end_state.get("judge_safe"),
        overall_safety_score=judge_score,
        metadata=raw.get("metadata", {}),
    )


def parse_traces(
    raw_episodes: List[Dict[str, Any]],
    framework: EvalFramework,
) -> List[EvalTrace]:
    """Parse a batch of raw evaluation episodes.

    Args:
        raw_episodes: List of raw episode dicts from the eval framework.
        framework: Which framework produced these episodes.

    Returns:
        List of parsed EvalTrace objects.

    Raises:
        ValueError: If framework is not supported.
    """
    parsers = {
        EvalFramework.HAICOSYSTEM: parse_haicosystem_episode,
        EvalFramework.OPENAGENTSAFETY: parse_openagentsafety_episode,
    }

    parser = parsers.get(framework)
    if parser is None:
        raise ValueError(
            f"No parser for framework {framework.value}. "
            f"Supported: {[f.value for f in parsers]}"
        )

    traces = []
    for ep in raw_episodes:
        try:
            traces.append(parser(ep))
        except Exception:
            ep_id = "unknown"
            if isinstance(ep, dict):
                ep_id = ep.get("episode_id") or ep.get("task_id") or "unknown"
            logger.warning(
                "Failed to parse episode %s, skipping",
                ep_id,
                exc_info=True,
            )

    return traces
