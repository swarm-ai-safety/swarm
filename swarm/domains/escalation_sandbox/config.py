"""Configuration for the Escalation Sandbox domain."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Maximum allowed turns to prevent memory exhaustion
MAX_TURNS_LIMIT = 10_000

# Allowed characters for agent names/IDs (prevents prompt injection)
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_ .'-]{1,64}$")


@dataclass
class AgentConfig:
    """Configuration for a single nation-state agent."""

    agent_id: str = ""
    name: str = ""
    agent_type: str = "scripted"  # scripted | llm
    persona: str = "default"  # default | hawk | dove | tit_for_tat | random | ...
    provider: str = ""  # anthropic | openai | ollama (for LLM agents)
    model_id: str = ""  # e.g., claude-sonnet-4-20250514
    has_nuclear: bool = True
    has_second_strike: bool = True
    military_strength: float = 100.0
    economic_strength: float = 100.0
    intelligence_quality: float = 0.5
    temperature: float = 0.7  # LLM sampling temperature


@dataclass
class CrisisConfig:
    """Configuration for the crisis scenario."""

    template: str = "border_dispute"
    timeline_turns: int = 20
    initial_escalation: int = 0
    asymmetric: bool = False
    timeline_pressure: bool = False
    pressure_fog_increase: float = 0.1  # per-turn fog increase as deadline nears


@dataclass
class FogOfWarConfig:
    """Configuration for fog-of-war noise mechanics."""

    enabled: bool = True
    noise_mu: float = 0.0
    noise_sigma: float = 0.5
    positive_skew: float = 0.3  # bias toward accidental escalation
    per_level_variance_scale: float = 0.1  # higher levels = more fog
    intelligence_reduction_factor: float = 0.5  # intelligence reduces fog


@dataclass
class SignalConfig:
    """Configuration for signal-action mechanics."""

    broadcast_before_action: bool = True
    trust_decay_rate: float = 0.95
    commitment_trap_enabled: bool = True
    commitment_reputation_cost: float = 5.0  # cost for backing down from commitment


@dataclass
class GovernanceLeverConfig:
    """Configuration for governance mechanisms."""

    mad_enabled: bool = True
    mad_retaliation_probability: float = 0.9
    treaty_commitments: List[Dict[str, Any]] = field(default_factory=list)
    treaty_max_level: int = 4  # max escalation level permitted by treaty
    treaty_defection_penalty: float = 10.0
    mediation_enabled: bool = False
    mediation_de_escalation_levels: int = 2  # how many levels mediator proposes
    mediation_acceptance_bonus: float = 5.0
    back_channel_enabled: bool = False
    back_channel_fog_reduction: float = 0.5
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 7  # pause at nuclear threshold
    circuit_breaker_negotiation_turns: int = 1
    de_escalation_friction_multiplier: float = 1.0


@dataclass
class MetricsConfig:
    """Configuration for which metrics to compute and log."""

    compute_all: bool = True
    log_per_turn: bool = True
    log_per_episode: bool = True


@dataclass
class EscalationConfig:
    """Top-level configuration for the Escalation Sandbox domain."""

    agents: List[AgentConfig] = field(default_factory=lambda: [
        AgentConfig(agent_id="nation_a", name="Nation A"),
        AgentConfig(agent_id="nation_b", name="Nation B"),
    ])
    crisis: CrisisConfig = field(default_factory=CrisisConfig)
    fog_of_war: FogOfWarConfig = field(default_factory=FogOfWarConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    governance: GovernanceLeverConfig = field(default_factory=GovernanceLeverConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    seed: Optional[int] = None
    max_turns: int = 20

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EscalationConfig:
        """Parse an EscalationConfig from a YAML-sourced dict."""
        if not data:
            return cls()

        # Parse agents
        agents = []
        for a in data.get("agents", []):
            agents.append(AgentConfig(**{
                k: a[k] for k in (
                    "agent_id", "name", "agent_type", "persona", "provider",
                    "model_id", "has_nuclear", "has_second_strike",
                    "military_strength", "economic_strength",
                    "intelligence_quality", "temperature",
                ) if k in a
            }))

        # Parse crisis
        crisis_data = data.get("crisis", {})
        crisis = CrisisConfig(**{
            k: crisis_data[k] for k in (
                "template", "timeline_turns", "initial_escalation",
                "asymmetric", "timeline_pressure", "pressure_fog_increase",
            ) if k in crisis_data
        })

        # Parse fog of war
        fog_data = data.get("fog_of_war", {})
        fog = FogOfWarConfig(**{
            k: fog_data[k] for k in (
                "enabled", "noise_mu", "noise_sigma", "positive_skew",
                "per_level_variance_scale", "intelligence_reduction_factor",
            ) if k in fog_data
        })

        # Parse signals
        signal_data = data.get("signals", {})
        signals = SignalConfig(**{
            k: signal_data[k] for k in (
                "broadcast_before_action", "trust_decay_rate",
                "commitment_trap_enabled", "commitment_reputation_cost",
            ) if k in signal_data
        })

        # Parse governance
        gov_data = data.get("governance", {})
        governance = GovernanceLeverConfig(**{
            k: gov_data[k] for k in (
                "mad_enabled", "mad_retaliation_probability",
                "treaty_commitments", "treaty_max_level",
                "treaty_defection_penalty", "mediation_enabled",
                "mediation_de_escalation_levels",
                "mediation_acceptance_bonus", "back_channel_enabled",
                "back_channel_fog_reduction", "circuit_breaker_enabled",
                "circuit_breaker_threshold",
                "circuit_breaker_negotiation_turns",
                "de_escalation_friction_multiplier",
            ) if k in gov_data
        })

        # Parse metrics
        metrics_data = data.get("metrics", {})
        metrics_cfg = MetricsConfig(**{
            k: metrics_data[k] for k in (
                "compute_all", "log_per_turn", "log_per_episode",
            ) if k in metrics_data
        })

        config = cls(
            agents=agents if agents else cls().agents,
            crisis=crisis,
            fog_of_war=fog,
            signals=signals,
            governance=governance,
            metrics=metrics_cfg,
            seed=data.get("seed"),
            max_turns=min(data.get("max_turns", 20), MAX_TURNS_LIMIT),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate configuration values and sanitize agent names.

        Raises ValueError for invalid configurations.
        """
        # Sanitize agent names and IDs
        for agent in self.agents:
            if agent.agent_id and not _SAFE_NAME_RE.match(agent.agent_id):
                agent.agent_id = re.sub(r"[^a-zA-Z0-9_]", "_", agent.agent_id)[:64]
            if agent.name and not _SAFE_NAME_RE.match(agent.name):
                agent.name = re.sub(r"[^a-zA-Z0-9_ .'-]", "_", agent.name)[:64]
            agent.intelligence_quality = max(0.0, min(1.0, agent.intelligence_quality))
            agent.military_strength = max(0.0, agent.military_strength)
            agent.economic_strength = max(0.0, agent.economic_strength)

        # Validate numeric bounds
        if self.max_turns < 1:
            self.max_turns = 1
        if self.max_turns > MAX_TURNS_LIMIT:
            self.max_turns = MAX_TURNS_LIMIT

        self.fog_of_war.noise_sigma = max(0.0, self.fog_of_war.noise_sigma)
        self.fog_of_war.intelligence_reduction_factor = max(
            0.0, min(1.0, self.fog_of_war.intelligence_reduction_factor),
        )

        self.signals.trust_decay_rate = max(0.0, min(1.0, self.signals.trust_decay_rate))

        if self.governance.circuit_breaker_threshold < 1:
            self.governance.circuit_breaker_threshold = 7
        self.governance.de_escalation_friction_multiplier = max(
            0.0, self.governance.de_escalation_friction_multiplier,
        )
