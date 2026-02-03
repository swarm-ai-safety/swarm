"""Scenario loader for YAML configuration files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

from src.agents.adversarial import AdversarialAgent
from src.agents.base import BaseAgent
from src.agents.deceptive import DeceptiveAgent
from src.agents.honest import HonestAgent
from src.agents.opportunistic import OpportunisticAgent
from src.core.orchestrator import Orchestrator, OrchestratorConfig
from src.core.payoff import PayoffConfig
from src.env.network import NetworkConfig, NetworkTopology
from src.env.state import RateLimits
from src.governance.config import GovernanceConfig

# Agent type registry for scripted agents
AGENT_TYPES: Dict[str, Type[BaseAgent]] = {
    "honest": HonestAgent,
    "opportunistic": OpportunisticAgent,
    "deceptive": DeceptiveAgent,
    "adversarial": AdversarialAgent,
}

# LLM agent support (lazy import to avoid requiring LLM dependencies)
_LLM_AGENT_CLASS = None
_LLM_CONFIG_CLASSES = None


def _get_llm_classes():
    """Lazy import LLM agent classes."""
    global _LLM_AGENT_CLASS, _LLM_CONFIG_CLASSES
    if _LLM_AGENT_CLASS is None:
        from src.agents.llm_agent import LLMAgent
        from src.agents.llm_config import LLMConfig, LLMProvider, PersonaType
        _LLM_AGENT_CLASS = LLMAgent
        _LLM_CONFIG_CLASSES = {
            "LLMConfig": LLMConfig,
            "LLMProvider": LLMProvider,
            "PersonaType": PersonaType,
        }
    return _LLM_AGENT_CLASS, _LLM_CONFIG_CLASSES


@dataclass
class ScenarioConfig:
    """Parsed scenario configuration."""

    scenario_id: str = "unnamed"
    description: str = ""
    motif: str = ""

    # Component configs
    orchestrator_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    rate_limits: RateLimits = field(default_factory=RateLimits)

    # Agent specifications
    agent_specs: List[Dict[str, Any]] = field(default_factory=list)

    # Success criteria
    success_criteria: Dict[str, Any] = field(default_factory=dict)

    # Output paths
    event_log_path: Optional[Path] = None
    metrics_csv_path: Optional[Path] = None


def parse_governance_config(data: Dict[str, Any]) -> GovernanceConfig:
    """
    Parse governance section from YAML into GovernanceConfig.

    Supports both flat keys and nested structure for clarity.

    Args:
        data: The governance section from YAML

    Returns:
        Populated GovernanceConfig
    """
    if not data:
        return GovernanceConfig()

    config = GovernanceConfig(
        # Transaction tax
        transaction_tax_rate=data.get("transaction_tax_rate", data.get("transaction_tax", 0.0)),
        transaction_tax_split=data.get("transaction_tax_split", 0.5),

        # Reputation
        reputation_decay_rate=data.get("reputation_decay_rate", data.get("reputation_decay", 1.0)),

        # Vote normalization
        vote_normalization_enabled=data.get("vote_normalization_enabled", False),
        max_vote_weight_per_agent=data.get("max_vote_weight_per_agent", 10.0),

        # Bandwidth
        bandwidth_cap=data.get("bandwidth_cap", 10),

        # Staking
        staking_enabled=data.get("staking_enabled", False),
        min_stake_to_participate=data.get("min_stake_to_participate", 0.0),
        stake_slash_rate=data.get("stake_slash_rate", 0.1),

        # Circuit breaker
        circuit_breaker_enabled=data.get("circuit_breaker_enabled", False),
        freeze_threshold_toxicity=data.get("freeze_threshold_toxicity", 0.7),
        freeze_threshold_violations=data.get("freeze_threshold_violations", 3),
        freeze_duration_epochs=data.get("freeze_duration_epochs", 2),

        # Random audit
        audit_enabled=data.get("audit_enabled", False),
        audit_probability=data.get("audit_probability", 0.1),
        audit_penalty_multiplier=data.get("audit_penalty_multiplier", 2.0),
        audit_threshold_p=data.get("audit_threshold_p", 0.5),

        # Collusion detection
        collusion_detection_enabled=data.get("collusion_detection_enabled", False),
        collusion_frequency_threshold=data.get("collusion_frequency_threshold", 2.0),
        collusion_correlation_threshold=data.get("collusion_correlation_threshold", 0.7),
        collusion_min_interactions=data.get("collusion_min_interactions", 3),
        collusion_score_threshold=data.get("collusion_score_threshold", 0.5),
        collusion_penalty_multiplier=data.get("collusion_penalty_multiplier", 1.0),
        collusion_realtime_penalty=data.get("collusion_realtime_penalty", False),
        collusion_realtime_rate=data.get("collusion_realtime_rate", 0.1),
        collusion_clear_history_on_epoch=data.get("collusion_clear_history_on_epoch", False),
    )

    config.validate()
    return config


def parse_payoff_config(data: Dict[str, Any]) -> PayoffConfig:
    """
    Parse payoff section from YAML into PayoffConfig.

    Args:
        data: The payoff section from YAML

    Returns:
        Populated PayoffConfig
    """
    if not data:
        return PayoffConfig()

    config = PayoffConfig(
        s_plus=data.get("s_plus", 2.0),
        s_minus=data.get("s_minus", 1.0),
        h=data.get("h", 2.0),
        theta=data.get("theta", 0.5),
        rho_a=data.get("rho_a", 0.0),
        rho_b=data.get("rho_b", 0.0),
        w_rep=data.get("w_rep", 1.0),
    )

    config.validate()
    return config


def parse_rate_limits(data: Dict[str, Any]) -> RateLimits:
    """
    Parse rate_limits section from YAML.

    Args:
        data: The rate_limits section from YAML

    Returns:
        Populated RateLimits
    """
    if not data:
        return RateLimits()

    return RateLimits(
        posts_per_epoch=data.get("posts_per_epoch", 10),
        interactions_per_step=data.get("interactions_per_step", 5),
        votes_per_epoch=data.get("votes_per_epoch", 50),
        tasks_per_epoch=data.get("tasks_per_epoch", 3),
    )


def parse_network_config(data: Dict[str, Any]) -> Optional[NetworkConfig]:
    """
    Parse network section from YAML into NetworkConfig.

    Args:
        data: The network section from YAML

    Returns:
        NetworkConfig if enabled, None otherwise
    """
    if not data:
        return None

    # Check if network is explicitly disabled
    if data.get("enabled") is False:
        return None

    # Parse topology
    topology_str = data.get("topology", "complete").lower()
    try:
        topology = NetworkTopology(topology_str)
    except ValueError:
        raise ValueError(f"Unknown network topology: {topology_str}")

    # Parse params (may be nested under 'params' key or flat)
    params = data.get("params", {})

    config = NetworkConfig(
        topology=topology,
        # Erdős-Rényi
        edge_probability=params.get("edge_probability", data.get("edge_probability", 0.5)),
        # Small-world
        k_neighbors=params.get("k", params.get("k_neighbors", data.get("k_neighbors", 4))),
        rewire_probability=params.get("p", params.get("rewire_probability", data.get("rewire_probability", 0.1))),
        # Scale-free
        m_edges=params.get("m", params.get("m_edges", data.get("m_edges", 2))),
        # Dynamic network
        dynamic=data.get("dynamic", False),
        edge_strengthen_rate=data.get("edge_strengthen_rate", 0.1),
        edge_decay_rate=data.get("edge_decay_rate", 0.05),
        min_edge_weight=data.get("min_edge_weight", 0.1),
        max_edge_weight=data.get("max_edge_weight", 1.0),
        # Reputation-based
        reputation_disconnect_threshold=data.get("reputation_disconnect_threshold"),
    )

    config.validate()
    return config


def load_scenario(path: Path) -> ScenarioConfig:
    """
    Load a scenario from a YAML file.

    Args:
        path: Path to the YAML scenario file

    Returns:
        Parsed ScenarioConfig

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If config validation fails
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse component configs
    governance_config = parse_governance_config(data.get("governance", {}))
    payoff_config = parse_payoff_config(data.get("payoff", {}))
    rate_limits = parse_rate_limits(data.get("rate_limits", {}))
    network_config = parse_network_config(data.get("network", {}))

    # Parse simulation settings
    sim_data = data.get("simulation", {})
    outputs_data = data.get("outputs", {})

    # Build orchestrator config
    orchestrator_config = OrchestratorConfig(
        n_epochs=sim_data.get("n_epochs", 10),
        steps_per_epoch=sim_data.get("steps_per_epoch", 10),
        seed=sim_data.get("seed"),
        payoff_config=payoff_config,
        governance_config=governance_config,
        network_config=network_config,
        log_path=Path(outputs_data["event_log"]) if outputs_data.get("event_log") else None,
        log_events=bool(outputs_data.get("event_log")),
    )

    return ScenarioConfig(
        scenario_id=data.get("scenario_id", "unnamed"),
        description=data.get("description", ""),
        motif=data.get("motif", ""),
        orchestrator_config=orchestrator_config,
        rate_limits=rate_limits,
        agent_specs=data.get("agents", []),
        success_criteria=data.get("success_criteria", {}),
        event_log_path=Path(outputs_data["event_log"]) if outputs_data.get("event_log") else None,
        metrics_csv_path=Path(outputs_data["metrics_csv"]) if outputs_data.get("metrics_csv") else None,
    )


def parse_llm_config(data: Dict[str, Any]) -> "LLMConfig":
    """
    Parse LLM configuration from YAML.

    Args:
        data: The llm section from YAML agent spec

    Returns:
        LLMConfig instance
    """
    LLMAgent, llm_classes = _get_llm_classes()
    LLMConfig = llm_classes["LLMConfig"]
    LLMProvider = llm_classes["LLMProvider"]
    PersonaType = llm_classes["PersonaType"]

    # Parse provider
    provider_str = data.get("provider", "anthropic").lower()
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        raise ValueError(f"Unknown LLM provider: {provider_str}")

    # Parse persona
    persona_str = data.get("persona", "open").lower()
    try:
        persona = PersonaType(persona_str)
    except ValueError:
        raise ValueError(f"Unknown persona type: {persona_str}")

    return LLMConfig(
        provider=provider,
        model=data.get("model", "claude-sonnet-4-20250514"),
        api_key=data.get("api_key"),  # Usually from env var
        base_url=data.get("base_url"),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 512),
        timeout=data.get("timeout", 30.0),
        max_retries=data.get("max_retries", 3),
        persona=persona,
        system_prompt=data.get("system_prompt"),
        cost_tracking=data.get("cost_tracking", True),
    )


def create_agents(agent_specs: List[Dict[str, Any]]) -> List[BaseAgent]:
    """
    Create agent instances from specifications.

    Supports both scripted agents (honest, opportunistic, etc.)
    and LLM-backed agents.

    Args:
        agent_specs: List of agent specifications from YAML

    Returns:
        List of instantiated agents
    """
    agents = []
    counters: Dict[str, int] = {}

    for spec in agent_specs:
        agent_type = spec.get("type", "honest")
        count = spec.get("count", 1)

        # Handle LLM agents
        if agent_type == "llm":
            LLMAgent, _ = _get_llm_classes()
            llm_config = parse_llm_config(spec.get("llm", {}))

            for _ in range(count):
                # Generate unique ID
                counters["llm"] = counters.get("llm", 0) + 1
                agent_id = f"llm_{counters['llm']}"

                agent = LLMAgent(
                    agent_id=agent_id,
                    llm_config=llm_config,
                )
                agents.append(agent)

        # Handle scripted agents
        elif agent_type in AGENT_TYPES:
            agent_class = AGENT_TYPES[agent_type]

            for _ in range(count):
                # Generate unique ID
                counters[agent_type] = counters.get(agent_type, 0) + 1
                agent_id = f"{agent_type}_{counters[agent_type]}"

                # Create agent with optional config
                agent = agent_class(agent_id=agent_id)
                agents.append(agent)

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    return agents


def build_orchestrator(scenario: ScenarioConfig) -> Orchestrator:
    """
    Build a fully configured Orchestrator from a scenario.

    Args:
        scenario: Parsed scenario configuration

    Returns:
        Configured Orchestrator with agents registered
    """
    # Create orchestrator
    orchestrator = Orchestrator(config=scenario.orchestrator_config)

    # Override rate limits
    orchestrator.state.rate_limits = scenario.rate_limits

    # Create and register agents
    agents = create_agents(scenario.agent_specs)
    for agent in agents:
        orchestrator.register_agent(agent)

    return orchestrator


def load_and_build(path: Path) -> Orchestrator:
    """
    Convenience function to load a scenario and build an orchestrator.

    Args:
        path: Path to YAML scenario file

    Returns:
        Ready-to-run Orchestrator
    """
    scenario = load_scenario(path)
    return build_orchestrator(scenario)
