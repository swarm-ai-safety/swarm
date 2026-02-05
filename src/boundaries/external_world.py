"""External world simulation for sandbox boundary interactions.

This module models the external environment that agents in the sandbox
may interact with, including external services, data sources, and entities.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExternalEntityType(Enum):
    """Types of external entities."""

    SERVICE = "service"  # API, web service
    DATA_SOURCE = "data_source"  # Database, file system
    HUMAN = "human"  # External human user
    AGENT = "agent"  # External AI agent
    UNKNOWN = "unknown"  # Unclassified entity


@dataclass
class ExternalEntity:
    """Base class for external entities."""

    entity_id: str
    name: str
    entity_type: ExternalEntityType = ExternalEntityType.UNKNOWN
    trust_level: float = 0.5  # 0 = untrusted, 1 = fully trusted
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trust level."""
        self.trust_level = max(0.0, min(1.0, self.trust_level))


@dataclass
class ExternalService(ExternalEntity):
    """An external service (API, web service, etc.)."""

    endpoint: str = ""
    rate_limit: int = 100  # requests per epoch
    latency_ms: float = 50.0  # average latency
    reliability: float = 0.99  # probability of successful response
    requires_auth: bool = False
    sensitive_data: bool = False  # whether responses contain sensitive data

    def __post_init__(self):
        super().__post_init__()
        self.entity_type = ExternalEntityType.SERVICE

    def call(self, request: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Simulate a service call."""
        rng = rng or random.Random()

        # Check reliability
        if rng.random() > self.reliability:
            return {
                "success": False,
                "error": "Service unavailable",
                "latency_ms": self.latency_ms * 2,
            }

        # Simulate response
        return {
            "success": True,
            "data": self._generate_response(request, rng),
            "latency_ms": self.latency_ms * (0.5 + rng.random()),
            "sensitive": self.sensitive_data,
        }

    def _generate_response(self, request: Dict[str, Any], rng: random.Random) -> Any:
        """Generate a simulated response."""
        # Override in subclasses for specific behavior
        return {"response": "ok", "request_id": rng.randint(1000, 9999)}


@dataclass
class ExternalDataSource(ExternalEntity):
    """An external data source (database, file system, etc.)."""

    data_type: str = "generic"
    size_bytes: int = 0
    sensitivity_level: float = 0.0  # 0 = public, 1 = highly sensitive
    freshness_hours: float = 24.0  # how old the data is
    access_cost: float = 0.0  # cost per access

    def __post_init__(self):
        super().__post_init__()
        self.entity_type = ExternalEntityType.DATA_SOURCE

    def query(
        self,
        query: str,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Simulate a data query."""
        rng = rng or random.Random()

        # Simulate query results
        return {
            "success": True,
            "rows": rng.randint(0, 100),
            "data_type": self.data_type,
            "sensitivity": self.sensitivity_level,
            "cost": self.access_cost,
        }


@dataclass
class ExternalWorld:
    """Simulates the external world outside the sandbox.

    Manages external entities and tracks interactions with them.
    """

    entities: Dict[str, ExternalEntity] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    total_outbound_calls: int = 0
    total_inbound_data: int = 0
    blocked_attempts: int = 0

    # Configuration
    default_trust_level: float = 0.3
    max_interactions_per_epoch: int = 1000

    def add_entity(self, entity: ExternalEntity) -> None:
        """Register an external entity."""
        self.entities[entity.entity_id] = entity

    def remove_entity(self, entity_id: str) -> None:
        """Remove an external entity."""
        self.entities.pop(entity_id, None)

    def get_entity(self, entity_id: str) -> Optional[ExternalEntity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def list_entities(
        self,
        entity_type: Optional[ExternalEntityType] = None,
        min_trust: float = 0.0,
    ) -> List[ExternalEntity]:
        """List entities matching criteria."""
        entities = list(self.entities.values())

        if entity_type is not None:
            entities = [e for e in entities if e.entity_type == entity_type]

        entities = [e for e in entities if e.trust_level >= min_trust]

        return entities

    def interact(
        self,
        agent_id: str,
        entity_id: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Execute an interaction with an external entity.

        Args:
            agent_id: The sandbox agent initiating the interaction
            entity_id: The external entity to interact with
            action: The action type (call, query, send, etc.)
            payload: Optional data to send
            rng: Random number generator for reproducibility

        Returns:
            Result of the interaction
        """
        rng = rng or random.Random()

        entity = self.entities.get(entity_id)
        if entity is None:
            return {
                "success": False,
                "error": f"Unknown entity: {entity_id}",
            }

        # Record the interaction
        interaction_record: Dict[str, Any] = {
            "agent_id": agent_id,
            "entity_id": entity_id,
            "entity_type": entity.entity_type.value,
            "action": action,
            "payload_size": len(str(payload)) if payload else 0,
            "trust_level": entity.trust_level,
        }

        # Execute based on entity type
        result: Dict[str, Any]
        if isinstance(entity, ExternalService):
            result = entity.call(payload or {}, rng)
        elif isinstance(entity, ExternalDataSource):
            result = entity.query(payload.get("query", "") if payload else "", rng)
        else:
            result = {"success": True, "data": None}

        # Update tracking
        self.total_outbound_calls += 1
        if result.get("success"):
            self.total_inbound_data += len(str(result.get("data", "")))

        interaction_record["result"] = result
        self.interaction_history.append(interaction_record)

        return result

    def get_interaction_stats(self) -> Dict[str, Any]:
        """Get statistics about external interactions."""
        if not self.interaction_history:
            return {
                "total_interactions": 0,
                "by_entity_type": {},
                "by_action": {},
                "success_rate": 0.0,
                "avg_trust_level": 0.0,
            }

        by_type: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        successes = 0
        total_trust = 0.0

        for interaction in self.interaction_history:
            entity_type = interaction["entity_type"]
            action = interaction["action"]

            by_type[entity_type] = by_type.get(entity_type, 0) + 1
            by_action[action] = by_action.get(action, 0) + 1

            if interaction.get("result", {}).get("success"):
                successes += 1
            total_trust += interaction.get("trust_level", 0)

        return {
            "total_interactions": len(self.interaction_history),
            "by_entity_type": by_type,
            "by_action": by_action,
            "success_rate": successes / len(self.interaction_history),
            "avg_trust_level": total_trust / len(self.interaction_history),
            "total_outbound_calls": self.total_outbound_calls,
            "total_inbound_data": self.total_inbound_data,
            "blocked_attempts": self.blocked_attempts,
        }

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch counters."""
        self.total_outbound_calls = 0
        self.total_inbound_data = 0
        self.blocked_attempts = 0

    def create_default_world(self) -> "ExternalWorld":
        """Create a world with default external entities."""
        # Add some default services
        self.add_entity(ExternalService(
            entity_id="web_search",
            name="Web Search API",
            endpoint="https://api.search.example.com",
            trust_level=0.7,
            rate_limit=50,
            latency_ms=100,
            reliability=0.98,
        ))

        self.add_entity(ExternalService(
            entity_id="code_repo",
            name="Code Repository API",
            endpoint="https://api.code.example.com",
            trust_level=0.8,
            rate_limit=100,
            latency_ms=50,
            reliability=0.99,
            requires_auth=True,
        ))

        self.add_entity(ExternalService(
            entity_id="external_llm",
            name="External LLM API",
            endpoint="https://api.llm.example.com",
            trust_level=0.5,
            rate_limit=20,
            latency_ms=500,
            reliability=0.95,
            sensitive_data=True,
        ))

        # Add some data sources
        self.add_entity(ExternalDataSource(
            entity_id="public_data",
            name="Public Dataset",
            data_type="structured",
            trust_level=0.9,
            sensitivity_level=0.0,
            size_bytes=1_000_000,
        ))

        self.add_entity(ExternalDataSource(
            entity_id="private_data",
            name="Private Database",
            data_type="structured",
            trust_level=0.6,
            sensitivity_level=0.8,
            size_bytes=10_000_000,
            access_cost=0.01,
        ))

        return self
