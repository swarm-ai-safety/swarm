"""Event schemas for the Prime Intellect bridge."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PIEventType(Enum):
    """Event types in the Prime Intellect bridge."""

    # Environment lifecycle
    ENVIRONMENT_CREATED = "environment_created"
    ENVIRONMENT_RESET = "environment_reset"
    EPISODE_STARTED = "episode_started"
    EPISODE_COMPLETED = "episode_completed"

    # Rollout events
    STEP_COMPLETED = "step_completed"
    REWARD_COMPUTED = "reward_computed"
    OBSERVATION_GENERATED = "observation_generated"

    # Training events
    TRAINING_JOB_SUBMITTED = "training_job_submitted"
    TRAINING_JOB_COMPLETED = "training_job_completed"
    CHECKPOINT_SAVED = "checkpoint_saved"

    # Evaluation / bridge events
    MODEL_LOADED = "model_loaded"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"

    # Error
    ERROR = "error"


@dataclass
class PIEvent:
    """An event in the Prime Intellect bridge."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: PIEventType = PIEventType.STEP_COMPLETED
    timestamp: datetime = field(default_factory=_utcnow)
    agent_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PIEvent":
        try:
            event_type = PIEventType(data["event_type"])
        except (ValueError, KeyError):
            event_type = PIEventType.ERROR

        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            try:
                timestamp = datetime.fromisoformat(raw_ts)
            except ValueError:
                timestamp = _utcnow()
        else:
            timestamp = _utcnow()

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=timestamp,
            agent_id=str(data.get("agent_id", "")),
            payload=data.get("payload", {}),
        )


@dataclass
class RolloutStep:
    """A single step in an RL rollout within a SWARM simulation."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str = ""
    step_number: int = 0
    agent_id: str = ""

    # The prompt/observation given to the model
    observation: str = ""

    # The model's response (action)
    completion: str = ""

    # SWARM metrics for this step
    p: float = 0.5
    v_hat: float = 0.0
    reward: float = 0.0
    toxicity: float = 0.0
    quality_gap: float = 0.0
    welfare: float = 0.0

    # Whether the episode is done
    done: bool = False
    truncated: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "episode_id": self.episode_id,
            "step_number": self.step_number,
            "agent_id": self.agent_id,
            "observation": self.observation,
            "completion": self.completion,
            "p": self.p,
            "v_hat": self.v_hat,
            "reward": self.reward,
            "toxicity": self.toxicity,
            "quality_gap": self.quality_gap,
            "welfare": self.welfare,
            "done": self.done,
            "truncated": self.truncated,
            "metadata": self.metadata,
        }


@dataclass
class EpisodeSummary:
    """Summary of a completed RL episode."""

    episode_id: str = ""
    agent_id: str = ""
    num_steps: int = 0
    total_reward: float = 0.0
    mean_p: float = 0.5
    mean_toxicity: float = 0.0
    final_quality_gap: float = 0.0
    final_welfare: float = 0.0
    terminated: bool = False
    truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "num_steps": self.num_steps,
            "total_reward": self.total_reward,
            "mean_p": self.mean_p,
            "mean_toxicity": self.mean_toxicity,
            "final_quality_gap": self.final_quality_gap,
            "final_welfare": self.final_welfare,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "metadata": self.metadata,
        }
