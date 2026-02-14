"""Client for reading AgentLaboratory data.

Supports offline parsing of AgentLab state checkpoints (pickle files)
and lab output directories to reconstruct event timelines.

AgentLaboratory is an optional dependency; pickle parsing uses lazy
imports. Tests use dict fixtures without requiring AgentLab.

Security: Pickle deserialization uses a RestrictedUnpickler that only
allows known-safe classes from AgentLaboratory and Python builtins.
See ``_ALLOWED_CLASSES`` for the allowlist.
"""

import io
import logging
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from swarm.bridges.agent_lab.config import AgentLabClientConfig
from swarm.bridges.agent_lab.events import (
    AgentLabEvent,
    AgentLabEventType,
    DialogueEvent,
    ReviewEvent,
    SolverIterationEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Restricted unpickler — allowlist of (module, class) pairs
# ---------------------------------------------------------------------------

# AgentLaboratory classes that appear in Paper*.pkl checkpoints.
# The workflow object serializes itself via ``pickle.dump(self, f)``
# with no custom __reduce__, so only __dict__ reconstruction is needed.
# Checkpoints may be pickled from __main__ or from module paths.
_AGENT_LAB_CLASSES: Set[Tuple[str, str]] = {
    # Top-level workflow
    ("ai_lab_repo", "LaboratoryWorkflow"),
    ("__main__", "LaboratoryWorkflow"),
    # Agent classes (all defined in agents.py)
    ("agents", "BaseAgent"),
    ("agents", "PhDStudentAgent"),
    ("agents", "PostdocAgent"),
    ("agents", "ProfessorAgent"),
    ("agents", "MLEngineerAgent"),
    ("agents", "SWEngineerAgent"),
    ("agents", "ReviewersAgent"),
    # Same classes when imported via package path
    ("ai_lab_repo.agents", "BaseAgent"),
    ("ai_lab_repo.agents", "PhDStudentAgent"),
    ("ai_lab_repo.agents", "PostdocAgent"),
    ("ai_lab_repo.agents", "ProfessorAgent"),
    ("ai_lab_repo.agents", "MLEngineerAgent"),
    ("ai_lab_repo.agents", "SWEngineerAgent"),
    ("ai_lab_repo.agents", "ReviewersAgent"),
}

# Python builtins and stdlib helpers needed by the default pickle protocol.
_SAFE_BUILTINS: Set[Tuple[str, str]] = {
    ("builtins", "True"),
    ("builtins", "False"),
    ("builtins", "None"),
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytes"),
    ("builtins", "bytearray"),
    ("builtins", "str"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "complex"),
    ("builtins", "bool"),
    ("builtins", "type"),
    ("builtins", "object"),
    ("builtins", "slice"),
    ("builtins", "range"),
    # copyreg._reconstructor is used by pickle to rebuild objects
    # whose classes don't define __reduce__.
    ("copyreg", "_reconstructor"),
    # collections types that may appear in nested state
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
}

_ALLOWED_CLASSES: Set[Tuple[str, str]] = _AGENT_LAB_CLASSES | _SAFE_BUILTINS


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows classes on an explicit allowlist.

    Raises ``pickle.UnpicklingError`` if the pickle stream tries to
    reconstruct a class not in ``_ALLOWED_CLASSES``.  This prevents
    arbitrary code execution via crafted pickle payloads.
    """

    def __init__(
        self,
        file: io.BufferedIOBase,
        *,
        extra_allowed: Set[Tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(file)
        self._allowed = _ALLOWED_CLASSES | (extra_allowed or set())

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._allowed:
            raise pickle.UnpicklingError(
                f"Blocked unsafe class: {module}.{name}. "
                "Only allowlisted AgentLab and builtin classes are "
                "permitted during checkpoint deserialization."
            )
        return super().find_class(module, name)


def restricted_loads(
    data: bytes,
    *,
    extra_allowed: Set[Tuple[str, str]] | None = None,
) -> Any:
    """Deserialize bytes using the restricted unpickler."""
    return _RestrictedUnpickler(
        io.BytesIO(data), extra_allowed=extra_allowed
    ).load()


def restricted_load(
    file: io.BufferedIOBase,
    *,
    extra_allowed: Set[Tuple[str, str]] | None = None,
) -> Any:
    """Deserialize a file using the restricted unpickler."""
    return _RestrictedUnpickler(
        file, extra_allowed=extra_allowed
    ).load()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentLabClient:
    """Interface to AgentLaboratory data sources.

    Primary mode is **offline**: parse pickled checkpoints and lab
    output directories. Online mode (subclassing ``LaboratoryWorkflow``
    to emit events at runtime) is deferred to a follow-up.
    """

    def __init__(self, config: AgentLabClientConfig | None = None) -> None:
        self._config = config or AgentLabClientConfig()

    def parse_checkpoint(self, path: str) -> List[AgentLabEvent]:
        """Parse a pickled AgentLab checkpoint file.

        Args:
            path: Path to a Paper*.pkl checkpoint.

        Returns:
            List of events reconstructed from the checkpoint state.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = self._load_pickle(checkpoint_path)
        return self._extract_events_from_state(state)

    def parse_lab_directory(self, lab_dir: str) -> List[AgentLabEvent]:
        """Parse an AgentLab output directory for events.

        Looks for solver logs, dialogue histories, review outputs,
        and cost tracking files.

        Args:
            lab_dir: Path to a lab output directory.

        Returns:
            List of events reconstructed from the directory contents.
        """
        dir_path = Path(lab_dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {lab_dir}")

        events: List[AgentLabEvent] = []

        # Look for checkpoint files within the directory
        for pkl_file in sorted(dir_path.glob("*.pkl")):
            try:
                state = self._load_pickle(pkl_file)
                events.extend(self._extract_events_from_state(state))
            except Exception:
                logger.exception("Failed to parse checkpoint %s", pkl_file)

        return events

    def extract_solver_history(
        self, solver_state: Dict[str, Any]
    ) -> List[SolverIterationEvent]:
        """Extract solver iteration events from solver state dict.

        Args:
            solver_state: Dictionary with solver history fields:
                - solver_type: "mle" or "paper"
                - scores: List of float scores per iteration
                - repair_counts: List of int repair attempts per iteration
                - errors: List of Optional[str] error messages
                - costs: List of float costs per iteration

        Returns:
            List of SolverIterationEvent records.
        """
        solver_type = solver_state.get("solver_type", "mle")
        scores = solver_state.get("scores", [])
        repair_counts = solver_state.get("repair_counts", [])
        errors = solver_state.get("errors", [])
        costs = solver_state.get("costs", [])

        iterations: List[SolverIterationEvent] = []
        for i, score in enumerate(scores):
            iterations.append(SolverIterationEvent(
                solver_type=solver_type,
                iteration_index=i,
                score=float(score),
                repair_attempts=int(repair_counts[i]) if i < len(repair_counts) else 0,
                execution_error=errors[i] if i < len(errors) else None,
                cost_usd=float(costs[i]) if i < len(costs) else 0.0,
            ))

        return iterations

    def extract_dialogue_events(
        self, agent_history: List[Dict[str, Any]]
    ) -> List[DialogueEvent]:
        """Extract dialogue events from agent conversation history.

        Args:
            agent_history: List of dialogue turn dicts with fields:
                - speaker_role: str
                - listener_role: str
                - phase: str
                - command_type: str
                - has_submission: bool

        Returns:
            List of DialogueEvent records.
        """
        dialogues: List[DialogueEvent] = []
        for turn in agent_history:
            dialogues.append(DialogueEvent(
                speaker_role=turn.get("speaker_role", ""),
                listener_role=turn.get("listener_role", ""),
                phase=turn.get("phase", ""),
                command_type=turn.get("command_type", ""),
                has_submission=turn.get("has_submission", False),
            ))
        return dialogues

    def extract_review_events(
        self, review_output: List[Dict[str, Any]]
    ) -> List[ReviewEvent]:
        """Extract review events from reviewer output.

        Args:
            review_output: List of review dicts with fields:
                - reviewer_index: int (0-2)
                - overall_score: float (1-10)
                - soundness: float
                - contribution: float
                - presentation: float
                - decision: str
                - confidence: float

        Returns:
            List of ReviewEvent records.
        """
        reviews: List[ReviewEvent] = []
        for rev in review_output:
            reviews.append(ReviewEvent(
                reviewer_index=rev.get("reviewer_index", 0),
                overall_score=rev.get("overall_score", 0.0),
                soundness=rev.get("soundness", 0.0),
                contribution=rev.get("contribution", 0.0),
                presentation=rev.get("presentation", 0.0),
                decision=rev.get("decision", ""),
                confidence=rev.get("confidence", 0.0),
                strengths=rev.get("strengths", []),
                weaknesses=rev.get("weaknesses", []),
            ))
        return reviews

    def _load_pickle(self, path: Path) -> Dict[str, Any]:
        """Load a pickle file using restricted deserialization.

        Uses ``_RestrictedUnpickler`` which only allows classes on an
        explicit allowlist (AgentLab workflow/agent classes and Python
        builtins).  Arbitrary code execution payloads are blocked.

        Returns the unpickled object as a dict-like structure.
        If pickle loading fails (e.g. missing AgentLab dependency),
        raises an ImportError with guidance.
        """
        import sys

        # Optionally add AgentLab to path for class resolution
        if self._config.agent_lab_path:
            agent_lab_path = str(Path(self._config.agent_lab_path).resolve())
            if agent_lab_path not in sys.path:
                sys.path.insert(0, agent_lab_path)

        try:
            with open(path, "rb") as f:
                result: Dict[str, Any] = restricted_load(f)
                return result
        except pickle.UnpicklingError:
            raise  # Re-raise security blocks as-is
        except (ModuleNotFoundError, AttributeError) as exc:
            raise ImportError(
                f"Failed to load AgentLab checkpoint {path}. "
                "Ensure AgentLaboratory is installed or set "
                "agent_lab_path in AgentLabClientConfig. "
                f"Original error: {exc}"
            ) from exc

    def _extract_events_from_state(
        self, state: Any
    ) -> List[AgentLabEvent]:
        """Extract events from a loaded AgentLab state object.

        Handles both dict-like state and actual AgentLab objects
        by probing for known attributes/keys.
        """
        events: List[AgentLabEvent] = []
        now = _utcnow()

        state_dict = self._to_dict(state)

        # Extract phase information
        phases = state_dict.get("phases", state_dict.get("phase_history", []))
        if isinstance(phases, list):
            for i, phase_data in enumerate(phases):
                phase_dict = self._to_dict(phase_data)
                phase_name = phase_dict.get("name", phase_dict.get("phase", f"phase_{i}"))
                status = phase_dict.get("status", "completed")

                event_type = (
                    AgentLabEventType.PHASE_COMPLETED
                    if status == "completed"
                    else AgentLabEventType.PHASE_FAILED
                )
                events.append(AgentLabEvent(
                    event_type=event_type,
                    timestamp=now,
                    phase=str(phase_name),
                    step=i,
                    payload=phase_dict,
                ))

        # Extract solver history
        for solver_key in ("mle_solver", "paper_solver", "solver"):
            solver_data = state_dict.get(solver_key)
            if solver_data is not None:
                solver_dict = self._to_dict(solver_data)
                solver_iters = self.extract_solver_history(solver_dict)
                for si in solver_iters:
                    events.append(AgentLabEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=AgentLabEventType.SOLVER_ITERATION,
                        timestamp=now,
                        agent_role="MLEngineerAgent" if "mle" in solver_key else "PhDStudentAgent",
                        phase=solver_key,
                        step=si.iteration_index,
                        payload=si.to_dict(),
                    ))

        # Extract dialogue history
        dialogue_data = state_dict.get("dialogue_history", state_dict.get("agent_history", []))
        if isinstance(dialogue_data, list):
            dialogues = self.extract_dialogue_events(dialogue_data)
            for j, d in enumerate(dialogues):
                events.append(AgentLabEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=AgentLabEventType.DIALOGUE_EXCHANGE,
                    timestamp=now,
                    agent_role=d.speaker_role,
                    phase=d.phase,
                    step=j,
                    payload=d.to_dict(),
                ))

        # Extract review data
        review_data = state_dict.get("reviews", state_dict.get("review_output", []))
        if isinstance(review_data, list):
            reviews = self.extract_review_events(review_data)
            for r in reviews:
                events.append(AgentLabEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=AgentLabEventType.REVIEW_SUBMITTED,
                    timestamp=now,
                    agent_role=f"ReviewersAgent_{r.reviewer_index}",
                    phase="review",
                    step=r.reviewer_index,
                    payload=r.to_dict(),
                ))

        # Extract cost data
        total_cost = state_dict.get("total_cost", state_dict.get("cost_usd"))
        if total_cost is not None:
            events.append(AgentLabEvent(
                event_id=str(uuid.uuid4()),
                event_type=AgentLabEventType.COST_UPDATED,
                timestamp=now,
                payload={"total_cost_usd": float(total_cost)},
            ))

        return events

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        """Convert an object to a dict, handling various representations."""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            result: Dict[str, Any] = obj.__dict__
            return result
        if hasattr(obj, "to_dict"):
            result2: Dict[str, Any] = obj.to_dict()
            return result2
        return {}
