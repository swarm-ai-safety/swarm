"""Proxy computer for v_hat computation from downstream observables."""

from typing import Any, Optional

from pydantic import BaseModel

from swarm.core.sigmoid import calibrated_sigmoid


class ProxyWeights(BaseModel):
    """
    Weights for combining downstream proxy signals into v_hat.

    Default values based on specification:
    - task_progress: 0.4 (primary signal)
    - rework_penalty: 0.2 (quality signal)
    - verifier_penalty: 0.2 (safety signal)
    - engagement_signal: 0.2 (counterparty response)
    """

    task_progress: float = 0.4
    rework_penalty: float = 0.2
    verifier_penalty: float = 0.2
    engagement_signal: float = 0.2

    model_config = {"frozen": False}

    def normalize(self) -> "ProxyWeights":
        """Return normalized weights that sum to 1."""
        total = (
            abs(self.task_progress)
            + abs(self.rework_penalty)
            + abs(self.verifier_penalty)
            + abs(self.engagement_signal)
        )

        if total == 0:
            return ProxyWeights(
                task_progress=0.25,
                rework_penalty=0.25,
                verifier_penalty=0.25,
                engagement_signal=0.25,
            )

        return ProxyWeights(
            task_progress=self.task_progress / total,
            rework_penalty=self.rework_penalty / total,
            verifier_penalty=self.verifier_penalty / total,
            engagement_signal=self.engagement_signal / total,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = self.model_dump()
        return result


class ProxyObservables(BaseModel):
    """
    Raw observable signals from an interaction.

    These are transformed into the v_hat proxy score.
    """

    task_progress_delta: float = 0.0  # [-1, +1] normalized progress
    rework_count: int = 0  # number of rework cycles
    verifier_rejections: int = 0  # number of verifier rejections
    tool_misuse_flags: int = 0  # number of tool misuse flags
    counterparty_engagement_delta: float = 0.0  # [-1, +1] engagement change

    model_config = {"frozen": False}

    @classmethod
    def from_interaction(cls, interaction: Any) -> "ProxyObservables":
        """Extract observables from a SoftInteraction."""
        return cls(
            task_progress_delta=interaction.task_progress_delta,
            rework_count=interaction.rework_count,
            verifier_rejections=interaction.verifier_rejections,
            tool_misuse_flags=interaction.tool_misuse_flags,
            counterparty_engagement_delta=interaction.counterparty_engagement_delta,
        )


class ProxyComputer:
    """
    Computes v_hat proxy scores from downstream observables.

    v_hat is a weighted combination of normalized signals, resulting
    in a score in [-1, +1]. This is then converted to p via calibrated sigmoid.
    """

    def __init__(
        self,
        weights: Optional[ProxyWeights] = None,
        sigmoid_k: float = 2.0,
        rework_decay: float = 0.3,
        rejection_decay: float = 0.4,
        misuse_decay: float = 0.5,
    ):
        """
        Initialize proxy computer.

        Args:
            weights: Weights for combining signals (default: ProxyWeights())
            sigmoid_k: Calibration sharpness for p computation
            rework_decay: Decay factor per rework cycle
            rejection_decay: Decay factor per verifier rejection
            misuse_decay: Decay factor per tool misuse flag
        """
        self.weights = (weights or ProxyWeights()).normalize()
        self.sigmoid_k = sigmoid_k
        self.rework_decay = rework_decay
        self.rejection_decay = rejection_decay
        self.misuse_decay = misuse_decay

    def _normalize_progress(self, delta: float) -> float:
        """Normalize progress delta to [-1, +1]."""
        return max(-1.0, min(1.0, delta))

    def _compute_rework_signal(self, count: int) -> float:
        """
        Convert rework count to signal in [-1, +1].

        More rework cycles = more negative signal.
        Uses exponential decay: signal = 1 - 2 * (1 - decay^count)
        """
        if count == 0:
            return 1.0  # No rework = positive signal

        # Exponential decay towards -1
        decay_factor = self.rework_decay**count
        return 2.0 * decay_factor - 1.0

    def _compute_rejection_signal(self, count: int) -> float:
        """
        Convert verifier rejection count to signal in [-1, +1].

        More rejections = more negative signal.
        """
        if count == 0:
            return 1.0

        decay_factor = self.rejection_decay**count
        return 2.0 * decay_factor - 1.0

    def _compute_misuse_signal(self, count: int) -> float:
        """
        Convert tool misuse flag count to signal in [-1, +1].

        Tool misuse is weighted heavily negative.
        """
        if count == 0:
            return 1.0

        decay_factor = self.misuse_decay**count
        return 2.0 * decay_factor - 1.0

    def _normalize_engagement(self, delta: float) -> float:
        """Normalize engagement delta to [-1, +1]."""
        return max(-1.0, min(1.0, delta))

    def compute_v_hat(self, observables: ProxyObservables) -> float:
        """
        Compute v_hat from downstream observables.

        Args:
            observables: Raw observable signals

        Returns:
            v_hat: Proxy score in [-1, +1]
        """
        # Compute individual signals
        progress_signal = self._normalize_progress(observables.task_progress_delta)
        rework_signal = self._compute_rework_signal(observables.rework_count)
        rejection_signal = self._compute_rejection_signal(
            observables.verifier_rejections
        )
        misuse_signal = self._compute_misuse_signal(observables.tool_misuse_flags)
        engagement_signal = self._normalize_engagement(
            observables.counterparty_engagement_delta
        )

        # Combine rejection and misuse into a single "verifier" signal
        verifier_signal = (rejection_signal + misuse_signal) / 2.0

        # Weighted combination
        v_hat = (
            self.weights.task_progress * progress_signal
            + self.weights.rework_penalty * rework_signal
            + self.weights.verifier_penalty * verifier_signal
            + self.weights.engagement_signal * engagement_signal
        )

        # Clamp to [-1, +1]
        return max(-1.0, min(1.0, v_hat))

    def compute_p(self, v_hat: float) -> float:
        """
        Convert v_hat to probability p via calibrated sigmoid.

        Args:
            v_hat: Proxy score in [-1, +1]

        Returns:
            p: P(v = +1) in [0, 1]
        """
        return calibrated_sigmoid(v_hat, self.sigmoid_k)

    def compute_labels(self, observables: ProxyObservables) -> tuple[float, float]:
        """
        Compute both v_hat and p from observables.

        Args:
            observables: Raw observable signals

        Returns:
            (v_hat, p): Proxy score and probability
        """
        v_hat = self.compute_v_hat(observables)
        p = self.compute_p(v_hat)
        return v_hat, p
