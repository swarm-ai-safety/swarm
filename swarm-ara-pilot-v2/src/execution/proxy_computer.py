"""
Proxy Computer: Observable-to-Probability Conversion

Implements the first two stages of the soft-label governance pipeline:
1. Observable normalization to [-1, +1]
2. Weighted combination into proxy score v_hat
3. Sigmoid calibration to soft label p

Implementation conforms to Algorithm section (Eq. 1–3) and Architecture section.
"""

from typing import Dict, Tuple

import numpy as np


class ProxyWeights:
    """
    Weighted combination of four observable signals.

    Default configuration prioritizes task_progress (w1=0.4, double weight)
    over indirect/subjective signals (rework, rejections, engagement).

    Justification (H01): task_progress is only direct outcome measure;
    others are noisy or manipulable. Sensitivity: High (Table 9: 5–60pp
    separation shift depending on weight distribution).

    Bounds: All weights non-negative, sum to 1.0. Acceptable range:
    w1 in [0.2, 0.8] with remaining distributed among w2, w3, w4.
    """

    def __init__(
        self,
        w_task_progress: float = 0.40,
        w_rework_count: float = 0.20,
        w_verifier_rejections: float = 0.20,
        w_engagement: float = 0.20
    ):
        """Initialize proxy weights with normalization."""
        self.raw_weights = np.array([
            w_task_progress,
            w_rework_count,
            w_verifier_rejections,
            w_engagement
        ])
        self.weights = self.normalize()

    def normalize(self) -> np.ndarray:
        """
        Normalize weights to sum to 1.0.

        Returns:
            Normalized weight vector w in [0, 1]^4 with sum = 1.0
        """
        total = np.sum(self.raw_weights)
        if total <= 0:
            raise ValueError("Weights must sum to positive value")
        return self.raw_weights / total

    def is_valid(self) -> bool:
        """Check if weights satisfy bounds: all >= 0, sum = 1.0."""
        return (
            np.all(self.weights >= 0) and
            np.allclose(np.sum(self.weights), 1.0)
        )


class ProxyComputer:
    """
    Converts observable signals to normalized proxy score v_hat ∈ [-1, +1]
    and applies sigmoid calibration to soft label p ∈ [0, 1].

    Pipeline:
    1. Normalize observables: o_j in [-1, +1] for each j in {1,2,3,4}
    2. Weighted combination: v_hat = sum(w_j * o_j)
    3. Sigmoid calibration: p = 1 / (1 + exp(-k * v_hat))

    Key design choices (Architecture section):
    - Observable normalization is independent (min-max scaling per observable)
    - Bounded v_hat enables stable calibration
    - Sigmoid steepness k=2.0 balances discrimination vs overconfidence
    - Calibration assumes A2 (sigmoid successfully maps proxy to well-calibrated p)
    """

    def __init__(
        self,
        weights: ProxyWeights = None,
        sigmoid_k: float = 2.0,
        observable_bounds: Dict[str, Tuple[float, float]] = None
    ):
        """
        Initialize proxy computer with weights and calibration parameters.

        Args:
            weights: ProxyWeights object; if None, uses default (0.4, 0.2, 0.2, 0.2)
            sigmoid_k: Steepness parameter for sigmoid (k=2.0 default, H02)
            observable_bounds: Dict mapping observable name to (min, max) for normalization
        """
        self.weights = weights or ProxyWeights()

        # Validate sigmoid steepness
        if sigmoid_k <= 0 or sigmoid_k > 100:
            raise ValueError(f"sigmoid_k must be in (0, 100], got {sigmoid_k}")
        self.sigmoid_k = sigmoid_k

        # Observable bounds for normalization (scenario-dependent)
        self.observable_bounds = observable_bounds or {
            'task_progress': (0.0, 1.0),
            'rework_count': (0.0, 10.0),  # empirical upper bound
            'verifier_rejections': (0.0, 5.0),  # empirical upper bound
            'engagement': (0.0, 1.0)
        }

    def normalize_observable(
        self,
        value: float,
        observable_name: str
    ) -> float:
        """
        Normalize single observable to [-1, +1] using min-max scaling.

        Formula: norm(x) = 2 * (x - min) / (max - min) - 1

        Args:
            value: Raw observable value
            observable_name: Name of observable (for bounds lookup)

        Returns:
            Normalized value in [-1, +1]
        """
        if observable_name not in self.observable_bounds:
            raise ValueError(f"Unknown observable: {observable_name}")

        min_val, max_val = self.observable_bounds[observable_name]
        if max_val <= min_val:
            raise ValueError(f"Invalid bounds for {observable_name}: [{min_val}, {max_val}]")

        # Clamp value to bounds
        value = np.clip(value, min_val, max_val)

        # Min-max normalization to [0, 1]
        normalized_01 = (value - min_val) / (max_val - min_val)

        # Scale to [-1, +1]
        return 2 * normalized_01 - 1

    def compute_v_hat(
        self,
        task_progress: float,
        rework_count: float,
        verifier_rejections: float,
        engagement: float
    ) -> float:
        """
        Compute proxy score v_hat ∈ [-1, +1] from observables.

        Implements Eq. 1–2 from Algorithm section:
        o_j = normalize(observable_j)
        v_hat = sum(w_j * o_j)

        Args:
            task_progress: Fraction of task completed [0, 1]
            rework_count: Number of times rework requested [0, ∞)
            verifier_rejections: Number of verification failures [0, ∞)
            engagement: Responsiveness/effort measure [0, 1]

        Returns:
            Proxy score v_hat ∈ [-1, +1]
        """
        # Normalize observables
        o1 = self.normalize_observable(task_progress, 'task_progress')
        o2 = -self.normalize_observable(rework_count, 'rework_count')  # negative: rework bad
        o3 = -self.normalize_observable(verifier_rejections, 'verifier_rejections')  # negative
        o4 = self.normalize_observable(engagement, 'engagement')

        # Weighted combination
        observables = np.array([o1, o2, o3, o4])
        v_hat = np.dot(self.weights.weights, observables)

        # Ensure bounded (should be guaranteed, but clip for safety)
        return np.clip(v_hat, -1.0, 1.0)

    def compute_p(self, v_hat: float) -> float:
        """
        Apply sigmoid calibration to get soft label p ∈ [0, 1].

        Implements Eq. 3 from Algorithm section:
        p = 1 / (1 + exp(-k * v_hat))

        Boundary condition (Constraints section): p must remain in [0, 1].
        Sigmoid is strictly bounded; k=2.0 prevents overflow in exp.

        Args:
            v_hat: Proxy score in [-1, +1]

        Returns:
            Soft label p = P(v = +1) ∈ [0, 1]
        """
        if not -1.0 <= v_hat <= 1.0:
            raise ValueError(f"v_hat must be in [-1, 1], got {v_hat}")

        # Compute sigmoid: p = 1 / (1 + exp(-k * v_hat))
        # Use numerically stable version to avoid overflow
        if v_hat > 0:
            exp_neg_kv = np.exp(-self.sigmoid_k * v_hat)
            p = 1.0 / (1.0 + exp_neg_kv)
        else:
            exp_kv = np.exp(self.sigmoid_k * v_hat)
            p = exp_kv / (1.0 + exp_kv)

        # Ensure bounded (numerical safety)
        return np.clip(p, 0.0, 1.0)

    def compute_labels(
        self,
        observables: Dict[str, float]
    ) -> Dict[str, float]:
        """
        End-to-end: observables -> v_hat -> p.

        Args:
            observables: Dict with keys {task_progress, rework_count, verifier_rejections, engagement}

        Returns:
            Dict with keys {v_hat, p}
        """
        v_hat = self.compute_v_hat(
            observables['task_progress'],
            observables['rework_count'],
            observables['verifier_rejections'],
            observables['engagement']
        )
        p = self.compute_p(v_hat)

        return {
            'v_hat': v_hat,
            'p': p
        }
