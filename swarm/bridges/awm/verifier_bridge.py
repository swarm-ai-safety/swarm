"""Bridge between AWM binary verification and SWARM soft labels.

AWM environments have binary pass/fail verification (did the DB end up
in the correct state?). This module converts that binary signal into
a soft probability p in [0.05, 0.95] — never exactly 0 or 1 to preserve
epistemic humility and the p in [0,1] invariant.
"""

from __future__ import annotations

from typing import Any, Dict


def binary_to_soft_p(
    passed: bool,
    confidence: float = 0.8,
) -> float:
    """Convert binary verification result to soft probability p.

    Maps:
    - passed=True  -> p = 0.5 + 0.45 * confidence  (max 0.95)
    - passed=False -> p = 0.5 - 0.45 * confidence  (min 0.05)

    The confidence parameter controls how far from 0.5 the result
    moves. At confidence=1.0, we get p=0.95 (pass) or p=0.05 (fail).
    At confidence=0.0, we always get p=0.5 (maximum uncertainty).

    Args:
        passed: Whether AWM verification passed
        confidence: How confident we are in the verification [0, 1]

    Returns:
        p in [0.05, 0.95]
    """
    confidence = max(0.0, min(1.0, confidence))

    if passed:
        p = 0.5 + 0.45 * confidence
    else:
        p = 0.5 - 0.45 * confidence

    # Clamp to [0.05, 0.95] for safety
    return max(0.05, min(0.95, p))


class AWMVerifierBridge:
    """Runs AWM verification and converts to soft labels.

    Wraps the AWM verification endpoint, handling errors gracefully
    and always producing a valid p in [0.05, 0.95].
    """

    def __init__(self, confidence: float = 0.8) -> None:
        self.confidence = confidence

    def verify_and_score(
        self,
        verification_result: Dict[str, Any],
    ) -> float:
        """Convert an AWM verification result dict to soft p.

        Args:
            verification_result: Dict with at least a 'passed' key

        Returns:
            p in [0.05, 0.95]
        """
        passed = bool(verification_result.get("passed", False))

        # Use per-result confidence if available, else default
        conf = float(
            verification_result.get("confidence", self.confidence)
        )

        return binary_to_soft_p(passed, conf)
