"""Calibration study modules.

Arm A — proxy fidelity: measures how well ProxyComputer recovers latent p
across a controlled grid of outcomes and sweeps the sigmoid k parameter.

Pre-registration: docs/research/calibration-prereg.md
"""

from swarm.calibration.fidelity import (
    BinStats,
    FidelityReport,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    sample_observables,
    sweep_sigmoid_k,
)

__all__ = [
    "BinStats",
    "FidelityReport",
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_bins",
    "sample_observables",
    "sweep_sigmoid_k",
]
