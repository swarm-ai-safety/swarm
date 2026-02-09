"""Calibrated sigmoid utilities for soft label computation."""

import math
from typing import Tuple


def calibrated_sigmoid(v_hat: float, k: float = 2.0) -> float:
    """
    Compute calibrated sigmoid: P(v = +1) = 1 / (1 + exp(-k * v_hat))

    Args:
        v_hat: Raw proxy score in [-1, +1]
        k: Calibration sharpness parameter (default 2.0)
            - k = 0: Always returns 0.5
            - k < 2: Soft/uncertain labels
            - k = 2: Moderate calibration
            - k > 2: Sharp/confident labels

    Returns:
        p: Probability in [0, 1]
    """
    # Clamp v_hat to avoid numerical issues
    v_hat = max(-10.0, min(10.0, v_hat))

    # Compute sigmoid
    exp_term = math.exp(-k * v_hat)
    return 1.0 / (1.0 + exp_term)


def inverse_sigmoid(p: float, k: float = 2.0) -> float:
    """
    Compute inverse sigmoid: v_hat = -ln((1-p)/p) / k

    Args:
        p: Probability in (0, 1)
        k: Calibration sharpness parameter

    Returns:
        v_hat: Raw proxy score

    Raises:
        ValueError: If p is not in (0, 1)
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    if k == 0:
        raise ValueError("k cannot be zero for inverse sigmoid")

    return -math.log((1.0 - p) / p) / k


def sigmoid_derivative(v_hat: float, k: float = 2.0) -> float:
    """
    Compute derivative of calibrated sigmoid: dp/dv_hat = k * p * (1 - p)

    Args:
        v_hat: Raw proxy score
        k: Calibration sharpness parameter

    Returns:
        Derivative value
    """
    p = calibrated_sigmoid(v_hat, k)
    return k * p * (1.0 - p)


def sigmoid_bounds(k: float = 2.0) -> Tuple[float, float]:
    """
    Compute the probability bounds for v_hat in [-1, +1].

    Args:
        k: Calibration sharpness parameter

    Returns:
        (p_min, p_max): Probability bounds at v_hat = -1 and v_hat = +1
    """
    p_min = calibrated_sigmoid(-1.0, k)
    p_max = calibrated_sigmoid(1.0, k)
    return p_min, p_max


def effective_uncertainty_band(
    k: float = 2.0, threshold: float = 0.1
) -> Tuple[float, float]:
    """
    Compute the v_hat range where p is within threshold of 0.5.

    Args:
        k: Calibration sharpness parameter
        threshold: Distance from 0.5 to consider uncertain

    Returns:
        (v_min, v_max): v_hat values where |p - 0.5| < threshold
    """
    # p = 0.5 + threshold => v_hat = inverse_sigmoid(0.5 + threshold, k)
    # p = 0.5 - threshold => v_hat = inverse_sigmoid(0.5 - threshold, k)
    v_max = inverse_sigmoid(0.5 + threshold, k)
    v_min = inverse_sigmoid(0.5 - threshold, k)
    return v_min, v_max
