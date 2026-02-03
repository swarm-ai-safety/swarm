"""Core computation modules for payoff and proxy calculations."""

from src.core.payoff import PayoffConfig, SoftPayoffEngine
from src.core.proxy import ProxyComputer, ProxyObservables, ProxyWeights
from src.core.sigmoid import calibrated_sigmoid, inverse_sigmoid

__all__ = [
    "calibrated_sigmoid",
    "inverse_sigmoid",
    "ProxyComputer",
    "ProxyWeights",
    "ProxyObservables",
    "SoftPayoffEngine",
    "PayoffConfig",
]
