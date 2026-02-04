"""Semi-permeable boundaries for sandbox-external world interactions.

This module models the boundary between the sandbox and external world,
tracking information flow, enforcing crossing policies, and detecting
potential leakage of sensitive data.
"""

from src.boundaries.external_world import (
    ExternalEntity,
    ExternalEntityType,
    ExternalService,
    ExternalDataSource,
    ExternalWorld,
)
from src.boundaries.information_flow import (
    FlowDirection,
    FlowType,
    InformationFlow,
    FlowTracker,
    FlowSummary,
)
from src.boundaries.policies import (
    CrossingDecision,
    BoundaryPolicy,
    RateLimitPolicy,
    ContentFilterPolicy,
    SensitivityPolicy,
    CompositePolicy,
    PolicyEngine,
)
from src.boundaries.leakage import (
    LeakageType,
    LeakageEvent,
    LeakageDetector,
    LeakageReport,
)

__all__ = [
    # External world
    "ExternalEntity",
    "ExternalEntityType",
    "ExternalService",
    "ExternalDataSource",
    "ExternalWorld",
    # Information flow
    "FlowDirection",
    "FlowType",
    "InformationFlow",
    "FlowTracker",
    "FlowSummary",
    # Policies
    "CrossingDecision",
    "BoundaryPolicy",
    "RateLimitPolicy",
    "ContentFilterPolicy",
    "SensitivityPolicy",
    "CompositePolicy",
    "PolicyEngine",
    # Leakage detection
    "LeakageType",
    "LeakageEvent",
    "LeakageDetector",
    "LeakageReport",
]
