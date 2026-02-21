"""SWARM-Hodoscope Bridge.

Connects SWARM simulation outputs to hodoscope's trajectory analysis
and visualization pipeline. Converts per-agent interaction sequences
into hodoscope-compatible trajectory JSON, enabling behavioral cluster
visualization across agent types and interaction quality levels.

Architecture:
    HodoscopeMapper
        └── converts SoftInteraction sequences → trajectory JSON dicts
    HodoscopeBridge
        └── orchestrates: mapper → hodoscope analyze → visualize

Requires: pip install swarm-safety[hodoscope]
"""

try:
    import hodoscope  # noqa: F401

    HODOSCOPE_AVAILABLE = True
except ImportError:
    HODOSCOPE_AVAILABLE = False

from swarm.bridges.hodoscope.bridge import HodoscopeBridge
from swarm.bridges.hodoscope.config import HodoscopeConfig
from swarm.bridges.hodoscope.mapper import HodoscopeMapper

__all__ = [
    "HODOSCOPE_AVAILABLE",
    "HodoscopeConfig",
    "HodoscopeMapper",
    "HodoscopeBridge",
]
