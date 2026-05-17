"""SWARM ↔ MiroShark bridge.

Translates SWARM scenario YAMLs into MiroShark seed briefings, walks the
MiroShark backend's create / prepare / start / export lifecycle, and saves
results into a SWARM-style ``runs/<ts>_<scenario>_miroshark/`` folder.

Backend default: ``http://localhost:5001``. Set ``MIROSHARK_API_URL`` to
override.
"""

from swarm.bridges.miroshark.client import MirosharkAPIError, MirosharkClient
from swarm.bridges.miroshark.config import MirosharkConfig
from swarm.bridges.miroshark.mapper import scenario_to_briefing
from swarm.bridges.miroshark.runner import parse_briefing_only, run_scenario

__all__ = [
    "MirosharkAPIError",
    "MirosharkClient",
    "MirosharkConfig",
    "parse_briefing_only",
    "run_scenario",
    "scenario_to_briefing",
]
