"""High-level orchestrator for hodoscope trajectory analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from swarm.analysis.aggregation import SimulationHistory
from swarm.bridges.hodoscope.config import HodoscopeConfig
from swarm.bridges.hodoscope.mapper import HodoscopeMapper


class HodoscopeBridge:
    """Orchestrates the full pipeline from SWARM data to hodoscope output.

    Provides high-level methods that chain the mapper with hodoscope's
    analyze/visualize/sample APIs.  The mapper itself requires no
    external dependencies; only ``analyze``, ``visualize``, and ``sample``
    need the ``hodoscope`` package installed.
    """

    def __init__(self, config: Optional[HodoscopeConfig] = None) -> None:
        self.config = config or HodoscopeConfig()
        self.mapper = HodoscopeMapper(self.config)

    def analyze_history(self, history: SimulationHistory) -> Path:
        """Full pipeline: history -> trajectory dir -> hodoscope analyze.

        Args:
            history: Complete simulation history with interactions.

        Returns:
            Path to the generated ``.hodoscope.json`` file.

        Raises:
            ImportError: If ``hodoscope`` is not installed.
        """
        from hodoscope import analyze  # type: ignore[import-untyped]

        trajectories = self.mapper.history_to_trajectories(history)
        traj_dir = self.mapper.write_trajectory_dir(trajectories)

        result_path = analyze(
            str(traj_dir),
            output=str(self.config.output_dir / ".hodoscope.json"),
            summarize_model=self.config.summarize_model,
            embedding_model=self.config.embedding_model,
            projection=self.config.projection,
        )
        return Path(result_path)

    def analyze_event_log(self, event_log_path: Path) -> Path:
        """Load a JSONL event log, reconstruct interactions, and analyze.

        Args:
            event_log_path: Path to the JSONL event log file.

        Returns:
            Path to the generated ``.hodoscope.json`` file.

        Raises:
            ImportError: If ``hodoscope`` is not installed.
        """
        from hodoscope import analyze  # type: ignore[import-untyped]

        from swarm.logging.event_log import EventLog

        event_log = EventLog(path=event_log_path)
        interactions = event_log.to_interactions()

        trajectories = self.mapper.interactions_to_trajectories(interactions, {})
        traj_dir = self.mapper.write_trajectory_dir(trajectories)

        result_path = analyze(
            str(traj_dir),
            output=str(self.config.output_dir / ".hodoscope.json"),
            summarize_model=self.config.summarize_model,
            embedding_model=self.config.embedding_model,
            projection=self.config.projection,
        )
        return Path(result_path)

    def visualize(
        self,
        hodoscope_json: Path,
        group_by: Optional[str] = None,
        open_browser: bool = False,
    ) -> Path:
        """Generate an HTML visualization from a .hodoscope.json file.

        Args:
            hodoscope_json: Path to hodoscope analysis output.
            group_by: Metadata field to color/group by (defaults to
                config.group_by).
            open_browser: Whether to open the result in a browser.

        Returns:
            Path to the generated HTML file.

        Raises:
            ImportError: If ``hodoscope`` is not installed.
        """
        from hodoscope import (
            visualize as hodo_visualize,  # type: ignore[import-untyped]
        )

        group_field = group_by or self.config.group_by
        html_path = hodo_visualize(
            str(hodoscope_json),
            group_by=group_field,
            open_browser=open_browser,
        )
        return Path(html_path)

    def sample(
        self,
        hodoscope_json: Path,
        n: int = 5,
        group_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract representative trajectory samples from each cluster.

        Args:
            hodoscope_json: Path to hodoscope analysis output.
            n: Number of samples per group.
            group_by: Metadata field to group by (defaults to config.group_by).

        Returns:
            Dict mapping group labels to lists of sample trajectories.

        Raises:
            ImportError: If ``hodoscope`` is not installed.
        """
        from hodoscope import sample as hodo_sample  # type: ignore[import-untyped]

        group_field = group_by or self.config.group_by
        return hodo_sample(  # type: ignore[no-any-return]
            str(hodoscope_json),
            n=n,
            group_by=group_field,
        )
