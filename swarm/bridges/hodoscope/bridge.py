"""High-level orchestrator for hodoscope trajectory analysis (v0.2 API)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from swarm.analysis.aggregation import SimulationHistory
from swarm.bridges.hodoscope.config import HodoscopeConfig
from swarm.bridges.hodoscope.mapper import HodoscopeMapper


class HodoscopeBridge:
    """Orchestrates the full pipeline from SWARM data to hodoscope output.

    Provides high-level methods that chain the mapper with hodoscope's
    analyze/visualize/sample APIs.  The mapper itself requires no
    external dependencies; only ``analyze``, ``visualize``, and ``sample``
    need the ``hodoscope`` package installed.

    Compatible with hodoscope >= 0.2.
    """

    def __init__(self, config: Optional[HodoscopeConfig] = None) -> None:
        self.config = config or HodoscopeConfig()
        self.mapper = HodoscopeMapper(self.config)

    def _make_hodoscope_config(self) -> Any:
        """Build a ``hodoscope.Config`` from our bridge config."""
        from hodoscope import Config as HodoConfig  # type: ignore[import-untyped]

        return HodoConfig(
            summarize_model=self.config.summarize_model,
            embedding_model=self.config.embedding_model,
            max_workers=self.config.max_workers,
            embed_dim=self.config.embed_dim,
        )

    def _run_analyze(self, traj_dir: Path) -> Path:
        """Run hodoscope.analyze on a trajectory directory.

        Returns:
            Path to the generated ``.hodoscope.json`` file.
        """
        from hodoscope import analyze  # type: ignore[import-untyped]

        output_path = str(self.config.output_dir / ".hodoscope.json")
        result_paths = analyze(
            (str(traj_dir),),
            output=output_path,
            config=self._make_hodoscope_config(),
            seed=self.config.seed,
        )
        return Path(result_paths[0])

    def analyze_history(self, history: SimulationHistory) -> Path:
        """Full pipeline: history -> trajectory dir -> hodoscope analyze.

        Args:
            history: Complete simulation history with interactions.

        Returns:
            Path to the generated ``.hodoscope.json`` file.

        Raises:
            ImportError: If ``hodoscope`` is not installed.
        """
        trajectories = self.mapper.history_to_trajectories(history)
        traj_dir = self.mapper.write_trajectory_dir(trajectories)
        return self._run_analyze(traj_dir)

    def analyze_event_log(self, event_log_path: Path) -> Path:
        """Load a JSONL event log, reconstruct interactions, and analyze.

        Args:
            event_log_path: Path to the JSONL event log file.

        Returns:
            Path to the generated ``.hodoscope.json`` file.

        Raises:
            ImportError: If ``hodoscope`` is not installed.
        """
        from swarm.logging.event_log import EventLog

        event_log = EventLog(path=event_log_path)
        interactions = event_log.to_interactions()

        trajectories = self.mapper.interactions_to_trajectories(interactions, {})
        traj_dir = self.mapper.write_trajectory_dir(trajectories)
        return self._run_analyze(traj_dir)

    def visualize(
        self,
        hodoscope_json: Path,
        group_by: Optional[str] = None,
        output_file: Optional[Path] = None,
        methods: Optional[List[str]] = None,
    ) -> Path:
        """Generate an HTML visualization from a .hodoscope.json file.

        Uses ``hodoscope.visualize_action_summaries`` which groups summaries
        and produces a Bokeh-based trajectory explorer.

        Args:
            hodoscope_json: Path to hodoscope analysis output.
            group_by: Metadata field to color/group by (defaults to
                config.group_by).
            output_file: Path for the output HTML file. Defaults to
                ``<output_dir>/trajectory_explorer.html``.
            methods: Projection methods to include (default: config.projection).

        Returns:
            Path to the generated HTML file.

        Raises:
            ImportError: If ``hodoscope`` is not installed.
        """
        from hodoscope import (  # type: ignore[import-untyped]
            group_summaries_from_list,
            read_analysis_json,
            visualize_action_summaries,
        )

        group_field = group_by or self.config.group_by
        proj_methods = methods or [self.config.projection]
        out_file = output_file or self.config.output_dir / "trajectory_explorer.html"

        data = read_analysis_json(hodoscope_json)
        summaries = data["summaries"]

        grouped = group_summaries_from_list(
            summaries,
            group_by=group_field,
        )

        visualize_action_summaries(
            grouped,
            output_file=str(out_file),
            methods=proj_methods,
        )
        return Path(out_file)

    def sample(
        self,
        hodoscope_json: Path,
        n: int = 5,
        group_by: Optional[str] = None,
        method: Optional[str] = None,
        filter_fn: Optional[Callable[[dict], bool]] = None,
    ) -> Dict[str, Any]:
        """Extract representative trajectory samples from each cluster.

        Uses hodoscope's FPS-based sampling to pick diverse, representative
        trajectories from each group.

        Args:
            hodoscope_json: Path to hodoscope analysis output.
            n: Number of samples per group.
            group_by: Metadata field to group by (defaults to config.group_by).
            method: Projection method for sampling (defaults to config.projection).
            filter_fn: Optional predicate to filter summaries before sampling.

        Returns:
            Dict mapping group labels to lists of sample trajectories.

        Raises:
            ImportError: If ``hodoscope`` is not installed.
        """
        from hodoscope import sample as hodo_sample  # type: ignore[import-untyped]

        group_field = group_by or self.config.group_by
        proj_method = method or self.config.projection

        return hodo_sample(  # type: ignore[no-any-return]
            (str(hodoscope_json),),
            group_by=group_field,
            n=n,
            method=proj_method,
            filter=filter_fn,
        )
