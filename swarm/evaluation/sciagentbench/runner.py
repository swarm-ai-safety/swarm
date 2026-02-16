"""Batch runner for SciAgentBench tasks across topology matrix.

Executes tasks under different topology modes (shared_episode, per_agent,
per_task) with fixed seeds for reproducibility.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from swarm.evaluation.sciagentbench.models import (
    BatchResult,
    SciAgentBenchTask,
    TaskResult,
    TopologyMode,
)

logger = logging.getLogger(__name__)


class BatchRunConfig(BaseModel):
    """Configuration for batch execution.
    
    Attributes:
        topology_modes: List of topology modes to run.
        base_seed: Base random seed for reproducibility.
        agent_ids: List of agent identifiers to evaluate.
        max_parallel_tasks: Maximum number of parallel task executions.
        verbose: Enable detailed logging.
    """

    topology_modes: List[TopologyMode] = Field(
        default_factory=lambda: [
            TopologyMode.SHARED_EPISODE,
            TopologyMode.PER_AGENT,
            TopologyMode.PER_TASK,
        ]
    )
    base_seed: int = 42
    agent_ids: List[str] = Field(default_factory=lambda: ["default_agent"])
    max_parallel_tasks: int = 1
    verbose: bool = False


class SciAgentBenchRunner:
    """Batch runner for SciAgentBench tasks.
    
    Executes a matrix of (tasks × topology_modes × agents) with fixed seeds.
    
    Example:
        runner = SciAgentBenchRunner(
            config=BatchRunConfig(
                topology_modes=[TopologyMode.SHARED_EPISODE, TopologyMode.PER_AGENT],
                agent_ids=["gpt4", "claude"],
                base_seed=42,
            )
        )
        
        results = runner.run_batch(
            tasks=[task1, task2],
            executor=my_task_executor,
        )
    """

    def __init__(self, config: Optional[BatchRunConfig] = None) -> None:
        """Initialize the runner.
        
        Args:
            config: Batch execution configuration.
        """
        self.config = config or BatchRunConfig()
        self._rng = np.random.default_rng(self.config.base_seed)

    def run_batch(
        self,
        tasks: List[SciAgentBenchTask],
        executor: Callable[[SciAgentBenchTask, str, int], TaskResult],
    ) -> Dict[TopologyMode, BatchResult]:
        """Run tasks across all configured topology modes.
        
        Args:
            tasks: List of tasks to execute.
            executor: Callable that executes a single task.
                Signature: (task, agent_id, seed) -> TaskResult
        
        Returns:
            Dictionary mapping topology_mode to BatchResult.
        """
        results: Dict[TopologyMode, BatchResult] = {}
        
        for topology_mode in self.config.topology_modes:
            logger.info(
                f"Running batch with topology={topology_mode.value}, "
                f"{len(tasks)} tasks, {len(self.config.agent_ids)} agents"
            )
            
            batch_result = self._run_topology_batch(
                tasks=tasks,
                topology_mode=topology_mode,
                executor=executor,
            )
            results[topology_mode] = batch_result
            
            if self.config.verbose:
                logger.info(
                    f"  {topology_mode.value}: "
                    f"{batch_result.successful_tasks}/{batch_result.total_tasks} "
                    f"successful, {batch_result.total_execution_time:.2f}s"
                )
        
        return results

    def _run_topology_batch(
        self,
        tasks: List[SciAgentBenchTask],
        topology_mode: TopologyMode,
        executor: Callable[[SciAgentBenchTask, str, int], TaskResult],
    ) -> BatchResult:
        """Execute tasks under a specific topology mode.
        
        Args:
            tasks: Tasks to execute.
            topology_mode: Execution topology.
            executor: Task executor function.
        
        Returns:
            BatchResult with all task outcomes.
        """
        start_time = time.monotonic()
        task_results: List[TaskResult] = []
        seeds = self._generate_seeds(topology_mode, len(tasks), len(self.config.agent_ids))
        
        if topology_mode == TopologyMode.SHARED_EPISODE:
            # All agents share the same episode - use one seed per task
            task_results = self._run_shared_episode(tasks, executor, seeds)
        elif topology_mode == TopologyMode.PER_AGENT:
            # Each agent gets isolated episode - use seed per (task, agent)
            task_results = self._run_per_agent(tasks, executor, seeds)
        elif topology_mode == TopologyMode.PER_TASK:
            # Each task gets own episode - use seed per task
            task_results = self._run_per_task(tasks, executor, seeds)
        
        total_time = time.monotonic() - start_time
        successful = sum(1 for r in task_results if r.success)
        
        # Compute summary metrics
        summary_metrics = self._compute_summary_metrics(task_results)
        
        return BatchResult(
            topology_mode=topology_mode,
            task_results=task_results,
            total_tasks=len(task_results),
            successful_tasks=successful,
            total_execution_time=total_time,
            seeds=seeds,
            summary_metrics=summary_metrics,
        )

    def _run_shared_episode(
        self,
        tasks: List[SciAgentBenchTask],
        executor: Callable[[SciAgentBenchTask, str, int], TaskResult],
        seeds: List[int],
    ) -> List[TaskResult]:
        """Execute tasks with all agents in shared episode.
        
        In shared_episode mode, all agents see the same environment state.
        Each task gets one seed, shared by all agents for that task.
        """
        results: List[TaskResult] = []
        
        for i, task in enumerate(tasks):
            seed = seeds[i]
            if self.config.verbose:
                logger.info(f"  Task {task.task_id} (seed={seed})")
            
            # All agents share this seed for this task
            for agent_id in self.config.agent_ids:
                result = executor(task, agent_id, seed)
                results.append(result)
        
        return results

    def _run_per_agent(
        self,
        tasks: List[SciAgentBenchTask],
        executor: Callable[[SciAgentBenchTask, str, int], TaskResult],
        seeds: List[int],
    ) -> List[TaskResult]:
        """Execute tasks with each agent in isolated episode.
        
        In per_agent mode, each agent gets its own environment instance.
        Each (task, agent) pair gets a unique seed.
        """
        results: List[TaskResult] = []
        seed_idx = 0
        
        for task in tasks:
            for agent_id in self.config.agent_ids:
                seed = seeds[seed_idx]
                seed_idx += 1
                
                if self.config.verbose:
                    logger.info(f"  Task {task.task_id}, agent={agent_id}, seed={seed}")
                
                result = executor(task, agent_id, seed)
                results.append(result)
        
        return results

    def _run_per_task(
        self,
        tasks: List[SciAgentBenchTask],
        executor: Callable[[SciAgentBenchTask, str, int], TaskResult],
        seeds: List[int],
    ) -> List[TaskResult]:
        """Execute tasks with one episode per task.
        
        In per_task mode, each task gets its own episode, and agents
        run sequentially within that episode. Each task gets one seed.
        """
        results: List[TaskResult] = []
        
        for i, task in enumerate(tasks):
            seed = seeds[i]
            if self.config.verbose:
                logger.info(f"  Task {task.task_id} (seed={seed})")
            
            # Agents run sequentially in this task's episode
            for agent_id in self.config.agent_ids:
                result = executor(task, agent_id, seed)
                results.append(result)
        
        return results

    def _generate_seeds(
        self,
        topology_mode: TopologyMode,
        n_tasks: int,
        n_agents: int,
    ) -> List[int]:
        """Generate seeds based on topology mode.
        
        Args:
            topology_mode: Execution topology.
            n_tasks: Number of tasks.
            n_agents: Number of agents.
        
        Returns:
            List of seeds appropriate for the topology mode.
        """
        if topology_mode == TopologyMode.SHARED_EPISODE:
            # One seed per task
            n_seeds = n_tasks
        elif topology_mode == TopologyMode.PER_AGENT:
            # One seed per (task, agent) pair
            n_seeds = n_tasks * n_agents
        elif topology_mode == TopologyMode.PER_TASK:
            # One seed per task
            n_seeds = n_tasks
        else:
            raise ValueError(f"Unknown topology mode: {topology_mode}")
        
        # Generate deterministic seeds from base seed
        return [int(self._rng.integers(0, 2**31)) for _ in range(n_seeds)]

    def _compute_summary_metrics(
        self, task_results: List[TaskResult]
    ) -> Dict[str, float]:
        """Compute aggregate metrics from task results.
        
        Args:
            task_results: Individual task outcomes.
        
        Returns:
            Dictionary of summary metrics.
        """
        if not task_results:
            return {}
        
        success_count = sum(1 for r in task_results if r.success)
        total_time = sum(r.execution_time for r in task_results)
        
        return {
            "success_rate": success_count / len(task_results),
            "total_execution_time": total_time,
            "avg_execution_time": total_time / len(task_results),
            "min_execution_time": min(r.execution_time for r in task_results),
            "max_execution_time": max(r.execution_time for r in task_results),
        }
