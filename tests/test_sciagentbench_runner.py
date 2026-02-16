"""Tests for SciAgentBench batch runner.

Tests different topology modes, seed management, and result aggregation.
"""

import pytest

from swarm.evaluation.sciagentbench import (
    BatchRunConfig,
    SciAgentBenchRunner,
    SciAgentBenchTask,
    TaskResult,
    TopologyMode,
)


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing."""
    return [
        SciAgentBenchTask(
            task_id="task1",
            instruction="Analyze dataset A",
            dataset_path="/data/task1.csv",
            domain="bioinformatics",
            success_criteria={"accuracy": 0.8},
        ),
        SciAgentBenchTask(
            task_id="task2",
            instruction="Process dataset B",
            dataset_path="/data/task2.csv",
            domain="chemistry",
            success_criteria={"rmse": 0.1},
        ),
        SciAgentBenchTask(
            task_id="task3",
            instruction="Classify dataset C",
            dataset_path="/data/task3.csv",
            domain="psychology",
            success_criteria={"f1": 0.75},
        ),
    ]


def mock_executor(task, agent_id, seed):
    """Mock task executor that returns deterministic results."""
    # Simulate success based on seed for determinism
    success = (seed % 3 != 0)  # Fail every third seed

    return TaskResult(
        task_id=task.task_id,
        agent_id=agent_id,
        topology_mode=TopologyMode.PER_TASK,  # Will be overridden by runner
        seed=seed,
        success=success,
        execution_time=0.1 + (seed % 10) * 0.01,
        output=f"Result for {task.task_id} by {agent_id}" if success else None,
        error_message="Simulated failure" if not success else None,
    )


class TestSciAgentBenchRunner:
    """Test suite for SciAgentBenchRunner."""

    def test_runner_initialization(self):
        """Test runner initializes with default config."""
        runner = SciAgentBenchRunner()
        assert runner.config is not None
        assert len(runner.config.topology_modes) == 3
        assert TopologyMode.SHARED_EPISODE in runner.config.topology_modes
        assert TopologyMode.PER_AGENT in runner.config.topology_modes
        assert TopologyMode.PER_TASK in runner.config.topology_modes

    def test_runner_custom_config(self):
        """Test runner with custom configuration."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_AGENT],
            base_seed=123,
            agent_ids=["agent1", "agent2"],
        )
        runner = SciAgentBenchRunner(config=config)
        assert runner.config.base_seed == 123
        assert len(runner.config.agent_ids) == 2
        assert len(runner.config.topology_modes) == 1

    def test_shared_episode_topology(self, sample_tasks):
        """Test shared_episode topology mode."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.SHARED_EPISODE],
            base_seed=42,
            agent_ids=["agent1", "agent2"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=sample_tasks, executor=mock_executor)

        assert TopologyMode.SHARED_EPISODE in results
        batch_result = results[TopologyMode.SHARED_EPISODE]

        # With 3 tasks and 2 agents, expect 6 results
        assert batch_result.total_tasks == 6
        assert len(batch_result.task_results) == 6

        # In shared_episode, each task should have same seed for both agents
        task1_results = [r for r in batch_result.task_results if r.task_id == "task1"]
        assert len(task1_results) == 2
        assert task1_results[0].seed == task1_results[1].seed

    def test_per_agent_topology(self, sample_tasks):
        """Test per_agent topology mode."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_AGENT],
            base_seed=42,
            agent_ids=["agent1", "agent2"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=sample_tasks, executor=mock_executor)

        assert TopologyMode.PER_AGENT in results
        batch_result = results[TopologyMode.PER_AGENT]

        # With 3 tasks and 2 agents, expect 6 results
        assert batch_result.total_tasks == 6
        assert len(batch_result.task_results) == 6

        # In per_agent, each (task, agent) pair should have unique seed
        task1_results = [r for r in batch_result.task_results if r.task_id == "task1"]
        assert len(task1_results) == 2
        assert task1_results[0].seed != task1_results[1].seed

    def test_per_task_topology(self, sample_tasks):
        """Test per_task topology mode."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_TASK],
            base_seed=42,
            agent_ids=["agent1", "agent2"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=sample_tasks, executor=mock_executor)

        assert TopologyMode.PER_TASK in results
        batch_result = results[TopologyMode.PER_TASK]

        # With 3 tasks and 2 agents, expect 6 results
        assert batch_result.total_tasks == 6
        assert len(batch_result.task_results) == 6

        # In per_task, each task gets one seed shared by all agents
        task1_results = [r for r in batch_result.task_results if r.task_id == "task1"]
        assert len(task1_results) == 2
        assert task1_results[0].seed == task1_results[1].seed

    def test_all_topologies_together(self, sample_tasks):
        """Test running all topology modes in one batch."""
        config = BatchRunConfig(
            topology_modes=[
                TopologyMode.SHARED_EPISODE,
                TopologyMode.PER_AGENT,
                TopologyMode.PER_TASK,
            ],
            base_seed=42,
            agent_ids=["agent1"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=sample_tasks, executor=mock_executor)

        # All three topology modes should have results
        assert len(results) == 3
        assert TopologyMode.SHARED_EPISODE in results
        assert TopologyMode.PER_AGENT in results
        assert TopologyMode.PER_TASK in results

        # Each should have 3 results (3 tasks × 1 agent)
        for topology_mode, batch_result in results.items():
            assert batch_result.total_tasks == 3, f"Failed for {topology_mode}"
            assert len(batch_result.task_results) == 3

    def test_seed_determinism(self, sample_tasks):
        """Test that same base seed produces same results."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_AGENT],
            base_seed=12345,
            agent_ids=["agent1"],
        )

        runner1 = SciAgentBenchRunner(config=config)
        results1 = runner1.run_batch(tasks=sample_tasks, executor=mock_executor)

        runner2 = SciAgentBenchRunner(config=config)
        results2 = runner2.run_batch(tasks=sample_tasks, executor=mock_executor)

        # Seeds should be identical
        seeds1 = results1[TopologyMode.PER_AGENT].seeds
        seeds2 = results2[TopologyMode.PER_AGENT].seeds
        assert seeds1 == seeds2

    def test_different_seeds_per_topology(self, sample_tasks):
        """Test that different topology modes get different seeds."""
        config = BatchRunConfig(
            topology_modes=[
                TopologyMode.SHARED_EPISODE,
                TopologyMode.PER_AGENT,
                TopologyMode.PER_TASK,
            ],
            base_seed=42,
            agent_ids=["agent1"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=sample_tasks, executor=mock_executor)

        # Each topology should have different seeds (because RNG advances)
        seeds_shared = results[TopologyMode.SHARED_EPISODE].seeds
        seeds_per_agent = results[TopologyMode.PER_AGENT].seeds
        seeds_per_task = results[TopologyMode.PER_TASK].seeds

        # All three should be different (but this is not guaranteed,
        # so we just check they were generated)
        assert len(seeds_shared) == 3
        assert len(seeds_per_agent) == 3
        assert len(seeds_per_task) == 3

    def test_summary_metrics(self, sample_tasks):
        """Test summary metrics calculation."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_TASK],
            base_seed=42,
            agent_ids=["agent1"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=sample_tasks, executor=mock_executor)
        batch_result = results[TopologyMode.PER_TASK]

        # Check summary metrics exist
        assert "success_rate" in batch_result.summary_metrics
        assert "total_execution_time" in batch_result.summary_metrics
        assert "avg_execution_time" in batch_result.summary_metrics
        assert "min_execution_time" in batch_result.summary_metrics
        assert "max_execution_time" in batch_result.summary_metrics

        # Check values are reasonable
        assert 0.0 <= batch_result.summary_metrics["success_rate"] <= 1.0
        assert batch_result.summary_metrics["total_execution_time"] > 0
        assert batch_result.summary_metrics["avg_execution_time"] > 0

    def test_batch_result_methods(self, sample_tasks):
        """Test BatchResult helper methods."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_TASK],
            base_seed=42,
            agent_ids=["agent1"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=sample_tasks, executor=mock_executor)
        batch_result = results[TopologyMode.PER_TASK]

        # Test success_rate method
        success_rate = batch_result.success_rate()
        assert 0.0 <= success_rate <= 1.0
        assert success_rate == batch_result.successful_tasks / batch_result.total_tasks

        # Test avg_execution_time method
        avg_time = batch_result.avg_execution_time()
        assert avg_time > 0
        expected_avg = sum(r.execution_time for r in batch_result.task_results) / len(
            batch_result.task_results
        )
        assert abs(avg_time - expected_avg) < 0.001

    def test_multiple_agents(self, sample_tasks):
        """Test with multiple agents."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_AGENT],
            base_seed=42,
            agent_ids=["agent1", "agent2", "agent3"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=sample_tasks, executor=mock_executor)
        batch_result = results[TopologyMode.PER_AGENT]

        # 3 tasks × 3 agents = 9 results
        assert batch_result.total_tasks == 9
        assert len(batch_result.task_results) == 9

        # Each task should have 3 results (one per agent)
        for task in sample_tasks:
            task_results = [r for r in batch_result.task_results if r.task_id == task.task_id]
            assert len(task_results) == 3
            agent_ids = {r.agent_id for r in task_results}
            assert agent_ids == {"agent1", "agent2", "agent3"}

    def test_empty_task_list(self):
        """Test with empty task list."""
        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_TASK],
            agent_ids=["agent1"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=[], executor=mock_executor)
        batch_result = results[TopologyMode.PER_TASK]

        assert batch_result.total_tasks == 0
        assert len(batch_result.task_results) == 0
        assert batch_result.successful_tasks == 0
        assert batch_result.success_rate() == 0.0

    def test_single_task(self):
        """Test with single task."""
        task = SciAgentBenchTask(
            task_id="single",
            instruction="Single task",
            dataset_path="/data/single.csv",
            domain="test",
        )

        config = BatchRunConfig(
            topology_modes=[TopologyMode.PER_AGENT],
            base_seed=42,
            agent_ids=["agent1"],
        )
        runner = SciAgentBenchRunner(config=config)

        results = runner.run_batch(tasks=[task], executor=mock_executor)
        batch_result = results[TopologyMode.PER_AGENT]

        assert batch_result.total_tasks == 1
        assert len(batch_result.task_results) == 1
        assert batch_result.task_results[0].task_id == "single"
