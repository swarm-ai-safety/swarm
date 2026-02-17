"""Example usage of SciAgentBench batch runner.

Demonstrates how to use the runner with different topology modes
to execute SciAgentBench-style tasks.
"""

import logging

from swarm.evaluation.sciagentbench import (
    BatchRunConfig,
    SciAgentBenchRunner,
    SciAgentBenchTask,
    TaskResult,
    TopologyMode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_task_executor(
    task: SciAgentBenchTask, agent_id: str, seed: int
) -> TaskResult:
    """Example task executor that simulates task execution.

    In a real implementation, this would:
    1. Set up the environment with the given seed
    2. Load the task dataset
    3. Execute the agent on the task
    4. Evaluate the output against success criteria
    5. Return the result

    Args:
        task: Task to execute.
        agent_id: Identifier for the agent.
        seed: Random seed for reproducibility.

    Returns:
        TaskResult with execution outcome.
    """
    import time

    logger.info(f"Executing {task.task_id} with {agent_id} (seed={seed})")

    # Simulate task execution
    start_time = time.monotonic()

    # Mock success/failure based on task and seed
    success = (seed % 3 != 0)  # Fail every third seed

    execution_time = time.monotonic() - start_time

    return TaskResult(
        task_id=task.task_id,
        agent_id=agent_id,
        topology_mode=TopologyMode.PER_TASK,  # Set by runner
        seed=seed,
        success=success,
        execution_time=execution_time,
        output=f"Analysis complete for {task.task_id}" if success else None,
        error_message="Mock failure" if not success else None,
        metadata={
            "dataset_size": 1000,
            "memory_used_mb": 256.0,
        },
    )


def main():
    """Run example batch evaluation."""

    # Define sample tasks
    tasks = [
        SciAgentBenchTask(
            task_id="bioinformatics_01",
            instruction="Perform gene expression analysis on the provided RNA-seq data",
            dataset_path="/data/rnaseq_sample.csv",
            domain="bioinformatics",
            expert_knowledge="Consider batch effects and normalization methods",
            success_criteria={"correlation": 0.85, "p_value": 0.05},
            timeout_seconds=300.0,
        ),
        SciAgentBenchTask(
            task_id="chemistry_02",
            instruction="Predict molecular properties from SMILES strings",
            dataset_path="/data/molecules.sdf",
            domain="computational_chemistry",
            expert_knowledge="Use RDKit for molecular descriptors",
            success_criteria={"rmse": 0.5, "r2": 0.75},
            timeout_seconds=600.0,
        ),
        SciAgentBenchTask(
            task_id="gis_03",
            instruction="Analyze spatial patterns in the urban development dataset",
            dataset_path="/data/urban_shapefile.zip",
            domain="gis",
            expert_knowledge="Consider spatial autocorrelation",
            success_criteria={"moran_i": 0.6},
            timeout_seconds=450.0,
        ),
    ]

    # Configure the batch runner
    config = BatchRunConfig(
        topology_modes=[
            TopologyMode.SHARED_EPISODE,
            TopologyMode.PER_AGENT,
            TopologyMode.PER_TASK,
        ],
        base_seed=42,
        agent_ids=["gpt4", "claude-opus", "deepseek"],
        verbose=True,
    )

    # Create and run the batch
    runner = SciAgentBenchRunner(config=config)

    logger.info("=" * 60)
    logger.info("Running SciAgentBench batch across topology matrix")
    logger.info(f"Tasks: {len(tasks)}")
    logger.info(f"Agents: {len(config.agent_ids)}")
    logger.info(f"Topologies: {[t.value for t in config.topology_modes]}")
    logger.info("=" * 60)

    results = runner.run_batch(
        tasks=tasks,
        executor=example_task_executor,
    )

    # Print results summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    for topology_mode, batch_result in results.items():
        logger.info(f"\n{topology_mode.value.upper()}:")
        logger.info(f"  Total tasks: {batch_result.total_tasks}")
        logger.info(f"  Successful: {batch_result.successful_tasks}")
        logger.info(f"  Success rate: {batch_result.success_rate():.2%}")
        logger.info(f"  Total time: {batch_result.total_execution_time:.2f}s")
        logger.info(f"  Avg time/task: {batch_result.avg_execution_time():.3f}s")
        logger.info(f"  Seeds used: {len(batch_result.seeds)}")

        # Show per-agent breakdown
        agent_stats = {}
        for result in batch_result.task_results:
            if result.agent_id not in agent_stats:
                agent_stats[result.agent_id] = {"total": 0, "success": 0}
            agent_stats[result.agent_id]["total"] += 1
            if result.success:
                agent_stats[result.agent_id]["success"] += 1

        logger.info("  Per-agent success rates:")
        for agent_id, stats in agent_stats.items():
            success_rate = stats["success"] / stats["total"]
            logger.info(f"    {agent_id}: {stats['success']}/{stats['total']} ({success_rate:.2%})")

    logger.info("\n" + "=" * 60)
    logger.info("Batch execution complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
