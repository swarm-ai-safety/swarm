# SciAgentBench Runner

Batch execution framework for SciAgentBench-style tasks across different topology configurations.

## Overview

The SciAgentBench runner enables systematic evaluation of language agents on scientific tasks with three execution topologies:

- **`shared_episode`**: All agents share the same episode/environment instance
- **`per_agent`**: Each agent gets its own isolated episode  
- **`per_task`**: Each task gets its own episode (task-level isolation)

All executions use **fixed seeds** for reproducibility.

## Quick Start

```python
from swarm.evaluation.sciagentbench import (
    SciAgentBenchRunner,
    SciAgentBenchTask,
    BatchRunConfig,
    TopologyMode,
    TaskResult,
)

# Define tasks
tasks = [
    SciAgentBenchTask(
        task_id="bio_01",
        instruction="Analyze gene expression data",
        dataset_path="/data/rnaseq.csv",
        domain="bioinformatics",
        success_criteria={"accuracy": 0.85},
    ),
]

# Configure runner
config = BatchRunConfig(
    topology_modes=[TopologyMode.SHARED_EPISODE, TopologyMode.PER_AGENT],
    base_seed=42,
    agent_ids=["gpt4", "claude"],
)

# Define executor
def my_executor(task, agent_id, seed):
    # Your task execution logic here
    # Returns TaskResult
    pass

# Run batch
runner = SciAgentBenchRunner(config=config)
results = runner.run_batch(tasks=tasks, executor=my_executor)

# Analyze results
for topology_mode, batch_result in results.items():
    print(f"{topology_mode}: {batch_result.success_rate():.2%} success rate")
```

## Topology Modes

### Shared Episode

All agents operate in the same environment instance. Use when:
- Agents should interact with the same world state
- You want to measure agent cooperation or competition
- Environment setup is expensive

**Seed behavior**: Each task gets one seed, shared by all agents.

```python
config = BatchRunConfig(topology_modes=[TopologyMode.SHARED_EPISODE])
```

### Per Agent

Each agent gets its own isolated environment. Use when:
- Independent agent evaluation is required
- Agents should not influence each other
- You want to compare agent capabilities directly

**Seed behavior**: Each (task, agent) pair gets a unique seed.

```python
config = BatchRunConfig(topology_modes=[TopologyMode.PER_AGENT])
```

### Per Task

Each task gets its own episode, agents run sequentially. Use when:
- Tasks are independent
- Agents should see the same initial state per task
- You want task-level reproducibility

**Seed behavior**: Each task gets one seed, shared by agents in that task.

```python
config = BatchRunConfig(topology_modes=[TopologyMode.PER_TASK])
```

## API Reference

### `SciAgentBenchTask`

Task configuration model.

**Fields:**
- `task_id` (str): Unique task identifier
- `instruction` (str): Task instruction text
- `dataset_path` (str): Path or URI to task dataset
- `domain` (str): Scientific domain
- `expert_knowledge` (Optional[str]): Domain-specific hints
- `success_criteria` (Dict[str, Any]): Success conditions
- `timeout_seconds` (float): Max execution time (default: 300.0)

### `TaskResult`

Result from executing a single task.

**Fields:**
- `task_id` (str): Task identifier
- `agent_id` (str): Agent identifier  
- `topology_mode` (TopologyMode): Execution topology
- `seed` (int): Random seed used
- `success` (bool): Whether task succeeded
- `execution_time` (float): Time in seconds
- `output` (Optional[str]): Agent output
- `error_message` (Optional[str]): Error if failed
- `metadata` (Dict[str, Any]): Additional metadata

### `BatchResult`

Aggregated results from a batch run.

**Fields:**
- `topology_mode` (TopologyMode): Topology used
- `task_results` (List[TaskResult]): Individual results
- `total_tasks` (int): Total tasks attempted
- `successful_tasks` (int): Tasks that succeeded
- `total_execution_time` (float): Wall-clock time
- `seeds` (List[int]): Seeds used
- `summary_metrics` (Dict[str, float]): Aggregate stats

**Methods:**
- `success_rate() -> float`: Overall success rate
- `avg_execution_time() -> float`: Average time per task

### `BatchRunConfig`

Configuration for batch execution.

**Fields:**
- `topology_modes` (List[TopologyMode]): Topologies to run (default: all three)
- `base_seed` (int): Base random seed (default: 42)
- `agent_ids` (List[str]): Agent identifiers (default: `["default_agent"]`)
- `max_parallel_tasks` (int): Max parallel executions (default: 1)
- `verbose` (bool): Enable detailed logging (default: False)

### `SciAgentBenchRunner`

Main batch runner class.

**Constructor:**
```python
runner = SciAgentBenchRunner(config: Optional[BatchRunConfig] = None)
```

**Methods:**

#### `run_batch()`

Execute tasks across all configured topologies.

```python
results = runner.run_batch(
    tasks: List[SciAgentBenchTask],
    executor: Callable[[SciAgentBenchTask, str, int], TaskResult],
) -> Dict[TopologyMode, BatchResult]
```

**Parameters:**
- `tasks`: List of tasks to execute
- `executor`: Callable with signature `(task, agent_id, seed) -> TaskResult`

**Returns:**
- Dictionary mapping `TopologyMode` to `BatchResult`

## Executor Function

The executor function implements task execution logic:

```python
def executor(
    task: SciAgentBenchTask,
    agent_id: str, 
    seed: int
) -> TaskResult:
    """Execute a single task.
    
    Args:
        task: Task configuration
        agent_id: Agent to evaluate
        seed: Random seed for this run
        
    Returns:
        TaskResult with execution outcome
    """
    # 1. Set random seed
    np.random.seed(seed)
    
    # 2. Load task dataset
    data = load_dataset(task.dataset_path)
    
    # 3. Execute agent
    output = agent.run(task.instruction, data, task.expert_knowledge)
    
    # 4. Evaluate output
    success = evaluate_output(output, task.success_criteria)
    
    # 5. Return result
    return TaskResult(
        task_id=task.task_id,
        agent_id=agent_id,
        topology_mode=TopologyMode.PER_TASK,
        seed=seed,
        success=success,
        execution_time=elapsed_time,
        output=output,
    )
```

## Examples

See `examples/sciagentbench_runner_example.py` for a complete working example.

### Multiple Agents

```python
config = BatchRunConfig(
    topology_modes=[TopologyMode.PER_AGENT],
    agent_ids=["gpt4", "claude-opus", "deepseek"],
    base_seed=42,
)
```

### Single Topology

```python
config = BatchRunConfig(
    topology_modes=[TopologyMode.SHARED_EPISODE],
    agent_ids=["my_agent"],
)
```

### Verbose Logging

```python
config = BatchRunConfig(
    verbose=True,  # Log each task execution
)
```

### Result Analysis

```python
results = runner.run_batch(tasks, executor)

for topology_mode, batch_result in results.items():
    print(f"\n{topology_mode.value}:")
    print(f"  Success rate: {batch_result.success_rate():.2%}")
    print(f"  Avg time: {batch_result.avg_execution_time():.3f}s")
    
    # Per-agent breakdown
    for agent_id in config.agent_ids:
        agent_results = [
            r for r in batch_result.task_results 
            if r.agent_id == agent_id
        ]
        success = sum(1 for r in agent_results if r.success)
        print(f"  {agent_id}: {success}/{len(agent_results)}")
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_sciagentbench_runner.py -v
```

Test coverage includes:
- All three topology modes
- Seed determinism and reproducibility  
- Multiple agents and tasks
- Empty task lists
- Summary metrics calculation
- Result aggregation

## Integration with ScienceAgentBench

This runner is designed to work with [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) tasks. To integrate:

1. Load ScienceAgentBench tasks from the dataset
2. Create `SciAgentBenchTask` objects with task metadata
3. Implement an executor that runs agents on tasks
4. Use the runner to execute across topology matrix
5. Export results in ScienceAgentBench format

## Design Decisions

### Fixed Seeds

All executions use deterministic seeds derived from `base_seed`:
- Ensures reproducibility across runs
- Different topologies get different seed sequences
- Seed generation uses NumPy's `default_rng()` for quality

### Topology Modes

Three modes cover common evaluation scenarios:
- `shared_episode`: Multi-agent interaction analysis
- `per_agent`: Independent capability assessment  
- `per_task`: Task-focused reproducibility

### Callable Executor

Executor as a callable provides flexibility:
- Easy to test (use mock executors)
- Supports any agent framework
- Decouples execution from orchestration
