# SciForge-Style Task Synthesis

This guide explains how to use SWARM's SciForge-style dependency graph extraction and replay verification system.

## Overview

The task synthesis pipeline automatically extracts structured task graphs from execution traces, enabling:

1. **Automatic task discovery** - Extract multi-step workflows from agent behavior
2. **Dependency inference** - Learn task dependencies from execution patterns
3. **Replay verification** - Validate that synthesized tasks can be reproduced
4. **Quality metrics** - Track synthesis success rates and reproducibility

## Core Concepts

### Execution Traces

An `AWMEpisodeTrace` captures a complete sequence of tool calls from an agent's task execution:

```python
from swarm.bridges.awm.mcp_client import AWMEpisodeTrace, ToolCallRecord

trace = AWMEpisodeTrace(
    episode_id="ep-001",
    agent_id="agent-1",
    task_description="Data pipeline task",
    tool_calls=[
        ToolCallRecord(tool_name="fetch_data", ...),
        ToolCallRecord(tool_name="transform_data", ...),
        ToolCallRecord(tool_name="write_output", ...),
    ],
)
```

### Trace Segments

A `TraceSegment` represents a logical subtask within the execution:

- **Boundaries** - Start/end indices in the tool call sequence
- **Tool clustering** - Groups related tool calls together
- **Phase detection** - Identifies transitions between task phases

### Dependency Graph

Dependencies are inferred from:
- **Execution order** - Later segments depend on earlier ones
- **Data flow** - Outputs consumed as inputs (future enhancement)
- **Resource usage** - Shared state access patterns (future enhancement)

### Composite Tasks

The `CompositeTask` structure captures the extracted workflow:
- Multiple `Subtask` objects with inferred capabilities
- Explicit dependency relationships forming a DAG
- Bounty allocation and quality metrics

## Usage

### Basic Synthesis

```python
from swarm.env.task_synthesis import TaskSynthesizer

# Create synthesizer with default settings
synthesizer = TaskSynthesizer()

# Synthesize task from trace
task = synthesizer.synthesize(
    trace=episode_trace,
    task_name="My Workflow",
    bounty=25.0,
)

print(f"Synthesized {len(task.subtasks)} subtasks")
print(f"Required capabilities: {task.required_capabilities}")
```

### Custom Segmentation

```python
from swarm.env.task_synthesis import TraceSegmenter, TaskSynthesizer

# Configure segmentation parameters
segmenter = TraceSegmenter(
    min_calls_per_segment=3,  # Minimum tool calls per subtask
    max_calls_per_segment=8,  # Maximum tool calls per subtask
)

synthesizer = TaskSynthesizer(segmenter=segmenter)
task = synthesizer.synthesize(trace)
```

### Replay Verification

```python
from swarm.replay.verifier import SynthesizedTaskVerifier

# Create verifier
verifier = SynthesizedTaskVerifier(
    replay_count=5,  # Run 5 replays with different seeds
    base_seed=42,
)

# Verify task
result = verifier.verify_task(task)

print(f"Success rate: {result.success_rate:.2%}")
print(f"Reproducibility: {result.reproducibility_score:.2f}")
print(f"Is verifiable: {result.is_verifiable}")
```

### Batch Verification

```python
# Synthesize multiple tasks
tasks = [synthesizer.synthesize(trace) for trace in traces]

# Verify all at once
results = verifier.verify_multiple_tasks(tasks)

# Create summary
from swarm.replay.verifier import VerificationSummary
summary = VerificationSummary.from_results(results)

print(f"Verifiable tasks: {summary.verifiable_tasks}/{summary.total_tasks}")
print(f"Avg reproducibility: {summary.avg_reproducibility:.2f}")
```

## Metrics

### Synthesis Metrics

Track synthesis quality with `SynthesisMetrics`:

```python
from swarm.env.task_synthesis import SynthesisMetrics

metrics = SynthesisMetrics()

# After each synthesis
segments = synthesizer.segmenter.segment(trace)
dependencies = synthesizer.inferencer.infer_dependencies(segments)
avg_deps = sum(len(d) for d in dependencies.values()) / len(dependencies)

metrics.update(
    segments_count=len(segments),
    avg_deps=avg_deps,
    success=True,
)

# Export metrics
data = metrics.to_dict()
print(f"Tasks synthesized: {data['total_tasks_synthesized']}")
print(f"Avg segments: {data['avg_segments_per_task']:.2f}")
```

### Verification Metrics

Each `TaskReplayResult` includes:
- `replay_count` - Number of replay runs
- `successful_replays` - Count of successful completions
- `success_rate` - Fraction of successful replays
- `avg_completion_fraction` - Avg % of subtasks completed
- `avg_quality` - Average quality score
- `reproducibility_score` - Consistency measure (0-1)
- `is_verifiable` - Boolean flag (>= 1 success, reproducibility >= 0.7)

## Advanced Usage

### Integrating with AWMHandler

Extract and synthesize tasks from AWM episodes:

```python
from swarm.core.awm_handler import AWMHandler
from swarm.env.task_synthesis import TaskSynthesizer

# After running simulations with AWM
handler = AWMHandler(...)
completed_episodes = handler.get_completed_episodes()

# Synthesize tasks from completed episodes
synthesizer = TaskSynthesizer()
tasks = [synthesizer.synthesize(ep) for ep in completed_episodes]

# Verify synthesized tasks
verifier = SynthesizedTaskVerifier()
results = verifier.verify_multiple_tasks(tasks)

# Report
verifiable = sum(1 for r in results if r.is_verifiable)
print(f"Extracted {verifiable} verifiable tasks from {len(episodes)} episodes")
```

### Custom Dependency Inference

Implement custom dependency logic by subclassing `DependencyInferencer`:

```python
from swarm.env.task_synthesis import DependencyInferencer

class DataFlowInferencer(DependencyInferencer):
    def infer_dependencies(self, segments):
        dependencies = super().infer_dependencies(segments)
        
        # Add data flow analysis
        for i, seg_i in enumerate(segments):
            for j, seg_j in enumerate(segments[:i]):
                if self._has_data_flow(seg_j, seg_i):
                    dependencies[seg_i.segment_id].add(seg_j.segment_id)
        
        return dependencies
    
    def _has_data_flow(self, source, target):
        # Custom logic to detect data dependencies
        pass
```

## Example Pipeline

See `examples/task_synthesis_demo.py` for a complete working example:

```bash
python examples/task_synthesis_demo.py
```

Output:
```
============================================================
SciForge-style Task Synthesis Pipeline
============================================================

Step 1: Generating sample execution trace...
  - Episode ID: demo-episode-001
  - Task: Build and test a simple data pipeline
  - Tool calls: 10
  - Verified: True

Step 2: Initializing synthesis components...
  - TraceSegmenter initialized
  - DependencyInferencer initialized
  - TaskSynthesizer initialized

Step 3: Segmenting trace into subtasks...
  - Extracted 4 segments

Step 4: Inferring dependencies...
  - Dependency graph: linear chain

Step 5: Synthesizing CompositeTask...
  - Task name: Data Pipeline Task
  - Subtasks: 4
  - Total bounty: $30.00

Step 6: Verifying synthesized task...
  - Success rate: 100.00%
  - Reproducibility: 0.95
  - Is verifiable: True

✓ SUCCESS: Synthesized task is verifiable!
```

## API Reference

### TraceSegmenter

```python
class TraceSegmenter:
    def __init__(
        self,
        min_calls_per_segment: int = 2,
        max_calls_per_segment: int = 10,
    ): ...
    
    def segment(self, trace: AWMEpisodeTrace) -> List[TraceSegment]: ...
```

### DependencyInferencer

```python
class DependencyInferencer:
    def infer_dependencies(
        self,
        segments: List[TraceSegment],
    ) -> Dict[str, Set[str]]: ...
```

### TaskSynthesizer

```python
class TaskSynthesizer:
    def __init__(
        self,
        segmenter: Optional[TraceSegmenter] = None,
        inferencer: Optional[DependencyInferencer] = None,
    ): ...
    
    def synthesize(
        self,
        trace: AWMEpisodeTrace,
        task_name: Optional[str] = None,
        bounty: float = 20.0,
    ) -> CompositeTask: ...
```

### SynthesizedTaskVerifier

```python
class SynthesizedTaskVerifier:
    def __init__(
        self,
        replay_count: int = 3,
        base_seed: int = 42,
    ): ...
    
    def verify_task(
        self,
        task: CompositeTask,
    ) -> TaskReplayResult: ...
    
    def verify_multiple_tasks(
        self,
        tasks: List[CompositeTask],
    ) -> List[TaskReplayResult]: ...
```

## Best Practices

1. **Segment size** - Tune `min_calls_per_segment` and `max_calls_per_segment` based on your task granularity
2. **Replay count** - Use at least 3 replays for reliable verification
3. **Batch processing** - Process multiple traces together for better metrics
4. **Quality threshold** - Filter for `is_verifiable=True` before using synthesized tasks
5. **Incremental refinement** - Iterate on segmentation parameters based on results

## Limitations

- **Current dependency inference is conservative** - Uses linear precedence (safe but may miss parallelism)
- **Capability inference is heuristic** - Based on tool usage patterns rather than semantic analysis
- **Verification is simulated** - Full replay integration requires complete task environment
- **No automatic data flow analysis** - Tool argument/result tracking not yet implemented

## Future Enhancements

- [ ] Data flow dependency detection
- [ ] Parallel subtask identification
- [ ] Semantic capability inference using LLMs
- [ ] Integration with full simulation replay
- [ ] Task similarity clustering
- [ ] Automatic bounty allocation optimization

## References

- SciForge paper: [link if available]
- SWARM composite tasks: `swarm/env/composite_tasks.py`
- AWM bridge: `swarm/bridges/awm/`
- Replay infrastructure: `swarm/replay/`
