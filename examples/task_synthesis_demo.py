"""Example: SciForge-style task synthesis from execution traces.

Demonstrates the complete pipeline:
1. Generate sample execution traces
2. Segment traces into subtasks
3. Infer dependencies
4. Synthesize CompositeTask structures
5. Verify tasks replay successfully
"""

from swarm.bridges.awm.mcp_client import AWMEpisodeTrace, ToolCallRecord
from swarm.env.task_synthesis import (
    DependencyInferencer,
    SynthesisMetrics,
    TaskSynthesizer,
    TraceSegmenter,
)
from swarm.replay.verifier import (
    SynthesizedTaskVerifier,
)


def create_sample_trace() -> AWMEpisodeTrace:
    """Create a sample execution trace for demonstration."""
    trace = AWMEpisodeTrace(
        episode_id="demo-episode-001",
        agent_id="demo-agent",
        environment_id="demo-env",
        task_description="Build and test a simple data pipeline",
    )

    # Simulate a multi-phase task execution
    # Phase 1: Data collection (3 calls)
    trace.tool_calls.extend([
        ToolCallRecord(
            tool_name="fetch_data",
            arguments={"source": "api", "limit": 100},
            success=True,
            result={"rows": 100},
        ),
        ToolCallRecord(
            tool_name="fetch_data",
            arguments={"source": "database", "query": "SELECT * FROM users"},
            success=True,
            result={"rows": 50},
        ),
        ToolCallRecord(
            tool_name="validate_data",
            arguments={"data": "fetched_data"},
            success=True,
            result={"valid": True},
        ),
    ])

    # Phase 2: Data transformation (4 calls)
    trace.tool_calls.extend([
        ToolCallRecord(
            tool_name="transform_data",
            arguments={"operation": "normalize"},
            success=True,
            result={"transformed": True},
        ),
        ToolCallRecord(
            tool_name="transform_data",
            arguments={"operation": "filter"},
            success=True,
            result={"filtered": True},
        ),
        ToolCallRecord(
            tool_name="aggregate_data",
            arguments={"group_by": "category"},
            success=True,
            result={"groups": 5},
        ),
        ToolCallRecord(
            tool_name="validate_data",
            arguments={"data": "transformed_data"},
            success=True,
            result={"valid": True},
        ),
    ])

    # Phase 3: Output and verification (3 calls)
    trace.tool_calls.extend([
        ToolCallRecord(
            tool_name="write_output",
            arguments={"destination": "file", "format": "json"},
            success=True,
            result={"written": True},
        ),
        ToolCallRecord(
            tool_name="run_tests",
            arguments={"test_suite": "integration"},
            success=True,
            result={"passed": 10, "failed": 0},
        ),
        ToolCallRecord(
            tool_name="generate_report",
            arguments={"format": "markdown"},
            success=True,
            result={"report_generated": True},
        ),
    ])

    trace.steps_used = len(trace.tool_calls)
    trace.verified = True

    return trace


def main():
    """Run the complete synthesis and verification pipeline."""
    print("=" * 60)
    print("SciForge-style Task Synthesis Pipeline")
    print("=" * 60)
    print()

    # Step 1: Create sample traces
    print("Step 1: Generating sample execution trace...")
    trace = create_sample_trace()
    print(f"  - Episode ID: {trace.episode_id}")
    print(f"  - Task: {trace.task_description}")
    print(f"  - Tool calls: {len(trace.tool_calls)}")
    print(f"  - Verified: {trace.verified}")
    print()

    # Step 2: Initialize synthesis components
    print("Step 2: Initializing synthesis components...")
    segmenter = TraceSegmenter(
        min_calls_per_segment=2,
        max_calls_per_segment=5,
    )
    inferencer = DependencyInferencer()
    synthesizer = TaskSynthesizer(
        segmenter=segmenter,
        inferencer=inferencer,
    )
    print("  - TraceSegmenter initialized")
    print("  - DependencyInferencer initialized")
    print("  - TaskSynthesizer initialized")
    print()

    # Step 3: Segment the trace
    print("Step 3: Segmenting trace into subtasks...")
    segments = segmenter.segment(trace)
    print(f"  - Extracted {len(segments)} segments:")
    for i, seg in enumerate(segments):
        print(f"    {i+1}. {seg.description}")
        print(f"       Tools: {', '.join(seg.primary_tools)}")
        print(f"       Calls: {seg.call_count}, Error rate: {seg.error_rate:.2%}")
    print()

    # Step 4: Infer dependencies
    print("Step 4: Inferring dependencies...")
    dependencies = inferencer.infer_dependencies(segments)
    print("  - Dependency graph:")
    for seg in segments:
        deps = dependencies.get(seg.segment_id, set())
        if deps:
            dep_indices = [
                i for i, s in enumerate(segments) if s.segment_id in deps
            ]
            print(f"    Segment {segments.index(seg)+1} depends on: {dep_indices}")
        else:
            print(f"    Segment {segments.index(seg)+1} has no dependencies")
    print()

    # Step 5: Synthesize task
    print("Step 5: Synthesizing CompositeTask...")
    task = synthesizer.synthesize(
        trace=trace,
        task_name="Data Pipeline Task",
        bounty=30.0,
    )
    print(f"  - Task ID: {task.task_id}")
    print(f"  - Task name: {task.name}")
    print(f"  - Subtasks: {len(task.subtasks)}")
    print(f"  - Required agents: {task.min_agents}-{task.max_agents}")
    print(f"  - Total bounty: ${task.total_bounty:.2f}")
    print(f"  - Required capabilities: {[c.value for c in task.required_capabilities]}")
    print()
    print("  Subtask breakdown:")
    for i, subtask in enumerate(task.subtasks):
        deps = [
            j for j, st in enumerate(task.subtasks)
            if st.subtask_id in subtask.dependencies
        ]
        print(f"    {i+1}. {subtask.name}")
        print(f"       Bounty share: {subtask.bounty_share:.2%}")
        print(f"       Capabilities: {[c.value for c in subtask.required_capabilities]}")
        print(f"       Dependencies: {deps if deps else 'None'}")
    print()

    # Step 6: Verify task
    print("Step 6: Verifying synthesized task...")
    verifier = SynthesizedTaskVerifier(
        replay_count=3,
        base_seed=42,
    )
    result = verifier.verify_task(task)
    print(f"  - Replay count: {result.replay_count}")
    print(f"  - Successful replays: {result.successful_replays}")
    print(f"  - Failed replays: {result.failed_replays}")
    print(f"  - Success rate: {result.success_rate:.2%}")
    print(f"  - Avg completion: {result.avg_completion_fraction:.2%}")
    print(f"  - Avg quality: {result.avg_quality:.2f}")
    print(f"  - Reproducibility: {result.reproducibility_score:.2f}")
    print(f"  - Is verifiable: {result.is_verifiable}")
    print()

    # Step 7: Update metrics
    print("Step 7: Collecting synthesis metrics...")
    metrics = SynthesisMetrics()
    avg_deps = sum(len(deps) for deps in dependencies.values()) / len(dependencies)
    metrics.update(
        segments_count=len(segments),
        avg_deps=avg_deps,
        success=result.is_verifiable,
    )
    print(f"  - Total traces processed: {metrics.total_traces_processed}")
    print(f"  - Tasks synthesized: {metrics.total_tasks_synthesized}")
    print(f"  - Segments extracted: {metrics.total_segments_extracted}")
    print(f"  - Avg segments per task: {metrics.avg_segments_per_task:.2f}")
    print(f"  - Avg dependencies per segment: {metrics.avg_dependencies_per_segment:.2f}")
    print()

    # Step 8: Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if result.is_verifiable:
        print("✓ SUCCESS: Synthesized task is verifiable!")
        print()
        print("The trace-to-subgraph pipeline successfully:")
        print("  1. Segmented the execution trace into subtasks")
        print("  2. Inferred task dependencies from execution order")
        print("  3. Synthesized a CompositeTask with proper structure")
        print("  4. Verified the task replays successfully")
        print()
        print("This demonstrates SciForge-style dependency graph")
        print("extraction and replay verification.")
    else:
        print("✗ PARTIAL: Task synthesized but verification incomplete")
        print(f"  - Reproducibility score too low: {result.reproducibility_score:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
