"""Tests for task synthesis from execution traces."""

import pytest

from swarm.bridges.awm.mcp_client import AWMEpisodeTrace, ToolCallRecord
from swarm.env.composite_tasks import CapabilityType
from swarm.env.task_synthesis import (
    DependencyInferencer,
    SynthesisMetrics,
    TaskSynthesizer,
    TraceSegment,
    TraceSegmenter,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_trace() -> AWMEpisodeTrace:
    """Create a sample episode trace for testing."""
    trace = AWMEpisodeTrace(
        episode_id="test-episode",
        agent_id="test-agent",
        environment_id="test-env",
        task_description="Test task",
    )

    # Add some tool calls
    for i in range(8):
        tool_name = f"tool_{i // 2}"  # Group tools for segmentation
        trace.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                arguments={"arg": i},
                success=True,
                result={"output": i},
            )
        )

    trace.steps_used = len(trace.tool_calls)
    return trace


@pytest.fixture
def empty_trace() -> AWMEpisodeTrace:
    """Create an empty trace."""
    return AWMEpisodeTrace(
        episode_id="empty-episode",
        agent_id="test-agent",
        environment_id="test-env",
        task_description="Empty task",
    )


# ── TraceSegmenter tests ────────────────────────────────────────────


def test_trace_segmenter_initialization():
    """Test TraceSegmenter initialization."""
    segmenter = TraceSegmenter(min_calls_per_segment=2, max_calls_per_segment=10)
    assert segmenter.min_calls_per_segment == 2
    assert segmenter.max_calls_per_segment == 10


def test_segment_empty_trace(empty_trace: AWMEpisodeTrace):
    """Test segmenting an empty trace."""
    segmenter = TraceSegmenter()
    segments = segmenter.segment(empty_trace)
    assert len(segments) == 0


def test_segment_simple_trace(sample_trace: AWMEpisodeTrace):
    """Test segmenting a simple trace."""
    segmenter = TraceSegmenter(min_calls_per_segment=2, max_calls_per_segment=4)
    segments = segmenter.segment(sample_trace)

    # Should create multiple segments
    assert len(segments) > 0

    # Check segment properties
    for segment in segments:
        assert segment.start_idx >= 0
        assert segment.end_idx > segment.start_idx
        assert segment.call_count > 0
        assert len(segment.primary_tools) > 0
        assert segment.description != ""


def test_segment_respects_max_size(sample_trace: AWMEpisodeTrace):
    """Test that segmenter respects max segment size."""
    max_size = 3
    segmenter = TraceSegmenter(min_calls_per_segment=1, max_calls_per_segment=max_size)
    segments = segmenter.segment(sample_trace)

    for segment in segments:
        assert segment.call_count <= max_size


def test_segment_trace_segment_properties():
    """Test TraceSegment property calculations."""
    calls = [
        ToolCallRecord(tool_name="tool1", success=True),
        ToolCallRecord(tool_name="tool2", success=False, is_error_response=True),
        ToolCallRecord(tool_name="tool3", success=True),
    ]

    segment = TraceSegment(
        segment_id="test-seg",
        start_idx=0,
        end_idx=3,
        tool_calls=calls,
        primary_tools={"tool1", "tool2", "tool3"},
    )

    assert segment.call_count == 3
    assert segment.error_rate == pytest.approx(1.0 / 3.0)


def test_segment_with_no_errors():
    """Test error rate for segment with no errors."""
    calls = [
        ToolCallRecord(tool_name="tool1", success=True),
        ToolCallRecord(tool_name="tool2", success=True),
    ]

    segment = TraceSegment(
        segment_id="test-seg",
        start_idx=0,
        end_idx=2,
        tool_calls=calls,
    )

    assert segment.error_rate == 0.0


# ── DependencyInferencer tests ──────────────────────────────────────


def test_dependency_inferencer_initialization():
    """Test DependencyInferencer initialization."""
    inferencer = DependencyInferencer()
    assert inferencer is not None


def test_infer_dependencies_empty():
    """Test dependency inference with empty segments."""
    inferencer = DependencyInferencer()
    dependencies = inferencer.infer_dependencies([])
    assert dependencies == {}


def test_infer_dependencies_single_segment():
    """Test dependency inference with single segment."""
    segment = TraceSegment(segment_id="seg1", start_idx=0, end_idx=2)
    inferencer = DependencyInferencer()
    dependencies = inferencer.infer_dependencies([segment])

    assert "seg1" in dependencies
    assert len(dependencies["seg1"]) == 0  # No dependencies


def test_infer_dependencies_multiple_segments():
    """Test dependency inference with multiple segments."""
    segments = [
        TraceSegment(segment_id="seg1", start_idx=0, end_idx=2),
        TraceSegment(segment_id="seg2", start_idx=2, end_idx=4),
        TraceSegment(segment_id="seg3", start_idx=4, end_idx=6),
    ]

    inferencer = DependencyInferencer()
    dependencies = inferencer.infer_dependencies(segments)

    # Should create linear dependencies
    assert len(dependencies["seg1"]) == 0
    assert "seg1" in dependencies["seg2"]
    assert "seg2" in dependencies["seg3"]


# ── TaskSynthesizer tests ───────────────────────────────────────────


def test_task_synthesizer_initialization():
    """Test TaskSynthesizer initialization."""
    synthesizer = TaskSynthesizer()
    assert synthesizer.segmenter is not None
    assert synthesizer.inferencer is not None


def test_task_synthesizer_with_custom_components():
    """Test TaskSynthesizer with custom components."""
    segmenter = TraceSegmenter(min_calls_per_segment=1, max_calls_per_segment=5)
    inferencer = DependencyInferencer()
    synthesizer = TaskSynthesizer(segmenter=segmenter, inferencer=inferencer)

    assert synthesizer.segmenter is segmenter
    assert synthesizer.inferencer is inferencer


def test_synthesize_from_empty_trace(empty_trace: AWMEpisodeTrace):
    """Test synthesizing task from empty trace."""
    synthesizer = TaskSynthesizer()
    task = synthesizer.synthesize(empty_trace)

    # Should create minimal task
    assert task.name != ""
    assert task.description == empty_trace.task_description
    assert len(task.subtasks) == 0


def test_synthesize_from_trace(sample_trace: AWMEpisodeTrace):
    """Test synthesizing task from trace."""
    synthesizer = TaskSynthesizer()
    task = synthesizer.synthesize(
        trace=sample_trace,
        task_name="Test Task",
        bounty=50.0,
    )

    # Check task properties
    assert task.name == "Test Task"
    assert task.description == sample_trace.task_description
    assert task.total_bounty == 50.0
    assert len(task.subtasks) > 0

    # Check subtasks
    total_share = sum(st.bounty_share for st in task.subtasks)
    assert total_share == pytest.approx(1.0)

    # Check that subtasks have capabilities
    for subtask in task.subtasks:
        assert len(subtask.required_capabilities) > 0

    # Check that task has aggregated capabilities
    assert len(task.required_capabilities) > 0


def test_synthesize_infers_capabilities(sample_trace: AWMEpisodeTrace):
    """Test that synthesis infers appropriate capabilities."""
    synthesizer = TaskSynthesizer()
    task = synthesizer.synthesize(sample_trace)

    # Should have at least execution capability
    all_caps = set()
    for subtask in task.subtasks:
        all_caps.update(subtask.required_capabilities)

    assert CapabilityType.EXECUTION in all_caps


def test_synthesize_wires_dependencies(sample_trace: AWMEpisodeTrace):
    """Test that synthesis wires subtask dependencies correctly."""
    synthesizer = TaskSynthesizer()
    task = synthesizer.synthesize(sample_trace)

    if len(task.subtasks) > 1:
        # Later subtasks should have dependencies
        has_dependencies = any(
            len(st.dependencies) > 0 for st in task.subtasks[1:]
        )
        assert has_dependencies


# ── SynthesisMetrics tests ──────────────────────────────────────────


def test_synthesis_metrics_initialization():
    """Test SynthesisMetrics initialization."""
    metrics = SynthesisMetrics()
    assert metrics.total_traces_processed == 0
    assert metrics.total_tasks_synthesized == 0
    assert metrics.synthesis_failures == 0


def test_synthesis_metrics_update_success():
    """Test updating metrics with successful synthesis."""
    metrics = SynthesisMetrics()
    metrics.update(segments_count=5, avg_deps=1.5, success=True)

    assert metrics.total_traces_processed == 1
    assert metrics.total_tasks_synthesized == 1
    assert metrics.total_segments_extracted == 5
    assert metrics.avg_segments_per_task == 5.0
    assert metrics.avg_dependencies_per_segment == 1.5
    assert metrics.synthesis_failures == 0


def test_synthesis_metrics_update_failure():
    """Test updating metrics with failed synthesis."""
    metrics = SynthesisMetrics()
    metrics.update(segments_count=0, avg_deps=0.0, success=False)

    assert metrics.total_traces_processed == 1
    assert metrics.total_tasks_synthesized == 0
    assert metrics.synthesis_failures == 1


def test_synthesis_metrics_multiple_updates():
    """Test multiple metric updates."""
    metrics = SynthesisMetrics()

    metrics.update(segments_count=4, avg_deps=1.0, success=True)
    metrics.update(segments_count=6, avg_deps=2.0, success=True)
    metrics.update(segments_count=0, avg_deps=0.0, success=False)

    assert metrics.total_traces_processed == 3
    assert metrics.total_tasks_synthesized == 2
    assert metrics.total_segments_extracted == 10
    assert metrics.avg_segments_per_task == 5.0  # (4 + 6) / 2
    assert metrics.avg_dependencies_per_segment == 1.5  # (1.0 + 2.0) / 2
    assert metrics.synthesis_failures == 1


def test_synthesis_metrics_to_dict():
    """Test converting metrics to dictionary."""
    metrics = SynthesisMetrics()
    metrics.update(segments_count=3, avg_deps=1.0, success=True)

    data = metrics.to_dict()

    assert isinstance(data, dict)
    assert "total_traces_processed" in data
    assert "total_tasks_synthesized" in data
    assert "synthesis_failures" in data
    assert data["total_traces_processed"] == 1
    assert data["total_tasks_synthesized"] == 1


# ── Integration tests ───────────────────────────────────────────────


def test_full_synthesis_pipeline():
    """Test the complete synthesis pipeline end-to-end."""
    # Create a realistic trace
    trace = AWMEpisodeTrace(
        episode_id="integration-test",
        agent_id="agent-1",
        environment_id="env-1",
        task_description="Complete multi-phase task",
    )

    # Phase 1: Setup
    for i in range(3):
        trace.tool_calls.append(
            ToolCallRecord(
                tool_name="setup",
                arguments={"step": i},
                success=True,
            )
        )

    # Phase 2: Processing
    for i in range(4):
        trace.tool_calls.append(
            ToolCallRecord(
                tool_name="process",
                arguments={"data": i},
                success=True,
            )
        )

    # Phase 3: Cleanup
    for i in range(2):
        trace.tool_calls.append(
            ToolCallRecord(
                tool_name="cleanup",
                arguments={"step": i},
                success=True,
            )
        )

    # Run synthesis
    synthesizer = TaskSynthesizer(
        segmenter=TraceSegmenter(min_calls_per_segment=2, max_calls_per_segment=4)
    )
    task = synthesizer.synthesize(trace, task_name="Integration Test Task")

    # Verify results
    assert task.name == "Integration Test Task"
    assert len(task.subtasks) >= 2  # At least 2 phases
    assert task.total_bounty > 0

    # Verify subtask structure
    for subtask in task.subtasks:
        assert subtask.name != ""
        assert subtask.bounty_share > 0
        assert len(subtask.required_capabilities) > 0

    # Verify metrics
    metrics = SynthesisMetrics()
    segments = synthesizer.segmenter.segment(trace)
    dependencies = synthesizer.inferencer.infer_dependencies(segments)
    avg_deps = sum(len(d) for d in dependencies.values()) / len(dependencies)
    metrics.update(
        segments_count=len(segments),
        avg_deps=avg_deps,
        success=True,
    )

    assert metrics.total_tasks_synthesized == 1
    assert metrics.total_segments_extracted == len(segments)
