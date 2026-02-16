"""Tests for replay verification of synthesized tasks."""

import pytest

from swarm.env.composite_tasks import CompositeTask, Subtask, CapabilityType
from swarm.replay.verifier import (
    SynthesizedTaskVerifier,
    TaskReplayResult,
    VerificationSummary,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def simple_task() -> CompositeTask:
    """Create a simple composite task for testing."""
    task = CompositeTask(
        name="Simple Test Task",
        description="A simple task for verification testing",
        goal="Complete all subtasks",
        min_agents=1,
        max_agents=2,
        total_bounty=10.0,
    )
    
    # Add two subtasks
    subtask1 = Subtask(
        name="First subtask",
        description="Do first thing",
        required_capabilities={CapabilityType.EXECUTION},
        bounty_share=0.5,
    )
    task.add_subtask(subtask1)
    
    subtask2 = Subtask(
        name="Second subtask",
        description="Do second thing",
        required_capabilities={CapabilityType.EXECUTION},
        dependencies={subtask1.subtask_id},
        bounty_share=0.5,
    )
    task.add_subtask(subtask2)
    
    return task


@pytest.fixture
def complex_task() -> CompositeTask:
    """Create a more complex task with multiple subtasks."""
    task = CompositeTask(
        name="Complex Test Task",
        description="A complex multi-phase task",
        goal="Complete all phases",
        min_agents=2,
        max_agents=4,
        total_bounty=30.0,
    )
    
    # Phase 1: Parallel subtasks
    subtask1 = Subtask(
        name="Phase 1A",
        required_capabilities={CapabilityType.RESEARCH},
        bounty_share=0.2,
    )
    task.add_subtask(subtask1)
    
    subtask2 = Subtask(
        name="Phase 1B",
        required_capabilities={CapabilityType.ANALYSIS},
        bounty_share=0.2,
    )
    task.add_subtask(subtask2)
    
    # Phase 2: Depends on both phase 1 subtasks
    subtask3 = Subtask(
        name="Phase 2",
        required_capabilities={CapabilityType.PLANNING},
        dependencies={subtask1.subtask_id, subtask2.subtask_id},
        bounty_share=0.3,
    )
    task.add_subtask(subtask3)
    
    # Phase 3: Final subtask
    subtask4 = Subtask(
        name="Phase 3",
        required_capabilities={CapabilityType.VERIFICATION},
        dependencies={subtask3.subtask_id},
        bounty_share=0.3,
    )
    task.add_subtask(subtask4)
    
    return task


@pytest.fixture
def malformed_task() -> CompositeTask:
    """Create a malformed task (for testing failure cases)."""
    task = CompositeTask(
        name="Malformed Task",
        description="Task with issues",
        goal="This won't work well",
        min_agents=5,  # More than max!
        max_agents=2,
        total_bounty=0.0,  # No bounty!
    )
    return task


# ── TaskReplayResult tests ──────────────────────────────────────────


def test_task_replay_result_success_rate():
    """Test success rate calculation."""
    result = TaskReplayResult(
        task_id="test-task",
        task_name="Test",
        replay_count=5,
        successful_replays=3,
        failed_replays=2,
    )
    
    assert result.success_rate == pytest.approx(0.6)


def test_task_replay_result_success_rate_zero_replays():
    """Test success rate with zero replays."""
    result = TaskReplayResult(
        task_id="test-task",
        task_name="Test",
        replay_count=0,
        successful_replays=0,
        failed_replays=0,
    )
    
    assert result.success_rate == 0.0


def test_task_replay_result_is_verifiable_true():
    """Test verifiable check for good result."""
    result = TaskReplayResult(
        task_id="test-task",
        task_name="Test",
        replay_count=3,
        successful_replays=3,
        failed_replays=0,
        reproducibility_score=0.9,
    )
    
    assert result.is_verifiable is True


def test_task_replay_result_is_verifiable_false_no_success():
    """Test verifiable check with no successful replays."""
    result = TaskReplayResult(
        task_id="test-task",
        task_name="Test",
        replay_count=3,
        successful_replays=0,
        failed_replays=3,
        reproducibility_score=0.9,
    )
    
    assert result.is_verifiable is False


def test_task_replay_result_is_verifiable_false_low_reproducibility():
    """Test verifiable check with low reproducibility."""
    result = TaskReplayResult(
        task_id="test-task",
        task_name="Test",
        replay_count=3,
        successful_replays=3,
        failed_replays=0,
        reproducibility_score=0.5,  # Too low
    )
    
    assert result.is_verifiable is False


def test_task_replay_result_to_dict():
    """Test converting result to dictionary."""
    result = TaskReplayResult(
        task_id="test-task",
        task_name="Test Task",
        replay_count=5,
        successful_replays=4,
        failed_replays=1,
        avg_completion_fraction=0.95,
        avg_quality=0.85,
        reproducibility_score=0.9,
        metadata={"key": "value"},
    )
    
    data = result.to_dict()
    
    assert isinstance(data, dict)
    assert data["task_id"] == "test-task"
    assert data["task_name"] == "Test Task"
    assert data["replay_count"] == 5
    assert data["successful_replays"] == 4
    assert data["success_rate"] == pytest.approx(0.8)
    assert data["is_verifiable"] is True
    assert "metadata" in data


# ── SynthesizedTaskVerifier tests ───────────────────────────────────


def test_verifier_initialization():
    """Test verifier initialization."""
    verifier = SynthesizedTaskVerifier(replay_count=5, base_seed=123)
    assert verifier.replay_count == 5
    assert verifier.base_seed == 123


def test_verifier_default_initialization():
    """Test verifier with default parameters."""
    verifier = SynthesizedTaskVerifier()
    assert verifier.replay_count == 3
    assert verifier.base_seed == 42


def test_verify_simple_task(simple_task: CompositeTask):
    """Test verifying a simple task."""
    verifier = SynthesizedTaskVerifier(replay_count=3)
    result = verifier.verify_task(simple_task)
    
    assert result.task_id == simple_task.task_id
    assert result.task_name == simple_task.name
    assert result.replay_count == 3
    assert result.successful_replays >= 0
    assert result.failed_replays >= 0
    assert result.successful_replays + result.failed_replays == result.replay_count


def test_verify_complex_task(complex_task: CompositeTask):
    """Test verifying a complex task."""
    verifier = SynthesizedTaskVerifier(replay_count=5)
    result = verifier.verify_task(complex_task)
    
    assert result.task_id == complex_task.task_id
    assert result.replay_count == 5
    # Complex task with proper structure should verify
    assert result.successful_replays > 0


def test_verify_malformed_task(malformed_task: CompositeTask):
    """Test verifying a malformed task."""
    verifier = SynthesizedTaskVerifier(replay_count=3)
    result = verifier.verify_task(malformed_task)
    
    assert result.task_id == malformed_task.task_id
    # Malformed task should have lower success/reproducibility
    assert result.is_verifiable is False


def test_verify_multiple_tasks(
    simple_task: CompositeTask,
    complex_task: CompositeTask,
):
    """Test verifying multiple tasks at once."""
    verifier = SynthesizedTaskVerifier(replay_count=3)
    results = verifier.verify_multiple_tasks([simple_task, complex_task])
    
    assert len(results) == 2
    assert results[0].task_id == simple_task.task_id
    assert results[1].task_id == complex_task.task_id
    
    # All should have completed verification
    for result in results:
        assert result.replay_count == 3


def test_verify_empty_task_list():
    """Test verifying empty list of tasks."""
    verifier = SynthesizedTaskVerifier()
    results = verifier.verify_multiple_tasks([])
    
    assert len(results) == 0


def test_verify_task_with_different_replay_counts():
    """Test that different replay counts work correctly."""
    task = CompositeTask(
        name="Test",
        description="Test task",
        goal="Goal",
        total_bounty=10.0,
    )
    
    for replay_count in [1, 3, 5, 10]:
        verifier = SynthesizedTaskVerifier(replay_count=replay_count)
        result = verifier.verify_task(task)
        assert result.replay_count == replay_count


# ── VerificationSummary tests ───────────────────────────────────────


def test_verification_summary_from_empty_results():
    """Test creating summary from empty results."""
    summary = VerificationSummary.from_results([])
    
    assert summary.total_tasks == 0
    assert summary.verifiable_tasks == 0
    assert summary.total_replays == 0
    assert summary.successful_replays == 0


def test_verification_summary_from_results():
    """Test creating summary from multiple results."""
    results = [
        TaskReplayResult(
            task_id="task1",
            task_name="Task 1",
            replay_count=3,
            successful_replays=3,
            failed_replays=0,
            reproducibility_score=0.9,
        ),
        TaskReplayResult(
            task_id="task2",
            task_name="Task 2",
            replay_count=3,
            successful_replays=2,
            failed_replays=1,
            reproducibility_score=0.8,
        ),
        TaskReplayResult(
            task_id="task3",
            task_name="Task 3",
            replay_count=3,
            successful_replays=0,
            failed_replays=3,
            reproducibility_score=0.3,
        ),
    ]
    
    summary = VerificationSummary.from_results(results)
    
    assert summary.total_tasks == 3
    assert summary.verifiable_tasks == 2  # First two are verifiable
    assert summary.total_replays == 9
    assert summary.successful_replays == 5
    assert summary.avg_success_rate == pytest.approx((1.0 + 2/3 + 0.0) / 3)
    assert summary.avg_reproducibility == pytest.approx((0.9 + 0.8 + 0.3) / 3)


def test_verification_summary_to_dict():
    """Test converting summary to dictionary."""
    results = [
        TaskReplayResult(
            task_id="task1",
            task_name="Task 1",
            replay_count=3,
            successful_replays=3,
            failed_replays=0,
            reproducibility_score=0.9,
        ),
    ]
    
    summary = VerificationSummary.from_results(results)
    data = summary.to_dict()
    
    assert isinstance(data, dict)
    assert data["total_tasks"] == 1
    assert data["verifiable_tasks"] == 1
    assert data["total_replays"] == 3
    assert data["successful_replays"] == 3


# ── Integration tests ───────────────────────────────────────────────


def test_end_to_end_verification_pipeline(
    simple_task: CompositeTask,
    complex_task: CompositeTask,
):
    """Test complete verification pipeline."""
    # Create verifier
    verifier = SynthesizedTaskVerifier(replay_count=3, base_seed=42)
    
    # Verify multiple tasks
    tasks = [simple_task, complex_task]
    results = verifier.verify_multiple_tasks(tasks)
    
    # Check results
    assert len(results) == 2
    
    # Create summary
    summary = VerificationSummary.from_results(results)
    
    assert summary.total_tasks == 2
    assert summary.total_replays == 6  # 2 tasks * 3 replays each
    
    # At least one should be verifiable
    assert summary.verifiable_tasks >= 1
    
    # Export summary
    summary_dict = summary.to_dict()
    assert isinstance(summary_dict, dict)
    assert "total_tasks" in summary_dict
    assert "verifiable_tasks" in summary_dict
