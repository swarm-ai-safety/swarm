"""Tests for composite tasks and emergent capability measurement."""

from swarm.env.composite_tasks import (
    CapabilityType,
    CompositeTask,
    CompositeTaskPool,
    CompositeTaskStatus,
    Subtask,
    SubtaskStatus,
    create_planning_coordination_task,
    create_problem_solving_task,
    create_research_synthesis_task,
)
from swarm.metrics.capabilities import (
    AgentCapabilityProfile,
    CapabilityAnalyzer,
    EmergentCapabilityMetrics,
    analyze_capability_distribution,
    compute_collective_intelligence_score,
)

# =============================================================================
# Subtask Tests
# =============================================================================


class TestSubtask:
    """Tests for Subtask class."""

    def test_create_subtask(self):
        """Should create subtask with default values."""
        subtask = Subtask(
            name="Test Subtask",
            description="A test subtask",
            required_capabilities={CapabilityType.RESEARCH},
        )

        assert subtask.name == "Test Subtask"
        assert subtask.status == SubtaskStatus.PENDING
        assert CapabilityType.RESEARCH in subtask.required_capabilities

    def test_is_available_no_dependencies(self):
        """Subtask with no dependencies should be available."""
        subtask = Subtask(name="Independent")
        assert subtask.is_available(completed_subtasks=set())

    def test_is_available_with_dependencies(self):
        """Subtask should check dependencies."""
        dep_id = "dep-123"
        subtask = Subtask(
            name="Dependent",
            dependencies={dep_id},
        )

        # Not available when dependency not complete
        assert not subtask.is_available(completed_subtasks=set())

        # Available when dependency complete
        assert subtask.is_available(completed_subtasks={dep_id})

    def test_can_claim_capability_check(self):
        """Should check agent capabilities."""
        subtask = Subtask(
            name="Specialized",
            required_capabilities={CapabilityType.ANALYSIS, CapabilityType.RESEARCH},
        )

        # Can't claim without all capabilities
        assert not subtask.can_claim({CapabilityType.RESEARCH})

        # Can claim with all capabilities
        assert subtask.can_claim({CapabilityType.ANALYSIS, CapabilityType.RESEARCH})

        # Can claim with superset
        assert subtask.can_claim(
            {
                CapabilityType.ANALYSIS,
                CapabilityType.RESEARCH,
                CapabilityType.PLANNING,
            }
        )

    def test_claim_and_complete(self):
        """Should support claim/start/complete workflow."""
        subtask = Subtask(name="Workflow Test")
        subtask.status = SubtaskStatus.AVAILABLE

        # Claim
        assert subtask.claim("agent_1")
        assert subtask.status == SubtaskStatus.CLAIMED
        assert subtask.assigned_to == "agent_1"

        # Start
        assert subtask.start()
        assert subtask.status == SubtaskStatus.IN_PROGRESS

        # Complete
        assert subtask.complete("output", 0.85)
        assert subtask.status == SubtaskStatus.COMPLETED
        assert subtask.quality_score == 0.85


# =============================================================================
# CompositeTask Tests
# =============================================================================


class TestCompositeTask:
    """Tests for CompositeTask class."""

    def test_create_composite_task(self):
        """Should create composite task."""
        task = CompositeTask(
            name="Test Task",
            description="A test composite task",
            goal="Complete testing",
            min_agents=2,
            max_agents=4,
            total_bounty=20.0,
        )

        assert task.name == "Test Task"
        assert task.status == CompositeTaskStatus.OPEN
        assert task.min_agents == 2

    def test_add_subtask(self):
        """Should add subtasks and set parent."""
        task = CompositeTask(name="Parent")
        subtask = Subtask(name="Child")

        task.add_subtask(subtask)

        assert len(task.subtasks) == 1
        assert subtask.parent_task_id == task.task_id
        assert task.get_subtask(subtask.subtask_id) is subtask

    def test_join_task(self):
        """Should allow agents to join."""
        task = CompositeTask(
            name="Team Task",
            min_agents=2,
            max_agents=3,
        )

        assert task.join_task("agent_1")
        assert "agent_1" in task.participating_agents
        assert task.status == CompositeTaskStatus.OPEN  # Need 2 agents

        assert task.join_task("agent_2")
        assert task.status == CompositeTaskStatus.IN_PROGRESS  # Now have 2

        assert task.join_task("agent_3")
        assert len(task.participating_agents) == 3

        # Can't exceed max
        assert not task.join_task("agent_4")

    def test_get_available_subtasks(self):
        """Should return subtasks with met dependencies."""
        task = CompositeTask(name="Dependency Test")

        st1 = Subtask(name="First")
        st2 = Subtask(name="Second", dependencies={st1.subtask_id})

        task.add_subtask(st1)
        task.add_subtask(st2)

        # Initially only st1 available
        available = task.get_available_subtasks()
        assert len(available) == 1
        assert available[0].subtask_id == st1.subtask_id

        # After completing st1, st2 becomes available
        st1.status = SubtaskStatus.COMPLETED
        available = task.get_available_subtasks()
        assert len(available) == 1
        assert available[0].subtask_id == st2.subtask_id

    def test_claim_subtask(self):
        """Should allow agents to claim subtasks."""
        task = CompositeTask(name="Claim Test")
        task.join_task("agent_1")

        subtask = Subtask(
            name="Claimable",
            required_capabilities={CapabilityType.RESEARCH},
        )
        subtask.status = SubtaskStatus.AVAILABLE
        task.add_subtask(subtask)

        # Can claim with required capability
        assert task.claim_subtask(
            subtask.subtask_id,
            "agent_1",
            {CapabilityType.RESEARCH},
        )

        # Can't claim without being in task
        assert not task.claim_subtask(
            subtask.subtask_id,
            "agent_2",
            {CapabilityType.RESEARCH},
        )

    def test_complete_task(self):
        """Should complete task when all subtasks done."""
        task = CompositeTask(
            name="Completion Test",
            min_agents=1,
            total_bounty=10.0,
        )
        task.join_task("agent_1")

        st1 = Subtask(name="Only Subtask", bounty_share=1.0)
        st1.status = SubtaskStatus.IN_PROGRESS
        st1.assigned_to = "agent_1"
        task.add_subtask(st1)

        # Complete the subtask
        task.complete_subtask(st1.subtask_id, "output", 0.9)

        assert task.status == CompositeTaskStatus.COMPLETED
        assert task.final_quality == 0.9

    def test_compute_payouts(self):
        """Should compute fair payouts."""
        task = CompositeTask(
            name="Payout Test",
            min_agents=2,
            total_bounty=20.0,
            completion_bonus=10.0,
        )
        task.join_task("agent_1")
        task.join_task("agent_2")

        # Set contributions
        task.agent_contributions["agent_1"] = 0.6
        task.agent_contributions["agent_2"] = 0.4
        task.status = CompositeTaskStatus.COMPLETED

        payouts = task.compute_payouts()

        # Base: 20 * 0.6 = 12 for agent_1, 20 * 0.4 = 8 for agent_2
        # Bonus: 10 / 2 = 5 each
        assert abs(payouts["agent_1"] - 17.0) < 0.01
        assert abs(payouts["agent_2"] - 13.0) < 0.01

    def test_emergent_metrics_computation(self):
        """Should compute emergent capability metrics on completion."""
        task = CompositeTask(
            name="Metrics Test",
            min_agents=2,
        )
        task.join_task("agent_1")
        task.join_task("agent_2")

        st1 = Subtask(name="First", bounty_share=0.5)
        st2 = Subtask(name="Second", dependencies={st1.subtask_id}, bounty_share=0.5)

        task.add_subtask(st1)
        task.add_subtask(st2)

        # Assign and complete
        st1.assigned_to = "agent_1"
        st1.status = SubtaskStatus.IN_PROGRESS
        task.complete_subtask(st1.subtask_id, "out1", 0.8)

        st2.assigned_to = "agent_2"
        st2.status = SubtaskStatus.IN_PROGRESS
        task.complete_subtask(st2.subtask_id, "out2", 0.9)

        # Task should be complete with metrics
        assert task.status == CompositeTaskStatus.COMPLETED
        # Coordination score should be 1.0 (balanced: 1 task each)
        assert task.coordination_score == 1.0


# =============================================================================
# CompositeTaskPool Tests
# =============================================================================


class TestCompositeTaskPool:
    """Tests for CompositeTaskPool class."""

    def test_add_and_get_task(self):
        """Should add and retrieve tasks."""
        pool = CompositeTaskPool()
        task = CompositeTask(name="Pool Test")

        pool.add_task(task)

        assert pool.get_task(task.task_id) is task

    def test_get_open_tasks(self):
        """Should return only open tasks."""
        pool = CompositeTaskPool()

        open_task = CompositeTask(name="Open")
        closed_task = CompositeTask(name="Closed")
        closed_task.status = CompositeTaskStatus.COMPLETED

        pool.add_task(open_task)
        pool.add_task(closed_task)

        open_tasks = pool.get_open_tasks()
        assert len(open_tasks) == 1
        assert open_tasks[0].task_id == open_task.task_id

    def test_get_joinable_tasks(self):
        """Should filter by agent capabilities."""
        pool = CompositeTaskPool()

        task1 = CompositeTask(
            name="Research Task",
            required_capabilities={CapabilityType.RESEARCH},
        )
        task2 = CompositeTask(
            name="Planning Task",
            required_capabilities={CapabilityType.PLANNING},
        )

        pool.add_task(task1)
        pool.add_task(task2)

        # Agent with research can join task1
        joinable = pool.get_joinable_tasks({CapabilityType.RESEARCH})
        assert len(joinable) == 1
        assert joinable[0].task_id == task1.task_id

        # Agent with both can join both
        joinable = pool.get_joinable_tasks(
            {
                CapabilityType.RESEARCH,
                CapabilityType.PLANNING,
            }
        )
        assert len(joinable) == 2

    def test_expire_overdue(self):
        """Should expire tasks past deadline."""
        pool = CompositeTaskPool()

        task = CompositeTask(
            name="Expiring",
            deadline_epoch=5,
        )
        pool.add_task(task)

        # Not expired yet
        expired = pool.expire_overdue(4)
        assert len(expired) == 0

        # Now expired
        expired = pool.expire_overdue(5)
        assert len(expired) == 1
        assert task.status == CompositeTaskStatus.EXPIRED


# =============================================================================
# Task Template Tests
# =============================================================================


class TestTaskTemplates:
    """Tests for task template functions."""

    def test_create_research_synthesis_task(self):
        """Should create well-formed research task."""
        task = create_research_synthesis_task("AI Safety", bounty=30.0)

        assert "Research" in task.name
        assert len(task.subtasks) == 5
        assert task.total_bounty == 30.0
        assert CapabilityType.RESEARCH in task.required_capabilities

        # Check dependencies are valid
        for st in task.subtasks:
            for dep_id in st.dependencies:
                assert task.get_subtask(dep_id) is not None

    def test_create_planning_coordination_task(self):
        """Should create planning task."""
        task = create_planning_coordination_task("Product Launch")

        assert "Planning" in task.name
        assert task.min_agents == 3
        assert len(task.subtasks) == 5

    def test_create_problem_solving_task(self):
        """Should create problem solving task with parallel subtasks."""
        task = create_problem_solving_task("Optimization Problem")

        assert len(task.subtasks) == 5

        # First two should have no dependencies (parallel)
        assert len(task.subtasks[0].dependencies) == 0
        assert len(task.subtasks[1].dependencies) == 0

        # Third should depend on both
        assert len(task.subtasks[2].dependencies) == 2


# =============================================================================
# CapabilityAnalyzer Tests
# =============================================================================


class TestCapabilityAnalyzer:
    """Tests for CapabilityAnalyzer class."""

    def test_register_agent(self):
        """Should register agents with capabilities."""
        analyzer = CapabilityAnalyzer()

        profile = analyzer.register_agent(
            "agent_1",
            {CapabilityType.RESEARCH, CapabilityType.ANALYSIS},
        )

        assert profile.agent_id == "agent_1"
        assert len(profile.capabilities) == 2

    def test_record_task_completion(self):
        """Should update profiles on task completion."""
        analyzer = CapabilityAnalyzer()
        analyzer.register_agent("agent_1", {CapabilityType.RESEARCH})
        analyzer.register_agent("agent_2", {CapabilityType.ANALYSIS})

        # Create and complete a task
        task = CompositeTask(name="Test", min_agents=2)
        task.join_task("agent_1")
        task.join_task("agent_2")

        st1 = Subtask(
            name="Research",
            required_capabilities={CapabilityType.RESEARCH},
            bounty_share=0.5,
        )
        st1.assigned_to = "agent_1"
        st1.status = SubtaskStatus.COMPLETED
        st1.quality_score = 0.8
        task.add_subtask(st1)

        st2 = Subtask(
            name="Analysis",
            required_capabilities={CapabilityType.ANALYSIS},
            bounty_share=0.5,
        )
        st2.assigned_to = "agent_2"
        st2.status = SubtaskStatus.COMPLETED
        st2.quality_score = 0.9
        task.add_subtask(st2)

        task.status = CompositeTaskStatus.COMPLETED
        task._compute_emergent_metrics()

        analyzer.record_task_completion(task)

        # Check profiles updated
        profile1 = analyzer.get_agent_profile("agent_1")
        assert profile1.subtasks_completed == 1
        assert profile1.avg_quality == 0.8
        assert "agent_2" in profile1.unique_collaborators

    def test_compute_metrics(self):
        """Should compute emergent metrics."""
        analyzer = CapabilityAnalyzer()
        analyzer.register_agent("a1", {CapabilityType.RESEARCH})
        analyzer.register_agent("a2", {CapabilityType.ANALYSIS})

        # Create completed task
        task = CompositeTask(name="Test", min_agents=2)
        task.join_task("a1")
        task.join_task("a2")
        task.status = CompositeTaskStatus.COMPLETED
        task.final_quality = 0.85
        task.coordination_score = 0.9
        task.synergy_score = 0.7
        task.information_flow_score = 0.8

        analyzer.record_task_completion(task)

        metrics = analyzer.compute_metrics()

        assert metrics.total_composite_tasks == 1
        assert metrics.avg_final_quality == 0.85
        assert metrics.avg_coordination_score == 0.9


# =============================================================================
# Metric Function Tests
# =============================================================================


class TestMetricFunctions:
    """Tests for standalone metric functions."""

    def test_compute_collective_intelligence_score(self):
        """Should compute weighted score."""
        metrics = EmergentCapabilityMetrics(
            avg_final_quality=0.8,
            avg_coordination_score=0.7,
            avg_synergy_score=0.6,
            avg_information_flow=0.8,
            task_efficiency=0.05,
            complementarity_score=0.9,
            knowledge_transfer=0.7,
        )

        score = compute_collective_intelligence_score(metrics)

        assert 0.0 <= score <= 1.0
        # Score should be around 0.7 given input values
        assert 0.5 <= score <= 0.9

    def test_analyze_capability_distribution(self):
        """Should analyze capability spread."""
        profiles = [
            AgentCapabilityProfile(
                agent_id="a1",
                capabilities={CapabilityType.RESEARCH, CapabilityType.ANALYSIS},
            ),
            AgentCapabilityProfile(
                agent_id="a2",
                capabilities={CapabilityType.PLANNING, CapabilityType.EXECUTION},
            ),
            AgentCapabilityProfile(
                agent_id="a3",
                capabilities={CapabilityType.RESEARCH, CapabilityType.VERIFICATION},
            ),
        ]

        stats = analyze_capability_distribution(profiles)

        assert stats["n_agents"] == 3
        assert stats["n_capability_types"] == 5
        assert stats["avg_capabilities_per_agent"] == 2.0
        assert 0.0 <= stats["capability_coverage"] <= 1.0

    def test_analyze_capability_distribution_empty(self):
        """Should handle empty list."""
        stats = analyze_capability_distribution([])
        assert stats == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestOrchestratorIntegration:
    """Tests for orchestrator integration."""

    def test_orchestrator_composite_task_disabled(self):
        """Should work with composite tasks disabled."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_composite_tasks=False)
        orchestrator = Orchestrator(config=config)

        assert orchestrator.composite_task_pool is None
        assert orchestrator.capability_analyzer is None

    def test_orchestrator_composite_task_enabled(self):
        """Should initialize composite task components."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_composite_tasks=True)
        orchestrator = Orchestrator(config=config)

        assert orchestrator.composite_task_pool is not None
        assert orchestrator.capability_analyzer is not None

    def test_add_composite_task(self):
        """Should add composite tasks to pool."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_composite_tasks=True)
        orchestrator = Orchestrator(config=config)

        task = create_research_synthesis_task("Test Topic")
        assert orchestrator.add_composite_task(task)

        retrieved = orchestrator.get_composite_task(task.task_id)
        assert retrieved is task

    def test_register_agent_capabilities(self):
        """Should register agent capabilities."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_composite_tasks=True)
        orchestrator = Orchestrator(config=config)

        assert orchestrator.register_agent_capabilities(
            "agent_1",
            {CapabilityType.RESEARCH, CapabilityType.ANALYSIS},
        )

        # Verify via capability_analyzer
        profile = orchestrator.capability_analyzer.get_agent_profile("agent_1")
        assert len(profile.capabilities) == 2

    def test_get_capability_metrics(self):
        """Should return capability metrics."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_composite_tasks=True)
        orchestrator = Orchestrator(config=config)

        metrics = orchestrator.get_capability_metrics()

        assert metrics is not None
        assert isinstance(metrics, EmergentCapabilityMetrics)

    def test_epoch_metrics_include_capabilities(self):
        """Epoch metrics should include capability metrics."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            n_epochs=1,
            steps_per_epoch=1,
            enable_composite_tasks=True,
        )
        orchestrator = Orchestrator(config=config)
        orchestrator.register_agent(HonestAgent("test"))

        metrics = orchestrator.run()

        assert len(metrics) == 1
        assert metrics[0].capability_metrics is not None
