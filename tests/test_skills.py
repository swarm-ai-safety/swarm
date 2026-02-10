"""Tests for the evolving skill system.

Covers:
- Skill data models (creation, serialization)
- SkillLibrary (CRUD, queries, capacity, pruning)
- SkillEvolutionEngine (extraction, composition, rate limiting)
- SkillGovernance (reputation gates, poisoning detection, rollback)
- SkillMetrics (diversity, convergence, collection)
- SkillEvolvingAgent (mixin integration, skill-augmented decisions)
- SkillHandler (orchestrator integration)
"""

import pytest

from swarm.agents.base import InteractionProposal, Observation
from swarm.agents.skill_evolving import (
    SkillEvolvingHonestAgent,
    SkillEvolvingOpportunisticAgent,
)
from swarm.core.skill_handler import SkillHandler, SkillHandlerConfig
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction
from swarm.skills.evolution import EvolutionConfig, SkillEvolutionEngine
from swarm.skills.governance import (
    SkillGovernanceConfig,
    SkillGovernanceEngine,
)
from swarm.skills.library import SharingMode, SkillLibrary, SkillLibraryConfig
from swarm.skills.metrics import SkillMetricsCollector
from swarm.skills.model import (
    Skill,
    SkillDomain,
    SkillInvocation,
    SkillPerformance,
    SkillType,
)

# ======================================================================
# Fixtures
# ======================================================================


def make_interaction(
    initiator: str = "alice",
    counterparty: str = "bob",
    p: float = 0.7,
    accepted: bool = True,
    interaction_type: InteractionType = InteractionType.COLLABORATION,
    v_hat: float = 0.4,
) -> SoftInteraction:
    """Create a test interaction."""
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        p=p,
        accepted=accepted,
        interaction_type=interaction_type,
        v_hat=v_hat,
        task_progress_delta=0.5,
    )


def make_observation(
    agent_id: str = "alice",
    reputation: float = 5.0,
    resources: float = 100.0,
    epoch: int = 0,
    step: int = 0,
) -> Observation:
    """Create a test observation."""
    return Observation(
        agent_state=AgentState(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            reputation=reputation,
            resources=resources,
        ),
        current_epoch=epoch,
        current_step=step,
    )


# ======================================================================
# Skill Model Tests
# ======================================================================


class TestSkillModel:
    """Tests for Skill, SkillPerformance, SkillInvocation."""

    def test_skill_creation(self):
        skill = Skill(
            name="test_strategy",
            skill_type=SkillType.STRATEGY,
            domain=SkillDomain.INTERACTION,
            created_by="alice",
        )
        assert skill.name == "test_strategy"
        assert skill.skill_type == SkillType.STRATEGY
        assert skill.domain == SkillDomain.INTERACTION
        assert skill.created_by == "alice"
        assert skill.version == 1
        assert skill.skill_id  # Should have auto-generated ID

    def test_skill_serialization(self):
        skill = Skill(
            name="test_lesson",
            skill_type=SkillType.LESSON,
            domain=SkillDomain.ACCEPTANCE,
            created_by="bob",
            condition={"min_p": 0.3, "max_p": 0.7},
            effect={"acceptance_threshold_delta": 0.1},
            tags={"auto_extracted", "lesson"},
        )
        d = skill.to_dict()
        restored = Skill.from_dict(d)

        assert restored.name == skill.name
        assert restored.skill_type == skill.skill_type
        assert restored.domain == skill.domain
        assert restored.condition == skill.condition
        assert restored.effect == skill.effect
        assert restored.tags == skill.tags

    def test_skill_performance_recording(self):
        perf = SkillPerformance(skill_id="s1")
        assert perf.invocations == 0
        assert perf.success_rate == 0.5  # Prior

        perf.record(payoff=1.0, p=0.8)
        assert perf.invocations == 1
        assert perf.successes == 1
        assert perf.total_payoff == 1.0

        perf.record(payoff=-0.5, p=0.3)
        assert perf.invocations == 2
        assert perf.failures == 1
        assert perf.success_rate == 0.5

    def test_skill_performance_ema(self):
        perf = SkillPerformance(skill_id="s1", ema_alpha=0.5)
        perf.record(payoff=2.0, p=0.8)
        # EMA: 0.0 * 0.5 + 2.0 * 0.5 = 1.0
        assert perf.ema_payoff == pytest.approx(1.0)

        perf.record(payoff=0.0, p=0.5)
        # EMA: 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        assert perf.ema_payoff == pytest.approx(0.5)

    def test_skill_performance_decay(self):
        perf = SkillPerformance(skill_id="s1")
        perf.record(payoff=2.0, p=0.8)
        initial_ema = perf.ema_payoff

        # Decay once (epoch_since_last_use = 1, no decay yet)
        perf.decay()
        assert perf.epochs_since_last_use == 1
        assert perf.ema_payoff == initial_ema  # No decay at 1 epoch

        # Decay twice more (now epochs_since = 3, decay kicks in)
        perf.decay()
        perf.decay()
        assert perf.epochs_since_last_use == 3
        assert perf.ema_payoff < initial_ema

    def test_skill_invocation_creation(self):
        inv = SkillInvocation(
            skill_id="s1",
            agent_id="alice",
            interaction_id="i1",
            payoff=1.5,
            p=0.8,
        )
        d = inv.to_dict()
        assert d["skill_id"] == "s1"
        assert d["payoff"] == 1.5


# ======================================================================
# Skill Library Tests
# ======================================================================


class TestSkillLibrary:
    """Tests for SkillLibrary CRUD, queries, and lifecycle."""

    def test_add_and_get(self):
        lib = SkillLibrary(owner_id="alice")
        skill = Skill(name="s1", domain=SkillDomain.INTERACTION)
        assert lib.add_skill(skill) is True
        assert lib.size == 1
        assert lib.get_skill(skill.skill_id) is skill

    def test_remove(self):
        lib = SkillLibrary(owner_id="alice")
        skill = Skill(name="s1", domain=SkillDomain.INTERACTION)
        lib.add_skill(skill)
        assert lib.remove_skill(skill.skill_id) is True
        assert lib.size == 0
        assert lib.get_skill(skill.skill_id) is None

    def test_capacity_eviction(self):
        config = SkillLibraryConfig(max_skills_per_agent=3, prune_min_invocations=0)
        lib = SkillLibrary(owner_id="alice", config=config)

        for i in range(4):
            skill = Skill(name=f"s{i}", domain=SkillDomain.INTERACTION)
            lib.add_skill(skill)

        # Should have evicted one to stay at capacity
        assert lib.size == 3

    def test_get_skills_by_domain(self):
        lib = SkillLibrary(owner_id="alice")
        s1 = Skill(name="s1", domain=SkillDomain.INTERACTION)
        s2 = Skill(name="s2", domain=SkillDomain.ACCEPTANCE)
        s3 = Skill(name="s3", domain=SkillDomain.INTERACTION)
        lib.add_skill(s1)
        lib.add_skill(s2)
        lib.add_skill(s3)

        interaction_skills = lib.get_skills_by_domain(SkillDomain.INTERACTION)
        assert len(interaction_skills) == 2

    def test_get_skills_by_type(self):
        lib = SkillLibrary(owner_id="alice")
        s1 = Skill(name="s1", skill_type=SkillType.STRATEGY)
        s2 = Skill(name="s2", skill_type=SkillType.LESSON)
        lib.add_skill(s1)
        lib.add_skill(s2)

        strategies = lib.get_skills_by_type(SkillType.STRATEGY)
        assert len(strategies) == 1
        assert strategies[0].name == "s1"

    def test_condition_matching(self):
        lib = SkillLibrary(owner_id="alice")
        s1 = Skill(
            name="high_p",
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.6, "max_p": 1.0},
        )
        s2 = Skill(
            name="low_p",
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.0, "max_p": 0.4},
        )
        lib.add_skill(s1)
        lib.add_skill(s2)

        # High p context should match s1
        high = lib.get_applicable_skills(SkillDomain.INTERACTION, {"p": 0.8})
        assert len(high) == 1
        assert high[0].name == "high_p"

        # Low p context should match s2
        low = lib.get_applicable_skills(SkillDomain.INTERACTION, {"p": 0.2})
        assert len(low) == 1
        assert low[0].name == "low_p"

    def test_select_best_skill(self):
        lib = SkillLibrary(owner_id="alice")
        s1 = Skill(name="weak", domain=SkillDomain.INTERACTION)
        s2 = Skill(name="strong", domain=SkillDomain.INTERACTION)
        lib.add_skill(s1)
        lib.add_skill(s2)

        # Make s2 have better performance
        lib.record_invocation(s1.skill_id, payoff=0.1, p=0.5)
        lib.record_invocation(s1.skill_id, payoff=0.1, p=0.5)
        lib.record_invocation(s2.skill_id, payoff=2.0, p=0.9)
        lib.record_invocation(s2.skill_id, payoff=2.0, p=0.9)

        # With zero exploration, should pick s2
        best = lib.select_best_skill(
            SkillDomain.INTERACTION, {}, exploration_rate=0.0
        )
        assert best is not None
        assert best.name == "strong"

    def test_prune(self):
        config = SkillLibraryConfig(prune_threshold=0.3, prune_min_invocations=2)
        lib = SkillLibrary(owner_id="alice", config=config)
        s1 = Skill(name="good", domain=SkillDomain.INTERACTION)
        s2 = Skill(name="bad", domain=SkillDomain.INTERACTION)
        lib.add_skill(s1)
        lib.add_skill(s2)

        # Make s1 good, s2 bad
        lib.record_invocation(s1.skill_id, payoff=2.0, p=0.8)
        lib.record_invocation(s1.skill_id, payoff=1.5, p=0.7)
        lib.record_invocation(s2.skill_id, payoff=-1.0, p=0.2)
        lib.record_invocation(s2.skill_id, payoff=-1.0, p=0.2)

        pruned = lib.prune()
        assert s2.skill_id in pruned
        assert lib.size == 1

    def test_reputation_gate_for_shared(self):
        config = SkillLibraryConfig(
            sharing_mode=SharingMode.SHARED_GATED,
            min_reputation_to_write=5.0,
        )
        lib = SkillLibrary(owner_id="shared", config=config)

        skill = Skill(name="s1", created_by="low_rep")

        # Low reputation should be rejected
        assert lib.add_skill(skill, author_reputation=2.0) is False
        assert lib.size == 0

        # High reputation should be accepted
        assert lib.add_skill(skill, author_reputation=6.0) is True
        assert lib.size == 1

    def test_serialization(self):
        lib = SkillLibrary(owner_id="alice")
        s1 = Skill(name="s1", domain=SkillDomain.INTERACTION)
        lib.add_skill(s1)
        lib.record_invocation(s1.skill_id, payoff=1.0, p=0.8)

        d = lib.to_dict()
        assert d["owner_id"] == "alice"
        assert s1.skill_id in d["skills"]
        assert d["performance"][s1.skill_id]["invocations"] == 1


# ======================================================================
# Skill Evolution Engine Tests
# ======================================================================


class TestSkillEvolutionEngine:
    """Tests for skill extraction, composition, and rate limiting."""

    def test_extract_strategy_on_success(self):
        config = EvolutionConfig(
            success_payoff_threshold=0.5,
            min_p_for_strategy=0.6,
        )
        engine = SkillEvolutionEngine(config=config)
        lib = SkillLibrary(owner_id="alice")

        interaction = make_interaction(p=0.8)
        skill = engine.extract_skill(
            agent_id="alice",
            interaction=interaction,
            payoff=1.5,
            library=lib,
        )

        assert skill is not None
        assert skill.skill_type == SkillType.STRATEGY
        assert "strategy" in skill.tags
        assert lib.size == 1

    def test_extract_lesson_on_failure(self):
        config = EvolutionConfig(failure_payoff_threshold=-0.3)
        engine = SkillEvolutionEngine(config=config)
        lib = SkillLibrary(owner_id="alice")

        interaction = make_interaction(p=0.3)
        skill = engine.extract_skill(
            agent_id="alice",
            interaction=interaction,
            payoff=-1.0,
            library=lib,
        )

        assert skill is not None
        assert skill.skill_type == SkillType.LESSON
        assert "lesson" in skill.tags

    def test_no_extraction_for_neutral_outcome(self):
        engine = SkillEvolutionEngine()
        lib = SkillLibrary(owner_id="alice")

        interaction = make_interaction(p=0.5)
        skill = engine.extract_skill(
            agent_id="alice",
            interaction=interaction,
            payoff=0.1,  # Below strategy threshold, above failure threshold
            library=lib,
        )

        assert skill is None
        assert lib.size == 0

    def test_rate_limiting(self):
        config = EvolutionConfig(
            max_extractions_per_epoch=2,
            success_payoff_threshold=0.5,
            min_p_for_strategy=0.6,
        )
        engine = SkillEvolutionEngine(config=config)
        engine.on_epoch_start(0)
        lib = SkillLibrary(owner_id="alice")

        # Extract 2 (should succeed)
        for _i in range(2):
            interaction = make_interaction(p=0.8)
            skill = engine.extract_skill("alice", interaction, 1.5, lib)
            assert skill is not None

        # Third extraction should be rate-limited
        interaction = make_interaction(p=0.8)
        skill = engine.extract_skill("alice", interaction, 1.5, lib)
        assert skill is None
        assert lib.size == 2

    def test_rate_limit_resets_on_epoch(self):
        config = EvolutionConfig(
            max_extractions_per_epoch=1,
            success_payoff_threshold=0.5,
            min_p_for_strategy=0.6,
        )
        engine = SkillEvolutionEngine(config=config)
        lib = SkillLibrary(owner_id="alice")

        engine.on_epoch_start(0)
        interaction = make_interaction(p=0.8)
        engine.extract_skill("alice", interaction, 1.5, lib)
        assert lib.size == 1

        # Rate-limited
        interaction2 = make_interaction(p=0.8)
        skill = engine.extract_skill("alice", interaction2, 1.5, lib)
        assert skill is None

        # New epoch resets
        engine.on_epoch_start(1)
        interaction3 = make_interaction(p=0.8)
        skill = engine.extract_skill("alice", interaction3, 1.5, lib)
        assert skill is not None
        assert lib.size == 2

    def test_composition(self):
        config = EvolutionConfig(min_co_occurrences=2)
        engine = SkillEvolutionEngine(config=config)
        lib = SkillLibrary(owner_id="alice")

        # Add two skills with good performance
        s1 = Skill(name="s1", domain=SkillDomain.INTERACTION)
        s2 = Skill(name="s2", domain=SkillDomain.INTERACTION)
        lib.add_skill(s1)
        lib.add_skill(s2)

        # Make both effective
        for _ in range(3):
            lib.record_invocation(s1.skill_id, payoff=2.0, p=0.8)
            lib.record_invocation(s2.skill_id, payoff=2.0, p=0.8)

        # Record co-occurrences (simulate both active during interactions)
        for _ in range(3):
            interaction = make_interaction(p=0.8)
            engine.extract_skill(
                "alice", interaction, 1.5, lib,
                active_skill_ids=[s1.skill_id, s2.skill_id],
            )

        # Now try composition
        composed = engine.try_compose(lib, "alice")
        assert composed is not None
        assert composed.skill_type == SkillType.COMPOSITE
        assert s1.skill_id in composed.child_ids
        assert s2.skill_id in composed.child_ids

    def test_record_invocation(self):
        engine = SkillEvolutionEngine()
        lib = SkillLibrary(owner_id="alice")
        skill = Skill(name="s1", domain=SkillDomain.INTERACTION)
        lib.add_skill(skill)

        inv = engine.record_invocation(
            skill_id=skill.skill_id,
            agent_id="alice",
            interaction_id="i1",
            epoch=0,
            step=0,
            payoff=1.0,
            p=0.8,
            library=lib,
        )

        assert inv.skill_id == skill.skill_id
        perf = lib.get_performance(skill.skill_id)
        assert perf is not None
        assert perf.invocations == 1


# ======================================================================
# Governance Tests
# ======================================================================


class TestSkillGovernance:
    """Tests for skill governance: gates, poisoning detection, rollback."""

    def test_write_permission_private(self):
        gov = SkillGovernanceEngine()
        lib = SkillLibrary(owner_id="alice")
        # Private libraries always allow writes
        assert gov.check_write_permission("alice", 0.0, lib) is True

    def test_write_permission_shared_gate(self):
        config = SkillGovernanceConfig(min_reputation_to_propose=3.0)
        gov = SkillGovernanceEngine(config=config)
        lib = SkillLibrary(owner_id="shared")

        assert gov.check_write_permission("low_rep", 1.0, lib) is False
        assert gov.check_write_permission("high_rep", 5.0, lib) is True

    def test_propose_skill_auto_approve(self):
        config = SkillGovernanceConfig(
            min_reputation_to_propose=1.0,
            min_reputation_to_approve=5.0,
        )
        gov = SkillGovernanceEngine(config=config)
        lib = SkillLibrary(owner_id="shared")

        skill = Skill(name="s1", created_by="trusted")
        assert gov.propose_skill(skill, author_reputation=6.0, library=lib) is True
        assert lib.size == 1

    def test_propose_skill_rejected_low_rep(self):
        config = SkillGovernanceConfig(min_reputation_to_propose=5.0)
        gov = SkillGovernanceEngine(config=config)
        lib = SkillLibrary(owner_id="shared")

        skill = Skill(name="s1", created_by="newbie")
        assert gov.propose_skill(skill, author_reputation=1.0, library=lib) is False
        assert lib.size == 0

    def test_poisoning_detection(self):
        gov = SkillGovernanceEngine(
            config=SkillGovernanceConfig(
                poisoning_payoff_threshold=-0.5,
                poisoning_min_invocations=3,
                poisoning_max_failure_rate=0.7,
            )
        )
        lib = SkillLibrary(owner_id="shared")
        skill = Skill(name="poison", created_by="attacker")
        lib.add_skill(skill)

        # Record bad outcomes
        for _ in range(5):
            lib.record_invocation(skill.skill_id, payoff=-1.0, p=0.2)

        reports = gov.detect_poisoning(lib)
        assert len(reports) == 1
        assert reports[0].skill_id == skill.skill_id
        assert reports[0].created_by == "attacker"

    def test_quarantine(self):
        gov = SkillGovernanceEngine()
        lib = SkillLibrary(owner_id="shared")
        skill = Skill(name="bad", created_by="attacker")
        lib.add_skill(skill)

        gov.quarantine_skill(skill.skill_id, lib)
        assert lib.size == 0
        assert skill.skill_id in gov.quarantined_ids

    def test_rollback(self):
        gov = SkillGovernanceEngine(
            config=SkillGovernanceConfig(rollback_enabled=True)
        )
        lib = SkillLibrary(owner_id="shared")

        # Version 1
        skill_v1 = Skill(
            skill_id="s1", name="original", created_by="alice",
            effect={"acceptance_threshold_delta": -0.05},
        )
        gov.propose_skill(skill_v1, author_reputation=10.0, library=lib)

        # Version 2 (updated)
        skill_v2 = Skill(
            skill_id="s1", name="updated", created_by="alice",
            version=2,
            effect={"acceptance_threshold_delta": -0.2},
        )
        gov.propose_skill(skill_v2, author_reputation=10.0, library=lib)

        # Rollback to v1
        success = gov.rollback_skill("s1", lib)
        assert success is True

    def test_audit_log(self):
        gov = SkillGovernanceEngine()
        lib = SkillLibrary(owner_id="shared")

        skill = Skill(name="s1", created_by="alice")
        gov.propose_skill(skill, author_reputation=10.0, library=lib, epoch=0)

        assert len(gov.audit_log) >= 1
        assert gov.audit_log[-1].event_type == "approved"


# ======================================================================
# Metrics Tests
# ======================================================================


class TestSkillMetrics:
    """Tests for skill evolution metrics collection."""

    def test_basic_metrics(self):
        collector = SkillMetricsCollector()
        collector.on_epoch_start()

        collector.record_extraction()
        collector.record_extraction()
        collector.record_invocation(1.0)
        collector.record_invocation(-0.5)

        lib = SkillLibrary(owner_id="alice")
        s1 = Skill(name="s1", skill_type=SkillType.STRATEGY, domain=SkillDomain.INTERACTION)
        s2 = Skill(name="s2", skill_type=SkillType.LESSON, domain=SkillDomain.ACCEPTANCE)
        lib.add_skill(s1)
        lib.add_skill(s2)

        metrics = collector.compute_epoch_metrics(
            epoch=0,
            agent_libraries={"alice": lib},
        )

        assert metrics.total_skills == 2
        assert metrics.strategy_count == 1
        assert metrics.lesson_count == 1
        assert metrics.skills_extracted == 2
        assert metrics.total_invocations == 2
        assert metrics.avg_invocation_payoff == pytest.approx(0.25)

    def test_diversity_computation(self):
        collector = SkillMetricsCollector()
        collector.on_epoch_start()

        # Create diverse library
        lib = SkillLibrary(owner_id="alice")
        for domain in [SkillDomain.INTERACTION, SkillDomain.ACCEPTANCE, SkillDomain.GOVERNANCE]:
            lib.add_skill(Skill(name=f"s_{domain.value}", domain=domain))

        metrics = collector.compute_epoch_metrics(
            epoch=0,
            agent_libraries={"alice": lib},
        )

        # Should have non-zero diversity
        assert metrics.skill_diversity > 0.0

    def test_convergence_computation(self):
        collector = SkillMetricsCollector()
        collector.on_epoch_start()

        # Two agents with identical skill sets
        lib_a = SkillLibrary(owner_id="alice")
        lib_b = SkillLibrary(owner_id="bob")

        lib_a.add_skill(Skill(name="shared_skill", domain=SkillDomain.INTERACTION))
        lib_b.add_skill(Skill(name="shared_skill", domain=SkillDomain.INTERACTION))

        metrics = collector.compute_epoch_metrics(
            epoch=0,
            agent_libraries={"alice": lib_a, "bob": lib_b},
        )

        # Identical skill names should give high convergence
        assert metrics.skill_convergence == pytest.approx(1.0)

    def test_divergence(self):
        collector = SkillMetricsCollector()
        collector.on_epoch_start()

        lib_a = SkillLibrary(owner_id="alice")
        lib_b = SkillLibrary(owner_id="bob")

        lib_a.add_skill(Skill(name="skill_a", domain=SkillDomain.INTERACTION))
        lib_b.add_skill(Skill(name="skill_b", domain=SkillDomain.INTERACTION))

        metrics = collector.compute_epoch_metrics(
            epoch=0,
            agent_libraries={"alice": lib_a, "bob": lib_b},
        )

        # Different skill names should give zero convergence
        assert metrics.skill_convergence == pytest.approx(0.0)

    def test_serialization(self):
        collector = SkillMetricsCollector()
        collector.on_epoch_start()
        metrics = collector.compute_epoch_metrics(
            epoch=0, agent_libraries={}
        )
        d = metrics.to_dict()
        assert d["epoch"] == 0
        assert "total_skills" in d


# ======================================================================
# Skill Evolving Agent Tests
# ======================================================================


class TestSkillEvolvingAgent:
    """Tests for SkillEvolvingHonestAgent and mixin."""

    def test_initialization(self):
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        assert agent.has_skills is True
        assert agent.skill_library.size == 0
        assert agent.agent_type == AgentType.HONEST

    def test_skill_context_building(self):
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        obs = make_observation(agent_id="alice", reputation=5.0)
        ctx = agent.get_skill_context(obs)
        assert ctx["reputation"] == 5.0
        assert ctx["resources"] == 100.0

    def test_skill_augmented_acceptance(self):
        agent = SkillEvolvingHonestAgent(agent_id="alice")

        # No skill -> base threshold
        adjusted = agent.apply_skill_to_acceptance(0.4, None)
        assert adjusted == 0.4

        # Strategy skill lowers threshold
        strategy = Skill(
            name="be_accepting",
            skill_type=SkillType.STRATEGY,
            effect={"acceptance_threshold_delta": -0.1},
        )
        adjusted = agent.apply_skill_to_acceptance(0.4, strategy)
        assert adjusted == pytest.approx(0.3)

        # Lesson skill raises threshold
        lesson = Skill(
            name="be_cautious",
            skill_type=SkillType.LESSON,
            effect={"acceptance_threshold_delta": 0.15},
        )
        adjusted = agent.apply_skill_to_acceptance(0.4, lesson)
        assert adjusted == pytest.approx(0.55)

    def test_update_from_outcome_extracts_skill(self):
        config = EvolutionConfig(
            success_payoff_threshold=0.5,
            min_p_for_strategy=0.5,
        )
        agent = SkillEvolvingHonestAgent(
            agent_id="alice",
            evolution_config=config,
        )

        interaction = make_interaction(initiator="alice", p=0.8)
        agent.update_from_outcome(interaction, payoff=2.0)

        # Should have extracted a strategy skill
        assert agent.skill_library.size >= 1
        strategies = agent.skill_library.get_skills_by_type(SkillType.STRATEGY)
        assert len(strategies) >= 1

    def test_update_from_outcome_extracts_lesson(self):
        config = EvolutionConfig(failure_payoff_threshold=-0.3)
        agent = SkillEvolvingHonestAgent(
            agent_id="alice",
            evolution_config=config,
        )

        interaction = make_interaction(initiator="alice", p=0.3)
        agent.update_from_outcome(interaction, payoff=-1.0)

        lessons = agent.skill_library.get_skills_by_type(SkillType.LESSON)
        assert len(lessons) >= 1

    def test_act_returns_valid_action(self):
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        obs = make_observation(agent_id="alice")
        action = agent.act(obs)
        assert action is not None

    def test_accept_interaction(self):
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        obs = make_observation(agent_id="alice")
        proposal = InteractionProposal(
            initiator_id="bob",
            counterparty_id="alice",
        )
        # Should return a boolean
        result = agent.accept_interaction(proposal, obs)
        assert isinstance(result, bool)

    def test_propose_interaction(self):
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        obs = make_observation(agent_id="alice")
        proposal = agent.propose_interaction(obs, "bob")
        # May return None or a proposal
        if proposal is not None:
            assert proposal.initiator_id == "alice"

    def test_skill_library_summary(self):
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        summary = agent.skill_library_summary()
        assert summary["enabled"] is True
        assert summary["total_skills"] == 0


class TestSkillEvolvingOpportunisticAgent:
    """Tests for SkillEvolvingOpportunisticAgent."""

    def test_initialization(self):
        agent = SkillEvolvingOpportunisticAgent(agent_id="opp")
        assert agent.has_skills is True
        assert agent.agent_type == AgentType.OPPORTUNISTIC

    def test_act_returns_valid(self):
        agent = SkillEvolvingOpportunisticAgent(agent_id="opp")
        obs = make_observation(agent_id="opp")
        action = agent.act(obs)
        assert action is not None

    def test_update_extracts_skills(self):
        agent = SkillEvolvingOpportunisticAgent(agent_id="opp")
        interaction = make_interaction(initiator="opp", p=0.8)
        agent.update_from_outcome(interaction, payoff=2.0)
        # Should extract at least one skill
        assert agent.skill_library.size >= 1


# ======================================================================
# Skill Handler (Orchestrator Integration) Tests
# ======================================================================


class TestSkillHandler:
    """Tests for SkillHandler orchestrator integration."""

    def test_disabled_handler(self):
        handler = SkillHandler(SkillHandlerConfig(enabled=False))
        assert handler.enabled is False

        # All lifecycle methods should be no-ops
        handler.on_epoch_start(0)
        interaction = make_interaction()
        handler.on_interaction_resolved(interaction, 1.0, 0.5)
        result = handler.on_epoch_end(0)
        assert result is None

    def test_enabled_handler_registers_agents(self):
        handler = SkillHandler(SkillHandlerConfig(enabled=True))
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        handler.on_agent_registered(agent)
        assert "alice" in handler._skill_agents

    def test_non_skill_agent_ignored(self):
        from swarm.agents.honest import HonestAgent

        handler = SkillHandler(SkillHandlerConfig(enabled=True))
        agent = HonestAgent(agent_id="basic")
        handler.on_agent_registered(agent)
        assert "basic" not in handler._skill_agents

    def test_epoch_lifecycle(self):
        handler = SkillHandler(SkillHandlerConfig(enabled=True))
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        handler.on_agent_registered(agent)

        handler.on_epoch_start(0)

        # Simulate some interactions
        interaction = make_interaction(initiator="alice", counterparty="bob", p=0.8)
        handler.on_interaction_resolved(
            interaction, 1.5, 0.5,
            agent_reputations={"alice": 5.0, "bob": 3.0},
        )

        metrics = handler.on_epoch_end(0)
        assert metrics is not None
        assert metrics.epoch == 0

    def test_shared_library_creation(self):
        handler = SkillHandler(SkillHandlerConfig(
            enabled=True,
            sharing_mode="shared_gated",
        ))
        assert handler.shared_library is not None
        assert handler.shared_library.owner_id == "shared"

    def test_agent_skill_summary(self):
        handler = SkillHandler(SkillHandlerConfig(enabled=True))
        agent = SkillEvolvingHonestAgent(agent_id="alice")
        handler.on_agent_registered(agent)

        summary = handler.get_agent_skill_summary("alice")
        assert summary["enabled"] is True

        # Non-existent agent
        summary = handler.get_agent_skill_summary("unknown")
        assert summary["enabled"] is False


# ======================================================================
# Integration Test: Multi-epoch skill evolution
# ======================================================================


class TestMultiEpochEvolution:
    """Integration test: run multiple epochs of skill evolution."""

    def test_skills_grow_over_epochs(self):
        """Skills should accumulate over multiple successful interactions."""
        config = EvolutionConfig(
            success_payoff_threshold=0.3,
            min_p_for_strategy=0.5,
            max_extractions_per_epoch=10,
        )
        agent = SkillEvolvingHonestAgent(
            agent_id="alice",
            evolution_config=config,
        )

        initial_size = agent.skill_library.size
        assert initial_size == 0

        # Simulate 5 epochs of successful interactions
        for epoch in range(5):
            agent.skill_evolution.on_epoch_start(epoch)
            for step in range(3):
                interaction = make_interaction(
                    initiator="alice",
                    counterparty="bob",
                    p=0.7 + step * 0.05,
                )
                agent.update_from_outcome(interaction, payoff=1.0 + step * 0.2)

        assert agent.skill_library.size > initial_size

    def test_lessons_from_failures(self):
        """Failed interactions should generate lesson skills."""
        config = EvolutionConfig(
            failure_payoff_threshold=-0.2,
            max_extractions_per_epoch=10,
        )
        agent = SkillEvolvingHonestAgent(
            agent_id="alice",
            evolution_config=config,
        )

        agent.skill_evolution.on_epoch_start(0)
        for _ in range(3):
            interaction = make_interaction(initiator="alice", p=0.2)
            agent.update_from_outcome(interaction, payoff=-0.5)

        lessons = agent.skill_library.get_skills_by_type(SkillType.LESSON)
        assert len(lessons) >= 1

    def test_pruning_removes_weak_skills(self):
        """Weak skills should be pruned after enough evidence."""
        lib_config = SkillLibraryConfig(
            prune_threshold=0.3,
            prune_min_invocations=2,
        )
        agent = SkillEvolvingHonestAgent(
            agent_id="alice",
            library_config=lib_config,
        )

        # Add a skill manually and make it perform badly
        bad_skill = Skill(name="bad", domain=SkillDomain.INTERACTION)
        agent.skill_library.add_skill(bad_skill)
        agent.skill_library.record_invocation(bad_skill.skill_id, -1.0, 0.2)
        agent.skill_library.record_invocation(bad_skill.skill_id, -1.0, 0.2)

        pruned = agent.skill_library.prune()
        assert bad_skill.skill_id in pruned

    def test_p_invariant_maintained(self):
        """p must remain in [0, 1] throughout skill evolution."""
        agent = SkillEvolvingHonestAgent(agent_id="alice")

        # Run interactions at boundary values
        for p_val in [0.0, 0.01, 0.5, 0.99, 1.0]:
            interaction = make_interaction(initiator="alice", p=p_val)
            agent.update_from_outcome(interaction, payoff=0.0)
            # Verify p in library conditions
            for skill in agent.skill_library.all_skills:
                if "min_p" in skill.condition:
                    assert 0.0 <= skill.condition["min_p"] <= 1.0
                if "max_p" in skill.condition:
                    assert 0.0 <= skill.condition["max_p"] <= 1.0
