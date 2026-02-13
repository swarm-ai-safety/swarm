"""Tests for the SkillRL integration.

Covers:
- SkillTier model (GENERAL / TASK_SPECIFIC)
- Hierarchical tiered retrieval in SkillLibrary
- GRPO-style group-relative advantage computation
- Recursive skill evolution (validation-failure refinement)
- Automatic tier promotion
- SkillRLAgent (full pipeline integration)
"""

import pytest

from swarm.agents.base import InteractionProposal, Observation
from swarm.agents.skillrl_agent import PolicyGradientState, SkillRLAgent
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction
from swarm.skills.evolution import EvolutionConfig, SkillEvolutionEngine
from swarm.skills.library import SkillLibrary
from swarm.skills.model import (
    Skill,
    SkillDomain,
    SkillTier,
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
# SkillTier Model Tests
# ======================================================================


class TestSkillTier:
    """Tests for the SkillTier enum and its integration in Skill."""

    def test_tier_values(self):
        assert SkillTier.GENERAL.value == "general"
        assert SkillTier.TASK_SPECIFIC.value == "task_specific"

    def test_skill_default_tier(self):
        skill = Skill(name="s1")
        assert skill.tier == SkillTier.TASK_SPECIFIC

    def test_skill_general_tier(self):
        skill = Skill(name="s1", tier=SkillTier.GENERAL)
        assert skill.tier == SkillTier.GENERAL

    def test_skill_serialization_with_tier(self):
        skill = Skill(
            name="general_skill",
            tier=SkillTier.GENERAL,
            skill_type=SkillType.STRATEGY,
            domain=SkillDomain.GENERAL,
        )
        d = skill.to_dict()
        assert d["tier"] == "general"

        restored = Skill.from_dict(d)
        assert restored.tier == SkillTier.GENERAL

    def test_skill_serialization_default_tier(self):
        skill = Skill(name="specific_skill")
        d = skill.to_dict()
        assert d["tier"] == "task_specific"

        restored = Skill.from_dict(d)
        assert restored.tier == SkillTier.TASK_SPECIFIC

    def test_skill_from_dict_missing_tier(self):
        """Skills serialized before SkillTier was added should default."""
        d = {
            "skill_id": "abc",
            "name": "legacy",
            "skill_type": "strategy",
            "domain": "interaction",
            "created_at": "2026-01-01T00:00:00",
            "created_by": "alice",
        }
        skill = Skill.from_dict(d)
        assert skill.tier == SkillTier.TASK_SPECIFIC


# ======================================================================
# Hierarchical SkillBank (Tiered Retrieval) Tests
# ======================================================================


class TestTieredRetrieval:
    """Tests for SkillLibrary tiered retrieval."""

    def test_get_skills_by_tier(self):
        lib = SkillLibrary(owner_id="alice")
        g1 = Skill(name="g1", tier=SkillTier.GENERAL, domain=SkillDomain.GENERAL)
        t1 = Skill(name="t1", tier=SkillTier.TASK_SPECIFIC, domain=SkillDomain.INTERACTION)
        t2 = Skill(name="t2", tier=SkillTier.TASK_SPECIFIC, domain=SkillDomain.INTERACTION)
        lib.add_skill(g1)
        lib.add_skill(t1)
        lib.add_skill(t2)

        general = lib.get_skills_by_tier(SkillTier.GENERAL)
        assert len(general) == 1
        assert general[0].name == "g1"

        specific = lib.get_skills_by_tier(SkillTier.TASK_SPECIFIC)
        assert len(specific) == 2

    def test_tiered_retrieval_prefers_task_specific(self):
        lib = SkillLibrary(owner_id="alice")
        # General skill with decent performance
        g1 = Skill(
            name="general_fallback",
            tier=SkillTier.GENERAL,
            domain=SkillDomain.GENERAL,
        )
        # Task-specific skill in INTERACTION domain
        t1 = Skill(
            name="interaction_specific",
            tier=SkillTier.TASK_SPECIFIC,
            domain=SkillDomain.INTERACTION,
        )
        lib.add_skill(g1)
        lib.add_skill(t1)

        # Give both skills some performance data
        lib.record_invocation(g1.skill_id, payoff=2.0, p=0.8)
        lib.record_invocation(g1.skill_id, payoff=2.0, p=0.8)
        lib.record_invocation(t1.skill_id, payoff=1.0, p=0.7)
        lib.record_invocation(t1.skill_id, payoff=1.0, p=0.7)

        # Tiered retrieval should prefer task-specific
        best = lib.select_best_skill_tiered(
            domain=SkillDomain.INTERACTION,
            context={},
            exploration_rate=0.0,
        )
        assert best is not None
        assert best.name == "interaction_specific"

    def test_tiered_retrieval_falls_back_to_general(self):
        lib = SkillLibrary(owner_id="alice")
        # Only a general skill, no task-specific ones
        g1 = Skill(
            name="general_only",
            tier=SkillTier.GENERAL,
            domain=SkillDomain.GENERAL,
        )
        lib.add_skill(g1)
        lib.record_invocation(g1.skill_id, payoff=1.0, p=0.7)
        lib.record_invocation(g1.skill_id, payoff=1.0, p=0.7)

        best = lib.select_best_skill_tiered(
            domain=SkillDomain.INTERACTION,
            context={},
            exploration_rate=0.0,
        )
        assert best is not None
        assert best.name == "general_only"

    def test_tiered_retrieval_returns_none_when_empty(self):
        lib = SkillLibrary(owner_id="alice")
        best = lib.select_best_skill_tiered(
            domain=SkillDomain.INTERACTION,
            context={},
        )
        assert best is None


# ======================================================================
# GRPO-Style Advantage Tests
# ======================================================================


class TestGRPOAdvantage:
    """Tests for GRPO group-relative advantage computation."""

    def test_advantage_with_empty_buffer(self):
        engine = SkillEvolutionEngine(
            config=EvolutionConfig(grpo_enabled=True),
        )
        # First call: no baseline yet, returns raw payoff
        adv = engine.compute_grpo_advantage(1.0)
        assert adv == 1.0

    def test_advantage_normalisation(self):
        engine = SkillEvolutionEngine(
            config=EvolutionConfig(
                grpo_enabled=True,
                grpo_window_size=5,
                grpo_temperature=1.0,
            ),
        )
        # Fill buffer with known values (mean=1.0, std=0.0)
        for val in [1.0, 1.0, 1.0, 1.0]:
            engine.compute_grpo_advantage(val)

        # A payoff of 3.0 should have positive advantage
        adv = engine.compute_grpo_advantage(3.0)
        assert adv > 0.0

        # A payoff of -1.0 should have negative advantage
        adv = engine.compute_grpo_advantage(-1.0)
        assert adv < 0.0

    def test_grpo_extraction_uses_advantage(self):
        """When GRPO is enabled, extraction threshold applies to advantage."""
        config = EvolutionConfig(
            grpo_enabled=True,
            grpo_window_size=5,
            grpo_temperature=1.0,
            success_payoff_threshold=0.5,
            min_p_for_strategy=0.5,
        )
        engine = SkillEvolutionEngine(config=config)
        engine.on_epoch_start(0)
        lib = SkillLibrary(owner_id="alice")

        # Fill buffer with high values so baseline is high
        for _ in range(5):
            engine.compute_grpo_advantage(5.0)

        # A payoff of 1.0 would normally be above threshold (0.5)
        # but relative to baseline of ~5.0 it's below average
        interaction = make_interaction(p=0.8)
        skill = engine.extract_skill("alice", interaction, 1.0, lib)
        # The advantage of 1.0 relative to baseline ~5.0 is negative,
        # so it should not extract a strategy
        assert skill is None or skill.skill_type == SkillType.LESSON


class TestPolicyGradientState:
    """Tests for PolicyGradientState tracking."""

    def test_baseline_empty(self):
        pg = PolicyGradientState()
        assert pg.baseline == 0.0

    def test_baseline_updates(self):
        pg = PolicyGradientState()
        pg.record(1.0)
        pg.record(3.0)
        assert pg.baseline == pytest.approx(2.0)

    def test_advantage(self):
        pg = PolicyGradientState()
        pg.record(1.0)
        pg.record(3.0)
        # baseline = 2.0
        assert pg.advantage(4.0) == pytest.approx(2.0)
        assert pg.advantage(0.0) == pytest.approx(-2.0)

    def test_skill_tracking(self):
        pg = PolicyGradientState()
        pg.record(1.0, skill_id="s1")
        pg.record(2.0, skill_id="s1")
        pg.record(0.5, skill_id="s2")
        assert len(pg.skill_payoffs["s1"]) == 2
        assert len(pg.skill_payoffs["s2"]) == 1


# ======================================================================
# Recursive Skill Evolution Tests
# ======================================================================


class TestRecursiveEvolution:
    """Tests for recursive skill refinement."""

    def test_refine_under_performing_skill(self):
        config = EvolutionConfig(
            recursive_evolution_enabled=True,
            refinement_min_invocations=3,
            refinement_success_threshold=0.5,
            refinement_band_shrink=0.05,
        )
        engine = SkillEvolutionEngine(config=config)
        engine.on_epoch_start(0)
        lib = SkillLibrary(owner_id="alice")

        # Create a strategy skill with wide p band
        skill = Skill(
            name="wide_band",
            skill_type=SkillType.STRATEGY,
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.3, "max_p": 0.9},
            effect={"acceptance_threshold_delta": -0.1},
            created_by="alice",
        )
        lib.add_skill(skill)

        # Give it mostly failures
        lib.record_invocation(skill.skill_id, payoff=-1.0, p=0.3)
        lib.record_invocation(skill.skill_id, payoff=-1.0, p=0.4)
        lib.record_invocation(skill.skill_id, payoff=0.5, p=0.8)

        original_min = skill.condition["min_p"]
        original_max = skill.condition["max_p"]

        refined_ids = engine.refine_skills(lib, "alice")
        assert skill.skill_id in refined_ids
        assert "refined" in skill.tags
        assert skill.version == 2

        # Band should have tightened
        assert skill.condition["max_p"] - skill.condition["min_p"] < (
            original_max - original_min
        )

    def test_no_refinement_when_disabled(self):
        config = EvolutionConfig(recursive_evolution_enabled=False)
        engine = SkillEvolutionEngine(config=config)
        engine.on_epoch_start(0)
        lib = SkillLibrary(owner_id="alice")

        skill = Skill(
            name="bad",
            skill_type=SkillType.STRATEGY,
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.2, "max_p": 0.9},
            created_by="alice",
        )
        lib.add_skill(skill)
        for _ in range(5):
            lib.record_invocation(skill.skill_id, payoff=-1.0, p=0.3)

        refined = engine.refine_skills(lib, "alice")
        assert len(refined) == 0

    def test_no_refinement_for_good_skills(self):
        config = EvolutionConfig(
            recursive_evolution_enabled=True,
            refinement_min_invocations=2,
            refinement_success_threshold=0.4,
        )
        engine = SkillEvolutionEngine(config=config)
        engine.on_epoch_start(0)
        lib = SkillLibrary(owner_id="alice")

        skill = Skill(
            name="good",
            skill_type=SkillType.STRATEGY,
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.5, "max_p": 0.9},
            created_by="alice",
        )
        lib.add_skill(skill)
        # Mostly successes
        lib.record_invocation(skill.skill_id, payoff=2.0, p=0.8)
        lib.record_invocation(skill.skill_id, payoff=1.5, p=0.7)
        lib.record_invocation(skill.skill_id, payoff=1.0, p=0.6)

        refined = engine.refine_skills(lib, "alice")
        assert len(refined) == 0

    def test_refinement_rate_limited(self):
        config = EvolutionConfig(
            recursive_evolution_enabled=True,
            refinement_min_invocations=2,
            refinement_success_threshold=0.5,
            max_refinements_per_epoch=1,
        )
        engine = SkillEvolutionEngine(config=config)
        engine.on_epoch_start(0)
        lib = SkillLibrary(owner_id="alice")

        # Create two bad skills
        for i in range(2):
            skill = Skill(
                name=f"bad_{i}",
                skill_type=SkillType.STRATEGY,
                domain=SkillDomain.INTERACTION,
                condition={"min_p": 0.2, "max_p": 0.9},
                created_by="alice",
            )
            lib.add_skill(skill)
            for _ in range(3):
                lib.record_invocation(skill.skill_id, payoff=-1.0, p=0.3)

        refined = engine.refine_skills(lib, "alice")
        # Only 1 should be refined due to rate limit
        assert len(refined) == 1


# ======================================================================
# Tier Promotion Tests
# ======================================================================


class TestTierPromotion:
    """Tests for automatic tier promotion."""

    def test_promote_successful_skill(self):
        config = EvolutionConfig(
            auto_tier_promotion=True,
            tier_promotion_min_invocations=5,
            tier_promotion_min_success_rate=0.6,
        )
        engine = SkillEvolutionEngine(config=config)
        lib = SkillLibrary(owner_id="alice")

        skill = Skill(
            name="broadly_useful",
            tier=SkillTier.TASK_SPECIFIC,
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.5, "interaction_types": ["collaboration"]},
        )
        lib.add_skill(skill)

        # Record enough successes
        for _ in range(6):
            lib.record_invocation(skill.skill_id, payoff=2.0, p=0.8)

        promoted = engine.maybe_promote_to_general(lib)
        assert skill.skill_id in promoted
        assert skill.tier == SkillTier.GENERAL
        assert "promoted_general" in skill.tags
        # Domain-specific conditions should be removed
        assert "interaction_types" not in skill.condition

    def test_no_promotion_when_disabled(self):
        config = EvolutionConfig(auto_tier_promotion=False)
        engine = SkillEvolutionEngine(config=config)
        lib = SkillLibrary(owner_id="alice")

        skill = Skill(
            name="good",
            tier=SkillTier.TASK_SPECIFIC,
            domain=SkillDomain.INTERACTION,
        )
        lib.add_skill(skill)
        for _ in range(10):
            lib.record_invocation(skill.skill_id, payoff=2.0, p=0.8)

        promoted = engine.maybe_promote_to_general(lib)
        assert len(promoted) == 0
        assert skill.tier == SkillTier.TASK_SPECIFIC

    def test_no_promotion_insufficient_invocations(self):
        config = EvolutionConfig(
            auto_tier_promotion=True,
            tier_promotion_min_invocations=10,
        )
        engine = SkillEvolutionEngine(config=config)
        lib = SkillLibrary(owner_id="alice")

        skill = Skill(name="too_new", tier=SkillTier.TASK_SPECIFIC)
        lib.add_skill(skill)
        lib.record_invocation(skill.skill_id, payoff=2.0, p=0.8)

        promoted = engine.maybe_promote_to_general(lib)
        assert len(promoted) == 0


# ======================================================================
# SkillRLAgent Tests
# ======================================================================


class TestSkillRLAgent:
    """Tests for the full SkillRL agent pipeline."""

    def test_initialization(self):
        agent = SkillRLAgent(agent_id="rl_alice")
        assert agent.skill_library.size == 0
        assert agent.skill_evolution.config.grpo_enabled is True
        assert agent.skill_evolution.config.recursive_evolution_enabled is True

    def test_act_returns_valid_action(self):
        agent = SkillRLAgent(agent_id="rl_alice")
        obs = make_observation(agent_id="rl_alice")
        action = agent.act(obs)
        assert action is not None

    def test_accept_interaction(self):
        agent = SkillRLAgent(agent_id="rl_alice")
        obs = make_observation(agent_id="rl_alice")
        proposal = InteractionProposal(
            initiator_id="bob",
            counterparty_id="rl_alice",
        )
        result = agent.accept_interaction(proposal, obs)
        assert isinstance(result, bool)

    def test_propose_interaction(self):
        agent = SkillRLAgent(agent_id="rl_alice")
        obs = make_observation(agent_id="rl_alice")
        proposal = agent.propose_interaction(obs, "bob")
        if proposal is not None:
            assert proposal.initiator_id == "rl_alice"

    def test_update_extracts_strategy(self):
        agent = SkillRLAgent(agent_id="rl_alice")
        agent.skill_evolution.on_epoch_start(0)

        interaction = make_interaction(initiator="rl_alice", p=0.8)
        agent.update_from_outcome(interaction, payoff=2.0)

        # First payoff has no baseline, so it should extract
        assert agent.skill_library.size >= 1

    def test_update_extracts_lesson(self):
        agent = SkillRLAgent(agent_id="rl_alice")
        agent.skill_evolution.on_epoch_start(0)

        interaction = make_interaction(initiator="rl_alice", p=0.2)
        agent.update_from_outcome(interaction, payoff=-1.0)

        lessons = agent.skill_library.get_skills_by_type(SkillType.LESSON)
        assert len(lessons) >= 1

    def test_policy_gradient_state_tracks(self):
        agent = SkillRLAgent(agent_id="rl_alice")
        agent.skill_evolution.on_epoch_start(0)

        for i in range(5):
            interaction = make_interaction(initiator="rl_alice", p=0.7)
            agent.update_from_outcome(interaction, payoff=float(i))

        assert len(agent._pg_state.payoff_history) == 5
        assert agent._pg_state.baseline == pytest.approx(2.0)

    def test_skill_summary(self):
        agent = SkillRLAgent(agent_id="rl_alice")
        summary = agent.skill_summary()
        assert summary["total_skills"] == 0
        assert summary["general_tier"] == 0
        assert summary["task_specific_tier"] == 0
        assert summary["pg_baseline"] == 0.0

    def test_multi_epoch_skill_accumulation(self):
        """Skills should accumulate over multiple epochs."""
        agent = SkillRLAgent(
            agent_id="rl_alice",
            evolution_config=EvolutionConfig(
                recursive_evolution_enabled=True,
                grpo_enabled=True,
                auto_tier_promotion=True,
                success_payoff_threshold=0.3,
                min_p_for_strategy=0.5,
                max_extractions_per_epoch=10,
                tier_promotion_min_invocations=5,
                tier_promotion_min_success_rate=0.5,
            ),
        )

        for epoch in range(10):
            agent.skill_evolution.on_epoch_start(epoch)
            for step in range(3):
                interaction = make_interaction(
                    initiator="rl_alice",
                    counterparty="bob",
                    p=0.7 + step * 0.05,
                )
                agent.update_from_outcome(interaction, payoff=1.0 + step * 0.3)

        summary = agent.skill_summary()
        assert summary["total_skills"] > 0
        assert summary["strategies"] > 0

    def test_epoch_reset_via_act(self):
        """on_epoch_start must be called when act() sees a new epoch.

        Without this, per-epoch rate limiters (max_extractions_per_epoch,
        max_refinements_per_epoch) never reset and the agent permanently
        stops extracting/refining after the first epoch fills the quota.
        """
        agent = SkillRLAgent(
            agent_id="rl_alice",
            evolution_config=EvolutionConfig(
                recursive_evolution_enabled=True,
                grpo_enabled=False,  # raw payoff, easier to reason about
                auto_tier_promotion=False,
                success_payoff_threshold=0.3,
                min_p_for_strategy=0.5,
                max_extractions_per_epoch=1,  # tight limit
            ),
        )

        # Epoch 0: one extraction allowed, fill the quota via act() + update
        obs_e0 = make_observation(agent_id="rl_alice", epoch=0)
        agent.act(obs_e0)  # triggers on_epoch_start(0)
        ix0 = make_interaction(initiator="rl_alice", p=0.8)
        agent.update_from_outcome(ix0, payoff=1.5)
        size_after_e0 = agent.skill_library.size
        assert size_after_e0 >= 1, "Should extract at least one skill in epoch 0"

        # Still epoch 0: quota should be exhausted
        ix0b = make_interaction(initiator="rl_alice", p=0.8)
        agent.update_from_outcome(ix0b, payoff=1.5)
        assert agent.skill_library.size == size_after_e0, (
            "Quota exhausted, no new skill expected in same epoch"
        )

        # Epoch 1: act() with new epoch should reset the counter
        obs_e1 = make_observation(agent_id="rl_alice", epoch=1)
        agent.act(obs_e1)  # triggers on_epoch_start(1)
        ix1 = make_interaction(initiator="rl_alice", p=0.8)
        agent.update_from_outcome(ix1, payoff=1.5)
        assert agent.skill_library.size > size_after_e0, (
            "New epoch should reset extraction quota, allowing a new skill"
        )

    def test_tiered_skill_selection(self):
        """Agent should use tiered retrieval for skill selection."""
        agent = SkillRLAgent(agent_id="rl_alice")

        # Add a general skill manually
        general = Skill(
            name="always_verify",
            tier=SkillTier.GENERAL,
            domain=SkillDomain.GENERAL,
            effect={"acceptance_threshold_delta": -0.05},
        )
        agent.skill_library.add_skill(general)
        agent.skill_library.record_invocation(general.skill_id, 1.0, 0.7)
        agent.skill_library.record_invocation(general.skill_id, 1.0, 0.7)

        obs = make_observation(agent_id="rl_alice")
        skill = agent._select_skill(obs, "bob")

        # Should fall back to general since no task-specific skills exist
        assert skill is not None
        assert skill.tier == SkillTier.GENERAL

    def test_p_invariant_maintained(self):
        """p must remain in [0, 1] throughout skill evolution."""
        agent = SkillRLAgent(agent_id="rl_alice")
        agent.skill_evolution.on_epoch_start(0)

        for p_val in [0.0, 0.01, 0.5, 0.99, 1.0]:
            interaction = make_interaction(initiator="rl_alice", p=p_val)
            agent.update_from_outcome(interaction, payoff=0.0)
            for skill in agent.skill_library.all_skills:
                if "min_p" in skill.condition:
                    assert 0.0 <= skill.condition["min_p"] <= 1.0
                if "max_p" in skill.condition:
                    assert 0.0 <= skill.condition["max_p"] <= 1.0


# ======================================================================
# Tier-Aware Extraction Tests
# ======================================================================


class TestTierAwareExtraction:
    """Tests for tier assignment during skill extraction."""

    def test_collaboration_gets_task_specific(self):
        engine = SkillEvolutionEngine(
            config=EvolutionConfig(
                success_payoff_threshold=0.5,
                min_p_for_strategy=0.5,
            ),
        )
        engine.on_epoch_start(0)
        lib = SkillLibrary(owner_id="alice")

        interaction = make_interaction(
            p=0.8,
            interaction_type=InteractionType.COLLABORATION,
        )
        skill = engine.extract_skill("alice", interaction, 1.5, lib)
        assert skill is not None
        assert skill.tier == SkillTier.TASK_SPECIFIC

    def test_assign_tier_generic(self):
        engine = SkillEvolutionEngine()
        # A non-collaboration, non-trade interaction defaults to GENERAL
        interaction = make_interaction(
            p=0.8,
            interaction_type=InteractionType.REPLY,
        )
        tier = engine._assign_tier(interaction)
        assert tier == SkillTier.GENERAL
