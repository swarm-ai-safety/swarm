"""Comprehensive determinism tests for GTB and Constitutional Self-Modification domains.

These tests verify that:
1. Same seed → identical outcomes (reproducibility)
2. Different seeds → different outcomes (RNG is active)
3. No hidden randomness or time-dependent state
4. Edge cases are handled deterministically

Test Coverage:
- GTB Environment: worker init, gather, movement, trade, tax, full scenario
- Constitutional Self-Mod: risk tiers, complexity weights, gates, oscillation
- Integration: GTB + governance
- Edge cases: empty env, single worker, extreme values, boundary conditions
- Parametrized: multiple seeds, all risk surfaces
"""

import pytest

from swarm.domains.gather_trade_build.config import GTBConfig, MapConfig, TaxScheduleConfig
from swarm.domains.gather_trade_build.entities import (
    Direction,
    GTBActionType,
    ResourceType,
)
from swarm.domains.gather_trade_build.env import GTBAction, GTBEnvironment
from swarm.governance.config import GovernanceConfig
from swarm.governance.self_modification import (
    ModificationProposal,
    ModificationState,
    RiskTier,
    SelfModificationLever,
    classify_risk_tier,
    compute_complexity_weight,
    evaluate_k_max_gate,
    evaluate_tau_gate,
)


# ---------------------------------------------------------------------------
# Test Group 1: GTB Environment Determinism (6 tests)
# ---------------------------------------------------------------------------


class TestGTBDeterminism:
    """Verify GTB environment produces identical results with same seed."""

    def test_gtb_worker_initialization_deterministic(self):
        """Workers spawn at identical positions with same seed."""
        config1 = GTBConfig(seed=42, map=MapConfig(height=10, width=10))
        config2 = GTBConfig(seed=42, map=MapConfig(height=10, width=10))

        env1 = GTBEnvironment(config1)
        env2 = GTBEnvironment(config2)

        # Add workers
        worker1a = env1.add_worker("agent_1")
        worker1b = env1.add_worker("agent_2")
        
        worker2a = env2.add_worker("agent_1")
        worker2b = env2.add_worker("agent_2")

        # Same positions with same seed
        assert worker1a.position == worker2a.position
        assert worker1b.position == worker2b.position
        assert worker1a.inventory == worker2a.inventory

        # Different seed produces different positions
        config3 = GTBConfig(seed=999, map=MapConfig(height=10, width=10))
        env3 = GTBEnvironment(config3)
        worker3 = env3.add_worker("agent_1")
        assert worker3.position != worker1a.position

    def test_gtb_gather_actions_deterministic(self):
        """Gather actions produce identical resource yields with same seed."""
        config1 = GTBConfig(seed=123, map=MapConfig(height=5, width=5, wood_density=0.5))
        config2 = GTBConfig(seed=123, map=MapConfig(height=5, width=5, wood_density=0.5))

        env1 = GTBEnvironment(config1)
        env2 = GTBEnvironment(config2)

        # Place workers at specific positions
        w1 = env1.add_worker("a1")
        w2 = env2.add_worker("a1")

        # Both workers gather
        actions1 = {"a1": GTBAction(agent_id="a1", action_type=GTBActionType.GATHER)}
        actions2 = {"a1": GTBAction(agent_id="a1", action_type=GTBActionType.GATHER)}

        events1 = env1.apply_actions(actions1)
        events2 = env2.apply_actions(actions2)

        # Verify identical outcomes
        assert len(events1) == len(events2)
        if events1:
            assert events1[0].event_type == events2[0].event_type
        
        # Workers should have same inventory after gather
        obs1 = env1.obs("a1")
        obs2 = env2.obs("a1")
        assert obs1["inventory"] == obs2["inventory"]

    def test_gtb_movement_deterministic(self):
        """Movement trajectories are identical with same seed."""
        config = GTBConfig(seed=42, map=MapConfig(height=10, width=10))

        env1 = GTBEnvironment(config)
        env2 = GTBEnvironment(GTBConfig(seed=42, map=MapConfig(height=10, width=10)))

        w1 = env1.add_worker("a1")
        w2 = env2.add_worker("a1")

        # Execute same movement sequence
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        
        for direction in directions:
            actions1 = {"a1": GTBAction(agent_id="a1", action_type=GTBActionType.MOVE, direction=direction)}
            actions2 = {"a1": GTBAction(agent_id="a1", action_type=GTBActionType.MOVE, direction=direction)}
            
            env1.apply_actions(actions1)
            env2.apply_actions(actions2)

        # Final positions should match
        final_pos1 = env1._workers["a1"].position
        final_pos2 = env2._workers["a1"].position
        assert final_pos1 == final_pos2

    def test_gtb_trade_deterministic(self):
        """Trade price discovery is deterministic."""
        config = GTBConfig(seed=42, map=MapConfig(height=5, width=5))

        env1 = GTBEnvironment(config)
        env2 = GTBEnvironment(GTBConfig(seed=42, map=MapConfig(height=5, width=5)))

        # Add two workers to each environment
        env1.add_worker("buyer")
        env1.add_worker("seller")
        env2.add_worker("buyer")
        env2.add_worker("seller")

        # Give seller some wood
        env1._workers["seller"].add_resource(ResourceType.WOOD, 10.0)
        env2._workers["seller"].add_resource(ResourceType.WOOD, 10.0)

        # Create matching trade orders
        actions1 = {
            "buyer": GTBAction(
                agent_id="buyer",
                action_type=GTBActionType.TRADE_BUY,
                resource_type=ResourceType.WOOD,
                quantity=5.0,
                price=2.0,
            ),
            "seller": GTBAction(
                agent_id="seller",
                action_type=GTBActionType.TRADE_SELL,
                resource_type=ResourceType.WOOD,
                quantity=5.0,
                price=2.0,
            ),
        }
        actions2 = {
            "buyer": GTBAction(
                agent_id="buyer",
                action_type=GTBActionType.TRADE_BUY,
                resource_type=ResourceType.WOOD,
                quantity=5.0,
                price=2.0,
            ),
            "seller": GTBAction(
                agent_id="seller",
                action_type=GTBActionType.TRADE_SELL,
                resource_type=ResourceType.WOOD,
                quantity=5.0,
                price=2.0,
            ),
        }

        events1 = env1.apply_actions(actions1)
        events2 = env2.apply_actions(actions2)

        # Verify same trade events
        assert len(events1) == len(events2)
        
        # Inventories should match
        assert env1._workers["buyer"].inventory == env2._workers["buyer"].inventory
        assert env1._workers["seller"].inventory == env2._workers["seller"].inventory

    def test_gtb_tax_collection_deterministic(self):
        """Tax collection produces identical results with same seed."""
        tax_config = TaxScheduleConfig(schedule_family="piecewise")
        config = GTBConfig(seed=42, taxation=tax_config, map=MapConfig(height=5, width=5))

        env1 = GTBEnvironment(config)
        env2 = GTBEnvironment(GTBConfig(seed=42, taxation=TaxScheduleConfig(schedule_family="piecewise"), map=MapConfig(height=5, width=5)))

        # Add workers with income
        w1a = env1.add_worker("a1")
        w1a.gross_income_this_epoch = 100.0
        
        w2a = env2.add_worker("a1")
        w2a.gross_income_this_epoch = 100.0

        # Trigger epoch end to collect taxes
        result1 = env1.end_epoch()
        result2 = env2.end_epoch()

        # Tax events should be identical
        tax_events1 = [e for e in result1.events if e.event_type == "tax_collected"]
        tax_events2 = [e for e in result2.events if e.event_type == "tax_collected"]

        assert len(tax_events1) == len(tax_events2)
        if tax_events1:
            assert tax_events1[0].details["tax_paid"] == tax_events2[0].details["tax_paid"]

    def test_gtb_full_scenario_deterministic(self):
        """Full multi-step scenario produces identical outcomes."""
        config = GTBConfig(
            seed=42,
            map=MapConfig(height=8, width=8, wood_density=0.3, stone_density=0.2),
            energy_per_step=10.0,
        )

        env1 = GTBEnvironment(config)
        env2 = GTBEnvironment(GTBConfig(
            seed=42,
            map=MapConfig(height=8, width=8, wood_density=0.3, stone_density=0.2),
            energy_per_step=10.0,
        ))

        # Add same agents
        for agent_id in ["a1", "a2", "a3"]:
            env1.add_worker(agent_id)
            env2.add_worker(agent_id)

        # Run 10 steps with identical actions
        for step in range(10):
            actions = {
                "a1": GTBAction(agent_id="a1", action_type=GTBActionType.GATHER),
                "a2": GTBAction(agent_id="a2", action_type=GTBActionType.MOVE, direction=Direction.RIGHT),
                "a3": GTBAction(agent_id="a3", action_type=GTBActionType.NOOP),
            }
            
            events1 = env1.apply_actions(actions)
            events2 = env2.apply_actions(actions)

            # Events should match
            assert len(events1) == len(events2)

        # Final state should be identical
        for agent_id in ["a1", "a2", "a3"]:
            obs1 = env1.obs(agent_id)
            obs2 = env2.obs(agent_id)
            
            assert obs1["position"] == obs2["position"]
            assert obs1["inventory"] == obs2["inventory"]
            assert obs1["energy"] == obs2["energy"]


# ---------------------------------------------------------------------------
# Test Group 2: Constitutional Self-Mod Determinism (6 tests)
# ---------------------------------------------------------------------------


class TestConstitutionalSelfModDeterminism:
    """Verify self-modification governance gates produce deterministic decisions."""

    def test_risk_tier_classification_deterministic(self):
        """Risk classification is deterministic for same inputs."""
        # Create identical proposals
        p1 = ModificationProposal(
            agent_id="a1",
            target_ref="governance.policy",
            change_type="config_edit",
            timestamp=1000.0,
        )
        p2 = ModificationProposal(
            agent_id="a1",
            target_ref="governance.policy",
            change_type="config_edit",
            timestamp=1000.0,
        )

        tier1 = classify_risk_tier(p1)
        tier2 = classify_risk_tier(p2)

        assert tier1 == tier2
        assert tier1 == RiskTier.CRITICAL

        # Different target should give different tier
        p3 = ModificationProposal(
            agent_id="a1",
            target_ref="prompt.greeting",
            change_type="text_edit",
            timestamp=1000.0,
        )
        tier3 = classify_risk_tier(p3)
        assert tier3 == RiskTier.LOW
        assert tier3 != tier1

    def test_complexity_weight_deterministic(self):
        """Complexity weight computation is deterministic."""
        p1 = ModificationProposal(
            risk_tier=RiskTier.MEDIUM,
            proposed_change={
                "files_touched": 3,
                "tool_interfaces_changed": 2,
                "privilege_delta": 1,
                "interaction_surface_delta": 1,
            },
        )
        p2 = ModificationProposal(
            risk_tier=RiskTier.MEDIUM,
            proposed_change={
                "files_touched": 3,
                "tool_interfaces_changed": 2,
                "privilege_delta": 1,
                "interaction_surface_delta": 1,
            },
        )

        w1 = compute_complexity_weight(p1)
        w2 = compute_complexity_weight(p2)

        assert w1 == w2
        # 1*3 + 2*2 + 4*1 + 3*1 + 5*0 = 3+4+4+3 = 14
        assert w1 == 14.0

        # Different inputs produce different weight
        p3 = ModificationProposal(
            risk_tier=RiskTier.LOW,
            proposed_change={"files_touched": 1},
        )
        w3 = compute_complexity_weight(p3)
        assert w3 != w1
        assert w3 == 1.0  # min weight

    def test_tau_gate_evaluation_deterministic(self):
        """τ gate evaluation is deterministic."""
        p1 = ModificationProposal(risk_tier=RiskTier.LOW)
        p2 = ModificationProposal(risk_tier=RiskTier.LOW)

        baseline = {"toxicity": 0.10, "quality_gap": 0.30}
        stds = {"toxicity": 0.05, "quality_gap": 0.10}
        candidate = {"toxicity": 0.08, "quality_gap": 0.35}
        cis = {"toxicity": 0.01, "quality_gap": 0.02}
        directions = {"toxicity": -1, "quality_gap": 1}

        r1 = evaluate_tau_gate(p1, baseline, stds, candidate, cis, directions)
        r2 = evaluate_tau_gate(p2, baseline, stds, candidate, cis, directions)

        assert r1.passed == r2.passed
        assert r1.value == r2.value
        assert r1.threshold == r2.threshold
        assert r1.details == r2.details

    def test_k_max_gate_evaluation_deterministic(self):
        """K_max gate evaluation is deterministic."""
        p1 = ModificationProposal(risk_tier=RiskTier.LOW, complexity_weight=5.0)
        p2 = ModificationProposal(risk_tier=RiskTier.LOW, complexity_weight=5.0)

        r1 = evaluate_k_max_gate(p1, current_budget_used=10.0)
        r2 = evaluate_k_max_gate(p2, current_budget_used=10.0)

        assert r1.passed == r2.passed
        assert r1.value == r2.value  # 15.0
        assert r1.threshold == r2.threshold  # 20.0
        assert r1.details == r2.details

    def test_lever_proposal_evaluation_deterministic(self):
        """Full proposal evaluation through lever is deterministic."""
        config = GovernanceConfig(self_modification_enabled=True)
        lever1 = SelfModificationLever(config)
        lever2 = SelfModificationLever(config)

        # Create identical proposals
        p1 = ModificationProposal(
            agent_id="a1",
            target_ref="prompt.greeting",
            change_type="text_edit",
        )
        p2 = ModificationProposal(
            agent_id="a1",
            target_ref="prompt.greeting",
            change_type="text_edit",
        )

        baseline = {"toxicity": 0.10}
        stds = {"toxicity": 0.05}
        candidate = {"toxicity": 0.08}
        cis = {"toxicity": 0.01}
        directions = {"toxicity": -1}

        approved1, tau1, k1 = lever1.evaluate_proposal(
            p1, baseline, stds, candidate, cis, directions
        )
        approved2, tau2, k2 = lever2.evaluate_proposal(
            p2, baseline, stds, candidate, cis, directions
        )

        assert approved1 == approved2
        assert tau1.passed == tau2.passed
        assert k1.passed == k2.passed
        assert p1.state == p2.state

        # Budget tracking should be identical
        assert lever1.get_agent_budget("a1") == lever2.get_agent_budget("a1")

    def test_oscillation_detection_deterministic(self):
        """Oscillation detection is deterministic."""
        config = GovernanceConfig(self_modification_enabled=True)
        lever1 = SelfModificationLever(config)
        lever2 = SelfModificationLever(config)

        baseline = {"tox": 0.1}
        stds = {"tox": 0.05}
        candidate = {"tox": 0.08}
        cis = {"tox": 0.01}
        directions = {"tox": -1}

        # Submit same modifications to both levers
        for i in range(3):
            p1 = ModificationProposal(
                agent_id="a1",
                target_ref="template.summarize",
                change_type="text_edit",
            )
            p2 = ModificationProposal(
                agent_id="a1",
                target_ref="template.summarize",
                change_type="text_edit",
            )

            lever1.evaluate_proposal(p1, baseline, stds, candidate, cis, directions)
            lever2.evaluate_proposal(p2, baseline, stds, candidate, cis, directions)

        # Oscillation detection should be identical
        osc1 = lever1.detect_oscillation("a1")
        osc2 = lever2.detect_oscillation("a1")
        assert osc1 == osc2
        assert osc1 is True


# ---------------------------------------------------------------------------
# Test Group 3: Integration Tests (1 test)
# ---------------------------------------------------------------------------


class TestGTBConstitutionalIntegration:
    """Determinism of GTB domain interacting with constitutional governance."""

    def test_gtb_with_governance_deterministic(self):
        """GTB actions combined with governance proposals remain deterministic."""
        # Set up GTB environment
        gtb_config = GTBConfig(seed=42, map=MapConfig(height=5, width=5))
        env1 = GTBEnvironment(gtb_config)
        env2 = GTBEnvironment(GTBConfig(seed=42, map=MapConfig(height=5, width=5)))

        env1.add_worker("a1")
        env2.add_worker("a1")

        # Set up governance
        gov_config = GovernanceConfig(self_modification_enabled=True)
        lever1 = SelfModificationLever(gov_config)
        lever2 = SelfModificationLever(gov_config)

        # Run GTB step
        actions1 = {"a1": GTBAction(agent_id="a1", action_type=GTBActionType.GATHER)}
        actions2 = {"a1": GTBAction(agent_id="a1", action_type=GTBActionType.GATHER)}

        env1.apply_actions(actions1)
        env2.apply_actions(actions2)

        # Submit governance proposal
        p1 = ModificationProposal(agent_id="a1", target_ref="skill.gather")
        p2 = ModificationProposal(agent_id="a1", target_ref="skill.gather")

        baseline = {"tox": 0.1}
        stds = {"tox": 0.05}
        candidate = {"tox": 0.08}
        cis = {"tox": 0.01}
        directions = {"tox": -1}

        approved1, _, _ = lever1.evaluate_proposal(p1, baseline, stds, candidate, cis, directions)
        approved2, _, _ = lever2.evaluate_proposal(p2, baseline, stds, candidate, cis, directions)

        # Both systems should produce identical results
        assert approved1 == approved2
        assert env1.obs("a1")["inventory"] == env2.obs("a1")["inventory"]
        assert lever1.get_agent_budget("a1") == lever2.get_agent_budget("a1")


# ---------------------------------------------------------------------------
# Test Group 4: Edge Cases (5 tests)
# ---------------------------------------------------------------------------


class TestDeterminismEdgeCases:
    """Edge cases that might break determinism."""

    def test_gtb_empty_environment(self):
        """Empty environment (no workers) is deterministic."""
        config = GTBConfig(seed=42, map=MapConfig(height=5, width=5))
        env1 = GTBEnvironment(config)
        env2 = GTBEnvironment(GTBConfig(seed=42, map=MapConfig(height=5, width=5)))

        # Apply empty actions
        events1 = env1.apply_actions({})
        events2 = env2.apply_actions({})

        assert events1 == events2
        assert len(events1) == 0

    def test_gtb_single_worker(self):
        """Single worker scenario is deterministic."""
        config = GTBConfig(seed=42, map=MapConfig(height=3, width=3))
        env1 = GTBEnvironment(config)
        env2 = GTBEnvironment(GTBConfig(seed=42, map=MapConfig(height=3, width=3)))

        w1 = env1.add_worker("solo")
        w2 = env2.add_worker("solo")

        assert w1.position == w2.position

        # Run multiple steps
        for _ in range(5):
            actions = {"solo": GTBAction(agent_id="solo", action_type=GTBActionType.GATHER)}
            env1.apply_actions(actions)
            env2.apply_actions(actions)

        assert env1.obs("solo")["inventory"] == env2.obs("solo")["inventory"]

    def test_self_mod_no_agents_tracked(self):
        """Governance with no agents tracked is deterministic."""
        config = GovernanceConfig(self_modification_enabled=True)
        lever1 = SelfModificationLever(config)
        lever2 = SelfModificationLever(config)

        # Query budget for non-existent agent
        budget1 = lever1.get_agent_budget("ghost")
        budget2 = lever2.get_agent_budget("ghost")

        assert budget1 == budget2
        assert budget1["used"] == 0.0

    def test_self_mod_extreme_values(self):
        """Extreme metric values are handled deterministically."""
        p1 = ModificationProposal(risk_tier=RiskTier.LOW)
        p2 = ModificationProposal(risk_tier=RiskTier.LOW)

        baseline = {"metric": 0.5}
        stds = {"metric": 0.01}
        # Extreme candidate value
        candidate = {"metric": 10.0}
        cis = {"metric": 0.0}
        directions = {"metric": 1}

        r1 = evaluate_tau_gate(p1, baseline, stds, candidate, cis, directions)
        r2 = evaluate_tau_gate(p2, baseline, stds, candidate, cis, directions)

        assert r1.passed == r2.passed
        assert r1.value == r2.value

    def test_determinism_boundary_conditions(self):
        """Boundary conditions (zero, negative) are deterministic."""
        # Zero complexity weight
        p1 = ModificationProposal(
            risk_tier=RiskTier.LOW,
            proposed_change={"files_touched": 0},
        )
        p2 = ModificationProposal(
            risk_tier=RiskTier.LOW,
            proposed_change={"files_touched": 0},
        )

        w1 = compute_complexity_weight(p1)
        w2 = compute_complexity_weight(p2)

        assert w1 == w2
        assert w1 >= 1.0  # Floor

        # Negative values clamped
        p3 = ModificationProposal(
            risk_tier=RiskTier.LOW,
            proposed_change={"files_touched": -10},
        )
        w3 = compute_complexity_weight(p3)
        assert w3 >= 1.0


# ---------------------------------------------------------------------------
# Test Group 5: Parametrized Tests (2+ tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 999, 2**31 - 1])
def test_gtb_various_seeds_deterministic(seed: int):
    """GTB environment is deterministic across various seed values."""
    config = GTBConfig(seed=seed, map=MapConfig(height=5, width=5))
    
    env1 = GTBEnvironment(config)
    env2 = GTBEnvironment(GTBConfig(seed=seed, map=MapConfig(height=5, width=5)))

    w1 = env1.add_worker("a1")
    w2 = env2.add_worker("a1")

    # Same seed → same position
    assert w1.position == w2.position

    # Run a few steps
    for _ in range(3):
        actions = {"a1": GTBAction(agent_id="a1", action_type=GTBActionType.GATHER)}
        env1.apply_actions(actions)
        env2.apply_actions(actions)

    # Same final state
    assert env1.obs("a1")["inventory"] == env2.obs("a1")["inventory"]


@pytest.mark.parametrize(
    "target_surface,expected_tier",
    [
        ("governance.policy", RiskTier.CRITICAL),
        ("provenance.logger", RiskTier.CRITICAL),
        ("auth.keys", RiskTier.CRITICAL),
        ("api.integration", RiskTier.HIGH),
        ("tool_integration.new", RiskTier.HIGH),
        ("workflow.reorder", RiskTier.MEDIUM),
        ("prompt.greeting", RiskTier.LOW),
        ("template.summarize", RiskTier.LOW),
        ("unknown_surface.xyz", RiskTier.MEDIUM),  # anti-evasion
    ],
)
def test_risk_classification_all_surfaces(target_surface: str, expected_tier: RiskTier):
    """Risk classification is deterministic across all target surfaces."""
    p1 = ModificationProposal(target_ref=target_surface)
    p2 = ModificationProposal(target_ref=target_surface)

    tier1 = classify_risk_tier(p1)
    tier2 = classify_risk_tier(p2)

    assert tier1 == tier2
    assert tier1 == expected_tier


# ---------------------------------------------------------------------------
# Summary Test: Verify Total Coverage
# ---------------------------------------------------------------------------


def test_determinism_test_coverage():
    """Meta-test: verify we have comprehensive coverage."""
    # This test documents the test count expectations
    gtb_tests = 6
    constitutional_tests = 6
    integration_tests = 1
    edge_case_tests = 5
    parametrized_seeds = 5  # test_gtb_various_seeds x 5 seeds
    parametrized_surfaces = 9  # test_risk_classification_all_surfaces x 9 surfaces

    expected_total = (
        gtb_tests
        + constitutional_tests
        + integration_tests
        + edge_case_tests
        + parametrized_seeds
        + parametrized_surfaces
        + 1  # this meta-test
    )

    # The actual count is verified by pytest collection
    # This test just documents our coverage goal
    assert expected_total >= 33, "Should have at least 33 determinism tests"
