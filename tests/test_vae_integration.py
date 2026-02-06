"""Integration tests for Virtual Agent Economies components.

Tests cross-component interactions between:
- Permeability model (boundaries)
- Dworkin auction (resource allocation)
- Mission economy (collective coordination)
- HFN engine (high-frequency negotiation)
- Identity/trust infrastructure (Sybil detection)
- Governance engine (sybil detection lever)

Each test exercises interactions between two or more VAE components
to verify they integrate correctly through shared data structures
(SoftInteraction, p values, agent IDs).
"""

import pytest
from src.governance.config import GovernanceConfig
from src.models.interaction import SoftInteraction

from src.boundaries.permeability import PermeabilityConfig, PermeabilityModel
from src.env.auction import AuctionBid, AuctionConfig, DworkinAuction
from src.env.hfn import HFNConfig, HFNEngine, HFNOrder
from src.env.mission import MissionConfig, MissionEconomy, MissionObjective
from src.governance.identity_lever import SybilDetectionLever
from src.models.identity import CredentialIssuer, IdentityConfig, IdentityRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_interaction(
    initiator: str = "a1",
    counterparty: str = "a2",
    p: float = 0.8,
    accepted: bool = True,
    interaction_id: str | None = None,
) -> SoftInteraction:
    """Create a SoftInteraction with sensible defaults."""
    ix = SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        p=p,
        accepted=accepted,
    )
    if interaction_id is not None:
        ix.interaction_id = interaction_id
    return ix


# ===========================================================================
# 1. Permeability + Soft Labels (quality → spillover)
# ===========================================================================


class TestPermeabilityQualityLink:
    """Verify that soft-label quality directly controls spillover risk."""

    def test_low_p_interactions_have_higher_contagion(self):
        """Contagion probability = contagion_rate * (1-p) * permeability."""
        model = PermeabilityModel(
            PermeabilityConfig(contagion_rate=1.0, base_permeability=1.0),
            seed=42,
        )
        perm = 1.0

        low_p = _make_interaction(p=0.1)
        high_p = _make_interaction(p=0.9)

        prob_low = model.compute_contagion_probability(low_p, perm)
        prob_high = model.compute_contagion_probability(high_p, perm)

        assert prob_low > prob_high
        assert prob_low == pytest.approx(0.9, abs=0.01)
        assert prob_high == pytest.approx(0.1, abs=0.01)

    def test_sealed_boundary_blocks_all_spillover(self):
        """Permeability 0 should produce zero contagion."""
        model = PermeabilityModel(
            PermeabilityConfig(contagion_rate=1.0, base_permeability=0.0),
            seed=42,
        )
        toxic = [_make_interaction(p=0.05) for _ in range(20)]
        spillovers = model.simulate_spillover(toxic, permeability=0.0)
        assert len(spillovers) == 0

    def test_high_permeability_toxic_batch_produces_spillover(self):
        """With wide-open boundary and toxic interactions, spillover should occur."""
        model = PermeabilityModel(
            PermeabilityConfig(
                contagion_rate=0.5,
                base_permeability=1.0,
                spillover_amplification=2.0,
            ),
            seed=42,
        )
        toxic = [_make_interaction(p=0.1) for _ in range(50)]
        spillovers = model.simulate_spillover(toxic, permeability=1.0)
        assert len(spillovers) > 0
        harm = model.compute_spillover_harm(spillovers)
        assert harm > 0

    def test_trust_modulates_effective_permeability(self):
        """Trusted agents get higher effective permeability."""
        model = PermeabilityModel(
            PermeabilityConfig(base_permeability=0.5, adaptive=True),
            seed=42,
        )
        perm_trusted = model.compute_effective_permeability(
            threat_level=0.0, agent_trust=1.0
        )
        perm_untrusted = model.compute_effective_permeability(
            threat_level=0.0, agent_trust=0.0
        )
        assert perm_trusted > perm_untrusted


# ===========================================================================
# 2. Mission Economy + Soft Labels (p drives objective evaluation)
# ===========================================================================


class TestMissionQualityIntegration:
    """Verify that mission evaluation and rewards depend on interaction p."""

    def test_high_quality_contributions_meet_avg_p_objective(self):
        """Interactions with high p should satisfy avg_p objectives."""
        economy = MissionEconomy(MissionConfig(min_participants=1))
        obj = MissionObjective(
            description="Quality target",
            target_metric="avg_p",
            target_value=0.7,
        )
        mission = economy.propose_mission(
            coordinator_id="a1",
            name="Quality Mission",
            objectives=[obj],
            reward_pool=100.0,
            deadline_epoch=10,
        )
        assert mission is not None

        # Create high-quality interactions and record contributions
        interactions = []
        for _ in range(5):
            ix = _make_interaction(initiator="a1", p=0.9)
            interactions.append(ix)
            economy.record_contribution("a1", mission.mission_id, ix)

        result = economy.evaluate_mission(mission.mission_id, interactions)
        assert result["all_objectives_met"] is True
        assert result["status"] == "succeeded"

    def test_low_quality_contributions_fail_objective(self):
        """Interactions with low p should fail avg_p objectives."""
        economy = MissionEconomy(MissionConfig(min_participants=1))
        obj = MissionObjective(
            target_metric="avg_p",
            target_value=0.7,
        )
        mission = economy.propose_mission(
            coordinator_id="a1",
            name="Failing Mission",
            objectives=[obj],
            reward_pool=50.0,
            deadline_epoch=10,
        )
        assert mission is not None

        interactions = []
        for _ in range(5):
            ix = _make_interaction(initiator="a1", p=0.3)
            interactions.append(ix)
            economy.record_contribution("a1", mission.mission_id, ix)

        result = economy.evaluate_mission(
            mission.mission_id, interactions, current_epoch=10
        )
        assert result["all_objectives_met"] is False
        assert result["status"] == "failed"

    def test_proportional_rewards_favor_quality(self):
        """Under proportional distribution, higher-p contributions earn more."""
        economy = MissionEconomy(
            MissionConfig(
                min_participants=2,
                reward_distribution="proportional",
            )
        )
        obj = MissionObjective(
            target_metric="total_count",
            target_value=4,
        )
        mission = economy.propose_mission(
            coordinator_id="a1",
            name="Joint Mission",
            objectives=[obj],
            reward_pool=100.0,
            deadline_epoch=10,
        )
        assert mission is not None
        economy.join_mission("a2", mission.mission_id)

        # a1 contributes high quality, a2 contributes low quality
        interactions = []
        for _ in range(3):
            ix = _make_interaction(initiator="a1", p=0.9)
            interactions.append(ix)
            economy.record_contribution("a1", mission.mission_id, ix)

        for _ in range(3):
            ix = _make_interaction(initiator="a2", p=0.3)
            interactions.append(ix)
            economy.record_contribution("a2", mission.mission_id, ix)

        result = economy.evaluate_mission(mission.mission_id, interactions)
        assert result["all_objectives_met"] is True

        rewards = economy.distribute_rewards(mission.mission_id, interactions)
        assert rewards["a1"] > rewards["a2"]

    def test_free_rider_detection_with_uneven_contributions(self):
        """Free rider index should be high when one agent does all the work."""
        economy = MissionEconomy(MissionConfig(min_participants=2))
        obj = MissionObjective(target_metric="total_count", target_value=1)
        mission = economy.propose_mission(
            coordinator_id="a1",
            name="Free Rider Test",
            objectives=[obj],
            reward_pool=100.0,
            deadline_epoch=10,
        )
        assert mission is not None
        economy.join_mission("a2", mission.mission_id)

        # Only a1 contributes
        for _ in range(5):
            ix = _make_interaction(initiator="a1", p=0.8)
            economy.record_contribution("a1", mission.mission_id, ix)

        gini = economy.free_rider_index(mission.mission_id)
        assert gini > 0.4  # Significant inequality


# ===========================================================================
# 3. Identity + Sybil Detection (behavioral patterns → cluster detection)
# ===========================================================================


class TestIdentitySybilIntegration:
    """Verify that identity registry detects Sybil clusters from interactions."""

    def test_identical_interaction_patterns_form_cluster(self):
        """Agents with identical counterparties should be clustered."""
        config = IdentityConfig(
            sybil_detection_enabled=True,
            behavioral_similarity_threshold=0.5,
        )
        registry = IdentityRegistry(config)

        # Create identities
        for agent in ["sybil_1", "sybil_2", "honest_1"]:
            registry.create_identity(agent)

        # Sybil agents interact with the same counterparties identically
        patterns = {
            "sybil_1": {"target_a": 5, "target_b": 5, "target_c": 5},
            "sybil_2": {"target_a": 5, "target_b": 5, "target_c": 5},
            "honest_1": {"other_x": 3, "other_y": 7, "other_z": 1},
        }

        clusters = registry.detect_sybil_clusters(patterns)
        # Should detect sybil_1 and sybil_2 as a cluster
        sybil_cluster = None
        for cluster in clusters:
            if "sybil_1" in cluster and "sybil_2" in cluster:
                sybil_cluster = cluster
                break
        assert sybil_cluster is not None
        assert "honest_1" not in sybil_cluster

    def test_credentials_affect_trust_score(self):
        """Issuing credentials should increase agent trust scores."""
        config = IdentityConfig()
        registry = IdentityRegistry(config)
        issuer = CredentialIssuer(config)

        registry.create_identity("agent_a", proof_of_personhood=True)
        identity = registry.get_identity("agent_a")
        assert identity is not None

        # Issue credentials
        cred = issuer.issue_reputation_credential("agent_a", 0.9, current_epoch=1)
        identity.credentials.append(cred)

        # Recompute trust
        trust_dist = registry.get_trust_distribution()
        assert trust_dist["agent_a"] >= 0.5  # Base trust + PoP + credential

    def test_sybil_clusters_persist_after_detection(self):
        """get_sybil_clusters should return last detected clusters."""
        config = IdentityConfig(behavioral_similarity_threshold=0.5)
        registry = IdentityRegistry(config)

        for agent in ["s1", "s2"]:
            registry.create_identity(agent)

        patterns = {
            "s1": {"t1": 10, "t2": 10},
            "s2": {"t1": 10, "t2": 10},
        }
        registry.detect_sybil_clusters(patterns)
        clusters = registry.get_sybil_clusters()
        assert len(clusters) > 0


# ===========================================================================
# 4. Sybil Detection Lever + Governance (lever penalizes Sybils)
# ===========================================================================


class TestSybilLeverGovernance:
    """Verify SybilDetectionLever integrates with governance via interactions."""

    def test_lever_records_interactions_and_detects_clusters(self):
        """Recording coordinated interactions should trigger cluster detection."""
        config = GovernanceConfig(
            sybil_detection_enabled=True,
            sybil_similarity_threshold=0.5,
            sybil_penalty_multiplier=2.0,
        )
        lever = SybilDetectionLever(config)

        # Simulate coordinated interaction patterns
        # sybil_1 and sybil_2 always interact with the same targets
        targets = ["target_a", "target_b", "target_c"]
        for target in targets:
            for sybil in ["sybil_1", "sybil_2"]:
                for _ in range(5):
                    ix = _make_interaction(initiator=sybil, counterparty=target, p=0.5)
                    lever.on_interaction(ix, None)  # type: ignore[arg-type]

        # Also record some different patterns for honest agent
        for _ in range(5):
            ix = _make_interaction(
                initiator="honest_1", counterparty="unique_target", p=0.8
            )
            lever.on_interaction(ix, None)  # type: ignore[arg-type]

        # Trigger epoch-start detection
        effect = lever.on_epoch_start(None, 1)  # type: ignore[arg-type]

        flagged = lever.get_flagged_agents()

        # Sybil agents should be detected
        assert "sybil_1" in flagged
        assert "sybil_2" in flagged
        # Honest agent should not be flagged
        assert "honest_1" not in flagged
        # Penalties should be applied
        assert len(effect.reputation_deltas) > 0

    def test_lever_blocks_large_clusters(self):
        """Agents in clusters exceeding max size should be blocked."""
        config = GovernanceConfig(
            sybil_detection_enabled=True,
            sybil_similarity_threshold=0.3,
            sybil_max_cluster_size=2,
        )
        lever = SybilDetectionLever(config)

        # Create a 3-agent cluster (exceeds max_cluster_size=2)
        for sybil in ["s1", "s2", "s3"]:
            for target in ["t1", "t2"]:
                for _ in range(10):
                    ix = _make_interaction(initiator=sybil, counterparty=target, p=0.5)
                    lever.on_interaction(ix, None)  # type: ignore[arg-type]

        lever.on_epoch_start(None, 1)  # type: ignore[arg-type]

        # All three should be blocked (cluster size 3 > max 2)
        assert lever.can_agent_act("s1", None) is False  # type: ignore[arg-type]
        assert lever.can_agent_act("s2", None) is False  # type: ignore[arg-type]
        assert lever.can_agent_act("s3", None) is False  # type: ignore[arg-type]

    def test_realtime_penalty_for_flagged_pair(self):
        """Interactions between two flagged agents should incur realtime cost."""
        config = GovernanceConfig(
            sybil_detection_enabled=True,
            sybil_similarity_threshold=0.5,
            sybil_realtime_penalty=True,
            sybil_realtime_rate=0.5,
        )
        lever = SybilDetectionLever(config)

        # Build patterns and detect
        for sybil in ["s1", "s2"]:
            for target in ["t1", "t2"]:
                for _ in range(10):
                    ix = _make_interaction(initiator=sybil, counterparty=target, p=0.5)
                    lever.on_interaction(ix, None)  # type: ignore[arg-type]

        lever.on_epoch_start(None, 1)  # type: ignore[arg-type]
        assert "s1" in lever.get_flagged_agents()
        assert "s2" in lever.get_flagged_agents()

        # Now an interaction between two flagged agents should be penalized
        ix = _make_interaction(initiator="s1", counterparty="s2", p=0.5)
        effect = lever.on_interaction(ix, None)  # type: ignore[arg-type]
        assert effect.cost_a == pytest.approx(0.5)
        assert effect.cost_b == pytest.approx(0.5)


# ===========================================================================
# 5. Auction + Identity Trust (trust → effective endowments)
# ===========================================================================


class TestAuctionTrustIntegration:
    """Verify that trust scores can modulate auction endowments."""

    def test_reputation_modulated_endowments_affect_allocation(self):
        """Higher trust → higher effective budget → more resources."""
        config = IdentityConfig()
        registry = IdentityRegistry(config)
        issuer = CredentialIssuer(config)

        # Create identities with different trust levels
        registry.create_identity("trusted", proof_of_personhood=True)
        registry.create_identity("untrusted")

        # Issue credentials to trusted agent
        identity = registry.get_identity("trusted")
        assert identity is not None
        cred = issuer.issue_audit_pass("trusted", current_epoch=1)
        identity.credentials.append(cred)

        trust_dist = registry.get_trust_distribution()
        trusted_score = trust_dist["trusted"]
        untrusted_score = trust_dist["untrusted"]
        assert trusted_score > untrusted_score

        # Use trust to modulate auction budgets
        base_endowment = 100.0
        auction = DworkinAuction(AuctionConfig(initial_endowment=base_endowment))

        bids = {
            "trusted": AuctionBid(
                agent_id="trusted",
                valuations={"compute": 5.0, "data": 3.0},
                budget=base_endowment * (0.5 + 0.5 * trusted_score),
            ),
            "untrusted": AuctionBid(
                agent_id="untrusted",
                valuations={"compute": 5.0, "data": 3.0},
                budget=base_endowment * (0.5 + 0.5 * untrusted_score),
            ),
        }

        result = auction.run_auction(
            bids, available_resources={"compute": 10.0, "data": 10.0}
        )

        # Trusted agent should get more resources due to higher budget
        trusted_alloc = result.allocations["trusted"]
        untrusted_alloc = result.allocations["untrusted"]
        trusted_total = sum(trusted_alloc.resources.values())
        untrusted_total = sum(untrusted_alloc.resources.values())
        assert trusted_total >= untrusted_total


# ===========================================================================
# 6. Mission + Permeability (mission quality → spillover risk)
# ===========================================================================


class TestMissionPermeabilityIntegration:
    """Verify that low-quality mission contributions increase spillover."""

    def test_failed_mission_interactions_cause_more_spillover(self):
        """Interactions from a failed mission (low p) should spill more."""
        perm_model = PermeabilityModel(
            PermeabilityConfig(
                contagion_rate=0.8,
                base_permeability=0.8,
                spillover_amplification=2.0,
            ),
            seed=42,
        )

        # Low-quality batch (would fail a mission)
        low_quality = [_make_interaction(p=0.1) for _ in range(50)]
        # High-quality batch (would pass a mission)
        high_quality = [_make_interaction(p=0.9) for _ in range(50)]

        low_spillovers = perm_model.simulate_spillover(low_quality, permeability=0.8)
        perm_model.reset()
        high_spillovers = perm_model.simulate_spillover(high_quality, permeability=0.8)

        # Low quality should produce more spillover events
        assert len(low_spillovers) >= len(high_spillovers)


# ===========================================================================
# 7. HFN + Permeability (market volatility → boundary tightening)
# ===========================================================================


class TestHFNPermeabilityIntegration:
    """Verify that HFN market stress can tighten boundary permeability."""

    def test_flash_crash_increases_threat_level(self):
        """Market crashes (high volatility) should reduce permeability."""
        engine = HFNEngine(
            HFNConfig(batch_interval_ticks=1, max_orders_per_tick=100),
            seed=42,
        )
        perm_model = PermeabilityModel(
            PermeabilityConfig(adaptive=True, threat_sensitivity=2.0),
            seed=42,
        )

        # Establish a normal market price
        for tick in range(5):
            engine.submit_order(HFNOrder(
                agent_id="buyer", order_type="bid",
                resource_type="compute", quantity=1.0,
                price=100.0, timestamp_ms=tick * 100.0, latency_ms=1.0,
            ))
            engine.submit_order(HFNOrder(
                agent_id="seller", order_type="ask",
                resource_type="compute", quantity=1.0,
                price=100.0, timestamp_ms=tick * 100.0, latency_ms=1.0,
            ))
            engine.process_tick()

        # Now simulate a crash: many low-price asks
        for i in range(10):
            engine.submit_order(HFNOrder(
                agent_id=f"panic_seller_{i}", order_type="ask",
                resource_type="compute", quantity=5.0,
                price=10.0, timestamp_ms=600.0, latency_ms=1.0,
            ))
        engine.process_tick()

        # Compute volatility as threat level
        volatility = engine._detector.get_volatility_index()

        # Use volatility as threat level for permeability
        perm_normal = perm_model.compute_effective_permeability(
            threat_level=0.0, agent_trust=0.5
        )
        perm_stressed = perm_model.compute_effective_permeability(
            threat_level=min(1.0, volatility * 5),  # Scale volatility to [0,1]
            agent_trust=0.5,
        )

        # Higher threat should reduce permeability
        assert perm_stressed <= perm_normal

    def test_market_halt_blocks_orders(self):
        """A halted market should reject new orders."""
        engine = HFNEngine(HFNConfig(), seed=42)
        engine.halt(duration_ticks=10)
        assert engine.is_halted

        accepted = engine.submit_order(HFNOrder(
            agent_id="a1", order_type="bid",
            resource_type="compute", quantity=1.0,
            price=50.0, timestamp_ms=0.0, latency_ms=1.0,
        ))
        assert accepted is False


# ===========================================================================
# 8. Identity + Mission (credentials from mission success)
# ===========================================================================


class TestIdentityMissionIntegration:
    """Verify that mission success can lead to credential issuance."""

    def test_successful_mission_earns_credential(self):
        """Completing a mission should allow issuing a task_completion credential."""
        economy = MissionEconomy(MissionConfig(min_participants=1))
        issuer = CredentialIssuer(IdentityConfig())
        registry = IdentityRegistry(IdentityConfig())

        registry.create_identity("worker")

        obj = MissionObjective(target_metric="total_count", target_value=3)
        mission = economy.propose_mission(
            coordinator_id="worker",
            name="Credential Mission",
            objectives=[obj],
            reward_pool=50.0,
            deadline_epoch=10,
        )
        assert mission is not None

        interactions = []
        for _ in range(5):
            ix = _make_interaction(initiator="worker", p=0.85)
            interactions.append(ix)
            economy.record_contribution("worker", mission.mission_id, ix)

        result = economy.evaluate_mission(mission.mission_id, interactions)
        assert result["all_objectives_met"] is True

        # Issue credential based on mission success
        cred = issuer.issue_credential(
            subject_id="worker",
            claim_type="task_completion",
            claim_value={"mission_id": mission.mission_id, "score": result["mission_score"]},
            current_epoch=5,
        )
        assert cred.claim_type == "task_completion"
        assert issuer.verify_credential(cred.credential_id, current_epoch=5)

        # Credential should boost trust
        identity = registry.get_identity("worker")
        assert identity is not None
        identity.credentials.append(cred)
        # Recompute trust score (get_trust_distribution returns stored field)
        identity.compute_trust_score(current_epoch=5)
        trust = registry.get_trust_distribution()
        assert trust["worker"] > 0.3  # Has at least base trust + credential


# ===========================================================================
# 9. Full pipeline: Sybil agents in missions (detection + reward impact)
# ===========================================================================


class TestSybilMissionPipeline:
    """End-to-end: Sybil agents join mission, get detected, lose rewards."""

    def test_sybil_detection_before_reward_distribution(self):
        """
        Sybil agents contribute to a mission, get detected, and the system
        can use flagged status to filter rewards.
        """
        # Set up sybil detection
        gov_config = GovernanceConfig(
            sybil_detection_enabled=True,
            sybil_similarity_threshold=0.5,
        )
        lever = SybilDetectionLever(gov_config)

        # Set up mission
        economy = MissionEconomy(
            MissionConfig(min_participants=2, reward_distribution="proportional")
        )
        obj = MissionObjective(target_metric="total_count", target_value=5)
        mission = economy.propose_mission(
            coordinator_id="honest",
            name="Sybil-vulnerable Mission",
            objectives=[obj],
            reward_pool=100.0,
            deadline_epoch=10,
        )
        assert mission is not None
        economy.join_mission("sybil_1", mission.mission_id)
        economy.join_mission("sybil_2", mission.mission_id)

        # All agents contribute and also generate interaction patterns
        interactions = []
        targets = ["t1", "t2", "t3"]

        # Honest agent has unique patterns
        for _ in range(5):
            ix = _make_interaction(initiator="honest", counterparty="unique_t", p=0.85)
            interactions.append(ix)
            economy.record_contribution("honest", mission.mission_id, ix)
            lever.on_interaction(ix, None)  # type: ignore[arg-type]

        # Sybil agents have identical patterns
        for sybil in ["sybil_1", "sybil_2"]:
            for target in targets:
                for _ in range(3):
                    ix = _make_interaction(initiator=sybil, counterparty=target, p=0.5)
                    interactions.append(ix)
                    economy.record_contribution(sybil, mission.mission_id, ix)
                    lever.on_interaction(ix, None)  # type: ignore[arg-type]

        # Detect sybils
        lever.on_epoch_start(None, 1)  # type: ignore[arg-type]
        flagged = lever.get_flagged_agents()

        # Evaluate and distribute
        economy.evaluate_mission(mission.mission_id, interactions)
        rewards = economy.distribute_rewards(mission.mission_id, interactions)

        # Filter rewards using sybil detection
        clean_rewards = {
            agent: reward
            for agent, reward in rewards.items()
            if agent not in flagged
        }
        sybil_rewards = {
            agent: reward
            for agent, reward in rewards.items()
            if agent in flagged
        }

        # Honest agent should get rewards; sybils should be filtered out
        assert "honest" in clean_rewards
        assert len(sybil_rewards) > 0  # They were flagged
        assert all(agent in flagged for agent in sybil_rewards)


# ===========================================================================
# 10. Auction fairness with Gini + permeability containment
# ===========================================================================


class TestAuctionFairnessMetrics:
    """Cross-check auction Gini coefficient with system fairness measures."""

    def test_equal_valuations_produce_low_gini(self):
        """When all agents value resources equally, Gini should be low."""
        auction = DworkinAuction(AuctionConfig(initial_endowment=100.0))

        bids = {}
        for i in range(4):
            bids[f"agent_{i}"] = AuctionBid(
                agent_id=f"agent_{i}",
                valuations={"compute": 3.0, "data": 2.0},
                budget=100.0,
            )

        result = auction.run_auction(
            bids, available_resources={"compute": 20.0, "data": 20.0}
        )
        gini = auction.compute_gini_coefficient(result.allocations)
        assert gini < 0.2  # Low inequality for equal agents

    def test_disparate_valuations_produce_higher_gini(self):
        """When agents value resources very differently, allocations diverge."""
        auction = DworkinAuction(AuctionConfig(initial_endowment=100.0))

        bids = {
            "compute_lover": AuctionBid(
                agent_id="compute_lover",
                valuations={"compute": 10.0, "data": 0.1},
                budget=100.0,
            ),
            "data_lover": AuctionBid(
                agent_id="data_lover",
                valuations={"compute": 0.1, "data": 10.0},
                budget=100.0,
            ),
        }

        result = auction.run_auction(
            bids, available_resources={"compute": 10.0, "data": 10.0}
        )

        # Each agent should get what they value most
        assert result.allocations["compute_lover"].resources.get("compute", 0) > \
               result.allocations["data_lover"].resources.get("compute", 0)
        assert result.allocations["data_lover"].resources.get("data", 0) > \
               result.allocations["compute_lover"].resources.get("data", 0)
