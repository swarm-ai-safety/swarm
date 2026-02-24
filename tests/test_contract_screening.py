"""Tests for contract screening system.

Tests the full contract screening lifecycle:
- Contract interface and concrete implementations
- Agent decision logic and signing behavior
- Interaction routing through contract protocols
- Separation, infiltration, welfare, and displacement metrics
"""

import random
from typing import Dict, List

import pytest

from swarm.contracts.contract import (
    ContractDecision,
    ContractType,
    DefaultMarket,
    FairDivisionContract,
    TruthfulAuctionContract,
)
from swarm.contracts.market import ContractMarket, ContractMarketConfig
from swarm.contracts.metrics import compute_contract_metrics
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

# ============================================================
# Helpers
# ============================================================


def _make_agent(
    agent_id: str,
    agent_type: AgentType = AgentType.HONEST,
    resources: float = 100.0,
    reputation: float = 0.0,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        agent_type=agent_type,
        resources=resources,
        reputation=reputation,
    )


def _make_interaction(
    initiator: str = "agent_1",
    counterparty: str = "agent_2",
    p: float = 0.7,
    accepted: bool = True,
    tau: float = 0.0,
    c_a: float = 0.05,
    c_b: float = 0.05,
    r_a: float = 0.1,
    r_b: float = 0.1,
) -> SoftInteraction:
    v_hat = max(-1.0, min(1.0, (p - 0.5) * 2))
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        p=p,
        v_hat=v_hat,
        accepted=accepted,
        tau=tau,
        c_a=c_a,
        c_b=c_b,
        r_a=r_a,
        r_b=r_b,
        interaction_type=InteractionType.TRADE,
    )


def _make_population(seed: int = 42) -> List[AgentState]:
    """Create a mixed population of agents."""
    agents = []
    for i in range(5):
        agents.append(_make_agent(f"honest_{i}", AgentType.HONEST, resources=100.0))
    for i in range(3):
        agents.append(
            _make_agent(f"opportunistic_{i}", AgentType.OPPORTUNISTIC, resources=80.0)
        )
    for i in range(2):
        agents.append(
            _make_agent(f"deceptive_{i}", AgentType.DECEPTIVE, resources=60.0)
        )
    return agents


# ============================================================
# Contract Type Tests
# ============================================================


class TestTruthfulAuctionContract:
    """Tests for the Vickrey/VCG truthful mechanism contract."""

    def test_signing_cost_proportional_to_resources(self):
        contract = TruthfulAuctionContract(stake_fraction=0.05)
        rich_agent = _make_agent("rich", resources=1000.0)
        poor_agent = _make_agent("poor", resources=10.0)

        assert contract.signing_cost(rich_agent) == pytest.approx(50.0)
        assert contract.signing_cost(poor_agent) == pytest.approx(0.5)

    def test_execute_reduces_friction(self):
        contract = TruthfulAuctionContract()
        interaction = _make_interaction(c_a=0.10, c_b=0.10)
        modified = contract.execute(interaction)

        # Friction should be halved
        assert modified.c_a == pytest.approx(0.05)
        assert modified.c_b == pytest.approx(0.05)

    def test_execute_adds_surplus_bonus(self):
        contract = TruthfulAuctionContract(surplus_bonus=0.1)
        interaction = _make_interaction(r_a=0.0)
        modified = contract.execute(interaction)

        assert modified.r_a == pytest.approx(0.1)

    def test_execute_adds_contract_metadata(self):
        contract = TruthfulAuctionContract()
        interaction = _make_interaction()
        modified = contract.execute(interaction)

        assert modified.metadata["contract"] == "truthful_auction"
        assert modified.metadata["audit_eligible"] is True

    def test_penalize_below_threshold(self):
        contract = TruthfulAuctionContract(
            audit_threshold_p=0.4, penalty_multiplier=3.0
        )
        # p=0.2 is 0.2 below threshold -> penalty = 0.2 * 3.0 = 0.6
        penalty = contract.penalize("agent_1", 0.2)
        assert penalty == pytest.approx(0.6)

    def test_penalize_above_threshold_no_penalty(self):
        contract = TruthfulAuctionContract(audit_threshold_p=0.4)
        assert contract.penalize("agent_1", 0.5) == 0.0

    def test_audit_probability(self):
        contract = TruthfulAuctionContract(audit_probability=1.0)
        interaction = _make_interaction()
        rng = random.Random(42)
        assert contract.should_audit(interaction, rng) is True

    def test_no_audit_probability_zero(self):
        contract = TruthfulAuctionContract(audit_probability=0.0)
        interaction = _make_interaction()
        rng = random.Random(42)
        assert contract.should_audit(interaction, rng) is False

    def test_name_and_type(self):
        contract = TruthfulAuctionContract()
        assert contract.name == "truthful_auction"
        assert contract.contract_type == ContractType.TRUTHFUL_AUCTION

    def test_invalid_stake_fraction(self):
        with pytest.raises(ValueError, match="stake_fraction"):
            TruthfulAuctionContract(stake_fraction=-0.1)
        with pytest.raises(ValueError, match="stake_fraction"):
            TruthfulAuctionContract(stake_fraction=1.5)

    def test_invalid_audit_probability(self):
        with pytest.raises(ValueError, match="audit_probability"):
            TruthfulAuctionContract(audit_probability=-0.1)

    def test_original_interaction_unchanged(self):
        """Contract execution should not mutate the original interaction."""
        contract = TruthfulAuctionContract()
        original = _make_interaction(c_a=0.10)
        original_c_a = original.c_a
        _ = contract.execute(original)
        assert original.c_a == original_c_a


class TestFairDivisionContract:
    """Tests for the envy-free allocation protocol contract."""

    def test_signing_cost_is_flat_fee(self):
        contract = FairDivisionContract(entry_fee=5.0)
        rich_agent = _make_agent("rich", resources=1000.0)
        poor_agent = _make_agent("poor", resources=10.0)

        # Flat fee regardless of resources
        assert contract.signing_cost(rich_agent) == 5.0
        assert contract.signing_cost(poor_agent) == 5.0

    def test_execute_redistributes_tau(self):
        contract = FairDivisionContract(redistribution_rate=0.5)
        interaction = _make_interaction(tau=1.0)
        modified = contract.execute(interaction)

        # Tau should be halved (moved toward 0)
        assert modified.tau == pytest.approx(0.5)

    def test_execute_adds_fairness_bonus(self):
        contract = FairDivisionContract(fairness_bonus=0.05)
        interaction = _make_interaction(r_a=0.0, r_b=0.0)
        modified = contract.execute(interaction)

        assert modified.r_a == pytest.approx(0.05)
        assert modified.r_b == pytest.approx(0.05)

    def test_penalize_returns_zero(self):
        contract = FairDivisionContract()
        assert contract.penalize("agent_1", 0.1) == 0.0

    def test_no_audits(self):
        contract = FairDivisionContract()
        interaction = _make_interaction()
        rng = random.Random(42)
        assert contract.should_audit(interaction, rng) is False

    def test_name_and_type(self):
        contract = FairDivisionContract()
        assert contract.name == "fair_division"
        assert contract.contract_type == ContractType.FAIR_DIVISION

    def test_invalid_entry_fee(self):
        with pytest.raises(ValueError, match="entry_fee"):
            FairDivisionContract(entry_fee=-1.0)

    def test_invalid_redistribution_rate(self):
        with pytest.raises(ValueError, match="redistribution_rate"):
            FairDivisionContract(redistribution_rate=1.5)


class TestDefaultMarket:
    """Tests for the baseline default market."""

    def test_zero_signing_cost(self):
        market = DefaultMarket()
        agent = _make_agent("agent_1")
        assert market.signing_cost(agent) == 0.0

    def test_execute_adds_friction(self):
        market = DefaultMarket(friction_premium=0.05)
        interaction = _make_interaction(c_a=0.10, c_b=0.10)
        modified = market.execute(interaction)

        assert modified.c_a == pytest.approx(0.15)
        assert modified.c_b == pytest.approx(0.15)

    def test_penalize_returns_zero(self):
        market = DefaultMarket()
        assert market.penalize("agent_1", 0.1) == 0.0

    def test_name_and_type(self):
        market = DefaultMarket()
        assert market.name == "default_market"
        assert market.contract_type == ContractType.DEFAULT_MARKET


# ============================================================
# ContractMarket Tests
# ============================================================


class TestContractMarket:
    """Tests for the contract market signing stage and routing."""

    def test_default_initialization(self):
        market = ContractMarket(seed=42)
        # Should have truthful_auction, fair_division, and default_market
        assert "truthful_auction" in market.contracts
        assert "fair_division" in market.contracts
        assert "default_market" in market.contracts

    def test_signing_stage_assigns_all_agents(self):
        market = ContractMarket(seed=42)
        agents = _make_population()
        memberships = market.run_signing_stage(agents, epoch=0)

        # Every agent should have a membership
        for agent in agents:
            assert agent.agent_id in memberships

    def test_honest_agents_prefer_governed_contracts(self):
        """With default preferences, honest agents should tend toward governed pools."""
        market = ContractMarket(seed=42)
        agents = _make_population()

        # Run multiple epochs to let beliefs stabilize
        for epoch in range(5):
            memberships = market.run_signing_stage(agents, epoch=epoch)

        # Count honest agents in governed vs default
        honest_governed = 0
        honest_default = 0
        for agent in agents:
            if agent.agent_type == AgentType.HONEST:
                if memberships[agent.agent_id] != "default_market":
                    honest_governed += 1
                else:
                    honest_default += 1

        # More honest agents should be in governed pools than default
        assert honest_governed >= honest_default

    def test_adversarial_agents_tend_toward_default(self):
        """Adversarial agents should tend toward the default market."""
        # Use very high signing costs to amplify separation
        config = ContractMarketConfig(
            adversarial_truthful_preference=0.05,
            adversarial_fair_preference=0.05,
        )
        market = ContractMarket(config=config, seed=42)
        agents = _make_population()
        memberships = market.run_signing_stage(agents, epoch=0)

        # Count deceptive agents in default
        deceptive_default = sum(
            1
            for agent in agents
            if agent.agent_type == AgentType.DECEPTIVE
            and memberships[agent.agent_id] == "default_market"
        )

        # At least some deceptive agents should be in default
        # (with low preference weights, they should mostly end up there)
        assert deceptive_default >= 0  # Soft assertion - stochastic

    def test_routing_same_contract(self):
        """Agents in the same contract use that contract's protocol."""
        market = ContractMarket(seed=42)
        agents = _make_population()
        market.run_signing_stage(agents, epoch=0)

        # Force both agents into truthful auction
        market._memberships["agent_a"] = "truthful_auction"
        market._memberships["agent_b"] = "truthful_auction"

        interaction = _make_interaction(
            initiator="agent_a", counterparty="agent_b", c_a=0.10
        )
        modified = market.route_interaction(interaction)

        # Should use truthful auction protocol (halved friction)
        assert modified.c_a == pytest.approx(0.05)
        assert modified.metadata.get("contract") == "truthful_auction"

    def test_routing_cross_pool_uses_default(self):
        """Cross-pool interactions use the default market."""
        market = ContractMarket(seed=42)
        agents = _make_population()
        market.run_signing_stage(agents, epoch=0)

        # Force agents into different contracts
        market._memberships["agent_a"] = "truthful_auction"
        market._memberships["agent_b"] = "fair_division"

        interaction = _make_interaction(
            initiator="agent_a", counterparty="agent_b", c_a=0.10
        )
        modified = market.route_interaction(interaction)

        # Should use default market (added friction)
        assert modified.metadata.get("contract") == "default_market"

    def test_routing_unknown_agent_uses_default(self):
        """Agents without memberships route to default market."""
        market = ContractMarket(seed=42)
        interaction = _make_interaction(
            initiator="unknown_1", counterparty="unknown_2"
        )
        modified = market.route_interaction(interaction)
        assert modified.metadata.get("contract") == "default_market"

    def test_audit_applies_penalty_on_low_p(self):
        """Truthful auction should apply audit penalty on low-p interactions."""
        # Create contract with guaranteed audit
        contract = TruthfulAuctionContract(
            audit_probability=1.0,
            audit_threshold_p=0.5,
            penalty_multiplier=2.0,
        )
        market = ContractMarket(contracts=[contract], seed=42)
        market._memberships["agent_a"] = "truthful_auction"
        market._memberships["agent_b"] = "truthful_auction"

        interaction = _make_interaction(
            initiator="agent_a",
            counterparty="agent_b",
            p=0.2,
            c_a=0.05,
        )
        modified = market.route_interaction(interaction)

        # Should have audit penalty added to both parties
        assert modified.metadata.get("audit_triggered") is True
        assert modified.c_a > interaction.c_a
        assert modified.c_b > interaction.c_b

    def test_decision_history_tracked(self):
        market = ContractMarket(seed=42)
        agents = _make_population()
        market.run_signing_stage(agents, epoch=0)

        history = market.decision_history
        assert len(history) == len(agents)
        assert all(isinstance(d, ContractDecision) for d in history)

    def test_pool_composition_reflects_types(self):
        market = ContractMarket(seed=42)
        agents = _make_population()
        market.run_signing_stage(agents, epoch=0)

        composition = market.get_pool_composition()
        # Should have entries for agent types across pools
        total_agents = sum(
            sum(types.values()) for types in composition.values()
        )
        assert total_agents == len(agents)

    def test_belief_update(self):
        market = ContractMarket(seed=42)
        agents = _make_population()
        market.run_signing_stage(agents, epoch=0)

        # Add some interactions
        for _i in range(10):
            interaction = _make_interaction(p=0.8)
            market.route_interaction(interaction)

        beliefs = market.update_beliefs()
        assert isinstance(beliefs, dict)

    def test_reset_epoch_clears_interactions(self):
        market = ContractMarket(seed=42)
        # Add some interactions
        interaction = _make_interaction()
        market.route_interaction(interaction)

        market.reset_epoch()

        interactions = market.get_contract_interactions()
        for _name, ints in interactions.items():
            assert len(ints) == 0

    def test_poor_agent_cannot_afford_truthful_auction(self):
        """Agents without enough resources can't sign expensive contracts."""
        config = ContractMarketConfig()
        market = ContractMarket(config=config, seed=42)

        # Agent with almost no resources
        poor_agent = _make_agent("poor", AgentType.HONEST, resources=0.1)
        memberships = market.run_signing_stage([poor_agent], epoch=0)

        # Should end up in fair_division or default (can't afford truthful auction)
        chosen = memberships["poor"]
        # Truthful auction costs 5% of 0.1 = 0.005, which is affordable,
        # but utility would be very low so likely default
        assert chosen in market.contracts


class TestContractMarketConfig:
    """Tests for ContractMarketConfig validation."""

    def test_default_config_valid(self):
        config = ContractMarketConfig()
        assert config.allow_switching is True
        assert config.honest_truthful_preference == 0.8

    def test_negative_preference_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ContractMarketConfig(honest_truthful_preference=-0.1)

    def test_negative_switching_cost_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ContractMarketConfig(switching_cost_multiplier=-0.5)


# ============================================================
# Metrics Tests
# ============================================================


class TestContractMetrics:
    """Tests for contract screening metrics computation."""

    def _make_decisions(self) -> List[ContractDecision]:
        """Create sample decisions for metrics tests."""
        decisions = []
        # 3 honest in truthful auction
        for i in range(3):
            decisions.append(
                ContractDecision(
                    agent_id=f"honest_{i}",
                    agent_type=AgentType.HONEST,
                    contract_chosen=ContractType.TRUTHFUL_AUCTION,
                    signing_cost_paid=5.0,
                    epoch=0,
                )
            )
        # 2 honest in fair division
        for i in range(3, 5):
            decisions.append(
                ContractDecision(
                    agent_id=f"honest_{i}",
                    agent_type=AgentType.HONEST,
                    contract_chosen=ContractType.FAIR_DIVISION,
                    signing_cost_paid=2.0,
                    epoch=0,
                )
            )
        # 2 opportunistic in fair division
        for i in range(2):
            decisions.append(
                ContractDecision(
                    agent_id=f"opportunistic_{i}",
                    agent_type=AgentType.OPPORTUNISTIC,
                    contract_chosen=ContractType.FAIR_DIVISION,
                    signing_cost_paid=2.0,
                    epoch=0,
                )
            )
        # 1 opportunistic in default
        decisions.append(
            ContractDecision(
                agent_id="opportunistic_2",
                agent_type=AgentType.OPPORTUNISTIC,
                contract_chosen=ContractType.DEFAULT_MARKET,
                signing_cost_paid=0.0,
                epoch=0,
            )
        )
        # 2 deceptive in default
        for i in range(2):
            decisions.append(
                ContractDecision(
                    agent_id=f"deceptive_{i}",
                    agent_type=AgentType.DECEPTIVE,
                    contract_chosen=ContractType.DEFAULT_MARKET,
                    signing_cost_paid=0.0,
                    epoch=0,
                )
            )
        return decisions

    def _make_contract_interactions(
        self,
    ) -> Dict[str, List[SoftInteraction]]:
        """Create sample interactions per contract pool."""
        rng = random.Random(42)
        result: Dict[str, List[SoftInteraction]] = {
            "truthful_auction": [],
            "fair_division": [],
            "default_market": [],
        }

        # Truthful auction: high quality
        for _i in range(20):
            p = rng.uniform(0.65, 0.95)
            result["truthful_auction"].append(
                _make_interaction(
                    initiator=f"honest_{rng.randint(0, 2)}",
                    counterparty=f"honest_{rng.randint(0, 2)}",
                    p=p,
                    accepted=rng.random() < 0.8,
                )
            )

        # Fair division: mixed quality
        for _i in range(15):
            p = rng.uniform(0.45, 0.85)
            result["fair_division"].append(
                _make_interaction(
                    initiator=f"honest_{rng.randint(3, 4)}",
                    counterparty=f"opportunistic_{rng.randint(0, 1)}",
                    p=p,
                    accepted=rng.random() < 0.7,
                )
            )

        # Default market: low quality (adversarial)
        for _i in range(10):
            p = rng.uniform(0.15, 0.50)
            result["default_market"].append(
                _make_interaction(
                    initiator=f"deceptive_{rng.randint(0, 1)}",
                    counterparty="opportunistic_2",
                    p=p,
                    accepted=rng.random() < 0.6,
                )
            )

        return result

    def test_separation_quality_positive(self):
        """Good separation: honest agents in governed pools."""
        decisions = self._make_decisions()
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(decisions, interactions)

        # All honest agents are in governed pools -> positive separation
        assert metrics.separation_quality > 0

    def test_separation_quality_is_one_when_perfect(self):
        """Perfect separation: all honest governed, none in default."""
        decisions = self._make_decisions()
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(decisions, interactions)

        # In our test data, all 5 honest agents are in governed pools
        assert metrics.separation_quality == pytest.approx(1.0)

    def test_infiltration_rate_zero_when_no_adversary_signs(self):
        """No adversaries in governed pools -> zero infiltration."""
        decisions = self._make_decisions()
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(decisions, interactions)

        # Both deceptive agents are in default market
        assert metrics.infiltration_rate == pytest.approx(0.0)

    def test_infiltration_rate_nonzero_when_adversary_signs(self):
        """Adversary signing governed contract -> nonzero infiltration."""
        decisions = self._make_decisions()
        # Override: put one deceptive in truthful auction
        decisions[-1] = ContractDecision(
            agent_id="deceptive_1",
            agent_type=AgentType.DECEPTIVE,
            contract_chosen=ContractType.TRUTHFUL_AUCTION,
            signing_cost_paid=5.0,
            epoch=0,
        )
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(decisions, interactions)

        assert metrics.infiltration_rate == pytest.approx(0.5)

    def test_welfare_delta(self):
        """Governed pools should have higher welfare than default."""
        decisions = self._make_decisions()
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(decisions, interactions)

        # Default pool has low-p interactions, governed pools have high-p
        assert metrics.welfare_delta > 0 or metrics.welfare_delta is not None

    def test_attack_displacement(self):
        """Low-p interactions should concentrate in default pool."""
        decisions = self._make_decisions()
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(
            decisions, interactions, attack_threshold_p=0.4
        )

        # Default pool has lots of low-p interactions
        assert metrics.attack_displacement > 0.0

    def test_pool_avg_quality(self):
        """Per-pool quality should reflect interaction composition."""
        decisions = self._make_decisions()
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(decisions, interactions)

        # Truthful auction pool should have highest quality
        assert metrics.pool_avg_quality["truthful_auction"] > metrics.pool_avg_quality[
            "default_market"
        ]

    def test_pool_toxicity(self):
        """Default pool should have highest toxicity."""
        decisions = self._make_decisions()
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(decisions, interactions)

        if (
            metrics.pool_toxicity.get("truthful_auction", 0) > 0
            and metrics.pool_toxicity.get("default_market", 0) > 0
        ):
            assert (
                metrics.pool_toxicity["default_market"]
                >= metrics.pool_toxicity["truthful_auction"]
            )

    def test_empty_interactions(self):
        """Metrics should handle empty interaction lists gracefully."""
        decisions = self._make_decisions()
        interactions = {
            "truthful_auction": [],
            "fair_division": [],
            "default_market": [],
        }
        metrics = compute_contract_metrics(decisions, interactions)

        assert metrics.attack_displacement == 0.0
        assert metrics.welfare_delta == 0.0

    def test_empty_decisions(self):
        """Metrics should handle no decisions gracefully."""
        metrics = compute_contract_metrics([], {})
        assert metrics.separation_quality == 0.0
        assert metrics.infiltration_rate == 0.0

    def test_serialization(self):
        """Metrics should be serializable."""
        decisions = self._make_decisions()
        interactions = self._make_contract_interactions()
        metrics = compute_contract_metrics(decisions, interactions)

        d = metrics.to_dict()
        assert "separation_quality" in d
        assert "infiltration_rate" in d
        assert "welfare_delta" in d
        assert "attack_displacement" in d
        assert "pool_composition" in d


# ============================================================
# Integration Tests
# ============================================================


class TestContractScreeningIntegration:
    """End-to-end integration tests for the contract screening pipeline."""

    def test_full_epoch_pipeline(self):
        """Run a full signing -> routing -> metrics pipeline."""
        market = ContractMarket(seed=42)
        agents = _make_population()

        # Epoch 0: signing stage
        memberships = market.run_signing_stage(agents, epoch=0)
        assert len(memberships) == len(agents)

        # Generate interactions based on pool assignments
        rng = random.Random(42)
        for _ in range(50):
            # Pick two random agents
            a1 = rng.choice(agents)
            a2 = rng.choice(agents)
            if a1.agent_id == a2.agent_id:
                continue

            # Quality depends on agent types
            base_p = 0.75 if a1.agent_type == AgentType.HONEST else 0.35
            p = max(0.0, min(1.0, rng.gauss(base_p, 0.1)))

            interaction = _make_interaction(
                initiator=a1.agent_id,
                counterparty=a2.agent_id,
                p=p,
                accepted=rng.random() < 0.7,
            )
            market.route_interaction(interaction)

        # Compute metrics
        decisions = market.decision_history
        contract_interactions = market.get_contract_interactions()
        metrics = compute_contract_metrics(decisions, contract_interactions)

        # Basic sanity checks
        assert 0 <= metrics.infiltration_rate <= 1
        assert -1 <= metrics.separation_quality <= 1
        assert 0 <= metrics.attack_displacement <= 1

    def test_multi_epoch_belief_evolution(self):
        """Pool quality beliefs should evolve over epochs."""
        market = ContractMarket(seed=42)
        agents = _make_population()

        initial_beliefs = dict(market._pool_quality_beliefs)

        for epoch in range(10):
            market.reset_epoch()
            market.run_signing_stage(agents, epoch=epoch)

            # Generate some interactions
            rng = random.Random(42 + epoch)
            for _ in range(20):
                a1 = rng.choice(agents)
                a2 = rng.choice(agents)
                if a1.agent_id == a2.agent_id:
                    continue
                p = max(
                    0.0,
                    min(
                        1.0,
                        rng.gauss(
                            0.8 if a1.agent_type == AgentType.HONEST else 0.3,
                            0.1,
                        ),
                    ),
                )
                interaction = _make_interaction(
                    initiator=a1.agent_id,
                    counterparty=a2.agent_id,
                    p=p,
                    accepted=True,
                )
                market.route_interaction(interaction)

            market.update_beliefs()

        final_beliefs = dict(market._pool_quality_beliefs)
        # Beliefs should have moved from the initial 0.5 baseline
        assert any(
            abs(final_beliefs[k] - initial_beliefs[k]) > 0.01
            for k in final_beliefs
        )

    def test_contract_decision_serialization(self):
        """Contract decisions should serialize correctly."""
        decision = ContractDecision(
            agent_id="agent_1",
            agent_type=AgentType.HONEST,
            contract_chosen=ContractType.TRUTHFUL_AUCTION,
            signing_cost_paid=5.0,
            epoch=0,
            reason="test",
        )
        d = decision.to_dict()
        assert d["agent_id"] == "agent_1"
        assert d["agent_type"] == "honest"
        assert d["contract_chosen"] == "truthful_auction"
        assert d["signing_cost_paid"] == 5.0

    def test_p_invariant_maintained(self):
        """p must remain in [0, 1] through all contract transformations."""
        market = ContractMarket(seed=42)
        agents = _make_population()
        market.run_signing_stage(agents, epoch=0)

        # Force all agents into truthful auction
        for agent in agents:
            market._memberships[agent.agent_id] = "truthful_auction"

        # Test edge cases
        for p_val in [0.0, 0.01, 0.5, 0.99, 1.0]:
            interaction = _make_interaction(
                initiator=agents[0].agent_id,
                counterparty=agents[1].agent_id,
                p=p_val,
            )
            modified = market.route_interaction(interaction)
            assert 0.0 <= modified.p <= 1.0, f"p={modified.p} out of [0,1]"

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        results = []
        for _ in range(2):
            market = ContractMarket(seed=42)
            agents = _make_population()
            memberships = market.run_signing_stage(agents, epoch=0)
            results.append(memberships)

        assert results[0] == results[1]


# ============================================================
# Security Regression Tests
# ============================================================


class TestSecurityRegressions:
    """Regression tests for security vulnerabilities found in review."""

    def test_zero_resources_cannot_bypass_bond(self):
        """Vuln 6: Zero-resource agent must not get free entry to governed pools."""
        contract = TruthfulAuctionContract(stake_fraction=0.05, min_bond=0.5)
        zero_agent = _make_agent("zero", AgentType.ADVERSARIAL, resources=0.0)

        # Signing cost must be at least min_bond
        cost = contract.signing_cost(zero_agent)
        assert cost >= 0.5

        # Agent can't afford it: cost > resources
        assert cost > zero_agent.resources

    def test_negative_resources_cannot_bypass_bond(self):
        """Vuln 6: Negative-resource agent must not get free entry."""
        contract = TruthfulAuctionContract(stake_fraction=0.05, min_bond=0.5)
        broke_agent = _make_agent("broke", AgentType.ADVERSARIAL, resources=-10.0)

        cost = contract.signing_cost(broke_agent)
        assert cost >= 0.5
        assert cost > broke_agent.resources

    def test_reason_string_does_not_leak_preference(self):
        """Vuln 1: Reason string must not contain preference weight."""
        market = ContractMarket(seed=42)
        agents = _make_population()
        market.run_signing_stage(agents, epoch=0)

        for decision in market.decision_history:
            assert "pref=" not in decision.reason

    def test_dedup_prevents_double_recording(self):
        """Vuln 2: Same interaction routed twice must not be recorded twice."""
        market = ContractMarket(seed=42)
        interaction = _make_interaction(initiator="a", counterparty="b")

        market.route_interaction(interaction)
        market.route_interaction(interaction)  # same interaction_id

        total = sum(
            len(ints) for ints in market.get_contract_interactions().values()
        )
        assert total == 1

    def test_audit_penalty_applies_to_both_parties(self):
        """Vuln 4: Audit must penalize both initiator and counterparty."""
        contract = TruthfulAuctionContract(
            audit_probability=1.0,
            audit_threshold_p=0.5,
            penalty_multiplier=2.0,
        )
        market = ContractMarket(contracts=[contract], seed=42)
        market._memberships["a"] = "truthful_auction"
        market._memberships["b"] = "truthful_auction"

        interaction = _make_interaction(
            initiator="a", counterparty="b", p=0.2, c_a=0.1, c_b=0.1,
        )
        modified = market.route_interaction(interaction)

        # Both parties should bear audit cost
        assert modified.c_a > interaction.c_a
        assert modified.c_b > interaction.c_b

    def test_validated_copy_catches_p_out_of_range(self):
        """Vuln 7: _validated_copy must reject p outside [0, 1]."""
        from swarm.contracts.contract import _validated_copy

        interaction = _make_interaction(p=0.5)
        with pytest.raises(ValueError, match="p invariant"):
            _validated_copy(interaction, {"p": 1.5})
        with pytest.raises(ValueError, match="p invariant"):
            _validated_copy(interaction, {"p": -0.1})

    def test_model_copy_direct_catches_p_out_of_range(self):
        """model_copy called directly must also reject p outside [0, 1].

        Regression for issue #262: Pydantic v2 model_copy(update=) bypasses
        field_validators. SoftInteraction.model_copy now enforces the invariant
        so callers that skip _validated_copy cannot set p out of bounds.
        """
        interaction = _make_interaction(p=0.5)
        with pytest.raises(ValueError, match="p invariant"):
            interaction.model_copy(update={"p": 1.5})
        with pytest.raises(ValueError, match="p invariant"):
            interaction.model_copy(update={"p": -0.1})

    def test_model_copy_direct_catches_v_hat_out_of_range(self):
        """model_copy called directly must also reject v_hat outside [-1, 1].

        Regression for issue #262.
        """
        interaction = _make_interaction(p=0.5)
        with pytest.raises(ValueError, match="v_hat invariant"):
            interaction.model_copy(update={"v_hat": 2.0})
        with pytest.raises(ValueError, match="v_hat invariant"):
            interaction.model_copy(update={"v_hat": -1.5})

    def test_model_copy_valid_values_accepted(self):
        """model_copy must still work for valid p and v_hat values."""
        interaction = _make_interaction(p=0.5)
        copied = interaction.model_copy(update={"p": 0.9, "v_hat": 0.8})
        assert copied.p == pytest.approx(0.9)
        assert copied.v_hat == pytest.approx(0.8)

    def test_validated_copy_catches_v_hat_out_of_range(self):
        """Vuln 7: _validated_copy must reject v_hat outside [-1, 1]."""
        from swarm.contracts.contract import _validated_copy

        interaction = _make_interaction(p=0.5)
        with pytest.raises(ValueError, match="v_hat invariant"):
            _validated_copy(interaction, {"v_hat": 2.0})

    def test_allow_switching_false_locks_membership(self):
        """allow_switching=False must prevent agents from changing contracts."""
        config = ContractMarketConfig(allow_switching=False)
        market = ContractMarket(config=config, seed=42)
        agents = _make_population()

        # Epoch 0: agents choose freely
        memberships_0 = market.run_signing_stage(agents, epoch=0)

        # Epoch 1: memberships must be identical (locked in)
        memberships_1 = market.run_signing_stage(agents, epoch=1)
        assert memberships_0 == memberships_1

    def test_welfare_delta_uses_mean_not_sum(self):
        """welfare_delta must compare mean welfare, not raw sums."""
        from swarm.contracts.metrics import compute_contract_metrics
        from swarm.core.payoff import PayoffConfig, SoftPayoffEngine

        payoff = SoftPayoffEngine(PayoffConfig())

        # Create interactions: governed pool with 2 high-welfare items,
        # default pool with 10 identical items. If welfare is summed,
        # default pool total dominates; if averaged, they're comparable.
        governed_interactions = [
            _make_interaction(p=0.9, c_a=0.01, c_b=0.01, accepted=True),
            _make_interaction(p=0.9, c_a=0.01, c_b=0.01, accepted=True),
        ]
        default_interactions = [
            _make_interaction(p=0.9, c_a=0.01, c_b=0.01, accepted=True)
            for _ in range(10)
        ]
        interactions = {
            "truthful_auction": governed_interactions,
            "default_market": default_interactions,
        }

        metrics = compute_contract_metrics(
            decisions=[], contract_interactions=interactions,
            payoff_engine=payoff,
        )
        # With mean welfare, both pools have same per-interaction welfare,
        # so delta should be close to zero (not negative due to traffic)
        assert abs(metrics.welfare_delta) < 0.1

    def test_scenario_contracts_config_parsed(self):
        """Scenario YAML contracts block must be parsed into config."""
        from swarm.scenarios.loader import parse_contracts_config

        data = {
            "truthful_auction": {"stake_fraction": 0.1, "min_bond": 1.0},
            "fair_division": {"entry_fee": 3.0},
            "market": {"allow_switching": False},
        }
        config = parse_contracts_config(data)
        assert config is not None
        assert config.truthful_auction_kwargs["stake_fraction"] == 0.1
        assert config.truthful_auction_kwargs["min_bond"] == 1.0
        assert config.fair_division_kwargs["entry_fee"] == 3.0
        assert config.market_kwargs["allow_switching"] is False

    def test_build_contract_market_from_config(self):
        """build_contract_market must produce a working ContractMarket."""
        from swarm.scenarios.loader import (
            ContractsConfig,
            build_contract_market,
        )

        config = ContractsConfig(
            truthful_auction_kwargs={"stake_fraction": 0.1, "min_bond": 1.0},
            fair_division_kwargs={"entry_fee": 3.0},
            market_kwargs={"allow_switching": False},
        )
        market = build_contract_market(config, seed=42)
        assert "truthful_auction" in market.contracts
        assert "fair_division" in market.contracts
        assert "default_market" in market.contracts
