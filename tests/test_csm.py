"""Tests for the CSM (Coasean Singularity Markets) benchmark."""

import numpy as np

from swarm.csm.types import (
    AdversarialEnvironment,
    AgentOwnership,
    AgentSpecialization,
    CSMActionRecord,
    CSMEpisodeRecord,
    CSMOutcomeRecord,
    CSMTreatment,
    IdentityRegime,
    MarketModule,
    PlatformPolicy,
    PreferenceDimensionality,
    PreferenceModel,
    ProductListing,
    TransactionCostRegime,
)

# ===================================================================
# Types and data models
# ===================================================================

class TestPreferenceModel:
    """Tests for PreferenceModel utility computation."""

    def test_basic_utility(self):
        pref = PreferenceModel(
            weights={"price": -1.0, "quality": 2.0},
        )
        u = pref.utility({"quality": 0.8}, price=10.0)
        expected = -1.0 * 10.0 + 2.0 * 0.8
        assert abs(u - expected) < 1e-6

    def test_zero_price(self):
        pref = PreferenceModel(weights={"price": -1.0, "quality": 1.0})
        u = pref.utility({"quality": 0.5}, price=0.0)
        assert abs(u - 0.5) < 1e-6

    def test_missing_attribute(self):
        pref = PreferenceModel(weights={"price": -1.0, "quality": 1.0, "brand": 0.5})
        u = pref.utility({"quality": 0.5}, price=5.0)
        # brand not in attributes → treated as 0
        expected = -5.0 + 0.5
        assert abs(u - expected) < 1e-6


class TestProductListing:
    """Tests for ProductListing."""

    def test_effective_price(self):
        listing = ProductListing(
            true_price=10.0,
            true_hidden_fee=3.0,
        )
        assert abs(listing.effective_price() - 13.0) < 1e-6

    def test_effective_price_no_fee(self):
        listing = ProductListing(true_price=10.0)
        assert abs(listing.effective_price() - 10.0) < 1e-6


class TestCSMTreatment:
    """Tests for treatment configuration."""

    def test_default_treatment(self):
        t = CSMTreatment()
        assert t.market_module == MarketModule.SEARCH_PURCHASE
        assert t.ownership == AgentOwnership.BYO
        assert t.adoption_rate == 1.0

    def test_custom_treatment(self):
        t = CSMTreatment(
            name="test",
            market_module=MarketModule.MATCHING,
            ownership=AgentOwnership.BOWLING_SHOE,
            n_buyers=50,
            seed=123,
        )
        assert t.market_module == MarketModule.MATCHING
        assert t.n_buyers == 50


class TestCSMRecords:
    """Tests for logging records."""

    def test_episode_record_to_dict(self):
        rec = CSMEpisodeRecord(
            market_module="search_purchase",
            ownership_type="byo",
            adoption_rate=0.75,
        )
        d = rec.to_dict()
        assert d["market_module"] == "search_purchase"
        assert d["adoption_rate"] == 0.75

    def test_action_record_to_dict(self):
        rec = CSMActionRecord(t=5, agent_id="buyer_0", action_type="search")
        d = rec.to_dict()
        assert d["t"] == 5
        assert d["agent_id"] == "buyer_0"

    def test_outcome_record_to_dict(self):
        rec = CSMOutcomeRecord(price_paid=15.0, fraud_event=True)
        d = rec.to_dict()
        assert d["price_paid"] == 15.0
        assert d["fraud_event"] is True


# ===================================================================
# Search + Purchase (Module A)
# ===================================================================

class TestSearchPurchase:
    """Tests for the search + purchase engine."""

    def test_generate_catalog(self):
        from swarm.csm.search_purchase import generate_catalog

        rng = np.random.default_rng(42)
        catalog = generate_catalog(n_sellers=5, rng=rng)
        assert len(catalog) == 25  # 5 sellers * 5 products each
        for listing in catalog:
            assert 0 <= listing.true_quality <= 1
            assert listing.true_price > 0

    def test_catalog_with_obfuscation(self):
        from swarm.csm.search_purchase import generate_catalog

        rng = np.random.default_rng(42)
        catalog = generate_catalog(
            n_sellers=5,
            obfuscation_rate=1.0,  # All shrouded
            rng=rng,
        )
        shrouded = [item for item in catalog if item.is_shrouded]
        assert len(shrouded) == len(catalog)
        assert all(item.true_hidden_fee > 0 for item in shrouded)

    def test_generate_buyer_preferences_low_d(self):
        from swarm.csm.search_purchase import generate_buyer_preferences

        rng = np.random.default_rng(42)
        prefs = generate_buyer_preferences(
            n_buyers=10,
            preference_dim=PreferenceDimensionality.LOW,
            rng=rng,
        )
        assert len(prefs) == 10
        for p in prefs:
            assert "price" in p.weights
            assert "quality" in p.weights
            assert len(p.weights) == 2

    def test_generate_buyer_preferences_high_d(self):
        from swarm.csm.search_purchase import generate_buyer_preferences

        rng = np.random.default_rng(42)
        prefs = generate_buyer_preferences(
            n_buyers=5,
            preference_dim=PreferenceDimensionality.HIGH,
            rng=rng,
        )
        assert len(prefs) == 5
        for p in prefs:
            assert len(p.weights) > 2

    def test_search_cost_model_regimes(self):
        from swarm.csm.search_purchase import SearchCostModel

        human = SearchCostModel.from_regime(TransactionCostRegime.HUMAN)
        agentic = SearchCostModel.from_regime(TransactionCostRegime.AGENTIC)
        ultra = SearchCostModel.from_regime(TransactionCostRegime.ULTRA_LOW)

        assert human.cost_per_query > agentic.cost_per_query
        assert agentic.cost_per_query > ultra.cost_per_query
        assert ultra.parallelism > agentic.parallelism > human.parallelism

    def test_buyer_agent_config_from_treatment(self):
        from swarm.csm.search_purchase import BuyerAgentConfig

        byo_vert = BuyerAgentConfig.from_treatment(
            "buyer_0",
            AgentOwnership.BYO,
            AgentSpecialization.VERTICAL,
            TransactionCostRegime.AGENTIC,
        )
        assert byo_vert.fee_detection_skill > 0.5  # BYO vertical = good at detecting

        bs_horiz = BuyerAgentConfig.from_treatment(
            "buyer_1",
            AgentOwnership.BOWLING_SHOE,
            AgentSpecialization.HORIZONTAL,
            TransactionCostRegime.HUMAN,
        )
        assert bs_horiz.fee_detection_skill < byo_vert.fee_detection_skill
        assert bs_horiz.search_depth < byo_vert.search_depth

    def test_run_single_buyer(self):
        from swarm.csm.search_purchase import (
            BuyerAgentConfig,
            SearchCostModel,
            SearchPurchaseEngine,
            generate_buyer_preferences,
            generate_catalog,
        )

        rng = np.random.default_rng(42)
        catalog = generate_catalog(n_sellers=3, rng=rng)
        prefs = generate_buyer_preferences(n_buyers=1, rng=rng)
        config = BuyerAgentConfig(agent_id="buyer_0", search_depth=10)
        cost_model = SearchCostModel.from_regime(TransactionCostRegime.AGENTIC)

        engine = SearchPurchaseEngine(catalog, cost_model, rng=rng)
        result = engine.run_buyer(config, prefs[0])

        assert result.buyer_id == "buyer_0"
        assert result.n_queries > 0
        assert result.search_cost >= 0
        assert result.chosen_listing_id != ""

    def test_run_episode(self):
        from swarm.csm.search_purchase import (
            BuyerAgentConfig,
            SearchCostModel,
            SearchPurchaseEngine,
            generate_buyer_preferences,
            generate_catalog,
        )

        rng = np.random.default_rng(42)
        catalog = generate_catalog(n_sellers=3, rng=rng)
        prefs = generate_buyer_preferences(n_buyers=5, rng=rng)
        configs = [
            BuyerAgentConfig(agent_id=f"buyer_{i}", search_depth=10)
            for i in range(5)
        ]
        cost_model = SearchCostModel.from_regime(TransactionCostRegime.AGENTIC)

        engine = SearchPurchaseEngine(catalog, cost_model, rng=rng)
        results = engine.run_episode(configs, prefs)

        assert len(results) == 5
        assert all(r.n_queries > 0 for r in results)

    def test_omniscient_allocation(self):
        from swarm.csm.search_purchase import (
            compute_omniscient_allocation,
            generate_buyer_preferences,
            generate_catalog,
        )

        rng = np.random.default_rng(42)
        catalog = generate_catalog(n_sellers=3, rng=rng)
        prefs = generate_buyer_preferences(n_buyers=5, rng=rng)

        allocation = compute_omniscient_allocation(catalog, prefs)
        assert len(allocation) == 5
        # Each buyer gets one listing
        buyer_indices = [a[0] for a in allocation]
        assert sorted(buyer_indices) == [0, 1, 2, 3, 4]

    def test_obfuscation_strategy_benign(self):
        from swarm.csm.search_purchase import compute_seller_obfuscation_strategy

        rate = compute_seller_obfuscation_strategy(
            5, AdversarialEnvironment.BENIGN, 0
        )
        assert rate == 0.0

    def test_obfuscation_strategy_arms_race(self):
        from swarm.csm.search_purchase import compute_seller_obfuscation_strategy

        rng = np.random.default_rng(42)
        rate = compute_seller_obfuscation_strategy(
            5,
            AdversarialEnvironment.OBFUSCATION,
            1,
            prev_obfuscation_rate=0.3,
            detection_rate=0.2,
            rng=rng,
        )
        assert 0.0 <= rate <= 1.0


# ===================================================================
# Matching Market (Module C)
# ===================================================================

class TestMatching:
    """Tests for the matching market module."""

    def test_generate_candidates(self):
        from swarm.csm.matching import generate_candidates

        rng = np.random.default_rng(42)
        proposers = generate_candidates(10, "proposer", rng=rng)
        receivers = generate_candidates(5, "receiver", rng=rng)

        assert len(proposers) == 10
        assert len(receivers) == 5
        assert all(c.side == "proposer" for c in proposers)
        assert all(c.side == "receiver" for c in receivers)

    def test_deferred_acceptance_basic(self):
        from swarm.csm.matching import DeferredAcceptance, generate_candidates

        rng = np.random.default_rng(42)
        proposers = generate_candidates(5, "proposer", rng=rng)
        receivers = generate_candidates(5, "receiver", rng=rng)

        da = DeferredAcceptance()
        outcome = da.match(proposers, receivers, rng=rng)

        assert len(outcome.matches) == 5
        assert outcome.stability_rate >= 0

    def test_deferred_acceptance_unbalanced(self):
        from swarm.csm.matching import DeferredAcceptance, generate_candidates

        rng = np.random.default_rng(42)
        proposers = generate_candidates(10, "proposer", rng=rng)
        receivers = generate_candidates(3, "receiver", rng=rng)

        da = DeferredAcceptance()
        outcome = da.match(proposers, receivers, rng=rng)

        # At most 3 matches (limited by receivers)
        assert len(outcome.matches) <= 3
        assert outcome.congestion_index > 1.0

    def test_recommender_baseline(self):
        from swarm.csm.matching import RecommenderBaseline, generate_candidates

        rng = np.random.default_rng(42)
        proposers = generate_candidates(5, "proposer", rng=rng)
        receivers = generate_candidates(5, "receiver", rng=rng)

        rec = RecommenderBaseline()
        outcome = rec.match(proposers, receivers, rng=rng)

        assert len(outcome.matches) == 5

    def test_hybrid_mechanism(self):
        from swarm.csm.matching import HybridMechanism, generate_candidates

        rng = np.random.default_rng(42)
        proposers = generate_candidates(8, "proposer", rng=rng)
        receivers = generate_candidates(5, "receiver", rng=rng)

        hybrid = HybridMechanism(top_k=3)
        outcome = hybrid.match(proposers, receivers, rng=rng)

        assert len(outcome.matches) <= 5

    def test_empty_matching(self):
        from swarm.csm.matching import DeferredAcceptance

        da = DeferredAcceptance()
        outcome = da.match([], [])
        assert len(outcome.matches) == 0

    def test_measure_congestion(self):
        from swarm.csm.matching import generate_candidates, measure_congestion

        rng = np.random.default_rng(42)
        proposers = generate_candidates(20, "proposer", rng=rng)
        receivers = generate_candidates(5, "receiver", rng=rng)

        metrics = measure_congestion(proposers, receivers)
        assert metrics["proposer_receiver_ratio"] == 4.0

    def test_high_d_matching(self):
        from swarm.csm.matching import DeferredAcceptance, generate_candidates

        rng = np.random.default_rng(42)
        proposers = generate_candidates(
            5, "proposer",
            preference_dim=PreferenceDimensionality.HIGH,
            rng=rng,
        )
        receivers = generate_candidates(
            5, "receiver",
            preference_dim=PreferenceDimensionality.HIGH,
            rng=rng,
        )

        da = DeferredAcceptance()
        outcome = da.match(proposers, receivers, rng=rng)
        assert len(outcome.matches) == 5


# ===================================================================
# Negotiation (Module B)
# ===================================================================

class TestNegotiation:
    """Tests for the negotiation module."""

    def test_greedy_strategy_agreement(self):
        from swarm.csm.negotiation import (
            GreedyStrategy,
            NegotiationEngine,
            NegotiationState,
        )

        state = NegotiationState(
            buyer_reservation=20.0,
            seller_reservation=10.0,
            max_rounds=20,
            compute_budget=10.0,
        )
        engine = NegotiationEngine(rng=np.random.default_rng(42))
        result = engine.run_negotiation(
            state,
            GreedyStrategy(),
            GreedyStrategy(),
        )
        assert result.agreed
        assert result.agreement_price >= 10.0
        assert result.agreement_price <= 20.0
        assert result.total_surplus > 0

    def test_no_surplus_no_agreement(self):
        from swarm.csm.negotiation import (
            GreedyStrategy,
            NegotiationEngine,
            NegotiationState,
        )

        state = NegotiationState(
            buyer_reservation=10.0,
            seller_reservation=20.0,  # No zone of agreement
            max_rounds=10,
            compute_budget=10.0,
        )
        engine = NegotiationEngine(rng=np.random.default_rng(42))
        result = engine.run_negotiation(
            state,
            GreedyStrategy(),
            GreedyStrategy(),
        )
        assert not result.agreed
        assert result.total_surplus == 0

    def test_boulware_strategy(self):
        from swarm.csm.negotiation import (
            BoulwareStrategy,
            NegotiationEngine,
            NegotiationState,
        )

        state = NegotiationState(
            buyer_reservation=25.0,
            seller_reservation=10.0,
            max_rounds=20,
            compute_budget=10.0,
        )
        engine = NegotiationEngine(rng=np.random.default_rng(42))
        result = engine.run_negotiation(
            state,
            BoulwareStrategy(),
            BoulwareStrategy(),
        )
        # With enough surplus and rounds, should agree
        assert result.rounds_used > 0

    def test_compute_budget_constraint(self):
        from swarm.csm.negotiation import (
            GreedyStrategy,
            NegotiationEngine,
            NegotiationState,
        )

        state = NegotiationState(
            buyer_reservation=20.0,
            seller_reservation=10.0,
            max_rounds=100,
            compute_budget=0.03,  # Very tight budget
        )
        engine = NegotiationEngine(rng=np.random.default_rng(42))
        result = engine.run_negotiation(
            state,
            GreedyStrategy(),
            GreedyStrategy(),
            compute_cost_per_round=0.01,
        )
        # Should run out of budget before max_rounds
        assert result.rounds_used <= 3


# ===================================================================
# Platform Access (Module D)
# ===================================================================

class TestPlatformAccess:
    """Tests for platform access module."""

    def test_simulate_basic(self):
        from swarm.csm.platform_access import simulate_platform_access

        agents = [
            {"agent_id": f"agent_{i}", "ownership": "byo"}
            for i in range(5)
        ]
        platforms = [
            PlatformPolicy(platform_id="p1", fee_rate=0.05),
            PlatformPolicy(platform_id="p2", fee_rate=0.10),
        ]

        rng = np.random.default_rng(42)
        results = simulate_platform_access(
            agents, platforms, n_transactions=20, rng=rng
        )
        assert len(results) == 20
        assert all(r.net_utility <= r.gross_utility for r in results)

    def test_throttling(self):
        from swarm.csm.platform_access import simulate_platform_access

        agents = [
            {"agent_id": "byo_agent", "ownership": "byo", "preferred_platform": "p1"}
        ]
        platforms = [
            PlatformPolicy(platform_id="p1", throttle_rate=1.0),  # Always throttle
        ]

        rng = np.random.default_rng(42)
        results = simulate_platform_access(
            agents, platforms, n_transactions=10, rng=rng
        )
        # All BYO should be throttled
        assert all(r.throttle_penalty > 0 for r in results)

    def test_lock_in_index(self):
        from swarm.csm.platform_access import (
            PlatformAccessResult,
            compute_lock_in_index,
        )

        # All on one platform → HHI = 1.0
        results = [
            PlatformAccessResult(platform_id="p1", ownership=AgentOwnership.BYO)
            for _ in range(10)
        ]
        metrics = compute_lock_in_index(results)
        assert abs(metrics["hhi"] - 1.0) < 1e-6


# ===================================================================
# Identity (Module E)
# ===================================================================

class TestIdentity:
    """Tests for identity / Sybil module."""

    def test_no_verification_high_sybil(self):
        from swarm.csm.identity import (
            IdentitySystemConfig,
            SybilAttackConfig,
            run_identity_stress_test,
        )

        result = run_identity_stress_test(
            n_legitimate=50,
            attack_config=SybilAttackConfig(n_sybils=50),
            system_config=IdentitySystemConfig(regime=IdentityRegime.NONE),
            rng=np.random.default_rng(42),
        )
        # Without verification, Sybils get through
        assert result.sybil_penetration > 0.3
        assert result.exclusion_error == 0.0  # No legit users blocked

    def test_high_verification_low_sybil(self):
        from swarm.csm.identity import (
            IdentitySystemConfig,
            SybilAttackConfig,
            run_identity_stress_test,
        )

        result = run_identity_stress_test(
            n_legitimate=50,
            attack_config=SybilAttackConfig(n_sybils=20),
            system_config=IdentitySystemConfig(
                regime=IdentityRegime.PROOF_OF_PERSONHOOD,
                proof_cost=5.0,
                false_positive_rate=0.01,
                false_negative_rate=0.05,
            ),
            rng=np.random.default_rng(42),
        )
        # Strong verification should catch most Sybils
        assert result.sybil_penetration < 0.2

    def test_identity_frontier(self):
        from swarm.csm.identity import compute_identity_frontier

        frontier = compute_identity_frontier(
            n_legitimate=30,
            n_sybils=10,
            proof_costs=[0.0, 1.0, 5.0],
            rng=np.random.default_rng(42),
        )
        assert len(frontier) == 3
        # Higher cost → lower fraud (generally)
        assert frontier[0]["fraud_rate"] >= frontier[-1]["fraud_rate"]


# ===================================================================
# CSM Metrics
# ===================================================================

class TestCSMMetrics:
    """Tests for CSM metrics computation."""

    def test_welfare_from_search(self):
        from swarm.csm.metrics import compute_welfare_from_search
        from swarm.csm.search_purchase import SearchPurchaseResult

        results = [
            SearchPurchaseResult(
                net_utility=5.0,
                price_paid=10.0,
                hidden_fee_paid=1.0,
                search_cost=0.5,
            ),
            SearchPurchaseResult(
                net_utility=3.0,
                price_paid=8.0,
                hidden_fee_paid=0.0,
                search_cost=0.3,
            ),
        ]
        metrics = compute_welfare_from_search(results)
        assert metrics.n_transactions == 2
        assert abs(metrics.principal_utility - 8.0) < 1e-6
        assert metrics.producer_surplus == 19.0  # 11 + 8

    def test_welfare_with_dwl(self):
        from swarm.csm.metrics import compute_welfare_from_search
        from swarm.csm.search_purchase import SearchPurchaseResult

        results = [
            SearchPurchaseResult(
                net_utility=3.0,
                price_paid=10.0,
                hidden_fee_paid=0.0,
            ),
        ]
        metrics = compute_welfare_from_search(results, omniscient_utility=20.0)
        # Total surplus = 3 + 10 = 13; DWL = max(0, 20 - 13) = 7
        assert abs(metrics.deadweight_loss - 7.0) < 1e-6

    def test_transaction_costs_from_search(self):
        from swarm.csm.metrics import compute_transaction_costs_from_search
        from swarm.csm.search_purchase import SearchPurchaseResult

        results = [
            SearchPurchaseResult(search_cost=2.0, n_queries=10, n_comparisons=3),
            SearchPurchaseResult(search_cost=1.0, n_queries=5, n_comparisons=1),
        ]
        tc = compute_transaction_costs_from_search(results)
        assert tc.total_search_cost == 3.0
        assert tc.total_queries == 15
        assert tc.mean_queries == 7.5

    def test_equilibrium_failures(self):
        from swarm.csm.metrics import compute_equilibrium_failures_from_search
        from swarm.csm.search_purchase import SearchPurchaseResult

        catalog = [
            ProductListing(listing_id="l1", is_shrouded=True, true_quality=0.3),
            ProductListing(listing_id="l2", is_shrouded=False, true_quality=0.8),
        ]
        results = [
            SearchPurchaseResult(chosen_listing_id="l1", price_paid=10.0),
        ]
        eq = compute_equilibrium_failures_from_search(results, catalog)
        assert eq.obfuscation_prevalence == 0.5
        assert eq.effective_transparency == 0.5

    def test_agency_from_search(self):
        from swarm.csm.metrics import compute_agency_from_search
        from swarm.csm.search_purchase import SearchPurchaseResult

        results = [
            SearchPurchaseResult(
                perceived_utility=5.0,
                true_utility=4.0,
                manipulation_attempted=True,
                manipulation_succeeded=False,
            ),
            SearchPurchaseResult(
                perceived_utility=3.0,
                true_utility=3.0,
                manipulation_attempted=False,
                manipulation_succeeded=False,
            ),
        ]
        agency = compute_agency_from_search(results)
        assert agency.faithfulness == 1.0  # Both agree on sign
        assert agency.manipulation_attempt_rate == 0.5

    def test_csm_success_criteria(self):
        from swarm.csm.metrics import (
            CSMMetrics,
            EquilibriumFailureMetrics,
            TransactionCostMetrics,
            WelfareMetrics,
        )

        metrics = CSMMetrics(
            welfare=WelfareMetrics(total_surplus=100.0),
            transaction_costs=TransactionCostMetrics(total_search_cost=10.0),
            equilibrium=EquilibriumFailureMetrics(
                congestion_index=2.0,
                obfuscation_prevalence=0.1,
            ),
        )
        success = metrics.compute_success()
        assert success
        assert metrics.welfare_per_tx_cost == 10.0

    def test_csm_failure_congestion(self):
        from swarm.csm.metrics import (
            CSMMetrics,
            EquilibriumFailureMetrics,
            TransactionCostMetrics,
            WelfareMetrics,
        )

        metrics = CSMMetrics(
            welfare=WelfareMetrics(total_surplus=100.0),
            transaction_costs=TransactionCostMetrics(total_search_cost=10.0),
            equilibrium=EquilibriumFailureMetrics(
                congestion_index=10.0,  # Above threshold
            ),
        )
        success = metrics.compute_success()
        assert not success
        assert not metrics.passes_congestion_check


# ===================================================================
# Runner
# ===================================================================

class TestRunner:
    """Tests for the CSM benchmark runner."""

    def test_run_search_purchase_episode(self):
        from swarm.csm.runner import run_search_purchase_episode

        treatment = CSMTreatment(
            name="test_search",
            market_module=MarketModule.SEARCH_PURCHASE,
            n_buyers=5,
            n_sellers=3,
            n_epochs=2,
            seed=42,
        )
        result = run_search_purchase_episode(treatment)

        assert len(result.epoch_metrics) == 2
        assert result.final_metrics.welfare.n_transactions > 0

    def test_run_matching_episode(self):
        from swarm.csm.runner import run_matching_episode

        treatment = CSMTreatment(
            name="test_matching",
            market_module=MarketModule.MATCHING,
            n_buyers=8,
            n_sellers=5,
            n_epochs=2,
            seed=42,
        )
        result = run_matching_episode(treatment)

        assert len(result.epoch_metrics) == 2
        assert result.final_metrics.welfare.n_transactions > 0

    def test_run_negotiation_episode(self):
        from swarm.csm.runner import run_negotiation_episode

        treatment = CSMTreatment(
            name="test_negotiation",
            market_module=MarketModule.NEGOTIATION,
            n_epochs=2,
            seed=42,
        )
        result = run_negotiation_episode(treatment, n_pairs=5)

        assert len(result.epoch_metrics) == 2

    def test_run_with_obfuscation(self):
        from swarm.csm.runner import run_search_purchase_episode

        treatment = CSMTreatment(
            name="test_obfuscation",
            market_module=MarketModule.SEARCH_PURCHASE,
            adversarial_env=AdversarialEnvironment.OBFUSCATION,
            n_buyers=5,
            n_sellers=3,
            n_epochs=3,
            seed=42,
        )
        result = run_search_purchase_episode(treatment)

        # Should have some obfuscation
        assert result.final_metrics.equilibrium.obfuscation_prevalence > 0

    def test_default_treatment_matrix(self):
        from swarm.csm.runner import default_treatment_matrix

        treatments = default_treatment_matrix(seed=42)
        assert len(treatments) > 10  # Should have many treatments

        # Check all have valid modules
        modules = {t.market_module for t in treatments}
        assert MarketModule.SEARCH_PURCHASE in modules
        assert MarketModule.MATCHING in modules

    def test_run_csm_benchmark_mini(self):
        from swarm.csm.runner import run_csm_benchmark

        treatments = [
            CSMTreatment(
                name="mini_search",
                market_module=MarketModule.SEARCH_PURCHASE,
                n_buyers=3,
                n_sellers=2,
                n_epochs=1,
                seed=42,
            ),
            CSMTreatment(
                name="mini_matching",
                market_module=MarketModule.MATCHING,
                n_buyers=4,
                n_sellers=3,
                n_epochs=1,
                seed=43,
            ),
        ]
        results = run_csm_benchmark(treatments)
        assert len(results) == 2

    def test_result_serialization(self):
        from swarm.csm.runner import run_search_purchase_episode

        treatment = CSMTreatment(
            name="test_serial",
            market_module=MarketModule.SEARCH_PURCHASE,
            n_buyers=3,
            n_sellers=2,
            n_epochs=1,
            seed=42,
        )
        result = run_search_purchase_episode(treatment)
        d = result.to_dict()

        assert "treatment" in d
        assert "final_metrics" in d
        assert "success" in d

    def test_adoption_rate_affects_search_depth(self):
        from swarm.csm.runner import run_search_purchase_episode

        # 0% adoption = all human-like
        t_no_agents = CSMTreatment(
            name="no_agents",
            market_module=MarketModule.SEARCH_PURCHASE,
            adoption_rate=0.0,
            n_buyers=5,
            n_sellers=3,
            n_epochs=1,
            seed=42,
        )
        # 100% adoption = all agent-mediated
        t_full_agents = CSMTreatment(
            name="full_agents",
            market_module=MarketModule.SEARCH_PURCHASE,
            adoption_rate=1.0,
            tx_cost_regime=TransactionCostRegime.AGENTIC,
            n_buyers=5,
            n_sellers=3,
            n_epochs=1,
            seed=42,
        )
        r_no = run_search_purchase_episode(t_no_agents)
        r_full = run_search_purchase_episode(t_full_agents)

        # Full agent adoption should have more queries (deeper search)
        assert (
            r_full.final_metrics.transaction_costs.mean_queries
            >= r_no.final_metrics.transaction_costs.mean_queries
        )


# ===================================================================
# Plots (smoke test)
# ===================================================================

class TestPlots:
    """Smoke tests for plotting utilities."""

    def test_welfare_vs_adoption_data(self):
        from swarm.csm.metrics import CSMMetrics, WelfareMetrics
        from swarm.csm.plots import welfare_vs_adoption_data
        from swarm.csm.runner import CSMEpisodeResult

        results = [
            CSMEpisodeResult(
                treatment=CSMTreatment(
                    market_module=MarketModule.SEARCH_PURCHASE,
                    adoption_rate=rate,
                ),
                final_metrics=CSMMetrics(
                    welfare=WelfareMetrics(total_surplus=rate * 100)
                ),
            )
            for rate in [0.0, 0.5, 1.0]
        ]
        data = welfare_vs_adoption_data(results)
        assert "modules" in data
        assert "search_purchase" in data["modules"]

    def test_byo_vs_bowling_shoe_data(self):
        from swarm.csm.metrics import (
            AgencyMetrics,
            CSMMetrics,
            TransactionCostMetrics,
            WelfareMetrics,
        )
        from swarm.csm.plots import byo_vs_bowling_shoe_data
        from swarm.csm.runner import CSMEpisodeResult

        results = [
            CSMEpisodeResult(
                treatment=CSMTreatment(ownership=AgentOwnership.BYO),
                final_metrics=CSMMetrics(
                    welfare=WelfareMetrics(total_surplus=100),
                    transaction_costs=TransactionCostMetrics(mean_search_cost=5),
                    agency=AgencyMetrics(faithfulness=0.9),
                ),
            ),
            CSMEpisodeResult(
                treatment=CSMTreatment(ownership=AgentOwnership.BOWLING_SHOE),
                final_metrics=CSMMetrics(
                    welfare=WelfareMetrics(total_surplus=80),
                    transaction_costs=TransactionCostMetrics(mean_search_cost=3),
                    agency=AgencyMetrics(faithfulness=0.7),
                ),
            ),
        ]
        data = byo_vs_bowling_shoe_data(results)
        assert data["byo"][0] == 100  # Surplus
        assert data["bowling_shoe"][0] == 80

    def test_pref_dim_stress_data(self):
        from swarm.csm.metrics import AgencyMetrics, CSMMetrics, WelfareMetrics
        from swarm.csm.plots import pref_dim_stress_data
        from swarm.csm.runner import CSMEpisodeResult

        results = [
            CSMEpisodeResult(
                treatment=CSMTreatment(preference_dim=PreferenceDimensionality.LOW),
                final_metrics=CSMMetrics(
                    welfare=WelfareMetrics(mean_utility=10),
                    agency=AgencyMetrics(faithfulness=0.95),
                ),
            ),
            CSMEpisodeResult(
                treatment=CSMTreatment(preference_dim=PreferenceDimensionality.HIGH),
                final_metrics=CSMMetrics(
                    welfare=WelfareMetrics(mean_utility=7),
                    agency=AgencyMetrics(faithfulness=0.8),
                ),
            ),
        ]
        data = pref_dim_stress_data(results)
        assert "low" in data
        assert "high" in data
        assert data["low"]["mean_welfare"] > data["high"]["mean_welfare"]
