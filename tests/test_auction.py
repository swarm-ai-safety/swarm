"""Tests for Dworkin-style auction mechanism."""

import pytest

from src.env.auction import (
    AuctionBid,
    AuctionConfig,
    DworkinAuction,
)


class TestAuctionConfig:
    """Tests for AuctionConfig validation."""

    def test_default_config_valid(self):
        config = AuctionConfig()
        config.validate()

    def test_invalid_endowment(self):
        with pytest.raises(ValueError, match="initial_endowment"):
            AuctionConfig(initial_endowment=-1).validate()

    def test_invalid_max_rounds(self):
        with pytest.raises(ValueError, match="max_rounds"):
            AuctionConfig(max_rounds=0).validate()

    def test_invalid_price_adjustment_rate(self):
        with pytest.raises(ValueError, match="price_adjustment_rate"):
            AuctionConfig(price_adjustment_rate=0).validate()

    def test_invalid_convergence_tolerance(self):
        with pytest.raises(ValueError, match="convergence_tolerance"):
            AuctionConfig(convergence_tolerance=0).validate()

    def test_invalid_envy_tolerance(self):
        with pytest.raises(ValueError, match="envy_tolerance"):
            AuctionConfig(envy_tolerance=-1).validate()


class TestDworkinAuction:
    """Tests for the DworkinAuction engine."""

    def test_empty_auction(self):
        auction = DworkinAuction()
        result = auction.run_auction({}, {})
        assert result.converged is True
        assert result.is_envy_free is True
        assert len(result.allocations) == 0

    def test_single_agent_single_resource(self):
        auction = DworkinAuction()
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 10.0},
                budget=100.0,
            )
        }
        resources = {"compute": 50.0}
        result = auction.run_auction(bids, resources)

        assert "agent_1" in result.allocations
        alloc = result.allocations["agent_1"]
        assert alloc.resources.get("compute", 0) > 0
        assert result.is_envy_free is True

    def test_equal_endowments_symmetric_preferences(self):
        """With equal budgets and identical preferences, allocation should be equal."""
        auction = DworkinAuction(AuctionConfig(max_rounds=100))
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 10.0},
                budget=100.0,
            ),
            "agent_2": AuctionBid(
                agent_id="agent_2",
                valuations={"compute": 10.0},
                budget=100.0,
            ),
        }
        resources = {"compute": 100.0}
        result = auction.run_auction(bids, resources)

        alloc_1 = result.allocations["agent_1"].resources.get("compute", 0)
        alloc_2 = result.allocations["agent_2"].resources.get("compute", 0)

        # Should be roughly equal
        assert abs(alloc_1 - alloc_2) < 10.0
        # Total shouldn't exceed supply
        assert alloc_1 + alloc_2 <= 100.0 + 0.01

    def test_heterogeneous_preferences_specialization(self):
        """Agents with different preferences should specialize."""
        auction = DworkinAuction(AuctionConfig(max_rounds=100))
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 10.0, "bandwidth": 1.0},
                budget=100.0,
            ),
            "agent_2": AuctionBid(
                agent_id="agent_2",
                valuations={"compute": 1.0, "bandwidth": 10.0},
                budget=100.0,
            ),
        }
        resources = {"compute": 100.0, "bandwidth": 100.0}
        result = auction.run_auction(bids, resources)

        # Agent 1 should get more compute, agent 2 more bandwidth
        a1_compute = result.allocations["agent_1"].resources.get("compute", 0)
        a2_bandwidth = result.allocations["agent_2"].resources.get("bandwidth", 0)
        a1_bandwidth = result.allocations["agent_1"].resources.get("bandwidth", 0)
        a2_compute = result.allocations["agent_2"].resources.get("compute", 0)

        assert a1_compute > a2_compute
        assert a2_bandwidth > a1_bandwidth

    def test_budget_constraint_binds(self):
        """Agents can't spend more than their budget."""
        auction = DworkinAuction()
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 100.0},
                budget=10.0,
            ),
        }
        resources = {"compute": 1000.0}
        result = auction.run_auction(bids, resources)

        alloc = result.allocations["agent_1"]
        assert alloc.price_paid <= 10.0 + 0.01

    def test_supply_constraint(self):
        """Total allocation shouldn't exceed supply."""
        auction = DworkinAuction(AuctionConfig(max_rounds=50))
        bids = {}
        for i in range(5):
            bids[f"agent_{i}"] = AuctionBid(
                agent_id=f"agent_{i}",
                valuations={"compute": 10.0},
                budget=100.0,
            )
        resources = {"compute": 50.0}
        result = auction.run_auction(bids, resources)

        total_allocated = sum(
            a.resources.get("compute", 0) for a in result.allocations.values()
        )
        assert total_allocated <= 50.0 + 0.01

    def test_envy_free_check(self):
        """Symmetric agents should result in envy-free allocation."""
        auction = DworkinAuction(
            AuctionConfig(
                max_rounds=100,
                envy_tolerance=5.0,
            )
        )
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 10.0},
                budget=100.0,
            ),
            "agent_2": AuctionBid(
                agent_id="agent_2",
                valuations={"compute": 10.0},
                budget=100.0,
            ),
        }
        resources = {"compute": 100.0}
        result = auction.run_auction(bids, resources)
        assert result.is_envy_free is True

    def test_gini_coefficient_equal(self):
        """Equal utilities should have Gini ~0."""
        auction = DworkinAuction(AuctionConfig(max_rounds=100))
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 10.0},
                budget=100.0,
            ),
            "agent_2": AuctionBid(
                agent_id="agent_2",
                valuations={"compute": 10.0},
                budget=100.0,
            ),
        }
        resources = {"compute": 100.0}
        result = auction.run_auction(bids, resources)

        gini = auction.compute_gini_coefficient(result.allocations)
        assert gini < 0.3  # Should be near 0 for symmetric case

    def test_gini_empty(self):
        auction = DworkinAuction()
        assert auction.compute_gini_coefficient({}) == 0.0

    def test_clearing_prices_positive(self):
        """All clearing prices should be positive."""
        auction = DworkinAuction()
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 10.0, "bandwidth": 5.0},
                budget=100.0,
            ),
        }
        resources = {"compute": 50.0, "bandwidth": 50.0}
        result = auction.run_auction(bids, resources)

        for _r, price in result.clearing_prices.items():
            assert price > 0

    def test_total_utility_positive(self):
        """Total utility should be positive with positive valuations."""
        auction = DworkinAuction()
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 10.0},
                budget=100.0,
            ),
        }
        resources = {"compute": 50.0}
        result = auction.run_auction(bids, resources)
        assert result.total_utility > 0

    def test_serialization(self):
        """Result should be serializable."""
        auction = DworkinAuction()
        bids = {
            "agent_1": AuctionBid(
                agent_id="agent_1",
                valuations={"compute": 10.0},
                budget=100.0,
            ),
        }
        resources = {"compute": 50.0}
        result = auction.run_auction(bids, resources)
        d = result.to_dict()
        assert "allocations" in d
        assert "clearing_prices" in d
        assert "is_envy_free" in d
