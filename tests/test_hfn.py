"""Tests for High-Frequency Negotiation engine."""

import pytest

from src.env.hfn import (
    FlashCrashDetector,
    FlashCrashEvent,
    HFNConfig,
    HFNEngine,
    HFNOrder,
    HFNTick,
)


class TestHFNConfig:
    """Tests for HFNConfig validation."""

    def test_default_config_valid(self):
        config = HFNConfig()
        config.validate()

    def test_invalid_tick_duration(self):
        with pytest.raises(ValueError, match="tick_duration_ms"):
            HFNConfig(tick_duration_ms=0).validate()

    def test_invalid_max_orders(self):
        with pytest.raises(ValueError, match="max_orders_per_tick"):
            HFNConfig(max_orders_per_tick=0).validate()

    def test_invalid_latency(self):
        with pytest.raises(ValueError, match="latency_noise_ms"):
            HFNConfig(latency_noise_ms=-1).validate()


class TestHFNEngine:
    """Tests for the HFN engine."""

    def test_submit_bid_and_ask(self):
        engine = HFNEngine(seed=42)
        bid = HFNOrder(
            agent_id="buyer", order_type="bid",
            resource_type="compute", quantity=10, price=5.0,
        )
        ask = HFNOrder(
            agent_id="seller", order_type="ask",
            resource_type="compute", quantity=10, price=4.0,
        )
        assert engine.submit_order(bid) is True
        assert engine.submit_order(ask) is True

    def test_reject_invalid_orders(self):
        engine = HFNEngine(seed=42)
        # Zero quantity
        bad = HFNOrder(
            agent_id="a", order_type="bid",
            resource_type="x", quantity=0, price=5.0,
        )
        assert engine.submit_order(bad) is False

        # Zero price
        bad2 = HFNOrder(
            agent_id="a", order_type="bid",
            resource_type="x", quantity=10, price=0,
        )
        assert engine.submit_order(bad2) is False

    def test_rate_limit(self):
        engine = HFNEngine(HFNConfig(max_orders_per_tick=2), seed=42)
        for i in range(3):
            order = HFNOrder(
                agent_id="a1", order_type="bid",
                resource_type="x", quantity=1, price=1.0,
            )
            result = engine.submit_order(order)
            if i < 2:
                assert result is True
            else:
                assert result is False

    def test_batch_clearing_matches_orders(self):
        engine = HFNEngine(HFNConfig(batch_interval_ticks=1), seed=42)

        bid = HFNOrder(
            agent_id="buyer", order_type="bid",
            resource_type="compute", quantity=10, price=5.0,
        )
        ask = HFNOrder(
            agent_id="seller", order_type="ask",
            resource_type="compute", quantity=10, price=4.0,
        )
        engine.submit_order(bid)
        engine.submit_order(ask)

        tick = engine.process_tick()
        assert tick.orders_executed > 0
        assert engine.market_price == pytest.approx(4.5)

    def test_no_match_when_bid_below_ask(self):
        engine = HFNEngine(HFNConfig(batch_interval_ticks=1), seed=42)

        bid = HFNOrder(
            agent_id="buyer", order_type="bid",
            resource_type="compute", quantity=10, price=3.0,
        )
        ask = HFNOrder(
            agent_id="seller", order_type="ask",
            resource_type="compute", quantity=10, price=5.0,
        )
        engine.submit_order(bid)
        engine.submit_order(ask)

        tick = engine.process_tick()
        assert tick.orders_executed == 0

    def test_halt_rejects_orders(self):
        engine = HFNEngine(seed=42)
        engine.halt(duration_ticks=5)
        assert engine.is_halted is True

        order = HFNOrder(
            agent_id="a1", order_type="bid",
            resource_type="x", quantity=1, price=1.0,
        )
        assert engine.submit_order(order) is False

    def test_halt_auto_recovers(self):
        engine = HFNEngine(HFNConfig(halt_duration_ticks=3), seed=42)
        engine.halt(duration_ticks=3)
        assert engine.is_halted is True

        for _ in range(3):
            engine.process_tick()

        assert engine.is_halted is False

    def test_cancel_order(self):
        engine = HFNEngine(seed=42)
        bid = HFNOrder(
            agent_id="a1", order_type="bid",
            resource_type="compute", quantity=10, price=5.0,
        )
        engine.submit_order(bid)
        depth_before = engine.get_order_book_depth()
        assert depth_before["bids"] == 1

        cancel = HFNOrder(
            agent_id="a1", order_type="cancel",
            resource_type="compute",
        )
        engine.submit_order(cancel)
        depth_after = engine.get_order_book_depth()
        assert depth_after["bids"] == 0

    def test_tick_history(self):
        engine = HFNEngine(seed=42)
        for _ in range(5):
            engine.process_tick()
        history = engine.get_tick_history()
        assert len(history) == 5
        assert history[0].tick_number == 1
        assert history[4].tick_number == 5

    def test_market_price_updates(self):
        engine = HFNEngine(HFNConfig(batch_interval_ticks=1), seed=42)
        initial_price = engine.market_price

        for i in range(3):
            bid = HFNOrder(
                agent_id=f"buyer_{i}", order_type="bid",
                resource_type="x", quantity=1, price=10.0 + i,
            )
            ask = HFNOrder(
                agent_id=f"seller_{i}", order_type="ask",
                resource_type="x", quantity=1, price=9.0 + i,
            )
            engine.submit_order(bid)
            engine.submit_order(ask)
            engine.process_tick()

        assert engine.market_price != initial_price

    def test_order_serialization(self):
        order = HFNOrder(
            agent_id="a1", order_type="bid",
            resource_type="compute", quantity=10, price=5.0,
        )
        d = order.to_dict()
        assert d["agent_id"] == "a1"
        assert d["order_type"] == "bid"

    def test_speed_advantage_gini_empty(self):
        engine = HFNEngine(seed=42)
        assert engine.speed_advantage_gini() == 0.0


class TestFlashCrashDetector:
    """Tests for the FlashCrashDetector."""

    def test_no_crash_stable_prices(self):
        detector = FlashCrashDetector(price_drop_threshold=0.1)
        for i in range(20):
            tick = HFNTick(tick_number=i, market_price=10.0)
            crash = detector.update(tick)
            assert crash is None

    def test_crash_detected_on_price_drop(self):
        detector = FlashCrashDetector(
            price_drop_threshold=0.1, window_ticks=10
        )
        # Stable prices
        for i in range(5):
            tick = HFNTick(tick_number=i, market_price=10.0)
            detector.update(tick)

        # Sudden drop
        tick = HFNTick(tick_number=5, market_price=8.0)
        crash = detector.update(tick, active_agent_ids=["a1", "a2"])

        assert crash is not None
        assert crash.price_drop_pct >= 0.1
        assert "a1" in crash.trigger_agent_ids

    def test_crash_recovery(self):
        detector = FlashCrashDetector(
            price_drop_threshold=0.1, window_ticks=20
        )
        # Stable
        for i in range(5):
            detector.update(HFNTick(tick_number=i, market_price=10.0))

        # Crash
        detector.update(HFNTick(tick_number=5, market_price=8.0))
        assert detector.is_in_crash is True

        # Recovery
        for i in range(6, 15):
            detector.update(HFNTick(tick_number=i, market_price=9.6))

        assert detector.is_in_crash is False

    def test_volatility_index_stable(self):
        detector = FlashCrashDetector()
        for i in range(10):
            detector.update(HFNTick(tick_number=i, market_price=10.0))

        assert detector.get_volatility_index() == 0.0

    def test_volatility_index_volatile(self):
        detector = FlashCrashDetector()
        prices = [10.0, 11.0, 9.0, 12.0, 8.0]
        for i, p in enumerate(prices):
            detector.update(HFNTick(tick_number=i, market_price=p))

        assert detector.get_volatility_index() > 0.0

    def test_crash_history(self):
        detector = FlashCrashDetector(
            price_drop_threshold=0.1, window_ticks=10
        )
        for i in range(5):
            detector.update(HFNTick(tick_number=i, market_price=10.0))

        detector.update(HFNTick(tick_number=5, market_price=8.0))

        history = detector.get_crash_history()
        assert len(history) == 1
        assert history[0].price_drop_pct >= 0.1

    def test_severity_proportional_to_drop(self):
        detector = FlashCrashDetector(
            price_drop_threshold=0.1, window_ticks=10
        )
        for i in range(5):
            detector.update(HFNTick(tick_number=i, market_price=10.0))

        tick = HFNTick(tick_number=5, market_price=5.0)  # 50% drop
        crash = detector.update(tick)

        assert crash is not None
        assert crash.severity > 1.0  # Greater than threshold ratio

    def test_flash_crash_serialization(self):
        crash = FlashCrashEvent(
            start_tick=5,
            end_tick=10,
            price_drop_pct=0.2,
            trigger_agent_ids=["a1"],
            severity=2.0,
        )
        d = crash.to_dict()
        assert d["price_drop_pct"] == 0.2
        assert d["severity"] == 2.0
