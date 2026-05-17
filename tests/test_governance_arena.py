"""Tests for swarm.analysis.governance_arena pure-logic functions.

These tests import individual helpers without running the Streamlit app.
The module-level ``st.set_page_config`` and ``st.markdown`` calls are
patched out at import time.
"""

from __future__ import annotations

import random
import sys
from typing import Any, Dict, List
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Patch Streamlit before importing the module under test.
# The module calls st.set_page_config() and st.markdown() at import time.
# ---------------------------------------------------------------------------

class _AttrDict(dict):
    """Dict that supports attribute access (mimics st.session_state)."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

_fake_session_state = _AttrDict({
    "run_complete": False,
    "run_results": None,
    "turn_log": [],
    "metrics": None,
    "events": [],
    "seed": 42,
    "preset": None,
    "leaderboard": [],
    "tax_rate": 5,
    "audit_probability": 12,
    "circuit_breaker_threshold": 7,
    "reputation_weight": 0.5,
    "collusion_sensitivity": 0.3,
    "mad_enabled": True,
    "treaty_max_level": 4,
    "mediation_enabled": False,
    "back_channel_enabled": False,
    "fog_sigma": 0.5,
    "persona_a": "tit_for_tat",
    "persona_b": "hawk",
    "max_turns": 20,
})

class _ColumnCtx:
    """Context manager returned by st.columns items."""
    def __enter__(self):
        return self
    def __exit__(self, *a: Any):
        pass
    def __getattr__(self, name: str) -> Any:
        return mock.MagicMock()

def _fake_columns(spec: Any) -> list:
    n = spec if isinstance(spec, int) else len(spec)
    return [_ColumnCtx() for _ in range(n)]

_st_stub = mock.MagicMock()
_st_stub.session_state = _fake_session_state
_st_stub.columns = _fake_columns
_st_stub.tabs = lambda labels: [_ColumnCtx() for _ in labels]
# Buttons must return False so module-level UI code doesn't trigger simulation
_st_stub.button = mock.MagicMock(return_value=False)
_st_stub.checkbox = mock.MagicMock(return_value=False)
_st_stub.slider = mock.MagicMock(return_value=0)
_st_stub.selectbox = mock.MagicMock(return_value="hawk")
_st_stub.number_input = mock.MagicMock(return_value=42)
sys.modules.setdefault("streamlit", _st_stub)

from swarm.analysis.governance_arena import (  # noqa: E402, I001
    PERSONA_OPTIONS,
    PRESETS,
    _add_to_leaderboard,
    _demo_agent_delta,
    _escalation_bar,
    _generate_receipt_text,
    _generate_share_text,
    _metric_tile,
    _run_demo_simulation,
    _transcript_entry,
)


# ===========================================================================
# 1. TestDemoAgentDelta — already pure
# ===========================================================================


class TestDemoAgentDelta:
    """Test the simple demo agent decision function."""

    def test_hawk_always_positive(self):
        rng = random.Random(42)
        for _ in range(50):
            delta = _demo_agent_delta("hawk", 3, 3, 5, rng)
            assert delta in (1, 2)

    def test_dove_deescalates_when_positive(self):
        rng = random.Random(42)
        assert _demo_agent_delta("dove", 3, 5, 5, rng) == -1

    def test_dove_stays_at_zero(self):
        rng = random.Random(42)
        assert _demo_agent_delta("dove", 0, 5, 5, rng) == 0

    def test_tit_for_tat_turn_zero(self):
        rng = random.Random(42)
        assert _demo_agent_delta("tit_for_tat", 0, 0, 0, rng) == 0

    def test_tit_for_tat_mirrors_opponent(self):
        rng = random.Random(42)
        # opponent at 5, me at 2 → diff = 3, clamped to 2
        assert _demo_agent_delta("tit_for_tat", 2, 5, 1, rng) == 2
        # opponent at 2, me at 5 → diff = -3, clamped to -1
        assert _demo_agent_delta("tit_for_tat", 5, 2, 1, rng) == -1

    def test_calculating_waits_then_escalates(self):
        rng = random.Random(42)
        for t in range(4):
            assert _demo_agent_delta("calculating", 3, 3, t, rng) == 0
        # After turn 4, should escalate
        delta = _demo_agent_delta("calculating", 3, 3, 4, rng)
        assert delta in (1, 2, 3)

    def test_gradual_always_one(self):
        rng = random.Random(42)
        for t in range(10):
            assert _demo_agent_delta("gradual", t, t, t, rng) == 1

    def test_random_in_range(self):
        rng = random.Random(42)
        for _ in range(100):
            delta = _demo_agent_delta("random", 3, 3, 5, rng)
            assert -1 <= delta <= 2


# ===========================================================================
# 2. TestDemoSimulation — uses config param (no session_state needed)
# ===========================================================================


class TestDemoSimulation:
    """Test the fallback demo simulation engine."""

    @staticmethod
    def _base_config(**overrides: Any) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "seed": 42,
            "max_turns": 20,
            "circuit_breaker_threshold": 10,  # disabled
            "persona_a": "hawk",
            "persona_b": "hawk",
            "fog_sigma": 0.0,
        }
        cfg.update(overrides)
        return cfg

    def test_hawk_vs_hawk_escalates(self):
        result = _run_demo_simulation(self._base_config(
            persona_a="hawk", persona_b="hawk", fog_sigma=0.0,
        ))
        assert result["outcome"] == "mutual_destruction"
        assert result["max_level"] == 9

    def test_dove_vs_dove_ceasefire(self):
        result = _run_demo_simulation(self._base_config(
            persona_a="dove", persona_b="dove", fog_sigma=0.0,
        ))
        assert result["outcome"] == "ceasefire"
        assert result["max_level"] <= 1

    def test_circuit_breaker_fires(self):
        result = _run_demo_simulation(self._base_config(
            persona_a="hawk", persona_b="hawk",
            circuit_breaker_threshold=3, fog_sigma=0.0,
        ))
        cb_events = [
            e for e in result["events"]
            if e.get("type") == "circuit_breaker_triggered"
        ]
        assert len(cb_events) >= 1

    def test_nuclear_turn_detected(self):
        result = _run_demo_simulation(self._base_config(
            persona_a="hawk", persona_b="hawk",
            circuit_breaker_threshold=10, fog_sigma=0.0,
        ))
        assert result["nuclear_turn"] is not None
        # Nuclear = level >= 7
        assert result["max_level"] >= 7

    def test_seed_reproducibility(self):
        cfg = self._base_config(seed=12345)
        r1 = _run_demo_simulation(cfg)
        r2 = _run_demo_simulation(cfg)
        assert r1["outcome"] == r2["outcome"]
        assert r1["max_level"] == r2["max_level"]
        assert r1["turns_played"] == r2["turns_played"]
        assert r1["turn_log"] == r2["turn_log"]

    def test_max_turns_respected(self):
        result = _run_demo_simulation(self._base_config(
            persona_a="gradual", persona_b="dove",
            max_turns=5, fog_sigma=0.0,
        ))
        assert result["turns_played"] <= 5

    def test_result_has_required_keys(self):
        result = _run_demo_simulation(self._base_config())
        for key in (
            "metrics", "turn_log", "events", "seed", "outcome",
            "nuclear_turn", "max_level", "turns_played", "welfare",
            "cooperation_score", "collateral",
        ):
            assert key in result, f"Missing key: {key}"


# ===========================================================================
# 3. TestPresets
# ===========================================================================


class TestPresets:
    """Test that governance presets are well-formed."""

    REQUIRED_KEYS = {
        "label", "icon", "css", "description",
        "tax_rate", "audit_probability", "circuit_breaker_threshold",
        "reputation_weight", "collusion_sensitivity", "mad_enabled",
        "treaty_max_level", "mediation_enabled", "back_channel_enabled",
        "fog_sigma", "persona_a", "persona_b",
    }

    def test_all_presets_have_required_keys(self):
        for name, preset in PRESETS.items():
            missing = self.REQUIRED_KEYS - set(preset.keys())
            assert not missing, f"Preset '{name}' missing keys: {missing}"

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_preset_values_in_range(self, preset_name: str):
        p = PRESETS[preset_name]
        assert 0 <= p["tax_rate"] <= 100
        assert 0 <= p["audit_probability"] <= 100
        assert 1 <= p["circuit_breaker_threshold"] <= 10
        assert 0.0 <= p["reputation_weight"] <= 1.0
        assert 0.0 <= p["collusion_sensitivity"] <= 1.0
        assert 1 <= p["treaty_max_level"] <= 9
        assert 0.0 <= p["fog_sigma"] <= 2.0
        assert p["persona_a"] in PERSONA_OPTIONS
        assert p["persona_b"] in PERSONA_OPTIONS

    def test_four_presets_exist(self):
        assert set(PRESETS.keys()) == {"chaos", "constitution", "minimal", "surveillance"}


# ===========================================================================
# 4. TestHTMLHelpers
# ===========================================================================


class TestHTMLHelpers:
    """Test HTML-generating helper functions."""

    def test_escalation_bar_returns_html(self):
        html = _escalation_bar(5)
        assert "<div" in html
        assert "esc-bar-track" in html

    @pytest.mark.parametrize("level,expected_color_fragment", [
        (0, "#2d8a4e"),    # green band
        (2, "#2d8a4e"),    # green band
        (3, "#8a8a2d"),    # yellow band
        (4, "#8a8a2d"),    # yellow band
        (5, "#cc6622"),    # orange band
        (6, "#cc6622"),    # orange band
        (7, "#cc2222"),    # red band
        (9, "#cc2222"),    # red band
    ])
    def test_escalation_bar_color_gradient(self, level: int, expected_color_fragment: str):
        html = _escalation_bar(level)
        assert expected_color_fragment in html

    def test_escalation_bar_label_text(self):
        html = _escalation_bar(3)
        assert "Cyber Operation (3/9)" in html

    def test_metric_tile_contains_value_and_label(self):
        html = _metric_tile("42", "Test Metric")
        assert "42" in html
        assert "Test Metric" in html

    def test_metric_tile_applies_css_class(self):
        html = _metric_tile("1", "X", css_class="my-class")
        assert "my-class" in html

    def test_transcript_entry_contains_turn(self):
        html = _transcript_entry(7, "Something happened")
        assert "Turn 7:" in html
        assert "Something happened" in html

    def test_transcript_entry_css_class(self):
        html = _transcript_entry(1, "text", css_class="nuclear")
        assert "nuclear" in html


# ===========================================================================
# 5. TestLeaderboard — uses explicit params
# ===========================================================================


class TestLeaderboard:
    """Test leaderboard management."""

    @staticmethod
    def _fake_results(**overrides: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "outcome": "ceasefire",
            "max_level": 2,
            "cooperation_score": 0.78,
            "nuclear_turn": None,
            "seed": 42,
            "turns_played": 15,
        }
        base.update(overrides)
        return base

    @staticmethod
    def _fake_settings() -> Dict[str, Any]:
        return {"tax_rate": 5, "audit_probability": 12, "circuit_breaker_threshold": 7}

    def test_entry_added_at_front(self):
        lb: List[Dict[str, Any]] = [{"preset": "old"}]
        _add_to_leaderboard(
            self._fake_results(), leaderboard=lb,
            preset="chaos", settings=self._fake_settings(),
        )
        assert lb[0]["preset"] == "chaos"
        assert lb[1]["preset"] == "old"

    def test_capped_at_50(self):
        lb: List[Dict[str, Any]] = [{"preset": f"entry-{i}"} for i in range(50)]
        _add_to_leaderboard(
            self._fake_results(), leaderboard=lb,
            preset="new", settings=self._fake_settings(),
        )
        assert len(lb) == 50
        assert lb[0]["preset"] == "new"

    def test_all_required_fields_present(self):
        lb: List[Dict[str, Any]] = []
        _add_to_leaderboard(
            self._fake_results(), leaderboard=lb,
            preset="minimal", settings=self._fake_settings(),
        )
        entry = lb[0]
        for key in ("timestamp", "preset", "outcome", "max_level", "cooperation",
                     "nuclear_turn", "seed", "turns", "tax", "audit", "breaker"):
            assert key in entry, f"Missing leaderboard field: {key}"

    def test_none_preset_becomes_custom(self):
        lb: List[Dict[str, Any]] = []
        _add_to_leaderboard(
            self._fake_results(), leaderboard=lb,
            preset=None, settings=self._fake_settings(),
        )
        assert lb[0]["preset"] == "Custom"


# ===========================================================================
# 6. TestShareAndReceipt — uses explicit settings
# ===========================================================================


class TestShareAndReceipt:
    """Test share text and receipt generation."""

    @staticmethod
    def _safe_results() -> Dict[str, Any]:
        return {
            "outcome": "ceasefire",
            "max_level": 3,
            "nuclear_turn": None,
            "seed": 42,
            "turns_played": 15,
            "cooperation_score": 0.78,
        }

    @staticmethod
    def _collapse_results() -> Dict[str, Any]:
        return {
            "outcome": "mutual_destruction",
            "max_level": 9,
            "nuclear_turn": 8,
            "seed": 99,
            "turns_played": 10,
            "cooperation_score": 0.11,
        }

    @staticmethod
    def _settings() -> Dict[str, Any]:
        return {
            "audit_probability": 15,
            "tax_rate": 5,
            "circuit_breaker_threshold": 6,
            "mad_enabled": True,
            "mediation_enabled": True,
            "persona_a": "dove",
            "persona_b": "tit_for_tat",
        }

    def test_share_text_prevented_collapse(self):
        text = _generate_share_text(self._safe_results(), settings=self._settings())
        assert "Prevented collapse" in text
        assert "Seed: 42" in text
        assert "huggingface.co" in text

    def test_share_text_collapse(self):
        text = _generate_share_text(self._collapse_results(), settings=self._settings())
        assert "Collapse at turn 8" in text
        assert "Seed: 99" in text

    def test_share_text_includes_settings(self):
        text = _generate_share_text(self._safe_results(), settings=self._settings())
        assert "Audits 15%" in text
        assert "Tax 5%" in text
        assert "Breaker 6" in text

    def test_receipt_text_includes_governance_settings(self):
        text = _generate_receipt_text(self._safe_results(), settings=self._settings())
        assert "Tax Rate: 5%" in text
        assert "Audit Probability: 15%" in text
        assert "Circuit Breaker: 6" in text
        assert "Agent A: dove" in text
        assert "Agent B: tit_for_tat" in text

    def test_receipt_text_includes_outcome(self):
        text = _generate_receipt_text(self._safe_results(), settings=self._settings())
        assert "Ceasefire" in text
        assert "Seed: 42" in text

    def test_receipt_text_nuclear_turn_none(self):
        text = _generate_receipt_text(self._safe_results(), settings=self._settings())
        # nuclear_turn=None is present in the dict, so .get() returns None, not 'Never'
        assert "Nuclear Turn: None" in text

    def test_receipt_text_nuclear_turn_present(self):
        text = _generate_receipt_text(self._collapse_results(), settings=self._settings())
        assert "Nuclear Turn: 8" in text
