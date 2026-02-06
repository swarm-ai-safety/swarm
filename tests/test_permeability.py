"""Tests for permeability model."""

import pytest
from src.models.interaction import SoftInteraction

from src.boundaries.permeability import (
    PermeabilityConfig,
    PermeabilityModel,
    SpilloverEvent,
)


def _make_interaction(p: float, accepted: bool = True) -> SoftInteraction:
    """Helper to create a test interaction."""
    return SoftInteraction(p=p, accepted=accepted)


class TestPermeabilityConfig:
    """Tests for PermeabilityConfig validation."""

    def test_default_config_valid(self):
        config = PermeabilityConfig()
        config.validate()

    def test_invalid_base_permeability(self):
        with pytest.raises(ValueError, match="base_permeability"):
            PermeabilityConfig(base_permeability=1.5).validate()

    def test_invalid_information_decay(self):
        with pytest.raises(ValueError, match="information_decay"):
            PermeabilityConfig(information_decay=-0.1).validate()

    def test_invalid_contagion_rate(self):
        with pytest.raises(ValueError, match="contagion_rate"):
            PermeabilityConfig(contagion_rate=1.5).validate()

    def test_invalid_spillover_amplification(self):
        with pytest.raises(ValueError, match="spillover_amplification"):
            PermeabilityConfig(spillover_amplification=-1).validate()


class TestPermeabilityModel:
    """Tests for the PermeabilityModel."""

    def test_zero_permeability_blocks_all(self):
        model = PermeabilityModel(
            PermeabilityConfig(base_permeability=0.0, adaptive=False),
            seed=42,
        )
        # With permeability 0, nothing should cross
        for _ in range(20):
            allowed, _ = model.should_cross(0.0, 0.8)
            assert allowed is False

    def test_full_permeability_high_quality(self):
        model = PermeabilityModel(
            PermeabilityConfig(base_permeability=1.0, adaptive=False),
            seed=42,
        )
        # With permeability 1 and high quality, almost everything crosses
        allowed_count = sum(
            1 for _ in range(100)
            if model.should_cross(1.0, 0.99)[0]
        )
        assert allowed_count > 90

    def test_adaptive_permeability_reduces_with_threat(self):
        model = PermeabilityModel(
            PermeabilityConfig(base_permeability=0.8, adaptive=True),
        )
        perm_low_threat = model.compute_effective_permeability(
            threat_level=0.1, agent_trust=0.5,
        )
        perm_high_threat = model.compute_effective_permeability(
            threat_level=0.9, agent_trust=0.5,
        )
        assert perm_high_threat < perm_low_threat

    def test_adaptive_permeability_increases_with_trust(self):
        model = PermeabilityModel(
            PermeabilityConfig(base_permeability=0.5, adaptive=True),
        )
        perm_low_trust = model.compute_effective_permeability(
            threat_level=0.3, agent_trust=0.1,
        )
        perm_high_trust = model.compute_effective_permeability(
            threat_level=0.3, agent_trust=0.9,
        )
        assert perm_high_trust > perm_low_trust

    def test_non_adaptive_ignores_threat(self):
        model = PermeabilityModel(
            PermeabilityConfig(base_permeability=0.5, adaptive=False),
        )
        perm = model.compute_effective_permeability(
            threat_level=0.9, agent_trust=0.1,
        )
        assert perm == 0.5

    def test_contagion_probability_proportional_to_harm(self):
        model = PermeabilityModel(PermeabilityConfig(contagion_rate=0.5))
        perm = 0.5

        high_quality = _make_interaction(p=0.9)
        low_quality = _make_interaction(p=0.1)

        prob_high = model.compute_contagion_probability(high_quality, perm)
        prob_low = model.compute_contagion_probability(low_quality, perm)

        assert prob_low > prob_high

    def test_contagion_probability_proportional_to_permeability(self):
        model = PermeabilityModel(PermeabilityConfig(contagion_rate=0.5))
        interaction = _make_interaction(p=0.3)

        prob_closed = model.compute_contagion_probability(interaction, 0.0)
        prob_open = model.compute_contagion_probability(interaction, 1.0)

        assert prob_open > prob_closed
        assert prob_closed == 0.0

    def test_simulate_spillover_accepted_only(self):
        model = PermeabilityModel(
            PermeabilityConfig(contagion_rate=1.0, base_permeability=1.0),
            seed=42,
        )
        accepted = [_make_interaction(p=0.1, accepted=True) for _ in range(10)]
        rejected = [_make_interaction(p=0.1, accepted=False) for _ in range(10)]

        spillovers_accepted = model.simulate_spillover(accepted, permeability=1.0)
        model.reset()
        spillovers_rejected = model.simulate_spillover(rejected, permeability=1.0)

        # Only accepted interactions can cause spillovers
        assert len(spillovers_accepted) > 0
        assert len(spillovers_rejected) == 0

    def test_spillover_harm_with_amplification(self):
        model = PermeabilityModel(
            PermeabilityConfig(
                contagion_rate=1.0,
                spillover_amplification=2.0,
            ),
            seed=42,
        )
        interactions = [_make_interaction(p=0.1) for _ in range(20)]
        spillovers = model.simulate_spillover(interactions, permeability=1.0)

        if spillovers:
            for s in spillovers:
                if not s.blocked:
                    # harm_magnitude = (1-p) * amplification = 0.9 * 2.0 = 1.8
                    assert s.harm_magnitude > 0

    def test_containment_rate(self):
        model = PermeabilityModel(
            PermeabilityConfig(contagion_rate=0.5),
            seed=42,
        )
        # Start with perfect containment
        assert model.containment_rate() == 1.0

    def test_compute_spillover_harm(self):
        model = PermeabilityModel()
        spillovers = [
            SpilloverEvent(harm_magnitude=1.0, blocked=False),
            SpilloverEvent(harm_magnitude=2.0, blocked=True),
            SpilloverEvent(harm_magnitude=3.0, blocked=False),
        ]
        harm = model.compute_spillover_harm(spillovers)
        assert harm == 4.0  # Only unblocked

    def test_optimal_permeability(self):
        model = PermeabilityModel()
        # Mix of high and low quality
        interactions = (
            [_make_interaction(p=0.9) for _ in range(5)]
            + [_make_interaction(p=0.1) for _ in range(5)]
        )
        opt = model.optimal_permeability(interactions, external_harm_weight=1.0)
        assert 0.0 <= opt <= 1.0

    def test_reset(self):
        model = PermeabilityModel(
            PermeabilityConfig(contagion_rate=1.0),
            seed=42,
        )
        interactions = [_make_interaction(p=0.1) for _ in range(10)]
        model.simulate_spillover(interactions, permeability=1.0)
        assert model.state.total_spillovers > 0 or model.state.total_blocked > 0

        model.reset()
        assert model.state.total_spillovers == 0
        assert model.state.total_blocked == 0
        assert len(model.get_spillover_history()) == 0

    def test_state_serialization(self):
        model = PermeabilityModel()
        d = model.state.to_dict()
        assert "effective_permeability" in d
        assert "cumulative_external_harm" in d

    def test_spillover_event_serialization(self):
        event = SpilloverEvent(
            source_p=0.3,
            harm_magnitude=1.5,
            direction="outbound",
            blocked=False,
        )
        d = event.to_dict()
        assert d["source_p"] == 0.3
        assert d["harm_magnitude"] == 1.5
        assert d["blocked"] is False
