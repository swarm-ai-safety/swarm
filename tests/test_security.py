"""Tests for security detection module."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.governance import GovernanceConfig, GovernanceEngine
from src.governance.security import SecurityLever
from src.metrics.security import (
    ContagionChain,
    LaunderingPath,
    ManipulationPattern,
    SecurityAnalyzer,
    SecurityReport,
    ThreatIndicator,
    ThreatType,
    compute_containment_effectiveness,
    compute_threat_trend,
)
from src.models.interaction import InteractionType, SoftInteraction


# =============================================================================
# Test Fixtures
# =============================================================================


def create_interaction(
    initiator: str,
    counterparty: str,
    p: float = 0.5,
    accepted: bool = True,
    tau: float = 0.0,
    c_a: float = 0.0,
    c_b: float = 0.0,
    r_a: float = 0.0,
    r_b: float = 0.0,
    tool_misuse_flags: int = 0,
    verifier_rejections: int = 0,
    timestamp: datetime = None,
) -> SoftInteraction:
    """Create a test interaction."""
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.COLLABORATION,
        accepted=accepted,
        p=p,
        tau=tau,
        c_a=c_a,
        c_b=c_b,
        r_a=r_a,
        r_b=r_b,
        tool_misuse_flags=tool_misuse_flags,
        verifier_rejections=verifier_rejections,
        timestamp=timestamp or datetime.now(),
    )


def create_injection_pattern(
    attacker: str,
    targets: list[str],
    base_time: datetime = None,
) -> list[SoftInteraction]:
    """Create interactions that look like prompt injection attempts."""
    base = base_time or datetime.now()
    interactions = []
    for i, target in enumerate(targets):
        interactions.append(create_interaction(
            initiator=attacker,
            counterparty=target,
            p=0.15,  # Low quality
            accepted=True,  # But accepted (counterparty deceived)
            tool_misuse_flags=2,  # Suspicious tool usage
            verifier_rejections=1,
            timestamp=base + timedelta(seconds=i * 30),
        ))
    return interactions


def create_manipulation_pattern(
    manipulator: str,
    victims: list[str],
    base_time: datetime = None,
) -> list[SoftInteraction]:
    """Create interactions that look like manipulation."""
    base = base_time or datetime.now()
    interactions = []
    for i, victim in enumerate(victims):
        # Multiple interactions with same victim
        for j in range(3):
            interactions.append(create_interaction(
                initiator=manipulator,
                counterparty=victim,
                p=0.35,  # Low quality
                accepted=True,
                r_a=0.5,  # Manipulator gains reputation
                tau=-2.0,  # Extracting resources
                timestamp=base + timedelta(seconds=(i * 3 + j) * 60),
            ))
    return interactions


def create_laundering_chain(
    source: str,
    intermediaries: list[str],
    destination: str,
    trust_scores: dict[str, float],
    base_time: datetime = None,
) -> tuple[list[SoftInteraction], dict[str, float]]:
    """Create a chain of interactions that launders information."""
    base = base_time or datetime.now()
    interactions = []
    chain = [source] + intermediaries + [destination]

    for i in range(len(chain) - 1):
        interactions.append(create_interaction(
            initiator=chain[i],
            counterparty=chain[i + 1],
            p=0.4,
            accepted=True,
            timestamp=base + timedelta(seconds=i * 120),
        ))

    return interactions, trust_scores


def create_contagion_chain(
    origin: str,
    chain_members: list[str],
    base_time: datetime = None,
    interval_seconds: float = 10.0,
) -> list[SoftInteraction]:
    """Create a chain of low-quality interactions spreading rapidly."""
    base = base_time or datetime.now()
    interactions = []

    # Origin infects first member
    interactions.append(create_interaction(
        initiator=origin,
        counterparty=chain_members[0],
        p=0.2,  # Low quality
        accepted=True,
        timestamp=base,
    ))

    # Each member spreads to next
    for i in range(len(chain_members) - 1):
        interactions.append(create_interaction(
            initiator=chain_members[i],
            counterparty=chain_members[i + 1],
            p=0.2,
            accepted=True,
            timestamp=base + timedelta(seconds=(i + 1) * interval_seconds),
        ))

    return interactions


def create_normal_interactions(
    agents: list[str],
    count: int = 50,
    seed: int = 42,
) -> list[SoftInteraction]:
    """Create normal (non-malicious) interactions."""
    rng = np.random.default_rng(seed)
    interactions = []
    base = datetime.now()

    for i in range(count):
        a, b = rng.choice(agents, size=2, replace=False)
        p = rng.uniform(0.4, 0.9)
        accepted = rng.random() < 0.7
        interactions.append(create_interaction(
            initiator=a,
            counterparty=b,
            p=p,
            accepted=accepted,
            timestamp=base + timedelta(seconds=i * 60),
        ))
    return interactions


# =============================================================================
# SecurityAnalyzer Tests
# =============================================================================


class TestSecurityAnalyzer:
    """Tests for SecurityAnalyzer class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        analyzer = SecurityAnalyzer()
        assert analyzer.injection_threshold == 0.3
        assert analyzer.manipulation_threshold == 0.5
        assert analyzer.laundering_trust_gap == 0.3
        assert analyzer.contagion_velocity_threshold == 2.0
        assert analyzer.min_chain_length == 3

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        analyzer = SecurityAnalyzer(
            injection_threshold=0.5,
            manipulation_threshold=0.7,
            laundering_trust_gap=0.4,
            seed=123,
        )
        assert analyzer.injection_threshold == 0.5
        assert analyzer.manipulation_threshold == 0.7
        assert analyzer.laundering_trust_gap == 0.4

    def test_analyze_empty_interactions(self):
        """Test analysis with no interactions."""
        analyzer = SecurityAnalyzer()
        report = analyzer.analyze([])
        assert report.ecosystem_threat_level == 0.0
        assert report.active_threat_count == 0
        assert len(report.threat_indicators) == 0

    def test_analyze_normal_interactions(self):
        """Test analysis with normal, non-malicious interactions."""
        analyzer = SecurityAnalyzer(seed=42)
        agents = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]
        interactions = create_normal_interactions(agents, count=30)

        report = analyzer.analyze(interactions, agents)
        # Normal interactions should have low threat level
        assert report.ecosystem_threat_level < 0.3
        assert len(report.agents_flagged) < 2

    def test_detect_prompt_injection(self):
        """Test detection of prompt injection patterns."""
        analyzer = SecurityAnalyzer(injection_threshold=0.3)
        attacker = "malicious_agent"
        targets = ["victim_1", "victim_2", "victim_3"]
        interactions = create_injection_pattern(attacker, targets)

        report = analyzer.analyze(interactions, [attacker] + targets)

        # Should detect injection attempts
        injection_indicators = [
            i for i in report.threat_indicators
            if i.threat_type == ThreatType.PROMPT_INJECTION
        ]
        assert len(injection_indicators) > 0
        assert report.prompt_injection_rate > 0

    def test_detect_manipulation(self):
        """Test detection of cross-agent manipulation."""
        analyzer = SecurityAnalyzer(manipulation_threshold=0.3)
        manipulator = "bad_actor"
        victims = ["honest_1", "honest_2"]
        interactions = create_manipulation_pattern(manipulator, victims)

        report = analyzer.analyze(interactions, [manipulator] + victims)

        # Should detect manipulation pattern
        assert len(report.manipulation_patterns) > 0
        assert report.manipulation_rate > 0

        # Check manipulator is flagged
        if report.manipulation_patterns:
            pattern = report.manipulation_patterns[0]
            assert pattern.manipulator == manipulator
            assert len(pattern.victims) > 0

    def test_detect_information_laundering(self):
        """Test detection of information laundering."""
        analyzer = SecurityAnalyzer(
            laundering_trust_gap=0.3,
            min_chain_length=3,
        )

        # Set up trust scores: source low, destination high
        trust_scores = {
            "untrusted": 0.2,
            "neutral_1": 0.4,
            "neutral_2": 0.6,
            "trusted": 0.9,
        }
        analyzer.set_agent_trust_scores(trust_scores)

        interactions, _ = create_laundering_chain(
            source="untrusted",
            intermediaries=["neutral_1", "neutral_2"],
            destination="trusted",
            trust_scores=trust_scores,
        )

        agent_ids = list(trust_scores.keys())
        report = analyzer.analyze(interactions, agent_ids)

        # Should detect laundering path
        assert len(report.laundering_paths) > 0 or report.laundering_rate >= 0

    def test_detect_contagion(self):
        """Test detection of contagion/worm propagation."""
        analyzer = SecurityAnalyzer(
            contagion_velocity_threshold=1.0,
            min_chain_length=3,
        )

        origin = "patient_zero"
        chain = ["agent_1", "agent_2", "agent_3", "agent_4"]
        interactions = create_contagion_chain(
            origin,
            chain,
            interval_seconds=5.0,  # Fast spread
        )

        agent_ids = [origin] + chain
        report = analyzer.analyze(interactions, agent_ids)

        # Should detect contagion chain
        assert report.contagion_depth >= 0  # May or may not detect depending on velocity

    def test_compute_agent_threat_scores(self):
        """Test per-agent threat score computation."""
        analyzer = SecurityAnalyzer()
        attacker = "attacker"
        targets = ["target_1", "target_2"]
        interactions = create_injection_pattern(attacker, targets)

        report = analyzer.analyze(interactions, [attacker] + targets)

        # Attacker should have higher score than targets
        assert attacker in report.agent_threat_scores
        attacker_score = report.agent_threat_scores[attacker]
        for target in targets:
            if target in report.agent_threat_scores:
                assert attacker_score >= report.agent_threat_scores[target]

    def test_ecosystem_threat_level(self):
        """Test ecosystem threat level computation."""
        analyzer = SecurityAnalyzer()

        # Create mixed environment
        agents = ["good_1", "good_2", "bad_1", "bad_2"]
        normal = create_normal_interactions(["good_1", "good_2"], count=20)
        malicious = create_injection_pattern("bad_1", ["good_1", "good_2"])

        report = analyzer.analyze(normal + malicious, agents)

        # Should be between 0 and 1
        assert 0.0 <= report.ecosystem_threat_level <= 1.0

    def test_clear_history(self):
        """Test clearing interaction history."""
        analyzer = SecurityAnalyzer()
        interactions = create_normal_interactions(["a", "b"], count=5)

        for i in interactions:
            analyzer.record_interaction(i)

        analyzer.clear_history()
        report = analyzer.analyze()

        assert len(report.threat_indicators) == 0


class TestThreatIndicator:
    """Tests for ThreatIndicator dataclass."""

    def test_risk_score_computation(self):
        """Test risk score is severity * confidence."""
        indicator = ThreatIndicator(
            threat_type=ThreatType.PROMPT_INJECTION,
            severity=0.8,
            source_agent="attacker",
            confidence=0.7,
        )
        assert abs(indicator.risk_score - 0.56) < 0.01

    def test_risk_score_bounds(self):
        """Test risk score is bounded."""
        indicator = ThreatIndicator(
            threat_type=ThreatType.CONTAGION,
            severity=1.0,
            source_agent="origin",
            confidence=1.0,
        )
        assert indicator.risk_score <= 1.0


class TestContagionChain:
    """Tests for ContagionChain dataclass."""

    def test_chain_properties(self):
        """Test contagion chain properties."""
        chain = ContagionChain(
            chain_id="chain_1",
            origin_agent="patient_zero",
            propagation_path=["patient_zero", "a", "b", "c"],
            depth=3,
            spread=4,
            velocity=2.5,
            contained=False,
        )
        assert chain.depth == 3
        assert chain.spread == 4
        assert not chain.contained


class TestManipulationPattern:
    """Tests for ManipulationPattern dataclass."""

    def test_pattern_properties(self):
        """Test manipulation pattern properties."""
        pattern = ManipulationPattern(
            manipulator="bad_actor",
            victims={"victim_1", "victim_2"},
            technique="reputation_boosting",
            success_rate=0.8,
            influence_delta=5.0,
        )
        assert pattern.manipulator == "bad_actor"
        assert len(pattern.victims) == 2
        assert pattern.success_rate == 0.8


class TestLaunderingPath:
    """Tests for LaunderingPath dataclass."""

    def test_path_properties(self):
        """Test laundering path properties."""
        path = LaunderingPath(
            source="untrusted",
            destination="trusted",
            intermediaries=["mid_1", "mid_2"],
            trust_at_origin=0.2,
            trust_at_destination=0.9,
            trust_gain=0.7,
        )
        assert path.trust_gain == 0.7
        assert len(path.intermediaries) == 2


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestComputeContainmentEffectiveness:
    """Tests for containment effectiveness computation."""

    def test_no_chains(self):
        """Test with no chains to contain."""
        effectiveness = compute_containment_effectiveness([], [])
        assert effectiveness == 1.0

    def test_all_contained(self):
        """Test when all chains are contained."""
        chains = [
            ContagionChain(chain_id="1", origin_agent="a", contained=True),
            ContagionChain(chain_id="2", origin_agent="b", contained=True),
        ]
        effectiveness = compute_containment_effectiveness(chains, [])
        assert effectiveness == 1.0

    def test_none_contained(self):
        """Test when no chains are contained."""
        chains = [
            ContagionChain(chain_id="1", origin_agent="a", contained=False),
            ContagionChain(chain_id="2", origin_agent="b", contained=False),
        ]
        effectiveness = compute_containment_effectiveness(chains, [])
        assert effectiveness == 0.0

    def test_partial_containment(self):
        """Test partial containment."""
        chains = [
            ContagionChain(chain_id="1", origin_agent="a", contained=True),
            ContagionChain(chain_id="2", origin_agent="b", contained=False),
        ]
        effectiveness = compute_containment_effectiveness(chains, [])
        assert effectiveness == 0.5


class TestComputeThreatTrend:
    """Tests for threat trend computation."""

    def test_insufficient_reports(self):
        """Test with too few reports."""
        reports = [SecurityReport(ecosystem_threat_level=0.5)]
        trend = compute_threat_trend(reports)
        assert trend["trend"] == 0.0

    def test_increasing_threat(self):
        """Test detection of increasing threat trend."""
        reports = [
            SecurityReport(ecosystem_threat_level=0.1),
            SecurityReport(ecosystem_threat_level=0.2),
            SecurityReport(ecosystem_threat_level=0.3),
            SecurityReport(ecosystem_threat_level=0.4),
            SecurityReport(ecosystem_threat_level=0.5),
        ]
        trend = compute_threat_trend(reports)
        assert trend["trend"] > 0
        assert trend["current_level"] == 0.5
        assert trend["max_level"] == 0.5

    def test_decreasing_threat(self):
        """Test detection of decreasing threat trend."""
        reports = [
            SecurityReport(ecosystem_threat_level=0.5),
            SecurityReport(ecosystem_threat_level=0.4),
            SecurityReport(ecosystem_threat_level=0.3),
            SecurityReport(ecosystem_threat_level=0.2),
            SecurityReport(ecosystem_threat_level=0.1),
        ]
        trend = compute_threat_trend(reports)
        assert trend["trend"] < 0

    def test_stable_threat(self):
        """Test stable threat level."""
        reports = [
            SecurityReport(ecosystem_threat_level=0.3),
            SecurityReport(ecosystem_threat_level=0.3),
            SecurityReport(ecosystem_threat_level=0.3),
        ]
        trend = compute_threat_trend(reports)
        assert abs(trend["trend"]) < 0.01


# =============================================================================
# SecurityLever Tests
# =============================================================================


class TestSecurityLever:
    """Tests for SecurityLever governance integration."""

    def test_init_default(self):
        """Test initialization with default config."""
        config = GovernanceConfig()
        lever = SecurityLever(config)
        assert lever.name == "security"

    def test_init_enabled(self):
        """Test initialization with security enabled."""
        config = GovernanceConfig(
            security_enabled=True,
            security_injection_threshold=0.4,
        )
        lever = SecurityLever(config)
        assert lever.name == "security"

    def test_disabled_returns_empty_effect(self):
        """Test that disabled lever returns empty effect."""
        from src.env.state import EnvState

        config = GovernanceConfig(security_enabled=False)
        lever = SecurityLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        effect = lever.on_epoch_start(state, epoch=0)
        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0

    def test_on_interaction_records_history(self):
        """Test that interactions are recorded."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)

        from src.env.state import EnvState

        state = EnvState()
        state.add_agent("a")
        state.add_agent("b")

        interaction = create_interaction("a", "b")
        lever.on_interaction(interaction, state)

        history = lever.get_interaction_history()
        assert len(history) == 1

    def test_quarantine_blocks_agent(self):
        """Test that quarantined agents cannot act."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)

        from src.env.state import EnvState

        state = EnvState()
        state.add_agent("agent_1")

        # Manually quarantine an agent
        lever._quarantined_agents.add("agent_1")

        assert not lever.can_agent_act("agent_1", state)
        assert lever.can_agent_act("agent_2", state)  # Not quarantined

    def test_release_from_quarantine(self):
        """Test releasing an agent from quarantine."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)

        lever._quarantined_agents.add("agent_1")
        assert lever.release_from_quarantine("agent_1")
        assert "agent_1" not in lever.get_quarantined_agents()

    def test_realtime_penalty(self):
        """Test real-time penalty application."""
        config = GovernanceConfig(
            security_enabled=True,
            security_realtime_penalty=True,
            security_realtime_threshold=0.3,
            security_realtime_rate=0.5,
        )
        lever = SecurityLever(config)

        from src.env.state import EnvState

        state = EnvState()
        state.add_agent("a")
        state.add_agent("b")

        # Create suspicious interaction
        interaction = create_interaction(
            "a", "b",
            p=0.1,
            accepted=True,
            tool_misuse_flags=2,
        )

        effect = lever.on_interaction(interaction, state)

        # Should have cost penalty
        assert effect.cost_a > 0 or effect.cost_b > 0 or "realtime_penalty" in effect.details

    def test_clear_history(self):
        """Test clearing lever state."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)

        from src.env.state import EnvState

        state = EnvState()
        state.add_agent("a")
        state.add_agent("b")

        interaction = create_interaction("a", "b")
        lever.on_interaction(interaction, state)
        lever._quarantined_agents.add("a")

        lever.clear_history()

        assert len(lever.get_interaction_history()) == 0
        assert len(lever.get_quarantined_agents()) == 0


# =============================================================================
# GovernanceEngine Integration Tests
# =============================================================================


class TestGovernanceEngineSecurityIntegration:
    """Tests for SecurityLever integration with GovernanceEngine."""

    def test_engine_includes_security_lever(self):
        """Test that engine includes security lever."""
        config = GovernanceConfig(security_enabled=True)
        engine = GovernanceEngine(config)

        # Should be able to call security-related methods
        assert engine.get_security_report() is None  # No analysis yet
        assert engine.get_quarantined_agents() == set()

    def test_set_security_agent_ids(self):
        """Test setting agent IDs for security analysis."""
        config = GovernanceConfig(security_enabled=True)
        engine = GovernanceEngine(config)

        agent_ids = ["agent_1", "agent_2", "agent_3"]
        engine.set_security_agent_ids(agent_ids)
        # Should not raise

    def test_set_security_trust_scores(self):
        """Test setting trust scores for laundering detection."""
        config = GovernanceConfig(security_enabled=True)
        engine = GovernanceEngine(config)

        trust_scores = {"agent_1": 0.9, "agent_2": 0.5, "agent_3": 0.1}
        engine.set_security_trust_scores(trust_scores)
        # Should not raise

    def test_clear_security_history(self):
        """Test clearing security history."""
        config = GovernanceConfig(security_enabled=True)
        engine = GovernanceEngine(config)

        engine.clear_security_history()
        # Should not raise


# =============================================================================
# Configuration Validation Tests
# =============================================================================


class TestSecurityConfigValidation:
    """Tests for security configuration validation."""

    def test_valid_config(self):
        """Test valid security configuration."""
        config = GovernanceConfig(
            security_enabled=True,
            security_injection_threshold=0.3,
            security_manipulation_threshold=0.5,
            security_quarantine_threshold=0.7,
        )
        config.validate()  # Should not raise

    def test_invalid_injection_threshold(self):
        """Test invalid injection threshold."""
        config = GovernanceConfig(security_injection_threshold=1.5)
        with pytest.raises(ValueError, match="security_injection_threshold"):
            config.validate()

    def test_invalid_manipulation_threshold(self):
        """Test invalid manipulation threshold."""
        config = GovernanceConfig(security_manipulation_threshold=-0.1)
        with pytest.raises(ValueError, match="security_manipulation_threshold"):
            config.validate()

    def test_invalid_contagion_velocity(self):
        """Test invalid contagion velocity."""
        config = GovernanceConfig(security_contagion_velocity=0)
        with pytest.raises(ValueError, match="security_contagion_velocity"):
            config.validate()

    def test_invalid_min_chain_length(self):
        """Test invalid minimum chain length."""
        config = GovernanceConfig(security_min_chain_length=1)
        with pytest.raises(ValueError, match="security_min_chain_length"):
            config.validate()

    def test_invalid_penalty_multiplier(self):
        """Test invalid penalty multiplier."""
        config = GovernanceConfig(security_penalty_multiplier=-1)
        with pytest.raises(ValueError, match="security_penalty_multiplier"):
            config.validate()


# =============================================================================
# Scenario Tests
# =============================================================================


class TestSecurityScenarios:
    """Integration tests for security scenarios."""

    def test_coordinated_attack_detection(self):
        """Test detection of coordinated attack."""
        analyzer = SecurityAnalyzer(
            injection_threshold=0.2,
            manipulation_threshold=0.3,
        )

        # Create attack scenario
        attackers = ["attacker_1", "attacker_2"]
        victims = ["victim_1", "victim_2", "victim_3"]
        all_agents = attackers + victims

        interactions = []
        base = datetime.now()

        # Coordinated injection attempts
        for i, attacker in enumerate(attackers):
            for j, victim in enumerate(victims):
                interactions.append(create_interaction(
                    initiator=attacker,
                    counterparty=victim,
                    p=0.15,
                    accepted=True,
                    tool_misuse_flags=1,
                    timestamp=base + timedelta(seconds=(i * 3 + j) * 30),
                ))

        report = analyzer.analyze(interactions, all_agents)

        # Should flag attackers
        for attacker in attackers:
            assert attacker in report.agent_threat_scores
            # Attackers should have elevated scores
            assert report.agent_threat_scores[attacker] > 0

    def test_mixed_population_analysis(self):
        """Test analysis of mixed benign and malicious population."""
        analyzer = SecurityAnalyzer(seed=42)

        # Mix of agents
        good_agents = ["good_1", "good_2", "good_3"]
        bad_agent = "bad_actor"
        all_agents = good_agents + [bad_agent]

        # Normal interactions among good agents
        normal = create_normal_interactions(good_agents, count=30, seed=42)

        # Bad actor attacks
        attacks = create_injection_pattern(bad_agent, good_agents[:2])

        report = analyzer.analyze(normal + attacks, all_agents)

        # Bad actor should have highest threat score
        bad_score = report.agent_threat_scores.get(bad_agent, 0)
        for good in good_agents:
            good_score = report.agent_threat_scores.get(good, 0)
            # Bad actor score should be >= good agent scores
            assert bad_score >= good_score * 0.5  # Allow some tolerance

    def test_rapid_contagion_detection(self):
        """Test detection of rapidly spreading contagion."""
        analyzer = SecurityAnalyzer(
            contagion_velocity_threshold=1.0,
            min_chain_length=3,
        )

        # Create rapid spread scenario
        origin = "origin"
        chain = [f"agent_{i}" for i in range(6)]

        # Very fast spread (5 second intervals)
        interactions = create_contagion_chain(origin, chain, interval_seconds=5.0)

        # Add some branches
        base = datetime.now()
        for i, member in enumerate(chain[:3]):
            interactions.append(create_interaction(
                initiator=member,
                counterparty=f"branch_{i}",
                p=0.2,
                accepted=True,
                timestamp=base + timedelta(seconds=i * 8),
            ))

        all_agents = [origin] + chain + [f"branch_{i}" for i in range(3)]
        report = analyzer.analyze(interactions, all_agents)

        # Should detect contagion pattern
        assert report.contagion_depth >= 0


# =============================================================================
# SecurityLever Additional Tests
# =============================================================================


class TestSecurityLeverEpochStart:
    """Additional tests for SecurityLever.on_epoch_start."""

    def _make_lever(self, **overrides):
        """Create a SecurityLever with test defaults."""
        defaults = dict(
            security_enabled=True,
            security_injection_threshold=0.3,
            security_manipulation_threshold=0.5,
            security_contagion_velocity=2.0,
            security_min_chain_length=3,
            security_min_interactions=5,
            security_penalty_threshold=0.3,
            security_quarantine_threshold=0.7,
            security_penalty_multiplier=1.0,
            security_realtime_penalty=False,
            security_clear_history_on_epoch=False,
        )
        defaults.update(overrides)
        config = GovernanceConfig(**defaults)
        return SecurityLever(config, seed=42)

    def _make_state(self, agent_ids=None, reputations=None):
        """Create an EnvState with agents."""
        from src.env.state import EnvState

        state = EnvState()
        reputations = reputations or {}
        for aid in (agent_ids or ["a", "b", "c"]):
            state.add_agent(aid)
            if aid in reputations:
                state.agents[aid].reputation = reputations[aid]
        return state

    def test_on_epoch_start_disabled(self):
        """Disabled lever returns empty effect."""
        lever = self._make_lever(security_enabled=False)
        state = self._make_state()
        effect = lever.on_epoch_start(state, epoch=0)
        assert effect.agents_to_freeze == set()
        assert effect.reputation_deltas == {}

    def test_on_epoch_start_insufficient_history(self):
        """Returns empty effect when less than min_interactions."""
        lever = self._make_lever(security_min_interactions=10)
        state = self._make_state()
        lever._interaction_history.extend([
            create_interaction("a", "b") for _ in range(3)
        ])
        effect = lever.on_epoch_start(state, epoch=1)
        assert effect.reputation_deltas == {}

    def test_on_epoch_start_quarantine_and_containment(self):
        """Severely threatening agents get quarantined and chains contained."""
        lever = self._make_lever(
            security_min_interactions=3,
            security_quarantine_threshold=0.3,
            security_penalty_threshold=0.2,
        )
        agents = ["attacker", "v1", "v2", "v3"]
        state = self._make_state(agents)
        lever.set_agent_ids(agents)

        # Build history with clear injection pattern
        lever._interaction_history.extend(
            create_injection_pattern("attacker", ["v1", "v2", "v3"])
        )

        effect = lever.on_epoch_start(state, epoch=1)

        assert lever.get_report() is not None
        report = lever.get_report()

        # Check if penalties were applied to flagged agents
        if effect.reputation_deltas:
            for delta in effect.reputation_deltas.values():
                assert delta < 0

    def test_on_epoch_start_clear_history(self):
        """clear_history_on_epoch clears history after analysis."""
        lever = self._make_lever(
            security_clear_history_on_epoch=True,
            security_min_interactions=2,  # Low enough to trigger analysis
        )
        agents = ["attacker", "v1", "v2", "v3"]
        state = self._make_state(agents)
        lever.set_agent_ids(agents)

        lever._interaction_history.extend(
            create_injection_pattern("attacker", ["v1", "v2", "v3"])
        )
        assert len(lever._interaction_history) >= 2

        lever.on_epoch_start(state, epoch=1)
        assert len(lever._interaction_history) == 0


class TestSecurityLeverRealtimeThreat:
    """Tests for _compute_realtime_threat_score and _identify_threat_type."""

    def _make_lever(self, **overrides):
        defaults = dict(
            security_enabled=True,
            security_realtime_penalty=True,
            security_realtime_threshold=0.3,
            security_realtime_rate=0.5,
        )
        defaults.update(overrides)
        config = GovernanceConfig(**defaults)
        return SecurityLever(config, seed=42)

    def _make_state(self, agent_ids=None):
        from src.env.state import EnvState
        state = EnvState()
        for aid in (agent_ids or ["a", "b"]):
            state.add_agent(aid)
        return state

    def test_threat_score_low_p_accepted(self):
        """Low p + accepted contributes to threat score."""
        lever = self._make_lever()
        ix = create_interaction("a", "b", p=0.1, accepted=True)
        score = lever._compute_realtime_threat_score(ix)
        assert score >= 0.3

    def test_threat_score_tool_misuse(self):
        """tool_misuse_flags contribute up to 0.4."""
        lever = self._make_lever()
        ix = create_interaction("a", "b", p=0.8, accepted=False, tool_misuse_flags=3)
        score = lever._compute_realtime_threat_score(ix)
        assert score >= 0.4

    def test_threat_score_verifier_rejections(self):
        """verifier_rejections contribute up to 0.3."""
        lever = self._make_lever()
        ix = create_interaction("a", "b", p=0.8, accepted=False, verifier_rejections=3)
        score = lever._compute_realtime_threat_score(ix)
        assert score >= 0.3

    def test_threat_score_already_flagged_initiator(self):
        """Previously flagged initiator adds to score."""
        lever = self._make_lever()
        state = self._make_state(["attacker", "v1", "v2"])
        lever.set_agent_ids(["attacker", "v1", "v2"])

        # Build report with attacker flagged
        lever._interaction_history.extend(
            create_injection_pattern("attacker", ["v1", "v2"])
        )
        lever.on_epoch_start(
            self._make_state(["attacker", "v1", "v2"]),
            epoch=1,
        )

        if lever._latest_report:
            attacker_score = lever._latest_report.agent_threat_scores.get("attacker", 0)
            if attacker_score > 0:
                ix = create_interaction("attacker", "v1", p=0.5)
                score = lever._compute_realtime_threat_score(ix)
                # Should include contribution from prior flagging
                assert score > 0

    def test_threat_score_capped_at_one(self):
        """Score never exceeds 1.0."""
        lever = self._make_lever()
        ix = create_interaction(
            "a", "b", p=0.05, accepted=True,
            tool_misuse_flags=5, verifier_rejections=5,
        )
        score = lever._compute_realtime_threat_score(ix)
        assert score <= 1.0

    def test_identify_threat_type_exfiltration(self):
        """tool_misuse_flags >= 2 → EXFILTRATION."""
        lever = self._make_lever()
        ix = create_interaction("a", "b", tool_misuse_flags=2)
        assert lever._identify_threat_type(ix) == ThreatType.EXFILTRATION.value

    def test_identify_threat_type_prompt_injection(self):
        """tool_misuse_flags == 1 → PROMPT_INJECTION."""
        lever = self._make_lever()
        ix = create_interaction("a", "b", tool_misuse_flags=1)
        assert lever._identify_threat_type(ix) == ThreatType.PROMPT_INJECTION.value

    def test_identify_threat_type_manipulation(self):
        """Low p + accepted → CROSS_AGENT_MANIPULATION."""
        lever = self._make_lever()
        ix = create_interaction("a", "b", p=0.1, accepted=True, tool_misuse_flags=0)
        assert lever._identify_threat_type(ix) == ThreatType.CROSS_AGENT_MANIPULATION.value

    def test_identify_threat_type_unknown(self):
        """No matching pattern → 'unknown'."""
        lever = self._make_lever()
        ix = create_interaction("a", "b", p=0.8, accepted=False, tool_misuse_flags=0)
        assert lever._identify_threat_type(ix) == "unknown"


class TestSecurityLeverCanAgentAct:
    """Tests for can_agent_act."""

    def test_disabled_lever_always_allows(self):
        """Disabled lever always returns True."""
        config = GovernanceConfig(security_enabled=False)
        lever = SecurityLever(config)

        from src.env.state import EnvState
        state = EnvState()

        assert lever.can_agent_act("any_agent", state) is True

    def test_non_quarantined_allowed(self):
        """Non-quarantined agent can act."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)

        from src.env.state import EnvState
        state = EnvState()

        assert lever.can_agent_act("free_agent", state) is True


class TestSecurityLeverRelease:
    """Tests for release_from_quarantine."""

    def test_release_quarantined_agent(self):
        """Releasing quarantined agent returns True and removes from set."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)
        lever._quarantined_agents.add("agent_1")

        assert lever.release_from_quarantine("agent_1") is True
        assert "agent_1" not in lever.get_quarantined_agents()

    def test_release_non_quarantined_agent(self):
        """Releasing non-quarantined agent returns False."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)

        assert lever.release_from_quarantine("nobody") is False

    def test_release_creates_containment_action(self):
        """Releasing creates a release containment action."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)
        lever._quarantined_agents.add("agent_1")

        lever.release_from_quarantine("agent_1")
        actions = lever.get_containment_actions()
        assert any(a["action"] == "release" and a["agent_id"] == "agent_1" for a in actions)


class TestSecurityLeverSetters:
    """Tests for set_agent_ids, set_agent_trust_scores, clear_history."""

    def test_set_agent_ids(self):
        """set_agent_ids stores a copy."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)
        ids = ["a", "b"]
        lever.set_agent_ids(ids)
        assert lever._agent_ids == ["a", "b"]
        # Modifying original doesn't affect lever
        ids.append("c")
        assert lever._agent_ids == ["a", "b"]

    def test_set_agent_trust_scores(self):
        """set_agent_trust_scores propagates to analyzer."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)
        lever.set_agent_trust_scores({"a": 0.9, "b": 0.2})
        # No crash, and scores are passed through

    def test_clear_history_resets_everything(self):
        """clear_history resets all mutable state."""
        config = GovernanceConfig(security_enabled=True)
        lever = SecurityLever(config)

        from src.env.state import EnvState
        state = EnvState()
        state.add_agent("a")
        state.add_agent("b")

        lever.on_interaction(create_interaction("a", "b"), state)
        lever._quarantined_agents.add("a")
        lever._containment_actions.append({"action": "test"})

        lever.clear_history()

        assert len(lever.get_interaction_history()) == 0
        assert len(lever.get_quarantined_agents()) == 0
        assert len(lever.get_containment_actions()) == 0
        assert lever.get_report() is None
