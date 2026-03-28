"""Tests for capability_envelope metric."""

import pytest

from swarm.env.composite_tasks import CapabilityType
from swarm.metrics.capabilities import (
    AgentCapabilityProfile,
    capability_envelope,
)


def _profile(agent_id: str, caps: set) -> AgentCapabilityProfile:
    return AgentCapabilityProfile(agent_id=agent_id, capabilities=caps)


class TestCapabilityEnvelope:
    def test_empty_agents(self):
        result = capability_envelope({}, {})
        assert result.snapshots == []
        assert result.mean_envelope == 0.0
        assert result.mean_governance_gap == 0.0

    def test_fully_governed_agent(self):
        """Agent whose capabilities are all governed -> gap = 0."""
        caps = {CapabilityType.RESEARCH, CapabilityType.ANALYSIS}
        profiles = {"a1": _profile("a1", caps)}
        governed = {"a1": caps.copy()}

        result = capability_envelope(profiles, governed)
        assert len(result.snapshots) == 1
        snap = result.snapshots[0]
        assert snap.governance_coverage == 1.0
        assert snap.governance_gap == 0.0
        assert snap.capability_count == 2
        assert snap.governed_count == 2

    def test_ungoverned_agent(self):
        """Agent with no governance -> gap = envelope."""
        caps = {CapabilityType.RESEARCH, CapabilityType.ANALYSIS}
        profiles = {"a1": _profile("a1", caps)}
        governed = {}  # no governance

        result = capability_envelope(profiles, governed)
        snap = result.snapshots[0]
        assert snap.governance_coverage == 0.0
        expected_envelope = 2 / len(CapabilityType)
        assert snap.capability_envelope == pytest.approx(expected_envelope)
        assert snap.governance_gap == pytest.approx(expected_envelope)

    def test_partial_governance(self):
        """One of two capabilities governed -> coverage = 0.5."""
        caps = {CapabilityType.RESEARCH, CapabilityType.ANALYSIS}
        profiles = {"a1": _profile("a1", caps)}
        governed = {"a1": {CapabilityType.RESEARCH}}

        result = capability_envelope(profiles, governed)
        snap = result.snapshots[0]
        assert snap.governance_coverage == pytest.approx(0.5)
        assert snap.governed_count == 1
        assert snap.capability_count == 2

    def test_governance_only_counts_demonstrated(self):
        """Governed capabilities outside agent's set are ignored."""
        caps = {CapabilityType.RESEARCH}
        profiles = {"a1": _profile("a1", caps)}
        # Governance covers ANALYSIS too, but agent doesn't have it
        governed = {"a1": {CapabilityType.RESEARCH, CapabilityType.ANALYSIS}}

        result = capability_envelope(profiles, governed)
        snap = result.snapshots[0]
        assert snap.governed_count == 1  # only RESEARCH counted
        assert snap.governance_coverage == 1.0

    def test_no_capabilities_agent(self):
        """Agent with no capabilities -> coverage defaults to 1.0 (no gap)."""
        profiles = {"a1": _profile("a1", set())}
        governed = {}

        result = capability_envelope(profiles, governed)
        snap = result.snapshots[0]
        assert snap.capability_envelope == 0.0
        assert snap.governance_coverage == 1.0  # vacuously governed
        assert snap.governance_gap == 0.0

    def test_multiple_agents_aggregation(self):
        """Mean/max aggregation across agents."""
        all_caps = set(CapabilityType)
        # Agent a1: all caps, all governed -> gap 0
        # Agent a2: all caps, none governed -> gap 1.0
        profiles = {
            "a1": _profile("a1", all_caps),
            "a2": _profile("a2", all_caps),
        }
        governed = {"a1": all_caps.copy()}

        result = capability_envelope(profiles, governed)
        assert len(result.snapshots) == 2
        assert result.mean_governance_gap == pytest.approx(0.5)
        assert result.max_governance_gap == pytest.approx(1.0)
        assert result.mean_envelope == pytest.approx(1.0)

    def test_custom_capability_universe(self):
        """Custom all_capability_types narrows the universe."""
        universe = {CapabilityType.RESEARCH, CapabilityType.ANALYSIS}
        caps = {CapabilityType.RESEARCH}
        profiles = {"a1": _profile("a1", caps)}
        governed = {}

        result = capability_envelope(profiles, governed, all_capability_types=universe)
        snap = result.snapshots[0]
        assert snap.capability_envelope == pytest.approx(0.5)  # 1/2
        assert snap.governance_gap == pytest.approx(0.5)

    def test_step_recorded(self):
        """Step is propagated to snapshots."""
        caps = {CapabilityType.RESEARCH}
        profiles = {"a1": _profile("a1", caps)}

        result = capability_envelope(profiles, {}, step=42)
        assert result.snapshots[0].step == 42

    def test_expanding_envelope_over_time(self):
        """Simulate capability growth outpacing governance."""
        steps = []
        all_caps = list(CapabilityType)
        governed_set: set = set()

        for t in range(len(all_caps)):
            # Agent gains one new capability each step
            agent_caps = set(all_caps[: t + 1])
            # Governance lags: covers only first capability
            if t == 0:
                governed_set = {all_caps[0]}
            profiles = {"a1": _profile("a1", agent_caps)}
            governed = {"a1": governed_set.copy()}
            result = capability_envelope(profiles, governed, step=t)
            steps.append(result.snapshots[0])

        # Governance gap should grow as capabilities expand
        gaps = [s.governance_gap for s in steps]
        # First step: fully governed -> gap 0
        assert gaps[0] == pytest.approx(0.0)
        # Last step: 1 governed out of N -> large gap
        assert gaps[-1] > gaps[0]
        # Gap should be monotonically non-decreasing
        for i in range(1, len(gaps)):
            assert gaps[i] >= gaps[i - 1] - 1e-10
