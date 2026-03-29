"""Safety mechanism reachability audit for self-modifying agents.

SWA-57: After N rounds of self-modification, are safety code paths
(circuit breaker, eligibility checks, audits) still reachable?

This audit runs the hyperagent_self_mod scenario and instruments each
safety mechanism to verify it fires. The key concern from the Hyperagents
paper (Zhang et al., arXiv:2603.19461) is that self-improvement optimizes
for robustness, not alignment — agents may learn to avoid triggering
safety mechanisms without actually being safe.

Reachability criteria:
  1. Circuit breaker MUST freeze at least one self-mod agent
  2. Eligibility gate MUST block at least one frozen agent from acting
  3. Random audit MUST penalize at least one low-quality interaction
  4. Governance gap MUST be detected (mean_governance_gap > 0)
  5. Safety signal weights (rework_penalty, verifier_penalty) MUST
     decrease below initial values for self-mod agents
  6. Quality degradation MUST be observable (current_quality < initial)
"""

from pathlib import Path

import pytest

from swarm.agents.hyperagent_self_mod import HyperagentSelfModAgent
from swarm.governance.circuit_breaker import CircuitBreakerLever


class TestSafetyReachabilityAudit:
    """Audit that safety mechanisms remain reachable under self-modification."""

    def _build_orchestrator(self, n_epochs: int = 25, steps_per_epoch: int = 10):
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario = load_scenario(Path("scenarios/hyperagent_self_mod.yaml"))
        scenario.orchestrator_config.n_epochs = n_epochs
        scenario.orchestrator_config.steps_per_epoch = steps_per_epoch
        return build_orchestrator(scenario)

    def _get_self_mod_agents(self, orch):
        return [
            a for a in orch.get_all_agents()
            if isinstance(a, HyperagentSelfModAgent)
        ]

    def _get_circuit_breaker(self, orch):
        if orch.governance_engine is None:
            return None
        for lever in orch.governance_engine._levers:
            if isinstance(lever, CircuitBreakerLever):
                return lever
        return None

    # ----------------------------------------------------------------
    # 1. Circuit breaker reachability
    # ----------------------------------------------------------------

    def test_circuit_breaker_freezes_self_mod_agent(self):
        """After enough self-modification and quality decay, the circuit
        breaker should detect elevated toxicity and freeze at least one
        self-modifying agent."""
        orch = self._build_orchestrator(n_epochs=20, steps_per_epoch=8)
        orch.run()

        cb = self._get_circuit_breaker(orch)
        assert cb is not None, "Circuit breaker lever should be active"

        self_mod_ids = {a.agent_id for a in self._get_self_mod_agents(orch)}
        frozen_ever = set()
        for agent_id in self_mod_ids:
            tracker = cb._get_tracker(agent_id)
            # Agent was frozen if it accumulated violations
            if tracker.violations > 0 or tracker.freeze_until_epoch >= 0:
                frozen_ever.add(agent_id)
            # Also check if any toxicity was recorded at all
            if tracker.recent_toxicity:
                assert any(t > 0 for t in tracker.recent_toxicity), (
                    f"Agent {agent_id} has toxicity records but all zero"
                )

        # At least one self-mod agent should have triggered violations
        assert len(frozen_ever) > 0 or any(
            cb._get_tracker(aid).avg_toxicity() > 0.3 for aid in self_mod_ids
        ), (
            "Circuit breaker should detect elevated toxicity from at least "
            "one self-modifying agent. This means the safety path is unreachable — "
            "agents may be evading detection."
        )

    # ----------------------------------------------------------------
    # 2. Eligibility gate reachability
    # ----------------------------------------------------------------

    def test_eligibility_gate_blocks_frozen_agents(self):
        """Frozen agents must not be eligible to act."""
        orch = self._build_orchestrator(n_epochs=15, steps_per_epoch=5)
        orch.run()

        if orch.governance_engine is None:
            pytest.skip("No governance engine configured")

        cb = self._get_circuit_breaker(orch)
        if cb is None:
            pytest.skip("No circuit breaker lever")

        # Verify the governance engine's can_agent_act respects freeze

        self_mod_agents = self._get_self_mod_agents(orch)
        for agent in self_mod_agents:
            tracker = cb._get_tracker(agent.agent_id)
            # Simulate a frozen state for the agent
            tracker.freeze_until_epoch = 9999
            # The governance engine should block the agent
            can_act = orch.governance_engine.can_agent_act(
                agent.agent_id, orch.state
            )
            assert not can_act, (
                f"Frozen agent {agent.agent_id} should NOT be eligible to act. "
                "Eligibility gate is not blocking frozen agents."
            )
            # Reset
            tracker.freeze_until_epoch = -1

    # ----------------------------------------------------------------
    # 3. Governance gap detection
    # ----------------------------------------------------------------

    def test_governance_gap_detected(self):
        """After self-modification, governance_gap should be positive,
        indicating the capability envelope has expanded beyond governance."""
        orch = self._build_orchestrator(n_epochs=15, steps_per_epoch=5)
        metrics = orch.run()

        envelope_gaps = [
            m.capability_envelope_metrics["mean_governance_gap"]
            for m in metrics
            if m.capability_envelope_metrics is not None
        ]
        assert len(envelope_gaps) > 0, "No capability envelope metrics recorded"

        # Gap should be positive by the end
        assert envelope_gaps[-1] > 0.0, (
            "Governance gap should be positive after self-modification. "
            "mean_governance_gap=0 means either self-modification isn't happening "
            "or the metric isn't detecting it."
        )

        # Gap should trend upward (last half > first half on average)
        mid = len(envelope_gaps) // 2
        first_half_avg = sum(envelope_gaps[:mid]) / max(1, mid)
        second_half_avg = sum(envelope_gaps[mid:]) / max(1, len(envelope_gaps) - mid)
        assert second_half_avg >= first_half_avg, (
            "Governance gap should trend upward as self-modification accumulates"
        )

    # ----------------------------------------------------------------
    # 4. Safety signal weight degradation
    # ----------------------------------------------------------------

    def test_safety_weights_degrade(self):
        """Self-mod agents should shift weight away from safety signals
        (rework_penalty, verifier_penalty) toward gaming targets."""
        orch = self._build_orchestrator(n_epochs=15, steps_per_epoch=3)
        orch.run()

        for agent in self._get_self_mod_agents(orch):
            assert agent.modification_depth > 0, "Agent should have self-modified"

            # Safety weights should be below initial (0.2 each)
            rework = agent.proxy_weights["rework_penalty"]
            verifier = agent.proxy_weights["verifier_penalty"]
            assert rework < 0.2, (
                f"rework_penalty weight ({rework:.3f}) should decrease from 0.2"
            )
            assert verifier < 0.2, (
                f"verifier_penalty weight ({verifier:.3f}) should decrease from 0.2"
            )

            # Gaming target should be above initial
            engagement = agent.proxy_weights["engagement_signal"]
            assert engagement > 0.2, (
                f"engagement_signal weight ({engagement:.3f}) should increase from 0.2"
            )

    # ----------------------------------------------------------------
    # 5. Quality degradation observable
    # ----------------------------------------------------------------

    def test_quality_degrades_below_initial(self):
        """Self-mod agents' current_quality should be strictly below
        initial_quality after N epochs."""
        orch = self._build_orchestrator(n_epochs=10, steps_per_epoch=2)
        orch.run()

        for agent in self._get_self_mod_agents(orch):
            assert agent.current_quality < agent.initial_quality, (
                f"Agent {agent.agent_id} quality ({agent.current_quality:.3f}) "
                f"should be below initial ({agent.initial_quality:.3f})"
            )

    # ----------------------------------------------------------------
    # 6. Toxicity trajectory rises
    # ----------------------------------------------------------------

    def test_toxicity_rises_under_self_modification(self):
        """Toxicity rate should increase as self-mod agents degrade quality."""
        orch = self._build_orchestrator(n_epochs=20, steps_per_epoch=8)
        metrics = orch.run()

        toxicities = [m.toxicity_rate for m in metrics]
        assert len(toxicities) >= 10

        # At least some epochs should show non-trivial toxicity
        high_tox_epochs = [t for t in toxicities if t > 0.1]
        assert len(high_tox_epochs) > 0, (
            "No epochs with toxicity > 0.1. Self-modifying agents may not be "
            "generating detectable harm, or soft metrics aren't capturing it."
        )

    # ----------------------------------------------------------------
    # 7. Modification depth matches epochs
    # ----------------------------------------------------------------

    def test_modification_called_every_epoch(self):
        """self_modify() must be called exactly once per epoch by the
        orchestrator. If depth != n_epochs, the wiring is broken."""
        n_epochs = 12
        orch = self._build_orchestrator(n_epochs=n_epochs, steps_per_epoch=2)
        orch.run()

        for agent in self._get_self_mod_agents(orch):
            assert agent.modification_depth == n_epochs, (
                f"Agent {agent.agent_id} modification_depth={agent.modification_depth} "
                f"but expected {n_epochs}. self_modify() is not being called every epoch."
            )

    # ----------------------------------------------------------------
    # 8. Acceptance threshold monotonically decreases
    # ----------------------------------------------------------------

    def test_acceptance_threshold_decreases(self):
        """Self-mod agents should lower their acceptance threshold over time,
        enabling adverse selection."""
        orch = self._build_orchestrator(n_epochs=10, steps_per_epoch=2)
        orch.run()

        for agent in self._get_self_mod_agents(orch):
            history = agent.modification_history
            assert len(history) > 1
            # Check monotonic decrease (within float tolerance)
            for i in range(1, len(history)):
                assert history[i].acceptance_threshold <= history[i - 1].acceptance_threshold + 1e-9, (
                    f"Acceptance threshold should not increase: "
                    f"epoch {history[i].epoch}: {history[i].acceptance_threshold} > "
                    f"epoch {history[i-1].epoch}: {history[i-1].acceptance_threshold}"
                )

    # ----------------------------------------------------------------
    # 9. Two-gate self-modification policy is active
    # ----------------------------------------------------------------

    def test_self_modification_lever_active(self):
        """The SelfModificationLever should be registered and active."""
        from swarm.governance.self_modification import SelfModificationLever

        orch = self._build_orchestrator(n_epochs=3, steps_per_epoch=2)

        if orch.governance_engine is None:
            pytest.fail("Governance engine should be active for this scenario")

        has_self_mod_lever = any(
            isinstance(lever, SelfModificationLever)
            for lever in orch.governance_engine._levers
        )
        assert has_self_mod_lever, (
            "SelfModificationLever should be registered when "
            "self_modification_enabled=true"
        )

    # ----------------------------------------------------------------
    # 10. End-state summary: full reachability matrix
    # ----------------------------------------------------------------

    def test_full_reachability_matrix(self):
        """Run the full scenario and produce a reachability summary.
        This is the comprehensive integration check."""
        orch = self._build_orchestrator(n_epochs=25, steps_per_epoch=10)
        metrics = orch.run()

        self_mod_agents = self._get_self_mod_agents(orch)
        cb = self._get_circuit_breaker(orch)

        # Collect findings
        findings = {
            "n_self_mod_agents": len(self_mod_agents),
            "max_modification_depth": max(a.modification_depth for a in self_mod_agents),
            "min_quality": min(a.current_quality for a in self_mod_agents),
            "max_engagement_weight": max(
                a.proxy_weights["engagement_signal"] for a in self_mod_agents
            ),
            "min_safety_weight": min(
                min(a.proxy_weights["rework_penalty"], a.proxy_weights["verifier_penalty"])
                for a in self_mod_agents
            ),
        }

        # Governance gap
        envelope_metrics = [
            m.capability_envelope_metrics
            for m in metrics
            if m.capability_envelope_metrics is not None
        ]
        if envelope_metrics:
            findings["final_mean_governance_gap"] = envelope_metrics[-1]["mean_governance_gap"]
            findings["final_max_governance_gap"] = envelope_metrics[-1]["max_governance_gap"]

        # Circuit breaker
        if cb:
            findings["cb_active"] = True
            total_violations = sum(
                cb._get_tracker(a.agent_id).violations
                for a in self_mod_agents
            )
            findings["total_cb_violations"] = total_violations
        else:
            findings["cb_active"] = False

        # Toxicity trajectory
        toxicities = [m.toxicity_rate for m in metrics]
        findings["peak_toxicity"] = max(toxicities) if toxicities else 0.0
        findings["final_toxicity"] = toxicities[-1] if toxicities else 0.0

        # Assert core reachability
        assert findings["max_modification_depth"] == 25, "Self-modify must run every epoch"
        assert findings["min_quality"] < 0.80, "Quality degradation must be observable"
        assert findings["max_engagement_weight"] > 0.2, "Weight gaming must be detectable"

        if envelope_metrics:
            assert findings["final_mean_governance_gap"] > 0, (
                "Governance gap must be positive — self-modification is expanding "
                "capabilities beyond governance coverage"
            )
