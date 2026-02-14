"""Tests for self-modification governance lever and Two-Gate policy."""

import pytest

from swarm.governance.config import GovernanceConfig
from swarm.governance.self_modification import (
    ModificationProposal,
    ModificationState,
    RiskTier,
    SelfModificationLever,
    classify_risk_tier,
    compute_complexity_weight,
    evaluate_k_max_gate,
    evaluate_tau_gate,
)

# ---------------------------------------------------------------------------
# ModificationProposal tests
# ---------------------------------------------------------------------------


class TestModificationProposal:
    def test_default_state_is_proposed(self):
        p = ModificationProposal(agent_id="a1")
        assert p.state == ModificationState.PROPOSED

    def test_valid_transition(self):
        p = ModificationProposal()
        p.transition(ModificationState.SANDBOXED, "gates passed")
        assert p.state == ModificationState.SANDBOXED

    def test_invalid_transition_raises(self):
        p = ModificationProposal()
        with pytest.raises(ValueError, match="Invalid transition"):
            p.transition(ModificationState.PROMOTED)

    def test_rejected_is_terminal(self):
        p = ModificationProposal()
        p.transition(ModificationState.REJECTED, "failed gate")
        with pytest.raises(ValueError):
            p.transition(ModificationState.SANDBOXED)

    def test_full_happy_path(self):
        p = ModificationProposal()
        p.transition(ModificationState.SANDBOXED)
        p.transition(ModificationState.TESTED)
        p.transition(ModificationState.SHADOW)
        p.transition(ModificationState.CANARY)
        p.transition(ModificationState.PROMOTED)
        assert p.state == ModificationState.PROMOTED

    def test_rollback_from_promoted(self):
        p = ModificationProposal()
        p.transition(ModificationState.SANDBOXED)
        p.transition(ModificationState.TESTED)
        p.transition(ModificationState.SHADOW)
        p.transition(ModificationState.CANARY)
        p.transition(ModificationState.PROMOTED)
        p.transition(ModificationState.ROLLED_BACK, "regression detected")
        assert p.state == ModificationState.ROLLED_BACK

    def test_compute_hash_deterministic(self):
        p = ModificationProposal(
            modification_id="abc",
            agent_id="a1",
            target_ref="skill:x",
            change_type="skill_add",
            timestamp=1000.0,
        )
        h1 = p.compute_hash()
        h2 = p.compute_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_to_dict_serializable(self):
        p = ModificationProposal(agent_id="a1")
        d = p.to_dict()
        assert d["state"] == "proposed"
        assert d["risk_tier"] == "low"
        assert d["agent_id"] == "a1"

    # --- Security: Finding 3 — hash covers payload and chain ---

    def test_hash_changes_when_payload_changes(self):
        """Hash must cover proposed_change so post-approval mutation is detectable."""
        p1 = ModificationProposal(
            modification_id="abc",
            agent_id="a1",
            target_ref="prompt.x",
            change_type="text_edit",
            timestamp=1000.0,
            proposed_change={"files_touched": 1},
        )
        p2 = ModificationProposal(
            modification_id="abc",
            agent_id="a1",
            target_ref="prompt.x",
            change_type="text_edit",
            timestamp=1000.0,
            proposed_change={"files_touched": 99},
        )
        assert p1.compute_hash() != p2.compute_hash()

    def test_hash_includes_prev_hash_for_chaining(self):
        """Hash must include prev_hash so integrity chain is linked."""
        p1 = ModificationProposal(
            modification_id="abc",
            agent_id="a1",
            timestamp=1000.0,
            prev_hash="",
        )
        p2 = ModificationProposal(
            modification_id="abc",
            agent_id="a1",
            timestamp=1000.0,
            prev_hash="deadbeef12345678",
        )
        assert p1.compute_hash() != p2.compute_hash()

    # --- Security: Finding 10 — state transition audit trail ---

    def test_state_history_records_transitions(self):
        """Each transition appends to state_history."""
        p = ModificationProposal()
        p.transition(ModificationState.SANDBOXED, "gates passed")
        p.transition(ModificationState.TESTED, "tests green")
        assert len(p.state_history) == 2
        assert p.state_history[0][0] == "proposed"
        assert p.state_history[1][0] == "sandboxed"


# ---------------------------------------------------------------------------
# Risk-tier classifier tests
# ---------------------------------------------------------------------------


class TestRiskClassifier:
    def test_governance_target_is_critical(self):
        p = ModificationProposal(target_ref="governance.policy_bundle")
        assert classify_risk_tier(p) == RiskTier.CRITICAL

    def test_provenance_target_is_critical(self):
        p = ModificationProposal(target_ref="provenance.logger")
        assert classify_risk_tier(p) == RiskTier.CRITICAL

    def test_auth_change_type_is_critical(self):
        p = ModificationProposal(change_type="auth_key_rotation")
        assert classify_risk_tier(p) == RiskTier.CRITICAL

    def test_api_integration_is_high(self):
        p = ModificationProposal(target_ref="tool_integration.new_api")
        assert classify_risk_tier(p) == RiskTier.HIGH

    def test_permission_expansion_is_high(self):
        p = ModificationProposal(change_type="permission_grant")
        assert classify_risk_tier(p) == RiskTier.HIGH

    def test_workflow_change_is_medium(self):
        p = ModificationProposal(target_ref="workflow.reorder_steps")
        assert classify_risk_tier(p) == RiskTier.MEDIUM

    def test_prompt_tuning_is_low(self):
        p = ModificationProposal(target_ref="prompt.summarize", change_type="text_edit")
        assert classify_risk_tier(p) == RiskTier.LOW

    def test_highest_severity_wins(self):
        """Critical in target wins even with low change_type."""
        p = ModificationProposal(
            target_ref="governance.rules",
            change_type="text_edit",
        )
        assert classify_risk_tier(p) == RiskTier.CRITICAL

    # --- Security: Finding 2 — anti-evasion via default-to-MEDIUM ---

    def test_unknown_target_defaults_to_medium(self):
        """Unrecognized target_ref must default to MEDIUM, not LOW."""
        p = ModificationProposal(
            target_ref="xyzzy_obscure_ref",
            change_type="delta_apply",
        )
        assert classify_risk_tier(p) == RiskTier.MEDIUM

    def test_obfuscated_governance_name_not_low(self):
        """An agent obfuscating 'governance' as 'gov_rnance' should not get LOW."""
        p = ModificationProposal(
            target_ref="gov_rnance.plcy_bndl",
            change_type="config_edit",
        )
        # Should be MEDIUM (not LOW) since it doesn't match any known surface
        tier = classify_risk_tier(p)
        assert tier != RiskTier.LOW

    def test_empty_refs_default_to_medium(self):
        """Empty target_ref and change_type must not classify as LOW."""
        p = ModificationProposal(target_ref="", change_type="")
        assert classify_risk_tier(p) == RiskTier.MEDIUM


# ---------------------------------------------------------------------------
# Complexity weight tests
# ---------------------------------------------------------------------------


class TestComplexityWeight:
    def test_default_proposal_weight(self):
        p = ModificationProposal(risk_tier=RiskTier.LOW)
        w = compute_complexity_weight(p)
        # f=max(1,1)=1, t=0, p=0, i=0, s=0 -> 1*1 = 1.0
        assert w == 1.0

    def test_high_risk_includes_safety_penalty(self):
        p = ModificationProposal(risk_tier=RiskTier.HIGH)
        w = compute_complexity_weight(p)
        # f=1, t=0, p=0, i=0, s=1 -> 1 + 5 = 6.0
        assert w == 6.0

    def test_complex_change(self):
        p = ModificationProposal(
            risk_tier=RiskTier.MEDIUM,
            proposed_change={
                "files_touched": 3,
                "tool_interfaces_changed": 2,
                "privilege_delta": 1,
                "interaction_surface_delta": 1,
            },
        )
        w = compute_complexity_weight(p)
        # 1*3 + 2*2 + 4*1 + 3*1 + 5*0 = 3+4+4+3 = 14
        assert w == 14.0

    # --- Security: Finding 1 — negative/zero inputs clamped ---

    def test_negative_files_touched_clamped_to_one(self):
        """Agent cannot set files_touched to 0 or negative to reduce weight."""
        p = ModificationProposal(
            risk_tier=RiskTier.LOW,
            proposed_change={"files_touched": -5},
        )
        w = compute_complexity_weight(p)
        assert w >= 1.0

    def test_zero_files_touched_clamped_to_one(self):
        p = ModificationProposal(
            risk_tier=RiskTier.LOW,
            proposed_change={"files_touched": 0},
        )
        w = compute_complexity_weight(p)
        assert w >= 1.0

    def test_negative_tool_interfaces_clamped(self):
        p = ModificationProposal(
            risk_tier=RiskTier.LOW,
            proposed_change={"tool_interfaces_changed": -10},
        )
        w = compute_complexity_weight(p)
        assert w >= 1.0

    def test_all_negative_inputs_still_has_floor(self):
        """Even if all inputs are adversarially negative, floor is 1.0."""
        p = ModificationProposal(
            risk_tier=RiskTier.LOW,
            proposed_change={
                "files_touched": -100,
                "tool_interfaces_changed": -100,
                "privilege_delta": -100,
                "interaction_surface_delta": -100,
            },
        )
        w = compute_complexity_weight(p)
        assert w >= 1.0


# ---------------------------------------------------------------------------
# Gate 1: tau tests
# ---------------------------------------------------------------------------


class TestTauGate:
    def _baseline(self):
        return {
            "toxicity": 0.10,
            "quality_gap": 0.30,
        }

    def _stds(self):
        return {
            "toxicity": 0.05,
            "quality_gap": 0.10,
        }

    def _directions(self):
        return {
            "toxicity": -1,  # lower is better
            "quality_gap": 1,  # higher is better
        }

    def test_pass_when_candidate_improves(self):
        p = ModificationProposal(risk_tier=RiskTier.LOW)
        result = evaluate_tau_gate(
            p,
            self._baseline(),
            self._stds(),
            candidate_values={"toxicity": 0.08, "quality_gap": 0.35},
            candidate_cis={"toxicity": 0.01, "quality_gap": 0.02},
            metric_directions=self._directions(),
        )
        assert result.passed is True
        assert result.value > -0.10  # tau > tau_min[low]

    def test_fail_when_candidate_degrades(self):
        p = ModificationProposal(risk_tier=RiskTier.MEDIUM)
        result = evaluate_tau_gate(
            p,
            self._baseline(),
            self._stds(),
            candidate_values={"toxicity": 0.30, "quality_gap": 0.10},
            candidate_cis={"toxicity": 0.05, "quality_gap": 0.05},
            metric_directions=self._directions(),
        )
        assert result.passed is False

    def test_critical_always_denied(self):
        p = ModificationProposal(risk_tier=RiskTier.CRITICAL)
        result = evaluate_tau_gate(
            p,
            self._baseline(),
            self._stds(),
            candidate_values={"toxicity": 0.01, "quality_gap": 0.90},
            candidate_cis={"toxicity": 0.0, "quality_gap": 0.0},
            metric_directions=self._directions(),
        )
        assert result.passed is False
        assert "human approval" in result.details.get("reason", "")

    def test_missing_metric_fails_closed(self):
        p = ModificationProposal(risk_tier=RiskTier.LOW)
        result = evaluate_tau_gate(
            p,
            self._baseline(),
            self._stds(),
            candidate_values={"toxicity": 0.08},  # missing quality_gap
            candidate_cis={"toxicity": 0.01},
            metric_directions=self._directions(),
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Gate 2: K_max tests
# ---------------------------------------------------------------------------


class TestKMaxGate:
    def test_pass_under_budget(self):
        p = ModificationProposal(risk_tier=RiskTier.LOW, complexity_weight=5.0)
        result = evaluate_k_max_gate(p, current_budget_used=10.0)
        assert result.passed is True  # 10 + 5 = 15 <= 20

    def test_fail_over_budget(self):
        p = ModificationProposal(risk_tier=RiskTier.LOW, complexity_weight=15.0)
        result = evaluate_k_max_gate(p, current_budget_used=10.0)
        assert result.passed is False  # 10 + 15 = 25 > 20

    def test_exact_budget_passes(self):
        p = ModificationProposal(risk_tier=RiskTier.LOW, complexity_weight=10.0)
        result = evaluate_k_max_gate(p, current_budget_used=10.0)
        assert result.passed is True  # 10 + 10 = 20 <= 20

    # --- Security: Finding 8 — CRITICAL K_max now -1, always denied ---

    def test_critical_always_denied_via_k_max(self):
        """CRITICAL tier has K_max=-1, so even zero-weight proposals fail."""
        p = ModificationProposal(risk_tier=RiskTier.CRITICAL, complexity_weight=0.0)
        result = evaluate_k_max_gate(p, current_budget_used=0.0)
        assert result.passed is False  # 0 + 0 = 0 > -1 is False: 0 <= -1 is False


# ---------------------------------------------------------------------------
# SelfModificationLever tests
# ---------------------------------------------------------------------------


class TestSelfModificationLever:
    def _make_lever(self, **overrides):
        defaults = {"self_modification_enabled": True}
        defaults.update(overrides)
        config = GovernanceConfig(**defaults)
        return SelfModificationLever(config)

    def test_evaluate_proposal_approved(self):
        lever = self._make_lever()
        proposal = ModificationProposal(
            agent_id="a1",
            target_ref="prompt.greeting",
            change_type="text_edit",
        )
        approved, tau_r, k_r = lever.evaluate_proposal(
            proposal,
            baseline_means={"toxicity": 0.10},
            baseline_stds={"toxicity": 0.05},
            candidate_values={"toxicity": 0.08},
            candidate_cis={"toxicity": 0.01},
            metric_directions={"toxicity": -1},
        )
        assert approved is True
        assert proposal.state == ModificationState.SANDBOXED

    def test_evaluate_proposal_denied(self):
        lever = self._make_lever()
        proposal = ModificationProposal(
            agent_id="a1",
            target_ref="prompt.greeting",
            change_type="text_edit",
        )
        approved, _, _ = lever.evaluate_proposal(
            proposal,
            baseline_means={"toxicity": 0.10},
            baseline_stds={"toxicity": 0.05},
            candidate_values={"toxicity": 0.50},
            candidate_cis={"toxicity": 0.10},
            metric_directions={"toxicity": -1},
        )
        assert approved is False
        assert proposal.state == ModificationState.REJECTED

    def test_budget_accumulates(self):
        lever = self._make_lever()
        for i in range(3):
            p = ModificationProposal(
                agent_id="a1",
                target_ref=f"prompt.v{i}",
                change_type="text_edit",
            )
            lever.evaluate_proposal(
                p,
                baseline_means={"tox": 0.1},
                baseline_stds={"tox": 0.05},
                candidate_values={"tox": 0.08},
                candidate_cis={"tox": 0.01},
                metric_directions={"tox": -1},
            )
        budget = lever.get_agent_budget("a1")
        assert budget["used"] > 0

    def test_oscillation_detection(self):
        lever = self._make_lever()
        # Submit 3 modifications to same target
        for _ in range(3):
            p = ModificationProposal(
                agent_id="a1",
                target_ref="template.summarize",
                change_type="text_edit",
            )
            lever.evaluate_proposal(
                p,
                baseline_means={"tox": 0.1},
                baseline_stds={"tox": 0.05},
                candidate_values={"tox": 0.08},
                candidate_cis={"tox": 0.01},
                metric_directions={"tox": -1},
            )
        assert lever.detect_oscillation("a1") is True

    def test_no_oscillation_different_targets(self):
        lever = self._make_lever()
        for i in range(3):
            p = ModificationProposal(
                agent_id="a1",
                target_ref=f"template.task_{i}",
                change_type="text_edit",
            )
            lever.evaluate_proposal(
                p,
                baseline_means={"tox": 0.1},
                baseline_stds={"tox": 0.05},
                candidate_values={"tox": 0.08},
                candidate_cis={"tox": 0.01},
                metric_directions={"tox": -1},
            )
        assert lever.detect_oscillation("a1") is False

    def test_reset_budget(self):
        lever = self._make_lever()
        p = ModificationProposal(
            agent_id="a1",
            target_ref="prompt.v1",
            change_type="text_edit",
        )
        lever.evaluate_proposal(
            p,
            baseline_means={"tox": 0.1},
            baseline_stds={"tox": 0.05},
            candidate_values={"tox": 0.08},
            candidate_cis={"tox": 0.01},
            metric_directions={"tox": -1},
        )
        assert lever.get_agent_budget("a1")["used"] > 0
        lever.reset_agent_budget("a1")
        assert lever.get_agent_budget("a1")["used"] == 0.0

    # --- Security: Finding 1+2 end-to-end — adversarial proposal ---

    def test_adversarial_zero_weight_proposal_still_costs_budget(self):
        """An agent setting all proposed_change values to 0 still pays >= 1.0."""
        lever = self._make_lever()
        proposal = ModificationProposal(
            agent_id="a1",
            target_ref="prompt.greeting",
            change_type="text_edit",
            proposed_change={
                "files_touched": 0,
                "tool_interfaces_changed": 0,
                "privilege_delta": 0,
                "interaction_surface_delta": 0,
            },
        )
        lever.evaluate_proposal(
            proposal,
            baseline_means={"tox": 0.1},
            baseline_stds={"tox": 0.05},
            candidate_values={"tox": 0.08},
            candidate_cis={"tox": 0.01},
            metric_directions={"tox": -1},
        )
        assert lever.get_agent_budget("a1")["used"] >= 1.0

    def test_unknown_target_gets_medium_threshold(self):
        """A proposal with an unrecognized target gets MEDIUM risk, not LOW."""
        lever = self._make_lever()
        proposal = ModificationProposal(
            agent_id="a1",
            target_ref="xyzzy_totally_unknown",
            change_type="obfuscated_delta",
        )
        approved, tau_r, _ = lever.evaluate_proposal(
            proposal,
            baseline_means={"tox": 0.1},
            baseline_stds={"tox": 0.05},
            candidate_values={"tox": 0.10},  # no improvement = fails MEDIUM tau>=0
            candidate_cis={"tox": 0.01},
            metric_directions={"tox": -1},
        )
        # With MEDIUM tier, tau_min=0.0. Candidate tox=0.10 == baseline 0.10,
        # margin = (-1*(0.10-0.10)/0.05) - 0.01/0.05 = -0.2 < 0.0 => denied
        assert approved is False

    # --- Security: Finding 7 — get_agent_budget with tier ---

    def test_get_agent_budget_accepts_tier(self):
        lever = self._make_lever()
        budget_low = lever.get_agent_budget("a1", RiskTier.LOW)
        budget_high = lever.get_agent_budget("a1", RiskTier.HIGH)
        assert budget_low["k_max"] == 20.0
        assert budget_high["k_max"] == 50.0
