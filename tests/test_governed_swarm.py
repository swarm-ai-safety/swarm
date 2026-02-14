"""Tests for the governed_swarm LangGraph Swarm bridge.

Tests are split into two groups:
  1. Pure-Python governance and provenance logic (no langgraph deps)
  2. LangGraph integration tests (skipped if langgraph not installed)
"""

from __future__ import annotations

import time

import pytest

from swarm.bridges.langgraph_swarm.governed_swarm import (
    CompositePolicy,
    CycleDetectionPolicy,
    GovernanceDecision,
    GovernancePolicy,
    GovernanceResult,
    InformationBoundaryPolicy,
    ProvenanceLogger,
    ProvenanceRecord,
    RateLimitPolicy,
    _cross_boundary_filter,
    analyze_swarm_run,
    default_governance_stack,
)

# =========================================================================
# ProvenanceRecord tests
# =========================================================================


class TestProvenanceRecord:
    def test_defaults(self):
        record = ProvenanceRecord()
        assert record.source_agent == ""
        assert record.target_agent == ""
        assert record.governance_decision == "pending"
        assert record.chain_depth == 0
        assert record.parent_record_id is None
        assert len(record.record_id) == 12

    def test_to_dict_roundtrip(self):
        record = ProvenanceRecord(
            source_agent="alice",
            target_agent="bob",
            handoff_reason="needs help",
            risk_score_at_handoff=0.5,
        )
        d = record.to_dict()
        assert d["source_agent"] == "alice"
        assert d["target_agent"] == "bob"
        assert d["risk_score_at_handoff"] == 0.5
        assert isinstance(d["timestamp"], float)

    def test_to_json(self):
        record = ProvenanceRecord(source_agent="x", target_agent="y")
        j = record.to_json()
        assert '"source_agent": "x"' in j
        assert '"target_agent": "y"' in j


# =========================================================================
# ProvenanceLogger tests
# =========================================================================


class TestProvenanceLogger:
    def test_empty_logger(self):
        logger = ProvenanceLogger()
        assert logger.records == []
        assert logger.get_chain() == []
        assert logger.to_audit_log() == []

    def test_log_appends(self):
        logger = ProvenanceLogger()
        r1 = ProvenanceRecord(source_agent="a", target_agent="b")
        r2 = ProvenanceRecord(source_agent="b", target_agent="c")
        logger.log(r1)
        logger.log(r2)
        assert len(logger.records) == 2

    def test_auto_parent_linking(self):
        logger = ProvenanceLogger()
        r1 = ProvenanceRecord(source_agent="a", target_agent="b")
        r2 = ProvenanceRecord(source_agent="b", target_agent="c")
        logger.log(r1)
        logger.log(r2)

        assert r1.parent_record_id is None  # First record has no parent
        assert r1.chain_depth == 0
        assert r2.parent_record_id == r1.record_id
        assert r2.chain_depth == 1

    def test_explicit_parent_not_overridden(self):
        logger = ProvenanceLogger()
        r1 = ProvenanceRecord(source_agent="a", target_agent="b")
        r2 = ProvenanceRecord(
            source_agent="b",
            target_agent="c",
            parent_record_id="custom-parent",
        )
        logger.log(r1)
        logger.log(r2)
        # Explicit parent is not overridden
        assert r2.parent_record_id == "custom-parent"

    def test_get_chain_filtered(self):
        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        logger.log(ProvenanceRecord(source_agent="b", target_agent="c"))
        logger.log(ProvenanceRecord(source_agent="c", target_agent="d"))

        chain_b = logger.get_chain("b")
        assert len(chain_b) == 2  # b is source in r1 and target in r2

        chain_d = logger.get_chain("d")
        assert len(chain_d) == 1

    def test_get_handoff_count(self):
        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        logger.log(ProvenanceRecord(source_agent="b", target_agent="a"))
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))

        assert logger.get_handoff_count("a", "b") == 2
        assert logger.get_handoff_count("b", "a") == 1
        assert logger.get_handoff_count("a", "c") == 0

    def test_detect_cycles_no_cycles(self):
        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        logger.log(ProvenanceRecord(source_agent="b", target_agent="c"))
        assert logger.detect_cycles(window=10) == []

    def test_detect_cycles_ping_pong(self):
        logger = ProvenanceLogger()
        # Create a ping-pong pattern: a->b three times
        for _ in range(4):
            logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        cycles = logger.detect_cycles(window=10)
        assert ("a", "b") in cycles

    def test_listener_callback(self):
        logger = ProvenanceLogger()
        received: list[ProvenanceRecord] = []
        logger.on_record(lambda r: received.append(r))

        r1 = ProvenanceRecord(source_agent="a", target_agent="b")
        logger.log(r1)
        assert len(received) == 1
        assert received[0] is r1

    def test_to_audit_log(self):
        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        audit = logger.to_audit_log()
        assert len(audit) == 1
        assert isinstance(audit[0], dict)
        assert audit[0]["source_agent"] == "a"


# =========================================================================
# GovernancePolicy tests
# =========================================================================


class TestGovernancePolicy:
    def test_default_policy_approves(self):
        policy = GovernancePolicy()
        result = policy.evaluate("a", "b", "test", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.APPROVE

    def test_governance_decision_values(self):
        assert GovernanceDecision.APPROVE.value == "approved"
        assert GovernanceDecision.DENY.value == "denied"
        assert GovernanceDecision.MODIFY.value == "modified"
        assert GovernanceDecision.ESCALATE.value == "escalated"


class TestCycleDetectionPolicy:
    def test_no_cycles_approves(self):
        policy = CycleDetectionPolicy(max_cycles=3, window=10)
        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))
        result = policy.evaluate("b", "c", "task", {}, logger)
        assert result.decision == GovernanceDecision.APPROVE
        assert result.risk_score == 0.0

    def test_cycle_detected_denies(self):
        policy = CycleDetectionPolicy(max_cycles=3, window=10)
        logger = ProvenanceLogger()
        # Create enough repetitions to trigger cycle detection
        for _ in range(4):
            logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))

        result = policy.evaluate("a", "b", "task", {}, logger)
        assert result.decision == GovernanceDecision.DENY
        assert result.risk_score == 0.8
        assert "Cycle detected" in result.reason

    def test_different_pair_not_affected(self):
        policy = CycleDetectionPolicy(max_cycles=3, window=10)
        logger = ProvenanceLogger()
        for _ in range(4):
            logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))

        # Different pair should not be blocked
        result = policy.evaluate("c", "d", "task", {}, logger)
        assert result.decision == GovernanceDecision.APPROVE


class TestRateLimitPolicy:
    def test_within_limits_approves(self):
        policy = RateLimitPolicy(max_handoffs=5, window_seconds=300.0)
        logger = ProvenanceLogger()
        logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))

        result = policy.evaluate("a", "b", "task", {}, logger)
        assert result.decision == GovernanceDecision.APPROVE
        assert result.risk_score == pytest.approx(1 / 5)

    def test_exceeds_rate_limit_escalates(self):
        policy = RateLimitPolicy(max_handoffs=3, window_seconds=300.0)
        logger = ProvenanceLogger()
        for _ in range(3):
            logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))

        result = policy.evaluate("a", "b", "task", {}, logger)
        assert result.decision == GovernanceDecision.ESCALATE
        assert result.risk_score == 0.9
        assert "Rate limit exceeded" in result.reason

    def test_old_records_outside_window(self):
        policy = RateLimitPolicy(max_handoffs=3, window_seconds=10.0)
        logger = ProvenanceLogger()
        # Create records with old timestamps (outside window)
        for _ in range(5):
            r = ProvenanceRecord(source_agent="a", target_agent="b")
            r.timestamp = time.time() - 100.0  # 100s ago
            logger.records.append(r)

        result = policy.evaluate("a", "b", "task", {}, logger)
        assert result.decision == GovernanceDecision.APPROVE


class TestInformationBoundaryPolicy:
    def test_same_group_approves(self):
        policy = InformationBoundaryPolicy(
            trust_groups={"a": "group1", "b": "group1"}
        )
        result = policy.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.APPROVE
        assert result.risk_score == 0.0

    def test_cross_boundary_modifies(self):
        policy = InformationBoundaryPolicy(
            trust_groups={"a": "finance", "b": "marketing"}
        )
        result = policy.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.MODIFY
        assert result.risk_score == 0.4
        assert result.modified_context is not None
        assert result.modified_context["context_filter"] == "cross_boundary_summary"

    def test_unknown_agents_default_group(self):
        policy = InformationBoundaryPolicy(trust_groups={"a": "finance"})
        # b not in trust_groups -> defaults to "default"
        # a is "finance", b is "default" -> cross-boundary
        result = policy.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.MODIFY

    def test_both_unknown_same_default(self):
        policy = InformationBoundaryPolicy(trust_groups={})
        result = policy.evaluate("x", "y", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.APPROVE


class TestCompositePolicy:
    def test_all_approve(self):
        policies = [GovernancePolicy(), GovernancePolicy()]
        composite = CompositePolicy(policies)
        result = composite.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.APPROVE

    def test_deny_short_circuits(self):
        class DenyPolicy(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.DENY,
                    reason="blocked",
                    risk_score=0.7,
                )

        class TrackingPolicy(GovernancePolicy):
            called = False

            def evaluate(self, *args, **kwargs):
                self.called = True
                return GovernanceResult(decision=GovernanceDecision.APPROVE)

        tracker = TrackingPolicy()
        composite = CompositePolicy([DenyPolicy(), tracker])
        result = composite.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.DENY
        assert not tracker.called

    def test_escalate_short_circuits(self):
        class EscalatePolicy(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.ESCALATE,
                    reason="needs human",
                    risk_score=0.9,
                )

        composite = CompositePolicy([EscalatePolicy(), GovernancePolicy()])
        result = composite.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.ESCALATE

    def test_modify_accumulated(self):
        class ModifyPolicy(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.MODIFY,
                    reason="filtered",
                    modified_context={"context_filter": "summary"},
                    risk_score=0.3,
                )

        composite = CompositePolicy([ModifyPolicy(), GovernancePolicy()])
        result = composite.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.MODIFY
        assert result.modified_context is not None
        assert result.modified_context["context_filter"] == "summary"

    def test_max_risk_propagated(self):
        class LowRisk(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.APPROVE, risk_score=0.1
                )

        class HighRisk(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.APPROVE, risk_score=0.6
                )

        composite = CompositePolicy([LowRisk(), HighRisk()])
        result = composite.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.risk_score == 0.6

    def test_deny_gets_max_risk_from_prior_policies(self):
        class HighRiskApprove(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.APPROVE, risk_score=0.7
                )

        class LowRiskDeny(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.DENY,
                    reason="blocked",
                    risk_score=0.2,
                )

        composite = CompositePolicy([HighRiskApprove(), LowRiskDeny()])
        result = composite.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.DENY
        # max_risk should reflect the higher value from the approve policy
        assert result.risk_score == 0.7

    def test_modify_redirect_last_wins(self):
        class Redirect1(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.MODIFY,
                    modified_target="agent_x",
                    risk_score=0.2,
                )

        class Redirect2(GovernancePolicy):
            def evaluate(self, *args, **kwargs):
                return GovernanceResult(
                    decision=GovernanceDecision.MODIFY,
                    modified_target="agent_y",
                    risk_score=0.3,
                )

        composite = CompositePolicy([Redirect1(), Redirect2()])
        result = composite.evaluate("a", "b", "task", {}, ProvenanceLogger())
        assert result.decision == GovernanceDecision.MODIFY
        assert result.modified_context is not None
        # Last redirect wins
        assert result.modified_context["redirected_target"] == "agent_y"


# =========================================================================
# Cross-boundary filter tests
# =========================================================================


class TestCrossBoundaryFilter:
    def test_short_list_unchanged(self):
        messages = ["m1", "m2", "m3"]
        assert _cross_boundary_filter(messages) == messages

    def test_long_list_truncated(self):
        messages = ["m1", "m2", "m3", "m4", "m5"]
        result = _cross_boundary_filter(messages)
        assert result == ["m3", "m4", "m5"]

    def test_empty_list(self):
        assert _cross_boundary_filter([]) == []


# =========================================================================
# analyze_swarm_run tests
# =========================================================================


class TestAnalyzeSwarmRun:
    def test_empty_provenance(self):
        logger = ProvenanceLogger()
        result = analyze_swarm_run(logger)
        assert result["total_handoffs"] == 0
        assert result["risk_level"] == "low"

    def test_all_approved_low_risk(self):
        logger = ProvenanceLogger()
        # Use distinct agent pairs to avoid triggering cycle detection
        pairs = [("a", "b"), ("b", "c"), ("c", "d")]
        for src, tgt in pairs:
            r = ProvenanceRecord(
                source_agent=src,
                target_agent=tgt,
                governance_decision="approved",
                risk_score_at_handoff=0.1,
            )
            logger.log(r)

        result = analyze_swarm_run(logger)
        assert result["total_handoffs"] == 3
        assert result["approved_handoffs"] == 3
        assert result["denied_handoffs"] == 0
        assert result["risk_level"] == "low"

    def test_high_denial_rate_raises_risk(self):
        logger = ProvenanceLogger()
        # 4 denied out of 5 = 80% denial rate
        for _ in range(4):
            r = ProvenanceRecord(
                source_agent="a",
                target_agent="b",
                governance_decision="denied",
                risk_score_at_handoff=0.5,
            )
            logger.log(r)
        r = ProvenanceRecord(
            source_agent="a",
            target_agent="c",
            governance_decision="approved",
            risk_score_at_handoff=0.1,
        )
        logger.log(r)

        result = analyze_swarm_run(logger)
        assert result["denied_handoffs"] == 4
        assert result["denial_rate"] == 0.8
        assert result["risk_level"] in ("high", "critical")

    def test_escalations_trigger_critical(self):
        logger = ProvenanceLogger()
        for _ in range(3):
            r = ProvenanceRecord(
                source_agent="a",
                target_agent="b",
                governance_decision="escalated",
                risk_score_at_handoff=0.9,
            )
            logger.log(r)

        result = analyze_swarm_run(logger)
        assert result["escalated_handoffs"] == 3
        assert result["risk_level"] == "critical"

    def test_cycles_trigger_medium(self):
        logger = ProvenanceLogger()
        # Create cycle pattern: a->b 4 times (triggers cycle detection)
        for _ in range(4):
            r = ProvenanceRecord(
                source_agent="a",
                target_agent="b",
                governance_decision="approved",
                risk_score_at_handoff=0.3,
            )
            logger.log(r)

        result = analyze_swarm_run(logger)
        assert len(result["cycle_pairs"]) > 0
        assert result["risk_level"] == "medium"

    def test_agent_risk_scores_averaged(self):
        logger = ProvenanceLogger()
        logger.log(
            ProvenanceRecord(
                source_agent="a",
                target_agent="b",
                governance_decision="approved",
                risk_score_at_handoff=0.2,
            )
        )
        logger.log(
            ProvenanceRecord(
                source_agent="a",
                target_agent="b",
                governance_decision="approved",
                risk_score_at_handoff=0.4,
            )
        )

        result = analyze_swarm_run(logger)
        # Agent "a" appears as source in both records -> avg = (0.2+0.4)/2 = 0.3
        assert result["agent_risk_scores"]["a"] == pytest.approx(0.3)

    def test_max_chain_depth(self):
        logger = ProvenanceLogger()
        # Each logged record gets auto-incrementing chain depth
        for _i in range(5):
            logger.log(
                ProvenanceRecord(
                    source_agent="a",
                    target_agent="b",
                    governance_decision="approved",
                    risk_score_at_handoff=0.1,
                )
            )

        result = analyze_swarm_run(logger)
        assert result["max_chain_depth"] == 4  # 0-indexed, 5 records -> depth 4


# =========================================================================
# default_governance_stack tests
# =========================================================================


class TestDefaultGovernanceStack:
    def test_returns_policy_and_logger(self):
        policy, logger = default_governance_stack()
        assert isinstance(policy, CompositePolicy)
        assert isinstance(logger, ProvenanceLogger)
        assert len(policy.policies) == 2  # CycleDetection + RateLimit

    def test_with_trust_groups(self):
        policy, logger = default_governance_stack(
            trust_groups={"a": "finance", "b": "marketing"}
        )
        assert len(policy.policies) == 3  # + InformationBoundary

    def test_custom_parameters(self):
        policy, _ = default_governance_stack(
            max_cycles=5, max_handoffs=50, rate_window_seconds=600.0
        )
        cycle_policy = policy.policies[0]
        rate_policy = policy.policies[1]
        assert isinstance(cycle_policy, CycleDetectionPolicy)
        assert isinstance(rate_policy, RateLimitPolicy)
        assert cycle_policy.max_cycles == 5
        assert rate_policy.max_handoffs == 50
        assert rate_policy.window_seconds == 600.0


# =========================================================================
# Integration: Composite policy with real provenance
# =========================================================================


class TestGovernanceIntegration:
    """Test realistic governance scenarios end-to-end."""

    def test_cycle_then_rate_limit_composite(self):
        """Cycle detection fires before rate limit in composite."""
        policy = CompositePolicy([
            CycleDetectionPolicy(max_cycles=2, window=10),
            RateLimitPolicy(max_handoffs=20, window_seconds=300.0),
        ])
        logger = ProvenanceLogger()

        # Create enough cycles to trigger cycle detection
        for _ in range(4):
            logger.log(ProvenanceRecord(source_agent="a", target_agent="b"))

        result = policy.evaluate("a", "b", "task", {}, logger)
        assert result.decision == GovernanceDecision.DENY
        assert "Cycle detected" in result.reason

    def test_info_boundary_with_rate_limit(self):
        """Cross-boundary handoff is modified but not denied."""
        policy = CompositePolicy([
            RateLimitPolicy(max_handoffs=20, window_seconds=300.0),
            InformationBoundaryPolicy(
                trust_groups={"finance_agent": "finance", "marketing_agent": "marketing"}
            ),
        ])
        logger = ProvenanceLogger()

        result = policy.evaluate(
            "finance_agent", "marketing_agent", "task", {}, logger
        )
        assert result.decision == GovernanceDecision.MODIFY
        assert result.modified_context is not None
        assert "cross_boundary_summary" in str(result.modified_context)

    def test_full_provenance_chain_analysis(self):
        """Build a realistic chain and analyze it."""
        logger = ProvenanceLogger()

        # Simulate a multi-agent workflow
        logger.log(
            ProvenanceRecord(
                source_agent="researcher",
                target_agent="writer",
                governance_decision="approved",
                risk_score_at_handoff=0.0,
            )
        )
        logger.log(
            ProvenanceRecord(
                source_agent="writer",
                target_agent="reviewer",
                governance_decision="modified",
                risk_score_at_handoff=0.4,
            )
        )
        logger.log(
            ProvenanceRecord(
                source_agent="reviewer",
                target_agent="writer",
                governance_decision="approved",
                risk_score_at_handoff=0.1,
            )
        )

        analysis = analyze_swarm_run(logger)
        assert analysis["total_handoffs"] == 3
        assert analysis["approved_handoffs"] == 2
        assert analysis["modified_handoffs"] == 1
        assert analysis["risk_level"] == "low"
        assert analysis["max_chain_depth"] == 2
