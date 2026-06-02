"""Tests for swarm/judges/ — arm B scaffolding.

Heaviest test category is orthogonality (the load-bearing property): the
judge must NEVER see ProxyComputer observables.
"""

from __future__ import annotations

import pytest

from swarm.judges import (
    FORBIDDEN_FIELDS,
    RUBRIC_VERSION,
    JudgeView,
    LLMJudge,
    MockJudge,
    assert_view_is_orthogonal,
    bin_counts,
    load_rubric,
    make_view,
    stratified_sample,
)
from tests.fixtures.interactions import (
    generate_benign_batch,
    generate_obfuscation_scenario,
    generate_toxic_batch,
)


class TestRubric:
    def test_rubric_loads(self) -> None:
        text = load_rubric()
        assert "FROZEN" in text
        assert RUBRIC_VERSION in text

    def test_rubric_lists_all_forbidden_fields(self) -> None:
        text = load_rubric()
        # All proxy observables should be explicitly listed
        for field in [
            "task_progress_delta",
            "rework_count",
            "verifier_rejections",
            "tool_misuse_flags",
            "counterparty_engagement_delta",
        ]:
            assert field in text, f"rubric does not mention forbidden field {field}"


class TestOrthogonality:
    """Load-bearing tests. If any of these fail, the anchor is compromised."""

    def test_view_strips_top_level_forbidden_fields(self) -> None:
        # Build an interaction with a juicy task_progress_delta and confirm
        # it does not appear anywhere in the judge payload.
        batch = generate_toxic_batch(count=5, seed=0)
        for interaction in batch:
            view = make_view(interaction)
            payload = view.to_judge_payload()
            for forbidden in FORBIDDEN_FIELDS:
                assert forbidden not in payload, (
                    f"forbidden field {forbidden} leaked to top level"
                )
            assert_view_is_orthogonal(payload)

    def test_view_strips_forbidden_keys_from_metadata(self) -> None:
        # Even if a malicious caller stuffs a forbidden field into metadata,
        # make_view must scrub it.
        class FakeInteraction:
            interaction_id = "abc"
            interaction_type = "REPLY"
            accepted = True
            initiator = "a"
            counterparty = "b"
            metadata = {"v_hat": 0.99, "p": 0.99, "agent_type": "honest"}

        view = make_view(FakeInteraction())
        assert "v_hat" not in view.metadata
        assert "p" not in view.metadata
        assert view.metadata["agent_type"] == "honest"
        assert_view_is_orthogonal(view.to_judge_payload())

    def test_assert_raises_on_forbidden_top_level(self) -> None:
        with pytest.raises(AssertionError):
            assert_view_is_orthogonal(
                {"interaction_id": "x", "task_progress_delta": 0.5}
            )

    def test_assert_raises_on_forbidden_in_metadata(self) -> None:
        with pytest.raises(AssertionError):
            assert_view_is_orthogonal({"metadata": {"rework_count": 2}})

    def test_judge_score_only_sees_view(self) -> None:
        # MockJudge should never crash with KeyError or similar even on a
        # toxic-fixture batch with rich observables.
        batch = generate_toxic_batch(count=10, seed=0)
        judge = MockJudge()
        for interaction in batch:
            view = make_view(interaction)
            verdict = judge.score(view)
            assert 0.0 <= verdict.score <= 1.0
            assert verdict.judge_name == "mock"
            assert verdict.rubric_version == RUBRIC_VERSION


class TestMockJudgeRubricFidelity:
    """MockJudge must implement the rubric's signal-interpretation rules."""

    def test_empty_metadata_gives_midscore(self) -> None:
        view = JudgeView(
            interaction_id="x",
            interaction_type="REPLY",
            accepted=True,
            initiator_label="a",
            counterparty_label="b",
            metadata={},
        )
        assert MockJudge().score(view).score == 0.5

    def test_honest_scores_high(self) -> None:
        view = JudgeView(
            interaction_id="x",
            interaction_type="COLLABORATION",
            accepted=True,
            initiator_label="honest_1",
            counterparty_label="honest_2",
            metadata={"agent_type": "honest"},
        )
        assert MockJudge().score(view).score >= 0.7

    def test_policy_obfuscation_scores_low_despite_innocuous_rationalization(self) -> None:
        # The dangerous case: high p (proxy fooled), but the agent_type
        # label tells us the truth. Judge must score LOW.
        view = JudgeView(
            interaction_id="x",
            interaction_type="TRADE",
            accepted=True,
            initiator_label="policy_obf_1",
            counterparty_label="honest_5",
            metadata={
                "agent_type": "policy_obfuscation",
                "rationalization": "Optimized for efficiency.",
                "rationalization_strength": 0.7,
            },
        )
        assert MockJudge().score(view).score <= 0.3

    def test_self_optimizer_with_deep_optimization_scores_lower(self) -> None:
        shallow = JudgeView(
            interaction_id="a", interaction_type="TRADE", accepted=True,
            initiator_label="opt_1", counterparty_label="h_1",
            metadata={"agent_type": "self_optimizer", "optimization_depth": 0},
        )
        deep = JudgeView(
            interaction_id="b", interaction_type="TRADE", accepted=True,
            initiator_label="opt_1", counterparty_label="h_1",
            metadata={"agent_type": "self_optimizer", "optimization_depth": 15},
        )
        assert MockJudge().score(deep).score < MockJudge().score(shallow).score

    def test_blatant_scores_low(self) -> None:
        view = JudgeView(
            interaction_id="x", interaction_type="TRADE", accepted=True,
            initiator_label="blatant_1", counterparty_label="honest_5",
            metadata={"agent_type": "blatant"},
        )
        assert MockJudge().score(view).score <= 0.2

    def test_deterministic(self) -> None:
        view = JudgeView(
            interaction_id="x", interaction_type="TRADE", accepted=True,
            initiator_label="a", counterparty_label="b",
            metadata={"agent_type": "honest"},
        )
        judge = MockJudge()
        a = judge.score(view)
        b = judge.score(view)
        assert a.score == b.score
        assert a.rationale == b.rationale


class TestLLMJudge:
    def test_score_not_implemented(self) -> None:
        # Until the LLM call path is wired in, calling score must raise
        # explicitly (not silently produce zeros).
        judge = LLMJudge(name="claude", config=None)
        view = JudgeView(
            interaction_id="x", interaction_type="REPLY", accepted=True,
            initiator_label="a", counterparty_label="b", metadata={},
        )
        with pytest.raises(NotImplementedError):
            judge.score(view)


class TestSampler:
    def test_stratification_respects_per_bin_cap(self) -> None:
        # 100 benign (high p) + 100 toxic (low p)
        benign = generate_benign_batch(count=100, acceptance_rate=1.0, seed=1)
        toxic = generate_toxic_batch(count=100, acceptance_rate=1.0, seed=2)
        sampled = stratified_sample(benign + toxic, per_bin=10, seed=42)
        # Bins are [0, .2, .4, .6, .8, 1.0]. We should get <=10 per
        # populated bin and the sample size should be modest.
        assert len(sampled) <= 50
        counts = bin_counts(sampled)
        for c in counts:
            assert c <= 10

    def test_only_accepted_returned(self) -> None:
        # Mix with low acceptance rate; sampler should filter to accepted only.
        batch = generate_toxic_batch(count=100, acceptance_rate=0.3, seed=3)
        sampled = stratified_sample(batch, per_bin=50, seed=0)
        for s in sampled:
            assert s.accepted is True

    def test_reproducible_with_seed(self) -> None:
        batch = generate_obfuscation_scenario(n_epochs=2, seed=5)
        flat = [i for epoch in batch for i in epoch]
        a = stratified_sample(flat, per_bin=5, seed=99)
        b = stratified_sample(flat, per_bin=5, seed=99)
        assert [i.interaction_id for i in a] == [i.interaction_id for i in b]
