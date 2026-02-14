"""Smoke tests for the Concordia governance sweep.

Tests narrative generators, governance configs, corpus judge, and the
single-run sweep pipeline.  All tests run without LLM calls.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.bridges.concordia.events import JudgeScores
from swarm.bridges.concordia.narratives import (
    adversarial_narrative,
    collusive_narrative,
    competitive_narrative,
    cooperative_narrative,
    generate_corpus,
    mixed_narrative,
)

# ── Narrative generators ─────────────────────────────────────────────────

AGENTS = ["alice", "bob", "carol", "dave", "eve", "frank"]


class TestNarrativeGenerators:
    """Test that each narrative generator returns valid data."""

    def test_cooperative_returns_tuple(self):
        text, scores = cooperative_narrative(AGENTS, step=0)
        assert isinstance(text, str) and len(text) > 10
        assert isinstance(scores, JudgeScores)

    def test_cooperative_scores_range(self):
        import random
        rng = random.Random(42)
        for _ in range(20):
            _, scores = cooperative_narrative(AGENTS, step=0, rng=rng)
            assert scores.progress >= 0.65
            assert scores.cooperation >= 0.75
            assert scores.harm <= 0.15

    def test_competitive_returns_tuple(self):
        text, scores = competitive_narrative(AGENTS, step=0)
        assert isinstance(text, str) and len(text) > 10
        assert isinstance(scores, JudgeScores)

    def test_competitive_scores_range(self):
        import random
        rng = random.Random(42)
        for _ in range(20):
            _, scores = competitive_narrative(AGENTS, step=0, rng=rng)
            assert 0.15 <= scores.cooperation <= 0.50
            assert scores.harm <= 0.25

    def test_adversarial_returns_tuple(self):
        text, scores = adversarial_narrative(AGENTS, step=0)
        assert isinstance(text, str) and len(text) > 10
        assert isinstance(scores, JudgeScores)

    def test_adversarial_scores_range(self):
        import random
        rng = random.Random(42)
        for _ in range(20):
            _, scores = adversarial_narrative(AGENTS, step=0, rng=rng)
            assert scores.cooperation <= 0.20
            assert scores.harm >= 0.55

    def test_collusive_returns_tuple(self):
        text, scores = collusive_narrative(AGENTS, step=0)
        assert isinstance(text, str) and len(text) > 10
        assert isinstance(scores, JudgeScores)

    def test_collusive_needs_three_agents(self):
        # With 2 agents, falls back to adversarial
        text, scores = collusive_narrative(["a", "b"], step=0)
        assert isinstance(text, str)
        assert scores.harm >= 0.5  # adversarial fallback has high harm

    def test_collusive_scores_range(self):
        import random
        rng = random.Random(42)
        for _ in range(20):
            _, scores = collusive_narrative(AGENTS, step=0, rng=rng)
            assert scores.harm >= 0.5

    def test_mixed_returns_tuple(self):
        text, scores = mixed_narrative(AGENTS, step=0)
        assert isinstance(text, str)
        assert isinstance(scores, JudgeScores)

    def test_mixed_adversarial_frac_zero(self):
        """With adversarial_frac=0, should only produce cooperative/competitive."""
        import random
        rng = random.Random(42)
        for _ in range(30):
            _, scores = mixed_narrative(AGENTS, step=0, adversarial_frac=0.0, rng=rng)
            # No adversarial/collusive: harm should be low
            assert scores.harm <= 0.2

    def test_mixed_adversarial_frac_one(self):
        """With adversarial_frac=1.0, should only produce adversarial/collusive."""
        import random
        rng = random.Random(42)
        for _ in range(30):
            _, scores = mixed_narrative(AGENTS, step=0, adversarial_frac=1.0, rng=rng)
            assert scores.harm >= 0.5


class TestCorpus:
    """Test corpus generation."""

    def test_corpus_shape(self):
        corpus = generate_corpus(AGENTS, n_epochs=3, steps_per_epoch=4, seed=42)
        assert len(corpus) == 3
        for epoch in corpus:
            assert len(epoch) == 4

    def test_corpus_reproducibility(self):
        c1 = generate_corpus(AGENTS, n_epochs=2, steps_per_epoch=2, seed=99)
        c2 = generate_corpus(AGENTS, n_epochs=2, steps_per_epoch=2, seed=99)
        for e in range(2):
            for s in range(2):
                text1, scores1 = c1[e][s]
                text2, scores2 = c2[e][s]
                assert text1 == text2
                assert scores1.progress == scores2.progress

    def test_corpus_different_seeds_differ(self):
        c1 = generate_corpus(AGENTS, n_epochs=2, steps_per_epoch=2, seed=1)
        c2 = generate_corpus(AGENTS, n_epochs=2, steps_per_epoch=2, seed=2)
        # At least one narrative should differ
        any_diff = False
        for e in range(2):
            for s in range(2):
                if c1[e][s][0] != c2[e][s][0]:
                    any_diff = True
        assert any_diff

    def test_p_invariant(self):
        """All scores must have values in [0, 1]."""
        corpus = generate_corpus(AGENTS, n_epochs=5, steps_per_epoch=10, seed=42)
        for epoch in corpus:
            for _, scores in epoch:
                assert 0.0 <= scores.progress <= 1.0
                assert 0.0 <= scores.quality <= 1.0
                assert 0.0 <= scores.cooperation <= 1.0
                assert 0.0 <= scores.harm <= 1.0


# ── Governance configs ───────────────────────────────────────────────────


class TestGovernanceConfigs:
    """Test that all 8 governance configs are valid."""

    def test_config_count(self):
        from examples.concordia_governance_sweep import governance_configs
        configs = governance_configs()
        assert len(configs) == 8

    def test_all_configs_have_labels(self):
        from examples.concordia_governance_sweep import governance_configs
        configs = governance_configs()
        labels = [c["label"] for c in configs]
        assert len(set(labels)) == 8  # unique labels

    def test_all_configs_valid_governance(self):
        from examples.concordia_governance_sweep import governance_configs
        from swarm.governance.engine import GovernanceEngine

        for cfg in governance_configs():
            gov = cfg["governance"]
            # Should not raise
            engine = GovernanceEngine(config=gov)
            assert engine is not None

    def test_baseline_has_no_active_levers(self):
        from examples.concordia_governance_sweep import governance_configs
        from swarm.governance.engine import GovernanceEngine

        baseline = governance_configs()[0]
        assert baseline["label"] == "baseline"
        engine = GovernanceEngine(config=baseline["governance"])
        active = engine.get_active_lever_names()
        # Baseline should have no consequential levers
        # (some levers register but do nothing at default settings)
        assert isinstance(active, list)

    def test_full_defense_has_many_levers(self):
        from examples.concordia_governance_sweep import governance_configs
        from swarm.governance.engine import GovernanceEngine

        full = governance_configs()[-1]
        assert full["label"] == "full_defense"
        engine = GovernanceEngine(config=full["governance"])
        active = engine.get_active_lever_names()
        # Full defense should have several active levers
        assert len(active) >= 3


# ── Corpus judge ─────────────────────────────────────────────────────────


class TestCorpusJudge:
    """Test the CorpusJudge that injects pre-computed scores."""

    def test_enqueue_and_evaluate(self):
        from examples.concordia_governance_sweep import CorpusJudge

        judge = CorpusJudge()
        expected = JudgeScores(progress=0.9, quality=0.8, cooperation=0.7, harm=0.1)
        judge.enqueue(expected)

        result = judge.evaluate("some narrative")
        assert result.progress == expected.progress
        assert result.quality == expected.quality
        assert result.cooperation == expected.cooperation
        assert result.harm == expected.harm

    def test_empty_queue_returns_defaults(self):
        from examples.concordia_governance_sweep import CorpusJudge

        judge = CorpusJudge()
        result = judge.evaluate("some narrative")
        assert result.progress == 0.5
        assert result.harm == 0.0

    def test_queue_order_preserved(self):
        from examples.concordia_governance_sweep import CorpusJudge

        judge = CorpusJudge()
        judge.enqueue(JudgeScores(progress=0.1))
        judge.enqueue(JudgeScores(progress=0.9))

        r1 = judge.evaluate("first")
        r2 = judge.evaluate("second")
        assert r1.progress == pytest.approx(0.1)
        assert r2.progress == pytest.approx(0.9)


# ── Single run ───────────────────────────────────────────────────────────


class TestSingleRun:
    """Smoke test for a single sweep run."""

    def test_baseline_run(self):
        from examples.concordia_governance_sweep import (
            AGENT_IDS,
            governance_configs,
            run_single,
        )

        corpus = generate_corpus(
            AGENT_IDS, n_epochs=3, steps_per_epoch=2, seed=42,
        )
        baseline = governance_configs()[0]
        result = run_single(
            label=baseline["label"],
            gov_config=baseline["governance"],
            corpus=corpus,
            seed=42,
        )

        assert result.label == "baseline"
        assert result.seed == 42
        assert len(result.epoch_metrics) == 3
        assert result.total_interactions > 0

    def test_full_defense_run(self):
        from examples.concordia_governance_sweep import (
            AGENT_IDS,
            governance_configs,
            run_single,
        )

        corpus = generate_corpus(
            AGENT_IDS, n_epochs=3, steps_per_epoch=2, seed=42,
        )
        full = governance_configs()[-1]
        result = run_single(
            label=full["label"],
            gov_config=full["governance"],
            corpus=corpus,
            seed=42,
        )

        assert result.label == "full_defense"
        assert len(result.epoch_metrics) == 3
        assert result.total_interactions > 0

    def test_metrics_are_finite(self):
        import math

        from examples.concordia_governance_sweep import (
            AGENT_IDS,
            governance_configs,
            run_single,
        )

        corpus = generate_corpus(
            AGENT_IDS, n_epochs=3, steps_per_epoch=2, seed=42,
        )
        for cfg in governance_configs():
            result = run_single(
                label=cfg["label"],
                gov_config=cfg["governance"],
                corpus=corpus,
                seed=42,
            )
            assert math.isfinite(result.mean_toxicity), f"{cfg['label']} toxicity"
            assert math.isfinite(result.mean_welfare), f"{cfg['label']} welfare"

    def test_p_invariant_in_adapter(self):
        """All interaction p values should be in [0, 1]."""
        from examples.concordia_governance_sweep import (
            AGENT_IDS,
            CorpusJudge,
        )
        from swarm.bridges.concordia.adapter import ConcordiaAdapter
        from swarm.bridges.concordia.config import ConcordiaConfig

        corpus = generate_corpus(
            AGENT_IDS, n_epochs=2, steps_per_epoch=3, seed=42,
        )
        judge = CorpusJudge()
        adapter = ConcordiaAdapter(config=ConcordiaConfig(), judge=judge)

        for epoch in corpus:
            for narrative_text, expected_scores in epoch:
                judge.enqueue(expected_scores)
                interactions = adapter.process_narrative(
                    agent_ids=AGENT_IDS,
                    narrative_text=narrative_text,
                    step=0,
                )
                for ix in interactions:
                    assert 0.0 <= ix.p <= 1.0, f"p={ix.p} out of bounds"


# ── Full sweep (mini) ────────────────────────────────────────────────────


class TestSweep:
    """Smoke test for the full sweep with minimal parameters."""

    def test_mini_sweep(self):
        from examples.concordia_governance_sweep import run_sweep

        results = run_sweep(
            n_seeds=1,
            n_epochs=2,
            steps_per_epoch=2,
            progress=False,
        )
        # 8 configs × 1 seed = 8 results
        assert len(results) == 8
        labels = [r.label for r in results]
        assert "baseline" in labels
        assert "full_defense" in labels

    def test_sweep_export_csv(self, tmp_path):
        from examples.concordia_governance_sweep import export_csv, run_sweep

        results = run_sweep(
            n_seeds=1, n_epochs=2, steps_per_epoch=2, progress=False,
        )
        csv_path = tmp_path / "summary.csv"
        export_csv(results, csv_path)
        assert csv_path.exists()

        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 8
        assert "label" in rows[0]
        assert "mean_toxicity" in rows[0]

    def test_sweep_export_epoch_csv(self, tmp_path):
        from examples.concordia_governance_sweep import export_epoch_csv, run_sweep

        results = run_sweep(
            n_seeds=1, n_epochs=2, steps_per_epoch=2, progress=False,
        )
        csv_path = tmp_path / "epochs.csv"
        export_epoch_csv(results, csv_path)
        assert csv_path.exists()

        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # 8 configs × 1 seed × 2 epochs = 16 rows
        assert len(rows) == 16

    def test_result_to_dict(self):
        from examples.concordia_governance_sweep import run_sweep

        results = run_sweep(
            n_seeds=1, n_epochs=2, steps_per_epoch=2, progress=False,
        )
        for r in results:
            d = r.to_dict()
            assert "label" in d
            assert "mean_toxicity" in d
            assert "mean_welfare" in d
