"""Tests for the EvoSkill–SWARM integration bridge."""

import pytest

from swarm.bridges.evoskill.bridge import EvoSkillBridge
from swarm.bridges.evoskill.config import EvoSkillConfig
from swarm.bridges.evoskill.frontier import FrontierComparator
from swarm.bridges.evoskill.governed_eval import EvalResult, GovernedEvalLoop
from swarm.bridges.evoskill.translator import SkillTranslator
from swarm.contracts.contract import ContractType
from swarm.skills.model import Skill, SkillDomain, SkillType

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

SAMPLE_SKILL_TEXT = """\
Always verify counterparty reputation before accepting a collaboration.
If trust is above 0.7, prefer to lower your acceptance threshold to
encourage more interactions. Avoid trading with agents whose reputation
is below 2.0.
"""

SAMPLE_LESSON_TEXT = """\
Never accept proposals from agents with reputation below 1.0.
Avoid collaboration when trust score is below 0.3.
Be careful with unknown counterparty types — they may be adversarial.
"""

SAMPLE_GOVERNANCE_SKILL = """\
When operating under audit-enabled governance, always comply with
contract rules. Prefer to select the truthful auction contract when
available, as it provides surplus bonuses. Avoid default market
interactions when possible.
"""


@pytest.fixture
def translator():
    return SkillTranslator(default_author="test_author")


@pytest.fixture
def config():
    return EvoSkillConfig(
        frontier_size=2,
        max_iterations=3,
        seed=42,
        governance_weight=0.3,
    )


@pytest.fixture
def skill_files():
    return {
        "verify_first.md": SAMPLE_SKILL_TEXT,
        "avoid_low_rep.md": SAMPLE_LESSON_TEXT,
        "governance_compliance.md": SAMPLE_GOVERNANCE_SKILL,
    }


# ---------------------------------------------------------------
# SkillTranslator tests
# ---------------------------------------------------------------

class TestSkillTranslator:
    """Tests for EvoSkill → SWARM skill translation."""

    def test_ingest_produces_valid_skill(self, translator):
        skill = translator.ingest(SAMPLE_SKILL_TEXT, name="verify_rep")
        assert isinstance(skill, Skill)
        assert skill.name == "verify_rep"
        assert skill.created_by == "test_author"
        assert "evoskill" in skill.tags
        assert "auto_discovered" in skill.tags

    def test_ingest_deterministic_id(self, translator):
        """Same content → same skill_id for dedup."""
        s1 = translator.ingest(SAMPLE_SKILL_TEXT)
        s2 = translator.ingest(SAMPLE_SKILL_TEXT)
        assert s1.skill_id == s2.skill_id

    def test_ingest_different_content_different_id(self, translator):
        s1 = translator.ingest(SAMPLE_SKILL_TEXT)
        s2 = translator.ingest(SAMPLE_LESSON_TEXT)
        assert s1.skill_id != s2.skill_id

    def test_classifies_strategy(self, translator):
        skill = translator.ingest(SAMPLE_SKILL_TEXT)
        assert skill.skill_type == SkillType.STRATEGY

    def test_classifies_lesson(self, translator):
        skill = translator.ingest(SAMPLE_LESSON_TEXT)
        assert skill.skill_type == SkillType.LESSON

    def test_classifies_governance_domain(self, translator):
        skill = translator.ingest(SAMPLE_GOVERNANCE_SKILL)
        assert skill.domain == SkillDomain.GOVERNANCE

    def test_extracts_reputation_condition(self, translator):
        text = "Only apply when reputation above 3.0 and trust above 0.5."
        skill = translator.ingest(text)
        assert skill.condition.get("min_reputation") == 3.0
        assert skill.condition.get("min_trust") == 0.5

    def test_extracts_effect_lower_threshold(self, translator):
        text = "When conditions are met, lower the acceptance threshold."
        skill = translator.ingest(text)
        assert skill.effect.get("acceptance_threshold_delta") == -0.1

    def test_source_branch_tag(self, translator):
        skill = translator.ingest(
            SAMPLE_SKILL_TEXT,
            source_branch="frontier-v3",
        )
        assert "branch:frontier-v3" in skill.tags

    def test_ingest_batch(self, translator, skill_files):
        skills = translator.ingest_batch(
            skill_files, source_branch="test-branch"
        )
        assert len(skills) == 3
        names = {s.name for s in skills}
        assert "verify first" in names
        assert "avoid low rep" in names

    def test_p_in_valid_range(self, translator):
        """Condition p values must be clamped to [0, 1]."""
        text = "Apply when quality above 0.9."
        skill = translator.ingest(text)
        p_val = skill.condition.get("min_p")
        if p_val is not None:
            assert 0.0 <= p_val <= 1.0

    def test_export_roundtrip(self, translator):
        """Ingest then export produces readable text."""
        skill = translator.ingest(SAMPLE_SKILL_TEXT, name="test_skill")
        exported = translator.export(skill)
        assert "# test_skill" in exported
        assert "strategy" in exported.lower()
        assert skill.skill_id in exported


# ---------------------------------------------------------------
# GovernedEvalLoop tests
# ---------------------------------------------------------------

class TestGovernedEvalLoop:
    """Tests for governed evaluation of candidate programs."""

    def test_evaluate_program_returns_result(self, config, skill_files):
        loop = GovernedEvalLoop(config=config)
        result = loop.evaluate_program(
            program_id="test-prog-1",
            skill_files=skill_files,
            benchmark_score=0.75,
            regime="truthful_auction",
            contract_type=ContractType.TRUTHFUL_AUCTION,
            seed=42,
        )
        assert isinstance(result, EvalResult)
        assert result.program_id == "test-prog-1"
        assert result.benchmark_score == 0.75
        assert result.regime == "truthful_auction"
        assert result.skills_ingested == 3

    def test_composite_score_in_range(self, config, skill_files):
        loop = GovernedEvalLoop(config=config)
        result = loop.evaluate_program(
            program_id="test-prog-2",
            skill_files=skill_files,
            benchmark_score=0.8,
            regime="default_market",
            contract_type=ContractType.DEFAULT_MARKET,
            seed=42,
        )
        assert 0.0 <= result.composite_score <= 1.0

    def test_governance_delta_keys(self, config, skill_files):
        loop = GovernedEvalLoop(config=config)
        result = loop.evaluate_program(
            program_id="test-prog-3",
            skill_files=skill_files,
            benchmark_score=0.6,
            regime="fair_division",
            contract_type=ContractType.FAIR_DIVISION,
            seed=42,
        )
        assert "toxicity_reduction" in result.governance_delta
        assert "welfare_improvement" in result.governance_delta
        assert "quality_gap_shift" in result.governance_delta

    def test_oracle_has_no_contract_key(self, config, skill_files):
        loop = GovernedEvalLoop(config=config)
        result = loop.evaluate_program(
            program_id="test-prog-4",
            skill_files=skill_files,
            benchmark_score=0.7,
            regime="truthful_auction",
            contract_type=ContractType.TRUTHFUL_AUCTION,
            seed=42,
        )
        # Oracle metrics should not have a contract key
        assert "contract" not in result.oracle_metrics

    def test_governed_has_contract_key(self, config, skill_files):
        loop = GovernedEvalLoop(config=config)
        result = loop.evaluate_program(
            program_id="test-prog-5",
            skill_files=skill_files,
            benchmark_score=0.7,
            regime="truthful_auction",
            contract_type=ContractType.TRUTHFUL_AUCTION,
            seed=42,
        )
        assert result.governed_metrics.get("contract") == "truthful_auction"

    def test_evaluate_all_regimes(self, config, skill_files):
        loop = GovernedEvalLoop(config=config)
        results = loop.evaluate_program_all_regimes(
            program_id="test-prog-6",
            skill_files=skill_files,
            benchmark_score=0.75,
            seed=42,
        )
        assert len(results) == 3
        assert "truthful_auction" in results
        assert "fair_division" in results
        assert "default_market" in results

    def test_reproducible_with_same_seed(self, config, skill_files):
        loop = GovernedEvalLoop(config=config)
        r1 = loop.evaluate_program(
            program_id="p", skill_files=skill_files,
            benchmark_score=0.7, regime="ta",
            contract_type=ContractType.TRUTHFUL_AUCTION, seed=123,
        )
        r2 = loop.evaluate_program(
            program_id="p", skill_files=skill_files,
            benchmark_score=0.7, regime="ta",
            contract_type=ContractType.TRUTHFUL_AUCTION, seed=123,
        )
        assert r1.composite_score == r2.composite_score

    def test_p_invariant_in_probe_interactions(self, config, skill_files):
        """All generated probe interactions must have p in [0, 1]."""
        loop = GovernedEvalLoop(config=config)
        import random
        rng = random.Random(42)
        translator = SkillTranslator()
        skills = translator.ingest_batch(skill_files)
        interactions = loop._generate_probe_interactions(skills, rng, 100)
        for ix in interactions:
            assert 0.0 <= ix.p <= 1.0, f"p={ix.p} out of range"
            assert -1.0 <= ix.v_hat <= 1.0, f"v_hat={ix.v_hat} out of range"


# ---------------------------------------------------------------
# FrontierComparator tests
# ---------------------------------------------------------------

class TestFrontierComparator:
    """Tests for frontier tracking and regime comparison."""

    def _make_result(self, pid, score, regime, tox_red=0.0, wel_imp=0.0):
        return EvalResult(
            program_id=pid,
            benchmark_score=score,
            composite_score=score,
            regime=regime,
            governance_delta={
                "toxicity_reduction": tox_red,
                "welfare_improvement": wel_imp,
            },
            skills_ingested=2,
        )

    def test_add_result_enters_frontier(self):
        comp = FrontierComparator(frontier_size=2)
        entered = comp.add_result(
            "ta", self._make_result("p1", 0.8, "ta"), iteration=0
        )
        assert entered

    def test_frontier_respects_size(self):
        comp = FrontierComparator(frontier_size=2)
        comp.add_result("ta", self._make_result("p1", 0.8, "ta"))
        comp.add_result("ta", self._make_result("p2", 0.9, "ta"))
        comp.add_result("ta", self._make_result("p3", 0.7, "ta"))
        frontier = comp.get_frontier("ta")
        assert len(frontier) == 2
        assert frontier[0].program_id == "p2"  # Best first
        assert frontier[1].program_id == "p1"

    def test_compare_pair_overlap(self):
        comp = FrontierComparator(frontier_size=2)
        comp.add_result("ta", self._make_result("p1", 0.8, "ta"))
        comp.add_result("ta", self._make_result("p2", 0.9, "ta"))
        comp.add_result("fd", self._make_result("p1", 0.85, "fd"))
        comp.add_result("fd", self._make_result("p3", 0.7, "fd"))
        div = comp.compare_pair("ta", "fd")
        # p1 is in both frontiers
        assert div.program_overlap > 0

    def test_compare_pair_no_overlap(self):
        comp = FrontierComparator(frontier_size=2)
        comp.add_result("ta", self._make_result("p1", 0.8, "ta"))
        comp.add_result("ta", self._make_result("p2", 0.9, "ta"))
        comp.add_result("fd", self._make_result("p3", 0.85, "fd"))
        comp.add_result("fd", self._make_result("p4", 0.7, "fd"))
        div = comp.compare_pair("ta", "fd")
        assert div.program_overlap == 0.0

    def test_compare_all(self):
        comp = FrontierComparator(frontier_size=2)
        for regime in ["ta", "fd", "dm"]:
            comp.add_result(regime, self._make_result("p1", 0.8, regime))
        divs = comp.compare_all()
        # 3 regimes → 3 pairs
        assert len(divs) == 3

    def test_summary_report_structure(self):
        comp = FrontierComparator(frontier_size=2)
        comp.add_result("ta", self._make_result("p1", 0.8, "ta"))
        report = comp.summary_report()
        assert "frontiers" in report
        assert "divergences" in report
        assert "ta" in report["frontiers"]

    def test_history_records_all(self):
        comp = FrontierComparator(frontier_size=1)
        comp.add_result("ta", self._make_result("p1", 0.8, "ta"), iteration=0)
        comp.add_result("ta", self._make_result("p2", 0.9, "ta"), iteration=1)
        assert len(comp.history) == 2


# ---------------------------------------------------------------
# EvoSkillBridge integration tests
# ---------------------------------------------------------------

class TestEvoSkillBridge:
    """Integration tests for the top-level bridge."""

    def test_evaluate_candidate(self, config, skill_files):
        bridge = EvoSkillBridge(config=config)
        results = bridge.evaluate_candidate(
            program_id="test-branch",
            skill_files=skill_files,
            benchmark_score=0.75,
            iteration=0,
        )
        assert len(results) == 3
        for _regime, result in results.items():
            assert isinstance(result, EvalResult)
            assert 0.0 <= result.composite_score <= 1.0

    def test_frontier_populated_after_evaluation(self, config, skill_files):
        bridge = EvoSkillBridge(config=config)
        bridge.evaluate_candidate(
            program_id="prog-1",
            skill_files=skill_files,
            benchmark_score=0.8,
            iteration=0,
        )
        for regime in config.contract_regimes:
            frontier = bridge.comparator.get_frontier(regime)
            assert len(frontier) >= 1

    def test_provenance_chain(self, config, skill_files):
        bridge = EvoSkillBridge(config=config)
        bridge.evaluate_candidate(
            program_id="prov-test",
            skill_files=skill_files,
            benchmark_score=0.7,
            iteration=0,
        )
        stats = bridge.provenance.get_stats()
        assert stats["total_records"] > 0

    def test_best_skills_for_regime(self, config, skill_files):
        bridge = EvoSkillBridge(config=config)
        bridge.evaluate_candidate(
            program_id="skill-test",
            skill_files=skill_files,
            benchmark_score=0.9,
            iteration=0,
        )
        # At least one regime should have ingested skills
        total_skills = sum(
            len(bridge.best_skills_for_regime(r))
            for r in config.contract_regimes
        )
        assert total_skills > 0

    def test_comparison_report(self, config, skill_files):
        bridge = EvoSkillBridge(config=config)
        bridge.evaluate_candidate(
            program_id="report-test",
            skill_files=skill_files,
            benchmark_score=0.7,
            iteration=0,
        )
        report = bridge.comparison_report()
        assert "frontiers" in report
        assert "divergences" in report
        assert "provenance" in report
        assert "iterations" in report
        assert len(report["iterations"]) == 3  # One per regime

    def test_export_frontier_skills(self, config, skill_files):
        bridge = EvoSkillBridge(config=config)
        bridge.evaluate_candidate(
            program_id="export-test",
            skill_files=skill_files,
            benchmark_score=0.85,
            iteration=0,
        )
        # Find a regime that has skills
        for regime in config.contract_regimes:
            exported = bridge.export_frontier_skills(regime)
            if exported:
                # Check exported files are readable markdown
                for filename, content in exported.items():
                    assert filename.endswith(".md")
                    assert "#" in content  # Has a header
                break

    def test_multiple_iterations_build_frontier(self, config, skill_files):
        bridge = EvoSkillBridge(config=config)
        for i in range(3):
            bridge.evaluate_candidate(
                program_id=f"prog-{i}",
                skill_files=skill_files,
                benchmark_score=0.6 + i * 0.1,
                iteration=i,
            )
        # With frontier_size=2, each regime should have at most 2 entries
        for regime in config.contract_regimes:
            frontier = bridge.comparator.get_frontier(regime)
            assert len(frontier) <= config.frontier_size

    def test_iterations_history(self, config, skill_files):
        bridge = EvoSkillBridge(config=config)
        bridge.evaluate_candidate(
            program_id="hist-test",
            skill_files=skill_files,
            benchmark_score=0.7,
            iteration=0,
        )
        # 3 regimes × 1 candidate = 3 iteration records
        assert len(bridge.iterations) == 3
