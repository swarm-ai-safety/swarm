"""Tests for causal credit propagation through interaction DAGs."""

from __future__ import annotations

import pytest

from swarm.metrics.causal_credit import (
    CausalCreditEngine,
    CausalSnapshot,
)
from swarm.models.interaction import SoftInteraction


def _ix(
    iid: str,
    p: float = 0.5,
    v_hat: float = 0.0,
    initiator: str = "agent_a",
    counterparty: str = "agent_b",
    causal_parents: list[str] | None = None,
) -> SoftInteraction:
    """Helper to build a minimal SoftInteraction."""
    return SoftInteraction(
        interaction_id=iid,
        p=p,
        v_hat=v_hat,
        initiator=initiator,
        counterparty=counterparty,
        causal_parents=causal_parents or [],
    )


class TestEmptyDAG:
    def test_empty_dag_snapshot(self):
        engine = CausalCreditEngine()
        snap = engine.compute_snapshot(step=0, interactions=[])
        assert snap.total_interactions == 0
        assert snap.dag_depth == 0
        assert snap.dag_width == 0
        assert snap.root_count == 0
        assert snap.leaf_count == 0
        assert snap.credit_by_agent == {}

    def test_empty_dag_propagate(self):
        engine = CausalCreditEngine()
        engine.build_dag([])
        assert engine.propagate_credit("nonexistent") == []

    def test_empty_dag_ancestors_descendants(self):
        engine = CausalCreditEngine()
        engine.build_dag([])
        assert engine.ancestors("x") == []
        assert engine.descendants("x") == []


class TestSingleRoot:
    def test_single_root_no_propagation(self):
        ix = _ix("root", p=0.8)
        engine = CausalCreditEngine()
        engine.build_dag([ix])

        attrs = engine.propagate_credit("root")
        assert attrs == []

    def test_single_root_snapshot(self):
        ix = _ix("root", p=0.8)
        engine = CausalCreditEngine()
        snap = engine.compute_snapshot(step=1, interactions=[ix])
        assert snap.total_interactions == 1
        assert snap.root_count == 1
        assert snap.leaf_count == 1
        assert snap.dag_depth == 0

    def test_single_root_cascade_risk(self):
        ix = _ix("root", p=0.8)
        engine = CausalCreditEngine()
        engine.build_dag([ix])
        assert engine.cascade_risk("root") == 0.0


class TestLinearChain:
    """A → B → C: verify credit decays exponentially backward."""

    def _build_chain(self, decay: float = 0.5):
        a = _ix("A", p=0.9, initiator="alice")
        b = _ix("B", p=0.7, initiator="bob", causal_parents=["A"])
        c = _ix("C", p=0.4, initiator="carol", causal_parents=["B"])
        engine = CausalCreditEngine(decay=decay)
        engine.build_dag([a, b, c])
        return engine

    def test_ancestors(self):
        engine = self._build_chain()
        assert set(engine.ancestors("C")) == {"A", "B"}
        assert engine.ancestors("A") == []

    def test_descendants(self):
        engine = self._build_chain()
        assert set(engine.descendants("A")) == {"B", "C"}
        assert engine.descendants("C") == []

    def test_credit_decay(self):
        engine = self._build_chain(decay=0.5)
        attrs = engine.propagate_credit("C", signal="p")

        by_target = {a.target_id: a for a in attrs}
        assert "B" in by_target
        assert "A" in by_target

        # C's p = 0.4
        # B is depth 1: 0.4 * 0.5^1 = 0.2
        assert by_target["B"].decayed_credit == pytest.approx(0.2)
        assert by_target["B"].path_length == 1

        # A is depth 2: 0.4 * 0.5^2 = 0.1
        assert by_target["A"].decayed_credit == pytest.approx(0.1)
        assert by_target["A"].path_length == 2

    def test_raw_credit_unchanged(self):
        engine = self._build_chain()
        attrs = engine.propagate_credit("C", signal="p")
        for a in attrs:
            assert a.raw_credit == pytest.approx(0.4)


class TestDiamondDAG:
    """A → B, A → C, B → D, C → D: verify D's credit reaches A via two paths."""

    def _build_diamond(self):
        a = _ix("A", p=0.9, initiator="alice")
        b = _ix("B", p=0.8, initiator="bob", causal_parents=["A"])
        c = _ix("C", p=0.7, initiator="carol", causal_parents=["A"])
        d = _ix("D", p=0.3, initiator="dave", causal_parents=["B", "C"])
        engine = CausalCreditEngine(decay=0.5)
        engine.build_dag([a, b, c, d])
        return engine

    def test_d_ancestors(self):
        engine = self._build_diamond()
        anc = engine.ancestors("D")
        assert set(anc) == {"A", "B", "C"}

    def test_a_descendants(self):
        engine = self._build_diamond()
        desc = engine.descendants("A")
        assert set(desc) == {"B", "C", "D"}

    def test_credit_reaches_a(self):
        engine = self._build_diamond()
        attrs = engine.propagate_credit("D", signal="p")
        targets = {a.target_id for a in attrs}
        assert "A" in targets

    def test_credit_values(self):
        engine = self._build_diamond()
        attrs = engine.propagate_credit("D", signal="p")
        by_target = {a.target_id: a for a in attrs}

        # D's p = 0.3
        # B,C at depth 1: 0.3 * 0.5 = 0.15
        assert by_target["B"].decayed_credit == pytest.approx(0.15)
        assert by_target["C"].decayed_credit == pytest.approx(0.15)

        # A at depth 2 (reached via BFS, first path wins in visited set)
        assert by_target["A"].decayed_credit == pytest.approx(0.075)
        assert by_target["A"].path_length == 2


class TestCascadeRisk:
    def test_high_cascade_risk(self):
        """Root causes chain of low-p descendants."""
        root = _ix("R", p=0.9)
        bad1 = _ix("B1", p=0.1, causal_parents=["R"])
        bad2 = _ix("B2", p=0.2, causal_parents=["R"])
        ok = _ix("OK", p=0.8, causal_parents=["R"])

        engine = CausalCreditEngine()
        engine.build_dag([root, bad1, bad2, ok])

        risk = engine.cascade_risk("R", p_threshold=0.3)
        # 2 out of 3 descendants below threshold
        assert risk == pytest.approx(2 / 3)

    def test_zero_cascade_risk(self):
        root = _ix("R", p=0.5)
        good = _ix("G", p=0.8, causal_parents=["R"])

        engine = CausalCreditEngine()
        engine.build_dag([root, good])

        assert engine.cascade_risk("R", p_threshold=0.3) == 0.0

    def test_no_descendants(self):
        leaf = _ix("L", p=0.1)
        engine = CausalCreditEngine()
        engine.build_dag([leaf])
        assert engine.cascade_risk("L") == 0.0


class TestMaxDepthCutoff:
    def test_depth_cutoff(self):
        """Deep chain respects max_depth limit."""
        # Build chain of length 10: N0 → N1 → ... → N9
        nodes = []
        for i in range(10):
            parents = [f"N{i-1}"] if i > 0 else []
            nodes.append(_ix(f"N{i}", p=0.5, causal_parents=parents))

        engine = CausalCreditEngine(decay=0.5, max_depth=3)
        engine.build_dag(nodes)

        # From N9, should only reach back 3 hops: N8, N7, N6
        attrs = engine.propagate_credit("N9")
        assert len(attrs) == 3
        targets = {a.target_id for a in attrs}
        assert targets == {"N8", "N7", "N6"}

    def test_ancestors_depth_cutoff(self):
        nodes = []
        for i in range(10):
            parents = [f"N{i-1}"] if i > 0 else []
            nodes.append(_ix(f"N{i}", p=0.5, causal_parents=parents))

        engine = CausalCreditEngine(max_depth=10)
        engine.build_dag(nodes)

        anc = engine.ancestors("N9", max_depth=2)
        assert set(anc) == {"N8", "N7"}


class TestAgentCreditSummary:
    def test_multi_agent_accumulation(self):
        """Multiple interactions, credit accumulates per agent."""
        a1 = _ix("a1", p=0.8, initiator="alice")
        a2 = _ix("a2", p=0.6, initiator="alice")
        b1 = _ix("b1", p=0.3, initiator="bob", causal_parents=["a1", "a2"])

        engine = CausalCreditEngine(decay=0.5)
        summary = engine.agent_credit_summary([a1, a2, b1], signal="p")

        # b1 propagates credit to alice (via a1 and a2)
        # b1.p = 0.3, depth 1 to both a1 and a2
        # alice gets 0.3 * 0.5 + 0.3 * 0.5 = 0.3
        assert "alice" in summary
        assert summary["alice"] == pytest.approx(0.3)

    def test_no_parents_no_credit(self):
        """Root-only interactions produce empty credit summary."""
        roots = [_ix(f"r{i}", p=0.5) for i in range(5)]
        engine = CausalCreditEngine()
        summary = engine.agent_credit_summary(roots)
        assert summary == {}


class TestCyclePrevention:
    def test_cycle_handled_gracefully(self):
        """If causal_parents accidentally create a cycle, engine doesn't hang."""
        # A → B → A (cycle!)
        a = _ix("A", p=0.5, causal_parents=["B"])
        b = _ix("B", p=0.5, causal_parents=["A"])

        engine = CausalCreditEngine()
        engine.build_dag([a, b])

        # Should terminate without hanging
        attrs_a = engine.propagate_credit("A")
        attrs_b = engine.propagate_credit("B")

        # In a cycle, BFS reaches the other node AND loops back to self
        targets_a = {a.target_id for a in attrs_a}
        targets_b = {a.target_id for a in attrs_b}
        assert "B" in targets_a
        assert "A" in targets_b
        # Key: terminates (visited set prevents infinite loop)
        assert len(attrs_a) <= 2
        assert len(attrs_b) <= 2

    def test_cycle_ancestors(self):
        a = _ix("A", p=0.5, causal_parents=["B"])
        b = _ix("B", p=0.5, causal_parents=["A"])

        engine = CausalCreditEngine()
        engine.build_dag([a, b])

        # Visited set prevents infinite loop; in a cycle, A reaches B
        # and B's parent is A, so A appears as its own ancestor
        assert "B" in set(engine.ancestors("A"))
        assert "A" in set(engine.ancestors("B"))

    def test_three_node_cycle(self):
        a = _ix("A", causal_parents=["C"])
        b = _ix("B", causal_parents=["A"])
        c = _ix("C", causal_parents=["B"])

        engine = CausalCreditEngine()
        engine.build_dag([a, b, c])

        # Should find all reachable nodes (including self via cycle) without hanging
        anc = engine.ancestors("A")
        assert {"B", "C"}.issubset(set(anc))


class TestSignals:
    def test_v_hat_signal(self):
        a = _ix("A", v_hat=0.6)
        b = _ix("B", v_hat=-0.2, causal_parents=["A"])

        engine = CausalCreditEngine(decay=0.5)
        engine.build_dag([a, b])

        attrs = engine.propagate_credit("B", signal="v_hat")
        assert len(attrs) == 1
        assert attrs[0].raw_credit == pytest.approx(-0.2)
        assert attrs[0].decayed_credit == pytest.approx(-0.1)

    def test_unknown_signal_raises(self):
        a = _ix("A")
        engine = CausalCreditEngine()
        engine.build_dag([a])

        with pytest.raises(ValueError, match="Unknown signal"):
            engine.propagate_credit("A", signal="nonexistent")


class TestSnapshot:
    def test_full_snapshot(self):
        a = _ix("A", p=0.9, initiator="alice")
        b = _ix("B", p=0.7, initiator="bob", causal_parents=["A"])
        c = _ix("C", p=0.3, initiator="carol", causal_parents=["A"])
        d = _ix("D", p=0.2, initiator="dave", causal_parents=["B", "C"])

        engine = CausalCreditEngine(decay=0.5)
        snap = engine.compute_snapshot(step=5, interactions=[a, b, c, d])

        assert snap.step == 5
        assert snap.total_interactions == 4
        assert snap.root_count == 1
        assert snap.leaf_count == 1
        assert snap.dag_depth == 2
        assert snap.dag_width >= 2  # B and C at depth 1
        assert 0 in snap.cascade_depth_histogram
        assert 1 in snap.cascade_depth_histogram

    def test_snapshot_type(self):
        engine = CausalCreditEngine()
        snap = engine.compute_snapshot(step=0, interactions=[])
        assert isinstance(snap, CausalSnapshot)


class TestCausalParentsOnInteraction:
    def test_default_empty(self):
        ix = SoftInteraction()
        assert ix.causal_parents == []

    def test_roundtrip_serialization(self):
        ix = SoftInteraction(causal_parents=["parent_1", "parent_2"])
        data = ix.to_dict()
        assert data["causal_parents"] == ["parent_1", "parent_2"]

        restored = SoftInteraction.from_dict(data)
        assert restored.causal_parents == ["parent_1", "parent_2"]

    def test_model_copy_preserves_parents(self):
        ix = SoftInteraction(causal_parents=["p1"])
        copy = ix.model_copy(update={"p": 0.8})
        assert copy.causal_parents == ["p1"]
