"""Tests for the artifact layer: models, registry, and causal wiring."""


from swarm.env.artifact_registry import ArtifactRegistry
from swarm.models.artifact import Artifact, ArtifactNeed, ArtifactSchema

# ── ArtifactSchema ──────────────────────────────────────────────────


class TestArtifactSchema:
    def test_compatible_superset(self):
        need = ArtifactSchema(kind="receipt", fields={"order_id": "str"})
        have = ArtifactSchema(
            kind="receipt",
            fields={"order_id": "str", "quality": "float"},
        )
        assert need.is_compatible(have)

    def test_incompatible_missing_field(self):
        need = ArtifactSchema(
            kind="receipt", fields={"order_id": "str", "extra": "int"}
        )
        have = ArtifactSchema(kind="receipt", fields={"order_id": "str"})
        assert not need.is_compatible(have)

    def test_incompatible_type_mismatch(self):
        need = ArtifactSchema(kind="receipt", fields={"order_id": "str"})
        have = ArtifactSchema(kind="receipt", fields={"order_id": "int"})
        assert not need.is_compatible(have)

    def test_empty_schema_compatible_with_anything(self):
        need = ArtifactSchema(kind="any", fields={})
        have = ArtifactSchema(kind="any", fields={"x": "str", "y": "int"})
        assert need.is_compatible(have)


# ── Artifact ────────────────────────────────────────────────────────


class TestArtifact:
    def test_to_dict_roundtrip(self):
        art = Artifact(
            artifact_id="art-1",
            kind="delivery_receipt",
            producer_id="agent_a",
            interaction_id="ix-1",
            data={"order_id": "ord-1", "quality_score": 0.9},
            step=5,
            p_at_production=0.8,
        )
        d = art.to_dict()
        assert d["kind"] == "delivery_receipt"
        assert d["p_at_production"] == 0.8
        assert "consumed_by" not in d  # not serialized

    def test_defaults(self):
        art = Artifact()
        assert art.kind == ""
        assert art.p_at_production == 0.5
        assert art.consumed_by == []


# ── ArtifactRegistry ───────────────────────────────────────────────


class TestArtifactRegistryPublish:
    def test_publish_and_lookup(self):
        reg = ArtifactRegistry()
        art = Artifact(artifact_id="a1", kind="receipt", step=10)
        reg.publish(art)
        assert reg.get("a1") is art
        assert len(reg) == 1

    def test_publish_reduces_pressure(self):
        reg = ArtifactRegistry()
        reg.declare_need(ArtifactNeed(kind="receipt", requester_id="b"))
        assert reg.pressure_scores()["receipt"] == 1.0

        reg.publish(Artifact(artifact_id="a1", kind="receipt", step=10))
        assert reg.pressure_scores()["receipt"] == 0.0


class TestArtifactRegistryMatch:
    def _registry_with_artifacts(self):
        reg = ArtifactRegistry()
        reg.publish(Artifact(
            artifact_id="fresh_good", kind="receipt",
            producer_id="alice", step=90, p_at_production=0.9,
        ))
        reg.publish(Artifact(
            artifact_id="fresh_bad", kind="receipt",
            producer_id="bob", step=90, p_at_production=0.2,
        ))
        reg.publish(Artifact(
            artifact_id="stale_good", kind="receipt",
            producer_id="carol", step=10, p_at_production=0.95,
        ))
        return reg

    def test_match_by_kind(self):
        reg = self._registry_with_artifacts()
        need = ArtifactNeed(kind="receipt", max_age_steps=50)
        matches = reg.match(need, current_step=100)
        # stale_good is age 90, should be excluded
        assert len(matches) == 2
        # Ordered by p_at_production descending
        assert matches[0].artifact_id == "fresh_good"
        assert matches[1].artifact_id == "fresh_bad"

    def test_match_with_min_p(self):
        reg = self._registry_with_artifacts()
        need = ArtifactNeed(kind="receipt", min_p=0.5, max_age_steps=50)
        matches = reg.match(need, current_step=100)
        assert len(matches) == 1
        assert matches[0].artifact_id == "fresh_good"

    def test_match_no_results(self):
        reg = self._registry_with_artifacts()
        need = ArtifactNeed(kind="nonexistent")
        assert reg.match(need, current_step=100) == []

    def test_match_for_agent_excludes_own(self):
        reg = ArtifactRegistry()
        reg.publish(Artifact(
            artifact_id="mine", kind="receipt",
            producer_id="alice", step=95,
        ))
        reg.publish(Artifact(
            artifact_id="theirs", kind="receipt",
            producer_id="bob", step=95,
        ))
        results = reg.match_for_agent("alice", current_step=100)
        assert len(results) == 1
        assert results[0]["artifact_id"] == "theirs"


class TestArtifactRegistryConsume:
    def test_consume_returns_parent_interaction_id(self):
        reg = ArtifactRegistry()
        reg.publish(Artifact(
            artifact_id="a1", kind="receipt",
            interaction_id="ix-parent", step=10,
        ))
        parent_id = reg.consume("a1", "ix-child")
        assert parent_id == "ix-parent"
        assert "ix-child" in reg.get("a1").consumed_by

    def test_consume_missing_returns_none(self):
        reg = ArtifactRegistry()
        assert reg.consume("nonexistent", "ix-child") is None


class TestArtifactRegistryPressure:
    def test_multiple_needs_accumulate(self):
        reg = ArtifactRegistry()
        reg.declare_need(ArtifactNeed(kind="receipt", requester_id="a"))
        reg.declare_need(ArtifactNeed(kind="receipt", requester_id="b"))
        reg.declare_need(ArtifactNeed(kind="proof", requester_id="c"))
        assert reg.pressure_scores()["receipt"] == 2.0
        assert reg.pressure_scores()["proof"] == 1.0

    def test_top_pressure(self):
        reg = ArtifactRegistry()
        for _ in range(5):
            reg.declare_need(ArtifactNeed(kind="hot", requester_id="x"))
        reg.declare_need(ArtifactNeed(kind="cold", requester_id="y"))
        top = reg.top_pressure(n=1)
        assert top[0][0] == "hot"
        assert top[0][1] == 5.0


class TestArtifactRegistryGC:
    def test_gc_removes_stale(self):
        reg = ArtifactRegistry()
        reg.publish(Artifact(artifact_id="old", kind="r", step=0))
        reg.publish(Artifact(artifact_id="new", kind="r", step=95))
        removed = reg.gc(current_step=100, max_age_steps=50)
        assert removed == 1
        assert reg.get("old") is None
        assert reg.get("new") is not None
        assert len(reg) == 1
