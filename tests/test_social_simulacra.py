"""Tests for Social Simulacra integration with SWARM-Concordia bridge."""

import random

import pytest

from swarm.bridges.concordia.adapter import ConcordiaAdapter
from swarm.bridges.concordia.events import ConcordiaEventType
from swarm.bridges.concordia.multiverse import (
    MultiverseConfig,
    MultiverseResult,
    MultiverseRunner,
    UniverseResult,
)
from swarm.bridges.concordia.simulacra import (
    CommunityConfig,
    ExpandedPersona,
    PersonaExpander,
    PersonaSeed,
    Post,
    Thread,
    ThreadGenerator,
    WhatIfInjector,
    thread_to_narrative_samples,
    threads_to_judge_ground_truth,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def community():
    return CommunityConfig(
        name="Test Community",
        description="A test community for unit tests",
        goal="Testing social simulacra integration",
        rules=["Be respectful", "Stay on topic"],
        norms=["Help newcomers"],
    )


@pytest.fixture
def seeds():
    return [
        PersonaSeed(
            name="Alice",
            description="A helpful and experienced member",
            community_role="veteran",
        ),
        PersonaSeed(
            name="Bob",
            description="A curious newcomer learning the ropes",
            community_role="newcomer",
        ),
        PersonaSeed(
            name="Carol",
            description="A moderator who keeps discussions civil",
            community_role="moderator",
        ),
    ]


@pytest.fixture
def personas(community, seeds):
    expander = PersonaExpander(community=community, seeds=seeds)
    return expander.expand(n=10, rng=random.Random(42))


# ---------------------------------------------------------------------------
# PersonaSeed tests
# ---------------------------------------------------------------------------


class TestPersonaSeed:
    def test_construction(self):
        seed = PersonaSeed(name="Test", description="A test persona")
        assert seed.name == "Test"
        assert seed.description == "A test persona"
        assert seed.community_role == ""

    def test_to_prompt_line_without_role(self):
        seed = PersonaSeed(name="Test", description="desc")
        assert seed.to_prompt_line() == "Test: desc"

    def test_to_prompt_line_with_role(self):
        seed = PersonaSeed(name="Test", description="desc", community_role="mod")
        assert seed.to_prompt_line() == "Test (mod): desc"


# ---------------------------------------------------------------------------
# ExpandedPersona tests
# ---------------------------------------------------------------------------


class TestExpandedPersona:
    def test_defaults(self):
        persona = ExpandedPersona()
        assert persona.name == ""
        assert persona.generated is True
        assert persona.persona_id  # should have a UUID

    def test_to_prompt_context(self):
        persona = ExpandedPersona(
            name="Alex", description="a tester", community_role="member"
        )
        ctx = persona.to_prompt_context()
        assert "Alex" in ctx
        assert "member" in ctx
        assert "a tester" in ctx

    def test_to_prompt_context_no_role(self):
        persona = ExpandedPersona(name="Alex", description="a tester")
        ctx = persona.to_prompt_context()
        assert "Alex" in ctx
        assert "(" not in ctx


# ---------------------------------------------------------------------------
# CommunityConfig tests
# ---------------------------------------------------------------------------


class TestCommunityConfig:
    def test_to_prompt_context(self, community):
        ctx = community.to_prompt_context()
        assert "Test Community" in ctx
        assert "Be respectful" in ctx
        assert "Help newcomers" in ctx

    def test_to_prompt_context_minimal(self):
        config = CommunityConfig(name="Minimal")
        ctx = config.to_prompt_context()
        assert "Minimal" in ctx


# ---------------------------------------------------------------------------
# Post and Thread tests
# ---------------------------------------------------------------------------


class TestPost:
    def test_is_reply(self):
        root = Post(content="Hello")
        assert root.is_reply is False

        reply = Post(content="Hi", parent_id="some-id")
        assert reply.is_reply is True


class TestThread:
    def test_empty_thread(self):
        t = Thread()
        assert t.root is None
        assert t.replies == []
        assert t.participants == []

    def test_single_post_thread(self):
        author = ExpandedPersona(name="Alice")
        root = Post(author=author, content="Hello")
        t = Thread(posts=[root])
        assert t.root is root
        assert len(t.replies) == 0
        assert len(t.participants) == 1

    def test_thread_with_replies(self):
        alice = ExpandedPersona(name="Alice")
        bob = ExpandedPersona(name="Bob")
        root = Post(author=alice, content="Hello")
        reply = Post(author=bob, content="Hi", parent_id=root.post_id, depth=1)
        t = Thread(posts=[root, reply])
        assert len(t.replies) == 1
        assert len(t.participants) == 2

    def test_to_narrative(self):
        alice = ExpandedPersona(name="Alice", community_role="mod")
        bob = ExpandedPersona(name="Bob")
        root = Post(author=alice, content="Hello world")
        reply = Post(
            author=bob, content="Hi there", parent_id=root.post_id, depth=1
        )
        t = Thread(posts=[root, reply])
        narrative = t.to_narrative()
        assert "Alice (mod): Hello world" in narrative
        assert "  Bob: Hi there" in narrative

    def test_participants_deduplication(self):
        alice = ExpandedPersona(name="Alice")
        root = Post(author=alice, content="Post 1")
        reply = Post(author=alice, content="Post 2", parent_id=root.post_id)
        t = Thread(posts=[root, reply])
        assert len(t.participants) == 1


# ---------------------------------------------------------------------------
# PersonaExpander tests
# ---------------------------------------------------------------------------


class TestPersonaExpander:
    def test_synthetic_expansion(self, community, seeds):
        expander = PersonaExpander(community=community, seeds=seeds)
        personas = expander.expand(n=20, rng=random.Random(42))
        # Should include 3 seeds + 20 generated
        assert len(personas) == 23

    def test_seeds_included(self, community, seeds):
        expander = PersonaExpander(community=community, seeds=seeds)
        personas = expander.expand(n=5, rng=random.Random(42))
        names = {p.name for p in personas}
        assert "Alice" in names
        assert "Bob" in names
        assert "Carol" in names

    def test_seed_personas_not_generated(self, community, seeds):
        expander = PersonaExpander(community=community, seeds=seeds)
        personas = expander.expand(n=5, rng=random.Random(42))
        seed_personas = [p for p in personas if not p.generated]
        assert len(seed_personas) == 3

    def test_generated_personas_have_seed_origin(self, community, seeds):
        expander = PersonaExpander(community=community, seeds=seeds)
        personas = expander.expand(n=5, rng=random.Random(42))
        generated = [p for p in personas if p.generated]
        for p in generated:
            assert p.seed_origin in {"Alice", "Bob", "Carol"}

    def test_deterministic_with_seed(self, community, seeds):
        expander = PersonaExpander(community=community, seeds=seeds)
        p1 = expander.expand(n=10, rng=random.Random(42))
        p2 = expander.expand(n=10, rng=random.Random(42))
        assert [p.name for p in p1] == [p.name for p in p2]

    def test_llm_expansion(self, community, seeds):
        responses = [
            "1. Dave (regular): A friendly regular who enjoys casual chat\n"
            "2. Eve (expert): A domain expert who shares deep insights\n"
            "3. Frank (lurker): A quiet observer who rarely posts"
        ]

        def mock_llm(prompt, temperature):
            return responses[0]

        expander = PersonaExpander(
            community=community, seeds=seeds, llm_client=mock_llm
        )
        personas = expander.expand(n=3, rng=random.Random(42))
        generated = [p for p in personas if p.generated]
        assert len(generated) >= 3
        gen_names = {p.name for p in generated}
        assert "Dave" in gen_names
        assert "Eve" in gen_names

    def test_expand_zero(self, community, seeds):
        expander = PersonaExpander(community=community, seeds=seeds)
        personas = expander.expand(n=0, rng=random.Random(42))
        # Only seeds
        assert len(personas) == 3


# ---------------------------------------------------------------------------
# ThreadGenerator tests
# ---------------------------------------------------------------------------


class TestThreadGenerator:
    def test_basic_generation(self, community, personas):
        gen = ThreadGenerator(community=community, personas=personas)
        thread = gen.generate_thread(rng=random.Random(42))
        assert thread.root is not None
        assert len(thread.posts) >= 1

    def test_thread_has_replies(self, community, personas):
        # With high reply_mean, should usually get replies
        gen = ThreadGenerator(
            community=community,
            personas=personas,
            reply_mean=0.95,
            max_replies=5,
        )
        thread = gen.generate_thread(rng=random.Random(42))
        assert len(thread.replies) >= 1

    def test_max_replies_respected(self, community, personas):
        gen = ThreadGenerator(
            community=community,
            personas=personas,
            reply_mean=1.0,
            max_replies=3,
        )
        thread = gen.generate_thread(rng=random.Random(42))
        assert len(thread.replies) <= 3

    def test_specific_author(self, community, personas):
        author = personas[0]
        gen = ThreadGenerator(community=community, personas=personas)
        thread = gen.generate_thread(author=author, rng=random.Random(42))
        assert thread.root.author.persona_id == author.persona_id

    def test_generate_multiple_threads(self, community, personas):
        gen = ThreadGenerator(community=community, personas=personas)
        threads = gen.generate_threads(5, rng=random.Random(42))
        assert len(threads) == 5
        for t in threads:
            assert t.root is not None

    def test_deterministic(self, community, personas):
        gen = ThreadGenerator(community=community, personas=personas)
        t1 = gen.generate_thread(rng=random.Random(42))
        t2 = gen.generate_thread(rng=random.Random(42))
        assert len(t1.posts) == len(t2.posts)
        for p1, p2 in zip(t1.posts, t2.posts, strict=True):
            assert p1.content == p2.content

    def test_empty_personas(self, community):
        gen = ThreadGenerator(community=community, personas=[])
        thread = gen.generate_thread(rng=random.Random(42))
        assert len(thread.posts) == 0

    def test_depth_capped(self, community, personas):
        gen = ThreadGenerator(
            community=community,
            personas=personas,
            reply_mean=1.0,
            max_replies=8,
        )
        thread = gen.generate_thread(rng=random.Random(42))
        for post in thread.posts:
            assert post.depth <= 5

    def test_with_llm_client(self, community, personas):
        call_count = 0

        def mock_llm(prompt, temperature):
            nonlocal call_count
            call_count += 1
            return f"Generated content #{call_count}"

        gen = ThreadGenerator(
            community=community,
            personas=personas,
            llm_client=mock_llm,
            reply_mean=0.8,
            max_replies=2,
        )
        thread = gen.generate_thread(rng=random.Random(42))
        assert thread.root is not None
        assert "Generated content" in thread.root.content


# ---------------------------------------------------------------------------
# WhatIfInjector tests
# ---------------------------------------------------------------------------


class TestWhatIfInjector:
    def _make_thread(self, personas):
        alice = personas[0]
        bob = personas[1]
        root = Post(author=alice, content="Original discussion post")
        reply = Post(
            author=bob,
            content="I agree with this",
            parent_id=root.post_id,
            depth=1,
        )
        return Thread(posts=[root, reply])

    def test_basic_injection(self, community, personas):
        thread = self._make_thread(personas)
        injector = WhatIfInjector(community=community, max_injected_replies=2)
        new_thread, injected = injector.inject(
            thread,
            "a troll who derails discussions",
            persona_name="TrollBot",
            rng=random.Random(42),
        )
        assert len(new_thread.posts) > len(thread.posts)
        assert injected.name == "TrollBot"
        assert "troll" in injected.description

    def test_injected_persona_is_counterfactual(self, community, personas):
        thread = self._make_thread(personas)
        injector = WhatIfInjector(community=community, max_injected_replies=1)
        _, injected = injector.inject(
            thread,
            "a domain expert",
            rng=random.Random(42),
        )
        assert injected.community_role == "injected_counterfactual"
        assert injected.seed_origin == "whatif"

    def test_thread_id_includes_whatif(self, community, personas):
        thread = self._make_thread(personas)
        injector = WhatIfInjector(community=community, max_injected_replies=1)
        new_thread, _ = injector.inject(
            thread,
            "a moderator",
            rng=random.Random(42),
        )
        assert "whatif" in new_thread.thread_id

    def test_community_response_generated(self, community, personas):
        thread = self._make_thread(personas)
        injector = WhatIfInjector(community=community, max_injected_replies=2)
        new_thread, injected = injector.inject(
            thread,
            "a troll",
            rng=random.Random(42),
        )
        # Should have original 2 + injected replies + community responses
        assert len(new_thread.posts) >= 4

    def test_troll_content_matches_role(self, community, personas):
        thread = self._make_thread(personas)
        injector = WhatIfInjector(community=community, max_injected_replies=1)
        new_thread, injected = injector.inject(
            thread,
            "a troll who flames everyone",
            persona_name="FlameBot",
            rng=random.Random(42),
        )
        injected_posts = [
            p for p in new_thread.posts if p.author.persona_id == injected.persona_id
        ]
        assert len(injected_posts) >= 1

    def test_expert_content(self, community, personas):
        thread = self._make_thread(personas)
        injector = WhatIfInjector(community=community, max_injected_replies=1)
        new_thread, _ = injector.inject(
            thread,
            "a domain expert with deep knowledge",
            rng=random.Random(42),
        )
        assert len(new_thread.posts) > len(thread.posts)

    def test_with_llm_client(self, community, personas):
        thread = self._make_thread(personas)

        def mock_llm(prompt, temperature):
            return "LLM-generated counterfactual content"

        injector = WhatIfInjector(
            community=community, llm_client=mock_llm, max_injected_replies=1
        )
        new_thread, _ = injector.inject(
            thread,
            "a mysterious newcomer",
            rng=random.Random(42),
        )
        injected_content = [p.content for p in new_thread.posts[len(thread.posts) :]]
        assert any("LLM-generated" in c for c in injected_content)

    def test_llm_sees_evolving_thread(self, community, personas):
        """Each LLM prompt during inject() must include previously injected
        posts so the counterfactual conversation conditions on its own turns."""
        thread = self._make_thread(personas)

        prompts_received: list[str] = []

        def recording_llm(prompt, temperature):
            prompts_received.append(prompt)
            return f"Reply #{len(prompts_received)}"

        injector = WhatIfInjector(
            community=community,
            llm_client=recording_llm,
            max_injected_replies=3,
        )
        injector.inject(
            thread,
            "a provocative debater",
            persona_name="Debater",
            rng=random.Random(42),
        )

        # With max_injected_replies=3 we get 3 injected + 2 community responses
        # = 5 LLM calls. Each successive prompt should contain earlier replies.
        assert len(prompts_received) >= 3

        # The second injected-reply prompt must contain "Reply #1"
        # (the content of the first injected post)
        assert "Reply #1" in prompts_received[2]


# ---------------------------------------------------------------------------
# Thread-to-SWARM bridge tests
# ---------------------------------------------------------------------------


class TestThreadToNarrativeSamples:
    def test_single_post(self):
        alice = ExpandedPersona(name="Alice")
        root = Post(author=alice, content="Hello world")
        thread = Thread(posts=[root])
        samples = thread_to_narrative_samples(thread)
        assert len(samples) == 1
        assert samples[0][0] == ["Alice"]

    def test_post_with_reply(self):
        alice = ExpandedPersona(name="Alice")
        bob = ExpandedPersona(name="Bob")
        root = Post(author=alice, content="Hello")
        reply = Post(
            author=bob, content="Hi", parent_id=root.post_id, depth=1
        )
        thread = Thread(posts=[root, reply])
        samples = thread_to_narrative_samples(thread)
        assert len(samples) == 1
        agent_ids, narrative = samples[0]
        assert "Alice" in agent_ids
        assert "Bob" in agent_ids
        assert "Hello" in narrative
        assert "Hi" in narrative

    def test_empty_thread(self):
        thread = Thread()
        samples = thread_to_narrative_samples(thread)
        assert len(samples) == 0

    def test_multiple_replies(self):
        alice = ExpandedPersona(name="Alice")
        bob = ExpandedPersona(name="Bob")
        carol = ExpandedPersona(name="Carol")
        root = Post(author=alice, content="Hello")
        r1 = Post(author=bob, content="Hi", parent_id=root.post_id, depth=1)
        r2 = Post(
            author=carol, content="Hey", parent_id=root.post_id, depth=1
        )
        thread = Thread(posts=[root, r1, r2])
        samples = thread_to_narrative_samples(thread)
        assert len(samples) == 2


class TestThreadsToJudgeGroundTruth:
    def test_without_trolls(self):
        alice = ExpandedPersona(name="Alice")
        bob = ExpandedPersona(name="Bob")
        root = Post(author=alice, content="Hi")
        reply = Post(
            author=bob, content="Hey", parent_id=root.post_id, depth=1
        )
        thread = Thread(posts=[root, reply])
        results = threads_to_judge_ground_truth(thread)
        assert len(results) == 1
        _, _, scores = results[0]
        assert scores.cooperation > 0.5
        assert scores.harm < 0.5

    def test_with_trolls(self):
        alice = ExpandedPersona(name="Alice")
        troll = ExpandedPersona(name="Troll")
        root = Post(author=alice, content="Hi")
        reply = Post(
            author=troll, content="You're wrong", parent_id=root.post_id, depth=1
        )
        thread = Thread(posts=[root, reply])
        results = threads_to_judge_ground_truth(
            thread, troll_persona_ids={"Troll"}
        )
        assert len(results) == 1
        _, _, scores = results[0]
        assert scores.harm > 0.5
        assert scores.cooperation < 0.5


# ---------------------------------------------------------------------------
# SWARM adapter integration tests
# ---------------------------------------------------------------------------


class TestSimulacraAdapterIntegration:
    def test_thread_scored_through_adapter(self, community, personas):
        gen = ThreadGenerator(
            community=community,
            personas=personas,
            reply_mean=0.9,
            max_replies=3,
        )
        thread = gen.generate_thread(rng=random.Random(42))
        samples = thread_to_narrative_samples(thread)

        adapter = ConcordiaAdapter()
        all_interactions = []
        for agent_ids, narrative in samples:
            interactions = adapter.process_narrative(
                agent_ids=agent_ids,
                narrative_text=narrative,
            )
            all_interactions.extend(interactions)

        assert len(all_interactions) > 0
        for interaction in all_interactions:
            assert 0.0 <= interaction.p <= 1.0
            assert interaction.metadata["bridge"] == "concordia"

    def test_whatif_scored_through_adapter(self, community, personas):
        gen = ThreadGenerator(community=community, personas=personas)
        thread = gen.generate_thread(rng=random.Random(42))

        injector = WhatIfInjector(community=community, max_injected_replies=2)
        new_thread, _ = injector.inject(
            thread,
            "a troll",
            rng=random.Random(42),
        )

        samples = thread_to_narrative_samples(new_thread)
        adapter = ConcordiaAdapter()
        all_interactions = []
        for agent_ids, narrative in samples:
            interactions = adapter.process_narrative(
                agent_ids=agent_ids,
                narrative_text=narrative,
            )
            all_interactions.extend(interactions)

        assert len(all_interactions) > 0
        for interaction in all_interactions:
            assert 0.0 <= interaction.p <= 1.0


# ---------------------------------------------------------------------------
# Multiverse tests
# ---------------------------------------------------------------------------


class TestMultiverseConfig:
    def test_defaults(self):
        config = MultiverseConfig()
        assert config.n_universes == 10
        assert len(config.temperatures) == 6
        assert config.threads_per_universe == 5


class TestUniverseResult:
    def test_defaults(self):
        result = UniverseResult()
        assert result.universe_id == 0
        assert result.toxicity_rate == 0.0
        assert result.n_interactions == 0


class TestMultiverseResult:
    def test_empty(self):
        result = MultiverseResult()
        assert len(result.universes) == 0

    def test_to_dict(self):
        result = MultiverseResult(
            toxicity_mean=0.3,
            toxicity_std=0.1,
            p_mean=0.6,
        )
        d = result.to_dict()
        assert d["toxicity_mean"] == 0.3
        assert d["p_mean"] == 0.6
        assert "n_universes" in d


class TestMultiverseRunner:
    def test_basic_run(self, community, personas):
        config = MultiverseConfig(
            n_universes=4,
            temperatures=[0.5, 0.7],
            threads_per_universe=2,
            base_seed=42,
        )
        runner = MultiverseRunner(
            community=community,
            personas=personas,
            config=config,
        )
        result = runner.run()
        assert len(result.universes) == 4
        assert result.toxicity_mean >= 0.0
        assert result.p_mean >= 0.0
        assert result.p_mean <= 1.0

    def test_deterministic(self, community, personas):
        config = MultiverseConfig(
            n_universes=4,
            temperatures=[0.6, 0.8],
            threads_per_universe=2,
            base_seed=42,
        )
        runner = MultiverseRunner(
            community=community,
            personas=personas,
            config=config,
        )
        r1 = runner.run()
        r2 = runner.run()
        assert r1.toxicity_mean == pytest.approx(r2.toxicity_mean)
        assert r1.p_mean == pytest.approx(r2.p_mean)

    def test_bias_variance_decomposition(self, community, personas):
        config = MultiverseConfig(
            n_universes=6,
            temperatures=[0.5, 0.7, 0.9],
            threads_per_universe=3,
            base_seed=42,
        )
        runner = MultiverseRunner(
            community=community,
            personas=personas,
            config=config,
        )
        result = runner.run(reference_p=0.5)
        # total_error = bias^2 + variance
        assert result.total_error == pytest.approx(
            result.bias_squared + result.variance
        )
        assert result.bias_squared >= 0.0
        assert result.variance >= 0.0

    def test_temperature_correlation_computed(self, community, personas):
        config = MultiverseConfig(
            n_universes=6,
            temperatures=[0.5, 0.7, 0.9],
            threads_per_universe=3,
            base_seed=42,
        )
        runner = MultiverseRunner(
            community=community,
            personas=personas,
            config=config,
        )
        result = runner.run()
        # Correlation should be in [-1, 1]
        assert -1.0 <= result.temperature_correlation <= 1.0

    def test_universe_count_matches_config(self, community, personas):
        """n_universes not evenly divisible by len(temperatures) must still
        produce exactly n_universes runs (remainder distributed round-robin)."""
        config = MultiverseConfig(
            n_universes=10,
            temperatures=[0.5, 0.7, 0.9],
            threads_per_universe=2,
            base_seed=42,
        )
        runner = MultiverseRunner(
            community=community,
            personas=personas,
            config=config,
        )
        result = runner.run()
        assert len(result.universes) == 10

        # First temperature gets extra universe (10 = 3*3 + 1)
        temps = [u.temperature for u in result.universes]
        assert temps.count(0.5) == 4  # base 3 + 1 remainder
        assert temps.count(0.7) == 3
        assert temps.count(0.9) == 3

    def test_per_universe_metrics(self, community, personas):
        config = MultiverseConfig(
            n_universes=2,
            temperatures=[0.7],
            threads_per_universe=2,
            base_seed=42,
        )
        runner = MultiverseRunner(
            community=community,
            personas=personas,
            config=config,
        )
        result = runner.run()
        for u in result.universes:
            assert u.n_threads == 2
            assert u.n_interactions >= 0
            assert 0.0 <= u.toxicity_rate <= 1.0
            assert 0.0 <= u.avg_p <= 1.0


# ---------------------------------------------------------------------------
# Event type tests
# ---------------------------------------------------------------------------


class TestSimulacraEventTypes:
    def test_new_event_types_exist(self):
        assert ConcordiaEventType.PERSONA_EXPANDED.value == "persona_expanded"
        assert ConcordiaEventType.THREAD_GENERATED.value == "thread_generated"
        assert ConcordiaEventType.WHATIF_INJECTED.value == "whatif_injected"
        assert (
            ConcordiaEventType.MULTIVERSE_UNIVERSE_COMPLETED.value
            == "multiverse_universe_completed"
        )
        assert (
            ConcordiaEventType.MULTIVERSE_ANALYSIS_COMPLETED.value
            == "multiverse_analysis_completed"
        )


# ---------------------------------------------------------------------------
# Import / __init__ tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_from_package(self):
        from swarm.bridges.concordia import (
            MultiverseRunner,
            PersonaSeed,
        )

        # Verify they're the right classes
        assert PersonaSeed.__name__ == "PersonaSeed"
        assert MultiverseRunner.__name__ == "MultiverseRunner"
