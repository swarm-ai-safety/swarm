"""Tests for Moltbook challenge generator and feed."""

import re

from swarm.env.moltbook import (
    ChallengeGenerator,
    ContentStatus,
    MathChallenge,
    MathOperation,
    MoltbookFeed,
)


def _expected_answer(operation: MathOperation, a: float, b: float) -> float:
    if operation == MathOperation.ADD:
        return round(a + b, 2)
    if operation == MathOperation.SUBTRACT:
        return round(a - b, 2)
    if operation == MathOperation.MULTIPLY:
        return round(a * b, 2)
    return round(a / b, 2) if b != 0 else 0.0


def test_challenge_generator_answer_matches_operands():
    gen = ChallengeGenerator(seed=123)
    challenge = gen.generate(difficulty=0.8)
    a, b = challenge.operands
    expected = _expected_answer(challenge.operation, a, b)
    assert challenge.answer == expected


def test_obfuscation_layers_present_at_high_difficulty():
    gen = ChallengeGenerator(seed=99)
    challenge = gen.generate(difficulty=1.0)
    raw = challenge.raw_text
    assert any(ch.isupper() for ch in raw)
    assert any(ch.islower() for ch in raw)
    assert any(ch in "^/~|]}<*+" for ch in raw)
    assert re.search(r"\d", raw) is None


def test_feed_verification_gate_and_expiry():
    feed = MoltbookFeed()
    challenge = MathChallenge(answer=10.0, expires_at_step=1)
    post = feed.submit_content(
        author_id="agent_1",
        content="Test post",
        submolt="test",
        current_step=0,
        challenge=challenge,
    )
    assert post.status == ContentStatus.PENDING_VERIFICATION
    assert feed.get_published_posts(submolt="test") == []

    assert feed.verify_content(post.post_id, 10.0, current_step=1)
    assert post.status == ContentStatus.PUBLISHED

    challenge2 = MathChallenge(answer=5.0, expires_at_step=0)
    post2 = feed.submit_content(
        author_id="agent_2",
        content="Late post",
        submolt="test",
        current_step=0,
        challenge=challenge2,
    )
    assert not feed.verify_content(post2.post_id, 5.0, current_step=2)
    assert post2.status == ContentStatus.EXPIRED
