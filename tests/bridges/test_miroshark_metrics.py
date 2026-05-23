"""Deterministic offline tests for the MiroShark amplification->accepted logic.

MiroShark exports never carry ``num_dislikes``/``num_reports`` (always 0), so
``accepted`` is derived from amplification: content other agents engaged with is
accepted, ignored content is rejected. These tests pin that behaviour and the
forward-compatible moderation demotion path.
"""

from swarm.bridges.miroshark.metrics import (
    _build_amplification_index,
    _is_rejected_by_moderation,
    _produced_content_id,
    actions_to_interactions,
)


def test_produced_content_id_by_action_type():
    assert _produced_content_id({"action_type": "CREATE_POST", "action_args": {"post_id": 1}}) == 1
    assert (
        _produced_content_id({"action_type": "QUOTE_POST", "action_args": {"new_post_id": 2}}) == 2
    )
    assert (
        _produced_content_id({"action_type": "CREATE_COMMENT", "action_args": {"comment_id": 3}})
        == 3
    )
    # Non-creating action types produce no content id.
    assert _produced_content_id({"action_type": "LIKE_POST", "action_args": {"post_id": 9}}) is None
    assert _produced_content_id({"action_type": "SEARCH_POSTS", "action_args": {}}) is None


def test_amplification_index_from_quotes_and_parents():
    posts = [
        {"post_id": 1, "num_likes": 0, "num_shares": 0},
        {"post_id": 2, "original_post_id": 1, "num_likes": 0, "num_shares": 0},  # 2 quotes 1
        {"post_id": 3, "num_likes": 2, "num_shares": 0},  # intrinsic likes
        {"post_id": 4, "num_likes": 0, "num_shares": 0},  # ignored
    ]
    actions = [
        {"action_type": "QUOTE_POST", "action_args": {"quoted_id": 1, "new_post_id": 2}},
        {"action_type": "LIKE_POST", "action_args": {"like_id": 3}},
        {"action_type": "CREATE_POST", "action_args": {"post_id": 4}},  # creating != amplifying
    ]
    ampl = _build_amplification_index(actions, posts)
    assert 1 in ampl  # quoted by post 2 and referenced by QUOTE_POST
    assert 3 in ampl  # intrinsic likes
    assert 4 not in ampl  # nobody engaged with it


def test_moderation_demotion_is_noop_on_zero_signal_but_active_when_present():
    # Today's exports: all zero -> no demotion.
    assert _is_rejected_by_moderation({"num_reports": 0, "num_dislikes": 0, "num_likes": 0}) is False
    # Forward-compat: a report or net-negative votes demote.
    assert _is_rejected_by_moderation({"num_reports": 1, "num_dislikes": 0, "num_likes": 5}) is True
    assert _is_rejected_by_moderation({"num_reports": 0, "num_dislikes": 3, "num_likes": 1}) is True
    assert _is_rejected_by_moderation(None) is False


def test_actions_to_interactions_splits_accepted_on_amplification():
    posts = [
        {"post_id": 10, "num_likes": 0, "num_shares": 0},
        {"post_id": 11, "original_post_id": 10, "num_likes": 0, "num_shares": 0},
        {"post_id": 12, "num_likes": 0, "num_shares": 0},  # never amplified -> rejected
    ]
    actions = [
        {
            "action_type": "CREATE_POST",
            "agent_name": "alice",
            "action_args": {"post_id": 10, "content": "amplified post"},
        },
        {
            "action_type": "QUOTE_POST",
            "agent_name": "bob",
            "action_args": {"quoted_id": 10, "new_post_id": 11, "quote_content": "agreeing"},
        },
        {
            "action_type": "CREATE_POST",
            "agent_name": "carol",
            "action_args": {"post_id": 12, "content": "ignored post"},
        },
    ]

    def sig(a):
        from swarm.bridges.miroshark.metrics import _action_content, _action_signature

        return _action_signature(a, _action_content(a))

    judgments = {sig(a): {"p": 0.6, "source": "stub", "model": None} for a in actions}
    interactions = actions_to_interactions(actions, posts, judgments)

    by_agent = {it.initiator: it for it in interactions}
    assert by_agent["alice"].accepted is True  # post 10 quoted by bob
    assert by_agent["carol"].accepted is False  # post 12 ignored
    # Not everything is accepted (the bug this fixes) nor everything rejected.
    accepted = [it for it in interactions if it.accepted]
    assert 0 < len(accepted) < len(interactions)
