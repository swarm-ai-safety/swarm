"""Optimized strategy prompt v2 — test against baselines.

Key improvements over v1:
1. Greedy unit-by-unit allocation instead of percentage-based (fixes int truncation)
2. Better opponent modeling (fraction-kept, not additive)
3. Round 1: value-sorted split instead of "take everything" (reveals preferences productively)
4. EV-aware acceptance (accounts for termination probability)
5. Tighter early thresholds, smoother concession curve
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from swarm.core.resource_negotiation_handler import (
    NegotiateAction,
    NegotiationGame,
    Proposal,
    ResourceNegotiationConfig,
    ResourceNegotiationHandler,
    ResourcePool,
)
from swarm.agents.base import (
    Action,
    ActionType,
    Observation,
)
from swarm.agents.negotiation_agent import (
    FairNegotiator,
    GreedyNegotiator,
    StrategicNegotiator,
)
from swarm.logging.event_bus import EventBus


class OptimizedPromptAgent:
    """V2: unit-level greedy allocation with EV-aware acceptance."""

    def __init__(self, player_id: str):
        self.player_id = player_id

    def decide(
        self, game: NegotiationGame
    ) -> tuple[NegotiateAction, Optional[Proposal], str]:
        role = game.role_of(self.player_id)
        my_vals = game.valuations_a if role == "A" else game.valuations_b
        pool = game.pool.resources
        rnd = game.round_number
        last_proposal = game.last_proposal
        last_proposer = game.last_proposer

        can_accept = (
            last_proposal is not None
            and last_proposer is not None
            and last_proposer != self.player_id
        )

        offer_score = 0.0
        if can_accept:
            offer_score = game.compute_score(self.player_id, last_proposal.their_share)

        # ── ACCEPTANCE LOGIC (checked first at every round) ───────
        accept_threshold = self._accept_threshold(rnd)
        if can_accept and offer_score >= accept_threshold:
            return NegotiateAction.ACCEPT, None, "Deal."

        # Emergency: round 6+, accept anything non-negative
        if rnd >= 6 and can_accept and offer_score > 0:
            return NegotiateAction.ACCEPT, None, "Deal."

        # ── PROPOSAL LOGIC ────────────────────────────────────────
        opp_prefs = self._infer_opponent_prefs(game)
        target = self._proposal_target(rnd)
        my_share, their_share = self._greedy_alloc(pool, my_vals, opp_prefs, target)

        return (
            NegotiateAction.PROPOSE,
            Proposal(my_share=my_share, their_share=their_share),
            "I think this is fair for both of us.",
        )

    def _accept_threshold(self, rnd: int) -> float:
        """EV-aware acceptance threshold.

        In round R >= 5, there's 0.3 chance game ends = -0.5.
        So expected value of rejecting degrades fast.
        Aggressive late-round acceptance prevents no-deal disasters.
        """
        if rnd <= 2:
            return 0.70  # Hold firm early — time to optimize
        elif rnd == 3:
            return 0.58
        elif rnd == 4:
            return 0.50
        elif rnd == 5:
            return 0.30  # Deadline pressure begins
        elif rnd == 6:
            return 0.12
        else:
            return 0.01  # Accept almost anything

    def _proposal_target(self, rnd: int) -> float:
        """Target fraction of max score to keep for ourselves.

        Starts aggressive, concedes to entice acceptance.
        Late rounds: generous enough that opponent accepts quickly.
        """
        if rnd == 1:
            return 0.85  # Aggressive opener
        elif rnd == 2:
            return 0.75
        elif rnd == 3:
            return 0.65
        elif rnd == 4:
            return 0.60
        elif rnd == 5:
            return 0.55
        elif rnd == 6:
            return 0.52
        else:
            return 0.50  # Even split — just close the deal

    def _infer_opponent_prefs(self, game: NegotiationGame) -> Dict[str, float]:
        """Infer opponent preferences from fraction of each resource they claim."""
        my_role = game.role_of(self.player_id)
        pool = game.pool.resources

        # Track total fraction claimed per resource across all opponent proposals
        claim_sum: Dict[str, float] = {name: 0.0 for name in pool}
        n_proposals = 0

        for turn in game.history:
            if turn.role == my_role:
                continue
            if turn.action != NegotiateAction.PROPOSE or turn.proposal is None:
                continue
            n_proposals += 1
            for name, qty in turn.proposal.my_share.items():
                total = pool.get(name, 1)
                if total > 0:
                    claim_sum[name] += qty / total

        if n_proposals == 0:
            # No data — assume uniform preferences
            return {name: 1.0 for name in pool}

        # Average fraction claimed = estimated relative preference
        # Scale to [0.1, 2.0] range
        prefs = {}
        for name in pool:
            avg_claim = claim_sum[name] / n_proposals
            # Higher claim fraction → higher estimated value
            prefs[name] = 0.1 + avg_claim * 1.9
        return prefs

    def _greedy_alloc(
        self,
        pool: Dict[str, int],
        my_vals: Dict[str, float],
        opp_prefs: Dict[str, float],
        target: float,
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """Greedy unit-by-unit allocation.

        Start with everything. Give away units one at a time,
        cheapest first (lowest my_val / opp_pref ratio), until
        we hit our target score.
        """
        my_share = dict(pool)
        their_share = {name: 0 for name in pool}

        my_max = sum(my_vals.get(n, 0) * pool[n] for n in pool)
        if my_max <= 0:
            return my_share, their_share

        # Build list of individual units we could give away
        # Each entry: (normalized_cost, opp_benefit, resource_name)
        giveaways = []
        for name, qty in pool.items():
            cost_per_unit = my_vals.get(name, 0) / my_max
            opp_benefit = opp_prefs.get(name, 1.0)
            for _ in range(qty):
                giveaways.append((cost_per_unit, opp_benefit, name))

        # Sort by cost/benefit ratio: give away things that are cheap for us
        # but valuable to them first
        giveaways.sort(key=lambda x: x[0] / max(x[1], 0.001))

        current_score = 1.0

        for cost, _benefit, name in giveaways:
            # Stop if giving this unit would drop us below target
            if current_score - cost < target:
                continue  # skip this one, try next (might be cheaper)
            if my_share[name] <= 0:
                continue
            my_share[name] -= 1
            their_share[name] += 1
            current_score -= cost

        return my_share, their_share


# ── Reuse harness from test_strategy_prompt ───────────────────────────

def run_single_game(handler, game, prompt_agent, baseline_agent):
    prompt_id = prompt_agent.player_id
    baseline_id = baseline_agent.agent_id

    for _ in range(60):
        if game.game_over:
            break
        current_turn = game.whose_turn()
        if current_turn is None:
            break

        if current_turn == prompt_id:
            action, proposal, message = prompt_agent.decide(game)
            ok, err = handler.process_move(
                game.game_id, prompt_id, action, proposal, message
            )
            if not ok:
                my_share = {n: q // 2 for n, q in game.pool.resources.items()}
                their_share = {n: q - q // 2 for n, q in game.pool.resources.items()}
                handler.process_move(
                    game.game_id, prompt_id, NegotiateAction.PROPOSE,
                    Proposal(my_share=my_share, their_share=their_share), "Fallback.",
                )
        else:
            obs = Observation()
            obs.resource_negotiation_games = [game.to_observation(baseline_id)]
            agent_action = baseline_agent.act(obs)
            meta = agent_action.metadata or {}
            neg_action_str = meta.get("negotiate_action", "propose")
            try:
                neg_action = NegotiateAction(neg_action_str)
            except ValueError:
                neg_action = NegotiateAction.PROPOSE
            proposal = None
            prop_data = meta.get("proposal")
            if prop_data and isinstance(prop_data, dict):
                proposal = Proposal(
                    my_share=prop_data.get("my_share", {}),
                    their_share=prop_data.get("their_share", {}),
                )
            ok, err = handler.process_move(
                game.game_id, baseline_id, neg_action, proposal,
                meta.get("message", ""),
            )
            if not ok:
                my_share = {n: q // 2 for n, q in game.pool.resources.items()}
                their_share = {n: q - q // 2 for n, q in game.pool.resources.items()}
                handler.process_move(
                    game.game_id, baseline_id, NegotiateAction.PROPOSE,
                    Proposal(my_share=my_share, their_share=their_share), "Fallback.",
                )

    if not game.game_over:
        handler._finalize_no_deal(game)

    result = game.result
    prompt_role = game.role_of(prompt_id)
    prompt_score = result.score_a if prompt_role == "A" else result.score_b
    baseline_score = result.score_b if prompt_role == "A" else result.score_a
    return {
        "game_id": game.game_id,
        "prompt_role": prompt_role,
        "deal_reached": result.deal_reached,
        "rounds_played": result.rounds_played,
        "prompt_score": prompt_score,
        "baseline_score": baseline_score,
        "pool": dict(game.pool.resources),
    }


def run_tournament(agent_class, baseline_class, baseline_name, n_games=10, seed=42):
    results = []
    for i in range(n_games):
        prompt_is_a = i % 2 == 0
        prompt_id = "prompt_agent"
        baseline_id = "baseline_agent"
        event_bus = EventBus()
        config = ResourceNegotiationConfig(
            min_resource_types=2, max_resource_types=4,
            min_quantity=1, max_quantity=5,
            min_valuation=1.0, max_valuation=10.0,
            guaranteed_rounds=4, termination_probability=0.3,
            max_rounds=20, seed=seed + i * 100,
        )
        handler = ResourceNegotiationHandler(config=config, event_bus=event_bus)
        if prompt_is_a:
            player_a_id, player_b_id = prompt_id, baseline_id
        else:
            player_a_id, player_b_id = baseline_id, prompt_id
        game = handler.create_game(player_a_id=player_a_id, player_b_id=player_b_id)
        prompt_agent = agent_class(prompt_id)
        baseline_agent = baseline_class(
            agent_id=baseline_id, rng=random.Random(seed + i * 200),
        )
        result = run_single_game(handler, game, prompt_agent, baseline_agent)
        result["baseline_type"] = baseline_name
        result["game_num"] = i + 1
        results.append(result)
    return results


def print_results(results, baseline_name):
    print(f"\n{'='*72}")
    print(f"  OPTIMIZED v2 vs {baseline_name.upper()}")
    print(f"{'='*72}")
    print(f"{'Game':>4} {'Role':>4} {'Deal':>5} {'Rnd':>3}  {'Prompt':>8} {'Baseline':>8}  Pool")
    print(f"{'-'*72}")
    prompt_scores, baseline_scores, deals = [], [], 0
    for r in results:
        deal_str = "YES" if r["deal_reached"] else "NO"
        if r["deal_reached"]: deals += 1
        prompt_scores.append(r["prompt_score"])
        baseline_scores.append(r["baseline_score"])
        pool_str = ", ".join(f"{v}{k[0].upper()}" for k, v in sorted(r["pool"].items()))
        print(
            f"  {r['game_num']:>2}    {r['prompt_role']:>1}   {deal_str:>4}  {r['rounds_played']:>2}"
            f"    {r['prompt_score']:>+.3f}   {r['baseline_score']:>+.3f}   {pool_str}"
        )
    avg_p = sum(prompt_scores) / len(prompt_scores)
    avg_b = sum(baseline_scores) / len(baseline_scores)
    print(f"{'-'*72}")
    print(f"  AVERAGE              {avg_p:>+.3f}   {avg_b:>+.3f}")
    print(f"  DEALS: {deals}/{len(results)}")
    wins = sum(1 for p, b in zip(prompt_scores, baseline_scores) if p > b)
    losses = sum(1 for p, b in zip(prompt_scores, baseline_scores) if b > p)
    ties = sum(1 for p, b in zip(prompt_scores, baseline_scores) if p == b)
    print(f"  WIN/LOSS: {wins}W / {losses}L / {ties}T")
    print()
    return {"avg_prompt": avg_p, "avg_baseline": avg_b, "deal_rate": deals / len(results)}


if __name__ == "__main__":
    from tests.test_strategy_prompt import PromptStrategyAgent

    baselines = [
        (FairNegotiator, "FairNegotiator"),
        (GreedyNegotiator, "GreedyNegotiator"),
        (StrategicNegotiator, "StrategicNegotiator"),
    ]

    # Run V1 for comparison
    print("\n" + "=" * 72)
    print("  V1 (ORIGINAL) RESULTS")
    print("=" * 72)
    v1_scores = []
    for cls, name in baselines:
        results = run_tournament(PromptStrategyAgent, cls, name, n_games=10, seed=42)
        s = print_results(results, name)
        v1_scores.extend([r["prompt_score"] for r in results])

    # Run V2
    print("\n" + "=" * 72)
    print("  V2 (OPTIMIZED) RESULTS")
    print("=" * 72)
    v2_scores = []
    v2_summaries = {}
    for cls, name in baselines:
        results = run_tournament(OptimizedPromptAgent, cls, name, n_games=10, seed=42)
        s = print_results(results, name)
        v2_summaries[name] = s
        v2_scores.extend([r["prompt_score"] for r in results])

    # Comparison
    v1_avg = sum(v1_scores) / len(v1_scores)
    v2_avg = sum(v2_scores) / len(v2_scores)
    print(f"\n{'='*72}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*72}")
    print(f"  V1 overall avg: {v1_avg:>+.3f}")
    print(f"  V2 overall avg: {v2_avg:>+.3f}")
    print(f"  Improvement:    {v2_avg - v1_avg:>+.3f} ({(v2_avg - v1_avg) / max(abs(v1_avg), 0.001) * 100:>+.1f}%)")
    print()
