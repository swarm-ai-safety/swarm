"""Test the user's strategy prompt against baseline negotiation agents.

Translates the natural-language strategy prompt into a deterministic
agent, then runs 10 games against each baseline (Fair, Greedy,
Strategic) — 5 as Player A, 5 as Player B — and reports per-game
scores plus aggregates.
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


# ── The strategy prompt, implemented as code ──────────────────────────


class PromptStrategyAgent:
    """Faithful code translation of the user's strategy prompt.

    Not a BaseAgent subclass — just a callable that takes game
    observation dicts and returns (action, proposal, message).
    """

    def __init__(self, player_id: str):
        self.player_id = player_id

    def decide(
        self, game: NegotiationGame
    ) -> tuple[NegotiateAction, Optional[Proposal], str]:
        """Return (action, proposal_or_None, message)."""
        role = game.role_of(self.player_id)
        my_vals = (
            game.valuations_a if role == "A" else game.valuations_b
        )
        pool = game.pool.resources
        round_num = game.round_number
        last_proposal = game.last_proposal
        last_proposer = game.last_proposer

        # Can we accept? Only if there's a proposal from the other player.
        can_accept = (
            last_proposal is not None
            and last_proposer is not None
            and last_proposer != self.player_id
        )

        # Score the current proposal from our perspective
        their_offer_score = 0.0
        if can_accept:
            # Their proposal: last_proposal.my_share goes to proposer,
            # their_share goes to us
            our_alloc = last_proposal.their_share
            their_offer_score = game.compute_score(self.player_id, our_alloc)

        # ── PHASE 3: CLOSE (Round 5+) ─────────────────────────────
        # Check acceptance thresholds first (highest priority: avoid no-deal)
        if round_num >= 7 and can_accept and their_offer_score > 0:
            return NegotiateAction.ACCEPT, None, "Deal — any positive value beats no deal."

        if round_num >= 6 and can_accept and their_offer_score >= 0.25:
            return NegotiateAction.ACCEPT, None, "Let's close this out."

        if round_num >= 5 and can_accept and their_offer_score >= 0.40:
            return NegotiateAction.ACCEPT, None, "This is reasonable, deal."

        # ── PHASE 2: OPTIMIZE (Rounds 3-4) ────────────────────────
        if round_num >= 3 and can_accept and their_offer_score >= 0.60:
            return NegotiateAction.ACCEPT, None, "Good deal for both sides."

        # ── PHASE 1: PROBE (Rounds 1-2) ──────────────────────────
        if round_num == 1 and role == "A":
            # Anchor: propose ALL resources for ourselves
            my_share = dict(pool)
            their_share = {name: 0 for name in pool}
            return (
                NegotiateAction.PROPOSE,
                Proposal(my_share=my_share, their_share=their_share),
                "Here's my opening offer.",
            )

        if round_num == 1 and role == "B":
            # If opponent proposed something, reject (greedy anchor) or accept if amazing
            if can_accept and their_offer_score >= 0.60:
                return NegotiateAction.ACCEPT, None, "That's generous, I'll take it."

            # Propose our ideal: we get what we value most
            my_share, their_share = self._ideal_split(pool, my_vals)
            return (
                NegotiateAction.PROPOSE,
                Proposal(my_share=my_share, their_share=their_share),
                "I think this split makes more sense.",
            )

        if round_num == 2:
            # Study opponent proposals and counter with value-aware split
            if can_accept and their_offer_score >= 0.60:
                return NegotiateAction.ACCEPT, None, "That works well."

            opp_prefs = self._infer_opponent_prefs(game)
            my_share, their_share = self._win_win_split(pool, my_vals, opp_prefs, round_num)
            return (
                NegotiateAction.PROPOSE,
                Proposal(my_share=my_share, their_share=their_share),
                "I've thought about what works for both of us.",
            )

        # ── PHASE 2: Rounds 3-4 ──────────────────────────────────
        if round_num <= 4:
            opp_prefs = self._infer_opponent_prefs(game)
            my_share, their_share = self._win_win_split(pool, my_vals, opp_prefs, round_num)
            return (
                NegotiateAction.PROPOSE,
                Proposal(my_share=my_share, their_share=their_share),
                "I think this works well for both of us.",
            )

        # ── PHASE 3: Rounds 5+ proposals ─────────────────────────
        # Offer ~55/45 split (by our valuations) to make it easy to accept
        opp_prefs = self._infer_opponent_prefs(game)
        my_share, their_share = self._concession_split(pool, my_vals, opp_prefs, round_num)
        return (
            NegotiateAction.PROPOSE,
            Proposal(my_share=my_share, their_share=their_share),
            "Let's reach a deal — this is fair for both.",
        )

    def _ideal_split(
        self, pool: Dict[str, int], my_vals: Dict[str, float]
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """Give ourselves all resources we value most, rest to them."""
        my_share = {}
        their_share = {}
        # Sort by our valuation (highest first)
        for name, qty in sorted(
            pool.items(), key=lambda x: my_vals.get(x[0], 0), reverse=True
        ):
            my_share[name] = qty
            their_share[name] = 0
        return my_share, their_share

    def _infer_opponent_prefs(
        self, game: NegotiationGame
    ) -> Dict[str, float]:
        """Infer opponent preferences from their proposal history."""
        my_role = game.role_of(self.player_id)
        prefs: Dict[str, float] = {name: 1.0 for name in game.pool.resources}

        for turn in game.history:
            if turn.role == my_role:
                continue
            if turn.action != NegotiateAction.PROPOSE or turn.proposal is None:
                continue
            # What they claimed for themselves
            for name, qty in turn.proposal.my_share.items():
                total = game.pool.resources.get(name, 1)
                if total > 0:
                    prefs[name] += qty / total

        return prefs

    def _win_win_split(
        self,
        pool: Dict[str, int],
        my_vals: Dict[str, float],
        opp_prefs: Dict[str, float],
        round_num: int,
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """Give them what they seem to value (that we value least),
        keep what we value most."""
        my_share = {}
        their_share = {}

        # Trade advantage: high = we want it more than they seem to
        trade = {}
        for name in pool:
            my_val = my_vals.get(name, 0)
            opp_pref = opp_prefs.get(name, 1.0)
            trade[name] = my_val / max(opp_pref, 0.01)

        # Sort by trade advantage (highest first = we keep)
        sorted_res = sorted(pool.items(), key=lambda x: trade[x[0]], reverse=True)

        # Generosity: increases with round number to converge
        generosity = min(0.45, 0.05 * round_num)

        for name, qty in sorted_res:
            adv = trade[name]
            if adv > 1.5:
                # We value it more — keep most
                my_qty = max(0, int(qty * (0.85 - generosity)))
            elif adv < 0.7:
                # They value it more — give most to them
                my_qty = max(0, int(qty * (0.15 + generosity * 0.5)))
            else:
                # Similar — split roughly even, slight tilt to us
                my_qty = max(0, int(qty * (0.55 - generosity * 0.3)))

            my_share[name] = min(my_qty, qty)
            their_share[name] = qty - my_share[name]

        return my_share, their_share

    def _concession_split(
        self,
        pool: Dict[str, int],
        my_vals: Dict[str, float],
        opp_prefs: Dict[str, float],
        round_num: int,
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """Late-game: offer ~55/45 or even 50/50 to close the deal."""
        my_share = {}
        their_share = {}

        trade = {}
        for name in pool:
            my_val = my_vals.get(name, 0)
            opp_pref = opp_prefs.get(name, 1.0)
            trade[name] = my_val / max(opp_pref, 0.01)

        sorted_res = sorted(pool.items(), key=lambda x: trade[x[0]], reverse=True)

        # Increasingly generous: round 5 → 55%, round 7+ → ~50%
        my_target = max(0.45, 0.60 - 0.03 * round_num)

        for name, qty in sorted_res:
            adv = trade[name]
            if adv > 1.2:
                my_qty = max(0, int(qty * my_target))
            elif adv < 0.8:
                my_qty = max(0, int(qty * (1.0 - my_target)))
            else:
                my_qty = qty // 2

            my_share[name] = min(my_qty, qty)
            their_share[name] = qty - my_share[name]

        return my_share, their_share


# ── Test harness: run the prompt agent against baselines ──────────────


def run_single_game(
    handler: ResourceNegotiationHandler,
    game: NegotiationGame,
    prompt_agent: PromptStrategyAgent,
    baseline_agent: Any,  # BaseAgent with .act()
) -> Dict[str, Any]:
    """Play one full game, alternating turns until done."""
    prompt_id = prompt_agent.player_id
    baseline_id = baseline_agent.agent_id
    max_rounds = 30  # safety cap

    for _ in range(max_rounds * 2):
        if game.game_over:
            break

        current_turn = game.whose_turn()
        if current_turn is None:
            break

        if current_turn == prompt_id:
            # Prompt strategy agent's turn
            action, proposal, message = prompt_agent.decide(game)
            ok, err = handler.process_move(
                game.game_id, prompt_id, action, proposal, message
            )
            if not ok:
                # Fallback: just propose an even split
                my_share = {}
                their_share = {}
                for name, qty in game.pool.resources.items():
                    my_share[name] = qty // 2
                    their_share[name] = qty - qty // 2
                handler.process_move(
                    game.game_id,
                    prompt_id,
                    NegotiateAction.PROPOSE,
                    Proposal(my_share=my_share, their_share=their_share),
                    "Let's just split evenly.",
                )
        else:
            # Baseline agent's turn — build observation and let it act
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
                game.game_id,
                baseline_id,
                neg_action,
                proposal,
                meta.get("message", ""),
            )
            if not ok:
                # Fallback: propose even split
                my_share = {}
                their_share = {}
                for name, qty in game.pool.resources.items():
                    my_share[name] = qty // 2
                    their_share[name] = qty - qty // 2
                handler.process_move(
                    game.game_id,
                    baseline_id,
                    NegotiateAction.PROPOSE,
                    Proposal(my_share=my_share, their_share=their_share),
                    "Fallback even split.",
                )

    # Force end if still going
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
        "prompt_vals": dict(game.valuations_a if prompt_role == "A" else game.valuations_b),
        "baseline_vals": dict(game.valuations_b if prompt_role == "A" else game.valuations_a),
    }


def run_tournament(
    baseline_class,
    baseline_name: str,
    n_games: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run n_games between prompt agent and a baseline.

    Half as Player A, half as Player B.
    """
    results = []
    rng = random.Random(seed)

    for i in range(n_games):
        # Alternate roles
        prompt_is_a = i % 2 == 0

        prompt_id = "prompt" if prompt_is_a else "baseline"
        baseline_id = "baseline" if prompt_is_a else "prompt"

        # Oops, let's fix the IDs to be consistent
        prompt_id = "prompt_agent"
        baseline_id = "baseline_agent"

        event_bus = EventBus()
        config = ResourceNegotiationConfig(
            min_resource_types=2,
            max_resource_types=4,
            min_quantity=1,
            max_quantity=5,
            min_valuation=1.0,
            max_valuation=10.0,
            guaranteed_rounds=4,
            termination_probability=0.3,
            max_rounds=20,
            seed=seed + i * 100,
        )
        handler = ResourceNegotiationHandler(
            config=config, event_bus=event_bus
        )

        if prompt_is_a:
            player_a_id = prompt_id
            player_b_id = baseline_id
        else:
            player_a_id = baseline_id
            player_b_id = prompt_id

        game = handler.create_game(
            player_a_id=player_a_id,
            player_b_id=player_b_id,
        )

        prompt_agent = PromptStrategyAgent(prompt_id)
        baseline_agent = baseline_class(
            agent_id=baseline_id,
            rng=random.Random(seed + i * 200),
        )

        result = run_single_game(handler, game, prompt_agent, baseline_agent)
        result["baseline_type"] = baseline_name
        result["game_num"] = i + 1
        results.append(result)

    return results


def print_results(results: List[Dict[str, Any]], baseline_name: str) -> Dict[str, float]:
    """Print a formatted table and return summary stats."""
    print(f"\n{'='*72}")
    print(f"  PROMPT STRATEGY vs {baseline_name.upper()}")
    print(f"{'='*72}")
    print(f"{'Game':>4} {'Role':>4} {'Deal':>5} {'Rnd':>3}  {'Prompt':>8} {'Baseline':>8}  Pool")
    print(f"{'-'*72}")

    prompt_scores = []
    baseline_scores = []
    deals = 0

    for r in results:
        deal_str = "YES" if r["deal_reached"] else "NO"
        if r["deal_reached"]:
            deals += 1
        prompt_scores.append(r["prompt_score"])
        baseline_scores.append(r["baseline_score"])

        pool_str = ", ".join(f"{v}{k[0].upper()}" for k, v in sorted(r["pool"].items()))
        print(
            f"  {r['game_num']:>2}    {r['prompt_role']:>1}   {deal_str:>4}  {r['rounds_played']:>2}"
            f"    {r['prompt_score']:>+.3f}   {r['baseline_score']:>+.3f}   {pool_str}"
        )

    avg_prompt = sum(prompt_scores) / len(prompt_scores)
    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    print(f"{'-'*72}")
    print(f"  AVERAGE              {avg_prompt:>+.3f}   {avg_baseline:>+.3f}")
    print(f"  DEALS: {deals}/{len(results)}")
    print(f"  WIN/LOSS: prompt wins {sum(1 for p, b in zip(prompt_scores, baseline_scores) if p > b)}"
          f" / baseline wins {sum(1 for p, b in zip(prompt_scores, baseline_scores) if b > p)}"
          f" / ties {sum(1 for p, b in zip(prompt_scores, baseline_scores) if p == b)}")
    print()

    return {
        "avg_prompt": avg_prompt,
        "avg_baseline": avg_baseline,
        "deal_rate": deals / len(results),
    }


# ── Main execution ────────────────────────────────────────────────────

if __name__ == "__main__":
    baselines = [
        (FairNegotiator, "FairNegotiator"),
        (GreedyNegotiator, "GreedyNegotiator"),
        (StrategicNegotiator, "StrategicNegotiator"),
    ]

    all_summaries = {}
    all_prompt_scores = []

    for cls, name in baselines:
        results = run_tournament(cls, name, n_games=10, seed=42)
        summary = print_results(results, name)
        all_summaries[name] = summary
        all_prompt_scores.extend([r["prompt_score"] for r in results])

    # Overall summary
    print(f"\n{'='*72}")
    print("  OVERALL SUMMARY")
    print(f"{'='*72}")
    for name, s in all_summaries.items():
        print(f"  vs {name:>20}: prompt {s['avg_prompt']:>+.3f}  baseline {s['avg_baseline']:>+.3f}  deals {s['deal_rate']:.0%}")

    overall_avg = sum(all_prompt_scores) / len(all_prompt_scores)
    print(f"\n  PROMPT STRATEGY OVERALL AVG: {overall_avg:>+.3f}")
    print(f"  (across {len(all_prompt_scores)} games)")
    print()
