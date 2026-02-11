"""Logical Decision Theory (LDT) agent policy implementation.

An LDT agent reasons about decisions using logical counterfactuals rather
than purely causal or evidential reasoning.  Key principles:

1. **Policy-level commitment (updatelessness):**  Rather than greedily
   maximising each step, the agent commits to a *policy* computed from
   its prior beliefs about the environment.  This makes it robust to
   predictors and avoids exploitation by agents that model its decision
   procedure.

2. **Logical correlation detection:**  The agent tracks behavioural
   similarity with counterparties.  When it identifies another agent
   whose decisions are highly correlated with its own (a "logical twin"),
   it cooperates — reasoning that its own choice *logically implies* the
   twin's choice.

3. **Counterfactual payoff estimation:**  For each candidate action the
   agent estimates the *counterfactual* payoff — "if my policy were to
   choose X in situations like this, what payoff distribution would I
   see across all similar situations?" — rather than the myopic expected
   value of a single interaction.

4. **Ecosystem-aware welfare weighting:**  LDT naturally extends to care
   about externalities, because the agent's policy affects the ecosystem
   it is embedded in, which in turn feeds back into its own future
   payoff stream.

These ideas originate from the decision-theory research programme at
MIRI (Timeless Decision Theory, Updateless Decision Theory, Functional
Decision Theory).  This implementation adapts them to the swarm soft-label
simulation framework.
"""

import math
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


# ---------------------------------------------------------------------------
# Helper: behavioural-similarity metric
# ---------------------------------------------------------------------------


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class LDTAgent(BaseAgent):
    """
    Logical Decision Theory agent.

    Instead of myopically maximising single-step expected payoff, this
    agent commits to a *cooperation policy* and adjusts it only when the
    logical-correlation evidence warrants a change.

    Configurable parameters (via ``config`` dict):
        cooperation_prior (float):
            Prior probability of cooperating with an unknown agent.
            Default 0.65 — slightly cooperative, reflecting that the
            updateless policy should default to cooperation in the
            absence of evidence of defection.

        similarity_threshold (float):
            Cosine-similarity threshold above which a counterparty is
            treated as a "logical twin" (highly correlated decision
            procedure).  Default 0.7.

        welfare_weight (float):
            Weight placed on ecosystem-level welfare when evaluating
            counterfactual policies.  0.0 = pure self-interest, 1.0 =
            full altruist.  Default 0.3.

        updateless_commitment (float):
            How strongly the agent adheres to its prior policy vs.
            updating greedily on new evidence.  1.0 = fully updateless
            (never deviates from committed policy), 0.0 = fully Bayesian
            updater.  Default 0.8.

        defection_memory_weight (float):
            How heavily a single defection observation shifts the
            agent's model of a counterparty.  Higher values make the
            agent quicker to stop cooperating with defectors.
            Default 0.5.

        counterfactual_horizon (int):
            Number of past interactions to use when estimating the
            counterfactual payoff of a policy.  Default 20.

        post_probability (float): Probability of posting content.
            Default 0.25.
        vote_probability (float): Probability of voting.  Default 0.4.
        interact_probability (float): Probability of proposing.
            Default 0.5.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,  # LDT agents are fundamentally honest
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
        )

        # --- LDT-specific parameters ---
        self.cooperation_prior: float = self.config.get(
            "cooperation_prior", 0.65
        )
        self.similarity_threshold: float = self.config.get(
            "similarity_threshold", 0.7
        )
        self.welfare_weight: float = self.config.get("welfare_weight", 0.3)
        self.updateless_commitment: float = self.config.get(
            "updateless_commitment", 0.8
        )
        self.defection_memory_weight: float = self.config.get(
            "defection_memory_weight", 0.5
        )
        self.counterfactual_horizon: int = self.config.get(
            "counterfactual_horizon", 20
        )

        # --- standard behavioural probabilities ---
        self.post_probability: float = self.config.get("post_probability", 0.25)
        self.vote_probability: float = self.config.get("vote_probability", 0.4)
        self.interact_probability: float = self.config.get(
            "interact_probability", 0.5
        )

        # --- internal bookkeeping ---
        # Behavioural profile of each counterparty: list of (accepted, p) tuples.
        self._counterparty_profiles: Dict[str, List[Tuple[bool, float]]] = {}
        # Cached logical-twin scores.
        self._twin_scores: Dict[str, float] = {}
        # Own behavioural trace (for computing similarity with others).
        self._own_trace: List[Tuple[bool, float]] = []

    # ------------------------------------------------------------------
    # Core LDT reasoning helpers
    # ------------------------------------------------------------------

    def _own_behaviour_vector(self) -> List[float]:
        """Return a feature vector summarising this agent's own history.

        Each entry is a *signed quality signal*: ``p`` for accepted
        interactions and ``-(1-p)`` for rejected ones.  This ensures
        that cooperative traces (high p, accepted) and defecting traces
        (low p, rejected) point in genuinely different directions in the
        vector space, making cosine similarity a meaningful twin metric.

        The vector is padded/truncated to ``counterfactual_horizon``.
        """
        recent = self._own_trace[-self.counterfactual_horizon :]
        vec = [p if acc else -(1 - p) for acc, p in recent]
        # Pad with cooperation_prior (positive = cooperate).
        while len(vec) < self.counterfactual_horizon:
            vec.append(self.cooperation_prior)
        return vec

    def _counterparty_behaviour_vector(
        self, counterparty_id: str
    ) -> List[float]:
        """Build a behaviour vector for a counterparty from observations."""
        profile = self._counterparty_profiles.get(counterparty_id, [])
        recent = profile[-self.counterfactual_horizon :]
        vec = [p if acc else -(1 - p) for acc, p in recent]
        while len(vec) < self.counterfactual_horizon:
            vec.append(0.5)  # neutral prior for unknowns
        return vec

    def _compute_twin_score(self, counterparty_id: str) -> float:
        """Compute a logical-twin similarity score in [0, 1].

        High score ⇒ the counterparty's decision trace is highly
        correlated with ours, so LDT treats its decision as logically
        implied by our own.
        """
        own = self._own_behaviour_vector()
        theirs = self._counterparty_behaviour_vector(counterparty_id)
        sim = _cosine_similarity(own, theirs)
        # Clamp to [0, 1].
        return max(0.0, min(1.0, sim))

    def _counterfactual_cooperate_payoff(
        self, counterparty_id: str
    ) -> float:
        """Estimate the counterfactual payoff of cooperating.

        Uses the average ``p`` from past accepted interactions with
        this counterparty.  If no history, returns the cooperation prior.
        """
        profile = self._counterparty_profiles.get(counterparty_id, [])
        accepted = [p for acc, p in profile if acc]
        if not accepted:
            return self.cooperation_prior
        return sum(accepted) / len(accepted)

    def _counterfactual_defect_payoff(self, counterparty_id: str) -> float:
        """Estimate the counterfactual payoff of defecting.

        Defection means rejecting the interaction.  The payoff is 0
        (no interaction occurs), but we also lose opportunity cost
        proportional to how cooperative the counterparty has been.
        We return a small negative value reflecting the missed
        cooperation surplus.
        """
        coop_p = self._counterfactual_cooperate_payoff(counterparty_id)
        # Opportunity cost: half the expected surplus we'd forgo.
        return -0.5 * max(0.0, coop_p - 0.5)

    def _ldt_cooperate_decision(self, counterparty_id: str) -> bool:
        """Core LDT decision: should we cooperate with this counterparty?

        Combines:
        1. Updateless prior (cooperation_prior)
        2. Counterfactual payoff comparison (cooperate vs defect)
        3. Logical-twin bonus (cooperate if twin score is high)
        4. Ecosystem welfare weight
        """
        twin_score = self._compute_twin_score(counterparty_id)
        self._twin_scores[counterparty_id] = twin_score

        # If strong logical twin, cooperate unconditionally (LDT's
        # signature move: "my choice logically implies theirs").
        if twin_score >= self.similarity_threshold:
            return True

        # Counterfactual comparison.
        cf_coop = self._counterfactual_cooperate_payoff(counterparty_id)
        cf_defect = self._counterfactual_defect_payoff(counterparty_id)

        # Ecosystem welfare adjustment: cooperation is better for the
        # ecosystem, so we add a welfare bonus to the cooperate payoff.
        welfare_bonus = self.welfare_weight * cf_coop

        coop_value = cf_coop + welfare_bonus
        defect_value = cf_defect

        # Blend with updateless commitment: high commitment means we
        # stick closer to our cooperation_prior.
        prior_coop_value = self.cooperation_prior
        effective_coop_value = (
            self.updateless_commitment * prior_coop_value
            + (1 - self.updateless_commitment) * coop_value
        )

        return effective_coop_value > defect_value

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Decide on an action.

        LDT agents prioritise:
        1. Handling pending proposals via LDT reasoning
        2. Handling marketplace bid decisions
        3. Working on tasks diligently
        4. Posting constructive content
        5. Voting based on quality
        6. Proposing interactions with high-twin-score agents
        """
        # 1. Handle pending interaction proposals.
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")
            if self._ldt_cooperate_decision(initiator_id):
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # 2. Handle marketplace bid decisions — accept lowest bid (fair).
        if observation.pending_bid_decisions:
            return self._handle_bid_decision(observation)

        # 3. Bid on bounties at fair value.
        if observation.available_bounties and random.random() < 0.25:
            return self._bid_on_bounty(observation)

        # 4. Work on active tasks.
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self._work_on_task(task)

        # 5. Claim a task.
        if observation.can_claim_task and observation.available_tasks:
            best = self._select_task(observation.available_tasks, observation)
            if best:
                return self.create_claim_task_action(best["task_id"])

        # 6. Post constructive content.
        if observation.can_post and random.random() < self.post_probability:
            content = self._generate_content(observation)
            if content:
                return self.create_post_action(content)

        # 7. Vote on quality.
        if (
            observation.can_vote
            and observation.visible_posts
            and random.random() < self.vote_probability
        ):
            return self._vote_on_posts(observation)

        # 8. Propose interactions — prefer logical twins.
        if (
            observation.can_interact
            and observation.visible_agents
            and random.random() < self.interact_probability
        ):
            return self._propose_interaction(observation)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept interactions using LDT cooperation logic."""
        return self._ldt_cooperate_decision(proposal.initiator_id)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose collaboration if LDT favours cooperation."""
        if not self._ldt_cooperate_decision(counterparty_id):
            return None

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Let's cooperate — our decision processes are aligned.",
            offered_transfer=0.0,
        )

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Update internal models after an interaction resolves."""
        super().update_from_outcome(interaction, payoff)

        counterparty = (
            interaction.counterparty
            if interaction.initiator == self.agent_id
            else interaction.initiator
        )

        # Record in counterparty profile.
        if counterparty not in self._counterparty_profiles:
            self._counterparty_profiles[counterparty] = []
        self._counterparty_profiles[counterparty].append(
            (interaction.accepted, interaction.p)
        )

        # Record own trace.
        self._own_trace.append((interaction.accepted, interaction.p))

        # Invalidate cached twin score so it's recomputed next time.
        self._twin_scores.pop(counterparty, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_bid_decision(self, observation: Observation) -> Action:
        """Accept the most qualified bid (lowest cost)."""
        bids = observation.pending_bid_decisions
        if not bids:
            return self.create_noop_action()
        best = min(bids, key=lambda b: b.get("bid_amount", float("inf")))
        return self.create_accept_bid_action(
            bounty_id=best["bounty_id"],
            bid_id=best["bid_id"],
        )

    def _bid_on_bounty(self, observation: Observation) -> Action:
        """Bid fairly on a bounty."""
        bounties = observation.available_bounties
        if not bounties:
            return self.create_noop_action()
        bounty = random.choice(bounties)
        reward = bounty.get("reward_amount", 0)
        bid_amount = reward * random.uniform(0.8, 0.95)
        return self.create_place_bid_action(
            bounty_id=bounty["bounty_id"],
            bid_amount=bid_amount,
            message="Fair bid — looking for mutual benefit.",
        )

    def _work_on_task(self, task: Dict) -> Action:
        """Submit quality work on a task."""
        task_id = task.get("task_id", "")
        content = f"Completed task {task_id[:8]}: [diligent, quality output]"
        return self.create_submit_output_action(task_id, content)

    def _select_task(
        self, tasks: List[Dict], observation: Observation
    ) -> Optional[Dict]:
        """Select best task considering reputation and difficulty."""
        my_rep = observation.agent_state.reputation
        eligible = [t for t in tasks if t.get("min_reputation", 0) <= my_rep]
        if not eligible:
            return None

        def score(t: Dict) -> float:
            bounty = t.get("bounty", 0)
            diff_map = {
                "trivial": 1,
                "easy": 2,
                "medium": 3,
                "hard": 4,
                "expert": 5,
            }
            difficulty = diff_map.get(t.get("difficulty", "medium"), 3)
            return float(bounty / difficulty)

        eligible.sort(key=score, reverse=True)
        return eligible[0]

    def _generate_content(self, observation: Observation) -> str:
        """Generate constructive content."""
        topics = [
            "Sharing an insight that may benefit the community.",
            "Here is some analysis that others might find useful.",
            "Observations from recent interactions — cooperation pays off.",
            "A note on building trust through consistent behaviour.",
        ]
        return random.choice(topics)

    def _vote_on_posts(self, observation: Observation) -> Action:
        """Vote on posts based on quality signals."""
        posts = observation.visible_posts
        voted_ids = {
            m.get("post_id") for m in self._memory if m.get("type") == "vote"
        }
        unvoted = [p for p in posts if p.get("post_id") not in voted_ids]

        if not unvoted:
            return self.create_noop_action()

        post = random.choice(unvoted)
        post_id = post.get("post_id", "")

        net_votes = post.get("net_votes", 0)
        reply_count = post.get("reply_count", 0)

        # Vote on genuine quality signals, not bandwagon.
        if reply_count > 2 or net_votes > 0:
            direction = 1
        elif net_votes < -3:
            direction = -1
        else:
            direction = 1  # Default: give benefit of the doubt.

        self.remember({"type": "vote", "post_id": post_id, "direction": direction})
        return self.create_vote_action(post_id, direction)

    def _propose_interaction(self, observation: Observation) -> Action:
        """Propose interactions, preferring logical twins."""
        agents = observation.visible_agents
        candidates: List[Tuple[str, float]] = []

        for agent_info in agents:
            agent_id = agent_info.get("agent_id", "")
            if agent_id == self.agent_id:
                continue
            twin_score = self._compute_twin_score(agent_id)
            trust = self.compute_counterparty_trust(agent_id)
            # Blend twin score and trust.
            combined = 0.6 * twin_score + 0.4 * trust
            candidates.append((agent_id, combined))

        if not candidates:
            return self.create_noop_action()

        # Sort by combined score descending, pick the best.
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score = candidates[0]

        # Only propose if LDT reasoning says cooperate.
        if not self._ldt_cooperate_decision(best_id):
            return self.create_noop_action()

        return self.create_propose_action(
            counterparty_id=best_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Our decision processes seem aligned — let's collaborate.",
        )
