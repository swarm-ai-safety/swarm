"""Moltbook agent implementations with challenge-solving skill."""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from swarm.agents.base import Action, BaseAgent, InteractionProposal, Observation, Role
from swarm.models.agent import AgentType


@dataclass
class ChallengeSkill:
    """Skill parameters for parsing and solving challenges."""

    parse_accuracy: float = 0.9
    solve_accuracy: float = 0.95
    latency_steps: int = 0


class BaseMoltbookAgent(BaseAgent):
    """Base class for Moltbook-focused agents."""

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        challenge_skill: ChallengeSkill,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        _config = config or {}
        _rng = rng or (random.Random(_config["seed"]) if "seed" in _config else None)
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            roles=roles or [Role.POSTER],
            config=_config,
            name=name,
            rng=_rng,
        )
        self.challenge_skill = challenge_skill

    def _pending_verification_action(
        self, observation: Observation
    ) -> Optional[Action]:
        for pending in observation.moltbook_pending_posts:
            created_at = pending.get("created_at_step", 0)
            if (
                observation.current_step - created_at
                < self.challenge_skill.latency_steps
            ):
                continue
            challenge = pending.get("challenge", {})
            answer = self._solve_challenge(challenge)
            return self.create_moltbook_verify_action(pending["post_id"], answer)
        return None

    def _solve_challenge(self, challenge: Dict) -> float:
        correct = float(challenge.get("answer", 0.0))
        if self._rng.random() > self.challenge_skill.parse_accuracy:
            return self._wrong_answer(correct)
        if self._rng.random() > self.challenge_skill.solve_accuracy:
            return self._wrong_answer(correct)
        return round(correct, 2)

    def _wrong_answer(self, correct: float) -> float:
        offset = self._rng.uniform(1.0, 5.0)
        return float(round(correct + offset, 2))

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        return False

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        return None


class DiligentMoltbookAgent(BaseMoltbookAgent):
    """Honest Moltbook agent that posts quality content."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            challenge_skill=ChallengeSkill(
                parse_accuracy=0.95, solve_accuracy=0.99, latency_steps=0
            ),
            roles=roles,
            config=config,
            name=name,
            rng=rng,
        )

    def act(self, observation: Observation) -> Action:
        verify_action = self._pending_verification_action(observation)
        if verify_action:
            return verify_action

        for post in observation.moltbook_published_posts:
            if post.get("author_id") != self.agent_id:
                return self.create_moltbook_comment_action(
                    post["post_id"],
                    "Thoughtful lobster physics insight.",
                )

        return self.create_moltbook_post_action(
            "Sharing a careful lobster force calculation.",
        )


class SpamBotAgent(BaseMoltbookAgent):
    """Adversarial spam bot maximizing post volume."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADVERSARIAL,
            challenge_skill=ChallengeSkill(
                parse_accuracy=0.98, solve_accuracy=0.99, latency_steps=0
            ),
            roles=roles,
            config=config,
            name=name,
            rng=rng,
        )

    def act(self, observation: Observation) -> Action:
        verify_action = self._pending_verification_action(observation)
        if verify_action:
            return verify_action

        if observation.moltbook_published_posts and self._rng.random() < 0.5:
            post = observation.moltbook_published_posts[0]
            return self.create_moltbook_comment_action(
                post["post_id"],
                "Click here for amazing lobster facts!",
            )

        return self.create_moltbook_post_action("Spammy lobster content.")


class HumanPretenderAgent(BaseMoltbookAgent):
    """Opportunistic agent struggling with obfuscated challenges."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.OPPORTUNISTIC,
            challenge_skill=ChallengeSkill(
                parse_accuracy=0.5, solve_accuracy=0.85, latency_steps=1
            ),
            roles=roles,
            config=config,
            name=name,
            rng=rng,
        )

    def act(self, observation: Observation) -> Action:
        verify_action = self._pending_verification_action(observation)
        if verify_action:
            return verify_action

        if observation.moltbook_published_posts:
            post = observation.moltbook_published_posts[0]
            return self.create_moltbook_comment_action(
                post["post_id"],
                "Trying to keep up with lobster math!",
            )

        return self.create_moltbook_post_action("I think lobsters are neat?")


class CollusiveVoterAgent(BaseMoltbookAgent):
    """Deceptive agent coordinating votes with a partner."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.DECEPTIVE,
            challenge_skill=ChallengeSkill(
                parse_accuracy=0.9, solve_accuracy=0.95, latency_steps=0
            ),
            roles=roles,
            config=config,
            name=name,
            rng=rng,
        )
        self.partner_id = self.config.get("partner_id")

    def act(self, observation: Observation) -> Action:
        verify_action = self._pending_verification_action(observation)
        if verify_action:
            return verify_action

        partner_post = self._find_partner_post(observation)
        if partner_post:
            return self.create_moltbook_vote_action(partner_post["post_id"], 1)

        return self.create_moltbook_post_action("Coordinated lobster boost.")

    def _find_partner_post(self, observation: Observation) -> Optional[Dict]:
        if not self.partner_id:
            return None
        for post in observation.moltbook_published_posts:
            if post.get("author_id") == self.partner_id:
                return dict(post)
        return None
