"""Mixin that adds evolving skills to any agent type.

Adapts the recursive skill-augmented RL approach from:
    Xia, P. et al. (2026). SkillRL: Evolving Agents via Recursive
    Skill-Augmented Reinforcement Learning. arXiv:2602.08234 [cs.LG].

SkillEvolvingMixin can be composed with any BaseAgent subclass to give
it a personal skill library that grows through interaction outcomes.

Usage:
    class SkillEvolvingHonestAgent(SkillEvolvingMixin, HonestAgent):
        pass

The mixin overrides update_from_outcome() to extract skills and
provides skill-augmented decision helpers.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
)
from swarm.skills.evolution import EvolutionConfig, SkillEvolutionEngine
from swarm.skills.library import SkillLibrary, SkillLibraryConfig
from swarm.skills.model import Skill, SkillDomain

if TYPE_CHECKING:
    from swarm.models.interaction import SoftInteraction


class SkillEvolvingMixin:
    """Mixin that adds skill evolution to any agent.

    Provides:
    - Personal skill library
    - Automatic skill extraction from interaction outcomes
    - Skill-augmented acceptance thresholds
    - Active skill tracking for co-occurrence analysis
    """

    def init_skills(
        self,
        library_config: Optional[SkillLibraryConfig] = None,
        evolution_config: Optional[EvolutionConfig] = None,
    ) -> None:
        """Initialize the skill evolution system.

        Must be called during __init__ of the concrete class.
        """
        agent_id = getattr(self, "agent_id", "unknown")
        self.skill_library = SkillLibrary(
            owner_id=agent_id,
            config=library_config,
        )
        self.skill_evolution = SkillEvolutionEngine(
            config=evolution_config,
        )
        self._active_skill_ids: List[str] = []
        self._skill_enabled = True

    @property
    def has_skills(self) -> bool:
        """Check if skill system is initialized."""
        return hasattr(self, "skill_library") and self._skill_enabled

    def get_skill_context(self, observation: Observation) -> Dict:
        """Build context dict for skill matching from current observation."""
        state = observation.agent_state
        return {
            "reputation": state.reputation,
            "resources": state.resources,
            "epoch": observation.current_epoch,
            "step": observation.current_step,
        }

    def select_skill_for_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
        p: Optional[float] = None,
    ) -> Optional[Skill]:
        """Select the best skill for an upcoming interaction.

        Updates _active_skill_ids so the evolution engine can track
        co-occurrences.
        """
        if not self.has_skills:
            return None

        context = self.get_skill_context(observation)
        context["counterparty_id"] = counterparty_id

        if p is not None:
            context["p"] = p

        trust = None
        if hasattr(self, "compute_counterparty_trust"):
            trust = self.compute_counterparty_trust(counterparty_id)
            context["trust"] = trust

        skill = self.skill_library.select_best_skill(
            domain=SkillDomain.INTERACTION,
            context=context,
        )

        if skill:
            self._active_skill_ids = [skill.skill_id]
        else:
            self._active_skill_ids = []

        return skill

    def apply_skill_to_acceptance(
        self,
        base_threshold: float,
        skill: Optional[Skill],
    ) -> float:
        """Adjust acceptance threshold based on active skill.

        Strategy skills lower the threshold (more accepting).
        Lesson skills raise it (more cautious).
        """
        if skill is None:
            return base_threshold

        effect = skill.effect
        delta = effect.get("acceptance_threshold_delta", 0.0)
        adjusted = base_threshold + delta

        # Clamp to valid range
        return float(max(0.0, min(1.0, adjusted)))

    def apply_skill_to_trust_weight(
        self,
        base_weight: float,
        skill: Optional[Skill],
    ) -> float:
        """Adjust trust weight based on active skill."""
        if skill is None:
            return base_weight

        effect = skill.effect
        delta = effect.get("trust_weight_delta", 0.0)
        adjusted = base_weight + delta
        return float(max(0.0, min(1.0, adjusted)))

    def skill_augmented_update(
        self,
        interaction: "SoftInteraction",
        payoff: float,
    ) -> Optional[Skill]:
        """Process interaction outcome through the skill evolution system.

        Should be called from update_from_outcome() in the concrete agent.
        Returns any newly extracted skill.
        """
        if not self.has_skills:
            return None

        agent_id = getattr(self, "agent_id", "unknown")

        # Record invocations for active skills
        for skill_id in self._active_skill_ids:
            self.skill_evolution.record_invocation(
                skill_id=skill_id,
                agent_id=agent_id,
                interaction_id=interaction.interaction_id,
                epoch=0,  # Will be overridden by orchestrator
                step=0,
                payoff=payoff,
                p=interaction.p,
                library=self.skill_library,
            )

        # Attempt skill extraction
        new_skill = self.skill_evolution.extract_skill(
            agent_id=agent_id,
            interaction=interaction,
            payoff=payoff,
            library=self.skill_library,
            active_skill_ids=self._active_skill_ids,
        )

        # Reset active skills
        self._active_skill_ids = []

        return new_skill

    def _skill_effectiveness(self, skill_id: str) -> float:
        """Return effectiveness for a skill, or 0.0 if no performance data."""
        perf = self.skill_library.get_performance(skill_id)
        return perf.effectiveness if perf else 0.0

    def skill_library_summary(self) -> Dict:
        """Get a summary of the agent's skill library."""
        if not self.has_skills:
            return {"enabled": False}

        from swarm.skills.model import SkillType

        skills = self.skill_library.all_skills
        return {
            "enabled": True,
            "total_skills": len(skills),
            "strategies": sum(1 for s in skills if s.skill_type == SkillType.STRATEGY),
            "lessons": sum(1 for s in skills if s.skill_type == SkillType.LESSON),
            "composites": sum(1 for s in skills if s.skill_type == SkillType.COMPOSITE),
            "top_skills": [
                {
                    "name": s.name,
                    "type": s.skill_type.value,
                    "effectiveness": self._skill_effectiveness(s.skill_id),
                }
                for s in sorted(
                    skills,
                    key=lambda x: self._skill_effectiveness(x.skill_id),
                    reverse=True,
                )[:5]
            ],
        }


# ------------------------------------------------------------------
# Concrete skill-evolving agent classes
# ------------------------------------------------------------------

class SkillEvolvingHonestAgent(SkillEvolvingMixin, BaseAgent):
    """Honest agent with evolving skill library.

    Wraps HonestAgent behavior with skill-augmented decision making.
    Strategy skills make it more effective at cooperation.
    Lesson skills help it avoid exploitation.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        library_config: Optional[SkillLibraryConfig] = None,
        evolution_config: Optional[EvolutionConfig] = None,
    ):
        from swarm.models.agent import AgentType

        config = config or {}
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            config=config,
            name=name,
        )

        self.acceptance_threshold = config.get("acceptance_threshold", 0.4)
        self.trust_weight = config.get("trust_weight", 0.3)
        self.post_probability = config.get("post_probability", 0.3)
        self.vote_probability = config.get("vote_probability", 0.5)
        self.interact_probability = config.get("interact_probability", 0.4)

        self.init_skills(library_config, evolution_config)

    def act(self, observation: Observation) -> Action:
        """Act with skill-augmented decision making."""
        import random

        from swarm.models.interaction import InteractionType

        # Handle pending proposals with skill augmentation
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")

            # Select skill for this interaction
            skill = self.select_skill_for_interaction(observation, initiator_id)

            # Skill-augmented acceptance
            threshold = self.apply_skill_to_acceptance(
                self.acceptance_threshold, skill
            )
            trust = self.compute_counterparty_trust(initiator_id)
            tw = self.apply_skill_to_trust_weight(self.trust_weight, skill)
            effective_score = trust * tw + 0.5 * (1 - tw)

            if effective_score >= threshold:
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # Post with some probability
        if observation.can_post and random.random() < self.post_probability:
            return self.create_post_action(
                "Sharing insights from experience."
            )

        # Vote on visible posts
        if (
            observation.can_vote
            and observation.visible_posts
            and random.random() < self.vote_probability
        ):
            post = random.choice(observation.visible_posts)
            return self.create_vote_action(post.get("post_id", ""), 1)

        # Propose interactions with skill guidance
        if observation.can_interact and random.random() < self.interact_probability:
            if observation.visible_agents:
                # Use skill to choose best target
                candidates = [
                    a for a in observation.visible_agents
                    if a.get("agent_id") != self.agent_id
                ]
                if candidates:
                    best = max(
                        candidates,
                        key=lambda a: self.compute_counterparty_trust(
                            a.get("agent_id", "")
                        ),
                    )
                    cid = best.get("agent_id", "")
                    self.select_skill_for_interaction(observation, cid)
                    return self.create_propose_action(
                        counterparty_id=cid,
                        interaction_type=InteractionType.COLLABORATION,
                        content="Would you like to collaborate?",
                    )

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Skill-augmented acceptance decision."""
        skill = self.select_skill_for_interaction(
            observation, proposal.initiator_id
        )
        threshold = self.apply_skill_to_acceptance(
            self.acceptance_threshold, skill
        )
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        tw = self.apply_skill_to_trust_weight(self.trust_weight, skill)
        effective_score = trust * tw + 0.5 * (1 - tw)
        return bool(effective_score >= threshold)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose interaction with skill guidance."""
        from swarm.models.interaction import InteractionType

        trust = self.compute_counterparty_trust(counterparty_id)
        if trust < 0.3:
            return None

        self.select_skill_for_interaction(observation, counterparty_id)

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Would you like to collaborate?",
        )

    def update_from_outcome(
        self,
        interaction: "SoftInteraction",
        payoff: float,
    ) -> None:
        """Update with skill extraction."""
        super().update_from_outcome(interaction, payoff)
        self.skill_augmented_update(interaction, payoff)


class SkillEvolvingOpportunisticAgent(SkillEvolvingMixin, BaseAgent):
    """Opportunistic agent with evolving skill library.

    Learns which exploitation strategies work and which get caught.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        library_config: Optional[SkillLibraryConfig] = None,
        evolution_config: Optional[EvolutionConfig] = None,
    ):
        from swarm.models.agent import AgentType

        config = config or {}
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.OPPORTUNISTIC,
            config=config,
            name=name,
        )

        self.acceptance_threshold = config.get("acceptance_threshold", 0.3)
        self.cherry_pick_threshold = config.get("cherry_pick_threshold", 0.6)

        self.init_skills(library_config, evolution_config)

    def act(self, observation: Observation) -> Action:
        """Opportunistic action selection with skill augmentation."""
        import random

        from swarm.models.interaction import InteractionType

        # Handle proposals - accept most to maximize interactions
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator = proposal.get("initiator_id", "")
            skill = self.select_skill_for_interaction(observation, initiator)

            # Lesson skills might warn against certain counterparties
            if skill and skill.effect.get("avoid_action"):
                return self.create_reject_action(proposal["proposal_id"])

            return self.create_accept_action(proposal["proposal_id"])

        # Cherry-pick high-value interactions
        if observation.can_interact and observation.visible_agents:
            candidates = [
                a for a in observation.visible_agents
                if a.get("agent_id") != self.agent_id
            ]
            if candidates:
                # Prefer high-reputation targets
                best = max(
                    candidates,
                    key=lambda a: a.get("reputation", 0),
                )
                cid = best.get("agent_id", "")
                self.select_skill_for_interaction(observation, cid)
                return self.create_propose_action(
                    counterparty_id=cid,
                    interaction_type=InteractionType.COLLABORATION,
                    content="Let's work on something together.",
                )

        if observation.can_post and random.random() < 0.2:
            return self.create_post_action("Sharing an update.")

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept most interactions opportunistically."""
        skill = self.select_skill_for_interaction(
            observation, proposal.initiator_id
        )
        if skill and skill.effect.get("avoid_action"):
            return False
        return True

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        from swarm.models.interaction import InteractionType

        self.select_skill_for_interaction(observation, counterparty_id)
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Interested in collaborating?",
        )

    def update_from_outcome(
        self,
        interaction: "SoftInteraction",
        payoff: float,
    ) -> None:
        super().update_from_outcome(interaction, payoff)
        self.skill_augmented_update(interaction, payoff)
