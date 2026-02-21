"""Bidirectional mapping between SWARM Observation and Letta core memory.

Handles serializing SWARM state into Letta memory blocks and extracting
trust updates from Letta's self-edited memory.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from swarm.agents.base import Observation
from swarm.bridges.letta.config import LettaConfig

logger = logging.getLogger(__name__)


class LettaMemoryMapper:
    """Maps between SWARM Observation and Letta core memory blocks."""

    def __init__(self, config: LettaConfig) -> None:
        self.config = config

    def observation_to_memory_blocks(
        self,
        obs: Observation,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Convert an Observation into core memory block updates.

        Returns a dict of ``{label: content}`` suitable for updating
        Letta core memory blocks.

        Args:
            obs: The SWARM Observation.
            agent_config: Per-agent config from the scenario YAML.

        Returns:
            Dict mapping block labels to serialized content strings.
        """
        blocks: Dict[str, str] = {}

        # swarm_state block: current simulation state
        state_data = {
            "epoch": obs.current_epoch,
            "step": obs.current_step,
            "can_post": obs.can_post,
            "can_interact": obs.can_interact,
            "can_vote": obs.can_vote,
            "reputation": obs.agent_state.reputation
            if hasattr(obs.agent_state, "reputation")
            else 0.0,
            "resources": obs.agent_state.resources
            if hasattr(obs.agent_state, "resources")
            else 0.0,
            "n_visible_posts": len(obs.visible_posts),
            "n_pending_proposals": len(obs.pending_proposals),
            "n_available_tasks": len(obs.available_tasks),
            "ecosystem_metrics": obs.ecosystem_metrics,
        }
        blocks["swarm_state"] = json.dumps(state_data, default=str)

        return blocks

    def observation_to_message(
        self,
        obs: Observation,
    ) -> str:
        """Build the step prompt message for a Letta agent.

        This message is sent via ``send_message()`` each step, describing
        what the agent can do and what has happened recently.

        Args:
            obs: The SWARM Observation.

        Returns:
            A natural-language prompt with embedded action instructions.
        """
        parts: List[str] = []

        parts.append(
            f"[SWARM Step] Epoch {obs.current_epoch}, Step {obs.current_step}"
        )

        # Available actions
        actions: List[str] = []
        if obs.can_post:
            actions.append("post (create a new post)")
        if obs.can_vote and obs.visible_posts:
            actions.append("vote (upvote/downvote a post)")
        if obs.can_interact and obs.visible_agents:
            actions.append("propose_interaction (propose to collaborate)")
        if obs.pending_proposals:
            actions.append("accept/reject (respond to pending proposals)")

        if actions:
            parts.append("Available actions: " + ", ".join(actions))

        # Visible posts (last 3)
        if obs.visible_posts:
            parts.append("Recent posts:")
            for post in obs.visible_posts[:3]:
                pid = post.get("post_id", "?")
                content = post.get("content", "")[:100]
                parts.append(f"  [{pid}] {content}")

        # Pending proposals
        if obs.pending_proposals:
            parts.append("Pending proposals to you:")
            for prop in obs.pending_proposals[:3]:
                pid = prop.get("proposal_id", "?")
                from_id = prop.get("initiator_id", "?")
                parts.append(f"  [{pid}] from {from_id}")

        # Visible agents
        if obs.visible_agents:
            agent_ids = [a.get("agent_id", "?") for a in obs.visible_agents[:5]]
            parts.append(f"Visible agents: {', '.join(agent_ids)}")

        # Ecosystem health
        if obs.ecosystem_metrics:
            parts.append(f"Ecosystem metrics: {json.dumps(obs.ecosystem_metrics)}")

        # Action format instructions
        parts.append(
            "\nRespond with a JSON action block like: "
            '{"action": "<action_type>", "target_id": "<optional>", '
            '"counterparty_id": "<optional>", "content": "<optional>", '
            '"vote_direction": <optional 1 or -1>}'
        )

        return "\n".join(parts)

    def extract_trust_updates(
        self,
        trust_block_content: str,
    ) -> Dict[str, float]:
        """Parse trust data from Letta's self-edited trust_scores block.

        Args:
            trust_block_content: Raw string content of the trust_scores block.

        Returns:
            Dict mapping agent_id to trust score float.
        """
        if not trust_block_content:
            return {}
        try:
            data = json.loads(trust_block_content)
            if isinstance(data, dict):
                return {
                    str(k): float(v)
                    for k, v in data.items()
                    if isinstance(v, (int, float))
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.debug("Failed to parse trust block content")
        return {}

    def governance_state_to_block(
        self,
        metrics: Dict[str, Any],
        proposals: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Serialize governance state for the shared governance block.

        Args:
            metrics: Current ecosystem metrics.
            proposals: Active governance proposals.

        Returns:
            JSON string for the shared governance_state block.
        """
        state = {
            "ecosystem_metrics": metrics,
            "active_proposals": proposals or [],
        }
        return json.dumps(state, default=str)
