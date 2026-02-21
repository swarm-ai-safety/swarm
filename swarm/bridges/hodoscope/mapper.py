"""Mapper from SWARM interactions to hodoscope trajectory format."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.analysis.aggregation import SimulationHistory
from swarm.bridges.hodoscope.config import HodoscopeConfig
from swarm.models.interaction import SoftInteraction


class HodoscopeMapper:
    """Converts SWARM interactions into hodoscope-compatible trajectory dicts.

    Each trajectory is a list of messages in OpenAI chat format with
    tool_calls, representing one agent's behavior over a time window.
    """

    def __init__(self, config: Optional[HodoscopeConfig] = None) -> None:
        self.config = config or HodoscopeConfig()

    def interaction_to_messages(
        self, interaction: SoftInteraction, agent_id: str
    ) -> List[Dict[str, Any]]:
        """Convert a single interaction into hodoscope-format messages.

        From the perspective of ``agent_id``, produces:
        - An ``assistant`` message with a ``tool_calls`` entry describing the
          agent's action (initiate or respond, accept or reject).
        - A ``tool`` message describing the outcome (payoff, counterparty, p).

        Args:
            interaction: The interaction to convert.
            agent_id: Which agent's perspective to take.

        Returns:
            List of message dicts in OpenAI chat format.
        """
        is_initiator = interaction.initiator == agent_id
        role_label = "initiator" if is_initiator else "counterparty"
        other_agent = (
            interaction.counterparty if is_initiator else interaction.initiator
        )

        # Determine the action taken by this agent
        if is_initiator:
            action = "propose"
            action_type = interaction.interaction_type.value
        else:
            action = "accept" if interaction.accepted else "reject"
            action_type = interaction.interaction_type.value

        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"

        # Assistant turn: the agent's action
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": action,
                        "arguments": json.dumps(
                            {
                                "interaction_type": action_type,
                                "counterparty": other_agent,
                                "role": role_label,
                            }
                        ),
                    },
                }
            ],
        }

        # Tool turn: the outcome
        payoff = interaction.r_a if is_initiator else interaction.r_b
        gov_cost = interaction.c_a if is_initiator else interaction.c_b

        tool_msg: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(
                {
                    "accepted": interaction.accepted,
                    "p": round(interaction.p, 4),
                    "v_hat": round(interaction.v_hat, 4),
                    "reputation_change": round(payoff, 4),
                    "governance_cost": round(gov_cost, 4),
                    "transfer": round(interaction.tau, 4),
                    "counterparty": other_agent,
                }
            ),
        }

        return [assistant_msg, tool_msg]

    def interactions_to_trajectories(
        self,
        interactions: List[SoftInteraction],
        agents_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert a list of interactions to hodoscope trajectory dicts.

        Groups interactions by agent (and optionally by epoch) based on
        ``self.config.trajectory_unit``, then converts each group into a
        trajectory with messages and metadata.

        Args:
            interactions: Flat list of interactions to convert.
            agents_map: Optional mapping of agent_id -> agent metadata
                (e.g. ``{"agent_1": {"agent_type": "honest", ...}}``).

        Returns:
            List of trajectory dicts, each with keys:
            ``trajectory_id``, ``messages``, ``metadata``.
        """
        if agents_map is None:
            agents_map = {}

        # Collect all agent IDs that participated
        agent_interactions: Dict[str, List[SoftInteraction]] = defaultdict(list)
        for ix in interactions:
            agent_interactions[ix.initiator].append(ix)
            agent_interactions[ix.counterparty].append(ix)

        trajectories: List[Dict[str, Any]] = []

        for agent_id, agent_ixs in agent_interactions.items():
            # Sort by timestamp
            agent_ixs_sorted = sorted(agent_ixs, key=lambda x: x.timestamp)

            if self.config.trajectory_unit == "agent_run":
                # One trajectory for the whole run
                traj = self._build_trajectory(
                    agent_id, agent_ixs_sorted, agents_map, epoch=None
                )
                trajectories.append(traj)

            elif self.config.trajectory_unit == "interaction":
                # One trajectory per interaction
                for ix in agent_ixs_sorted:
                    traj = self._build_trajectory(
                        agent_id,
                        [ix],
                        agents_map,
                        epoch=ix.metadata.get("epoch"),
                    )
                    trajectories.append(traj)

            else:
                # Default: agent_epoch — group by epoch
                epoch_groups: Dict[int, List[SoftInteraction]] = defaultdict(list)
                for ix in agent_ixs_sorted:
                    epoch = ix.metadata.get("epoch", 0) if ix.metadata else 0
                    epoch_groups[epoch].append(ix)

                for epoch, epoch_ixs in sorted(epoch_groups.items()):
                    traj = self._build_trajectory(
                        agent_id, epoch_ixs, agents_map, epoch=epoch
                    )
                    trajectories.append(traj)

        return trajectories

    def history_to_trajectories(
        self, history: SimulationHistory
    ) -> List[Dict[str, Any]]:
        """Convert a full SimulationHistory to trajectory dicts.

        Convenience method that extracts interactions from the history
        and builds an agents_map from agent snapshots.

        Args:
            history: Complete simulation history.

        Returns:
            List of trajectory dicts.
        """
        # Build agents_map from final snapshots
        agents_map: Dict[str, Dict[str, Any]] = {}
        final_states = history.get_final_agent_states()
        for agent_id, snapshot in final_states.items():
            agents_map[agent_id] = {
                "agent_type": snapshot.agent_type or "unknown",
                "reputation": snapshot.reputation,
                "total_payoff": snapshot.total_payoff,
            }

        return self.interactions_to_trajectories(
            history.interactions, agents_map
        )

    def write_trajectory_dir(
        self,
        trajectories: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Write one JSON file per trajectory to the output directory.

        Args:
            trajectories: List of trajectory dicts to write.
            output_dir: Target directory (defaults to config.output_dir).

        Returns:
            Path to the output directory.
        """
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for traj in trajectories:
            tid = traj["trajectory_id"]
            filepath = out / f"{tid}.json"
            with open(filepath, "w") as f:
                json.dump(traj, f, indent=2, default=str)

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_trajectory(
        self,
        agent_id: str,
        interactions: List[SoftInteraction],
        agents_map: Dict[str, Dict[str, Any]],
        epoch: Optional[int],
    ) -> Dict[str, Any]:
        """Build a single trajectory dict from a group of interactions."""
        messages: List[Dict[str, Any]] = []
        p_values: List[float] = []
        accepted_count = 0
        initiated_count = 0

        for ix in interactions:
            msgs = self.interaction_to_messages(ix, agent_id)
            messages.extend(msgs)
            p_values.append(ix.p)
            if ix.accepted:
                accepted_count += 1
            if ix.initiator == agent_id:
                initiated_count += 1

        # Infer agent_type from interactions metadata or agents_map
        agent_type = "unknown"
        if agent_id in agents_map and "agent_type" in agents_map[agent_id]:
            agent_type = agents_map[agent_id]["agent_type"]
        else:
            # Try to infer from interaction metadata
            for ix in interactions:
                if ix.metadata and "agent_type" in ix.metadata:
                    if ix.initiator == agent_id or ix.counterparty == agent_id:
                        agent_type = ix.metadata["agent_type"]
                        break

        avg_p = sum(p_values) / len(p_values) if p_values else 0.5
        epoch_str = f"_e{epoch}" if epoch is not None else ""
        trajectory_id = f"{agent_id}{epoch_str}_{uuid.uuid4().hex[:8]}"

        metadata: Dict[str, Any] = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "avg_p": round(avg_p, 4),
            "n_interactions": len(interactions),
            "n_accepted": accepted_count,
            "n_initiated": initiated_count,
            "acceptance_rate": round(
                accepted_count / len(interactions), 4
            )
            if interactions
            else 0.0,
        }
        if epoch is not None:
            metadata["epoch"] = epoch

        # Merge agent-level metadata from agents_map
        if agent_id in agents_map:
            for k, v in agents_map[agent_id].items():
                if k not in metadata:
                    metadata[k] = v

        return {
            "trajectory_id": trajectory_id,
            "messages": messages,
            "metadata": metadata,
        }
