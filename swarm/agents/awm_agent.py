"""Scripted agent that interacts with AWM (Agent World Model) environments.

This agent follows a simple strategy when assigned an AWM task:
1. Observe available tools
2. Execute a sequence of tool calls to complete the task
3. Submit via AWM_EXECUTE_TASK action

Different behavioral modes control quality and adversarial behavior.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType


class AWMAgent(BaseAgent):
    """Agent that interacts with AWM database-backed task environments.

    Config keys:
        mode: "diligent" | "lazy" | "adversarial" (default: "diligent")
        tool_call_count: Number of tool calls per task (default: 5)
        malformed_rate: Probability of sending malformed calls (default: 0.0)
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.HONEST,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
        **kwargs,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            roles=roles or [Role.WORKER],
            config=config,
            name=name,
            rng=rng,
        )
        cfg = config or {}
        self.mode = cfg.get("mode", "diligent")
        self.tool_call_count = cfg.get("tool_call_count", 5)
        self.malformed_rate = cfg.get("malformed_rate", 0.0)
        self.step_mode: bool = cfg.get("step_mode", False)
        self._current_plan: List[Dict] = []
        self._plan_index: int = 0

    def act(self, observation: Observation) -> Action:
        """Decide action based on current observation."""
        if observation.awm_task is not None:
            if self.step_mode:
                return self._act_step_mode(observation)
            return self._act_batch_mode(observation)
        return self.create_noop_action()

    def _act_batch_mode(self, observation: Observation) -> Action:
        """Batch mode: submit all tool calls in a single action."""
        tool_calls = self._plan_tool_calls(observation)
        return self.create_awm_execute_task_action(tool_calls=tool_calls)

    def _act_step_mode(self, observation: Observation) -> Action:
        """Step mode: submit one tool call per step, then finish."""
        # If no active episode, generate a fresh plan
        if not observation.awm_episode_active:
            self._current_plan = self._plan_tool_calls(observation)
            self._plan_index = 0

        # If plan exhausted or no steps remaining, finish
        if (
            self._plan_index >= len(self._current_plan)
            or observation.awm_steps_remaining <= 0
        ):
            return self.create_awm_finish_task_action()

        # Submit next tool call from plan
        call = self._current_plan[self._plan_index]
        self._plan_index += 1
        return self.create_awm_tool_call_action(
            tool_name=call.get("tool_name", ""),
            arguments=call.get("arguments", {}),
        )

    def _plan_tool_calls(self, observation: Observation) -> List[Dict]:
        """Plan tool calls based on mode and available tools."""
        available = observation.awm_available_tools
        if not available:
            return []

        tool_names = [t.get("name", "") for t in available if t.get("name")]
        if not tool_names:
            return []

        calls = []
        n_calls = self.tool_call_count

        if self.mode == "lazy":
            # Lazy agent makes fewer calls
            n_calls = max(1, n_calls // 3)
        elif self.mode == "adversarial":
            # Adversarial agent mixes in malformed calls
            pass

        for _ in range(n_calls):
            if self._rng.random() < self.malformed_rate:
                # Send malformed call
                calls.append({
                    "tool_name": "nonexistent_tool_xyz",
                    "arguments": {"invalid": True},
                })
            else:
                # Pick a valid tool
                tool_name = self._rng.choice(tool_names)
                calls.append({
                    "tool_name": tool_name,
                    "arguments": self._generate_arguments(tool_name),
                })

        return calls

    def _generate_arguments(self, tool_name: str) -> Dict:
        """Generate plausible arguments for a tool call."""
        if tool_name == "query_database":
            return {"query": "SELECT * FROM orders LIMIT 10"}
        elif tool_name == "update_record":
            return {"table": "orders", "id": 1, "data": {"status": "completed"}}
        elif tool_name == "create_record":
            return {"table": "orders", "data": {"customer_id": 1, "total": 99.99}}
        elif tool_name == "delete_record":
            return {"table": "orders", "id": 1}
        elif tool_name == "list_tables":
            return {}
        return {}

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """AWM agents accept interactions based on mode."""
        if self.mode == "adversarial":
            return bool(self._rng.random() < 0.3)
        return bool(self._rng.random() < 0.7)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """AWM agents don't initiate interactions."""
        return None
