"""Agent that interacts with AWM (Agent World Model) environments.

Supports two planning modes:
- **Scripted** (default): Rule-based tool call sequences based on mode
  (diligent / lazy / adversarial).
- **LLM** (opt-in via ``llm_planning: true`` in config): Delegates tool
  selection to an ``LLMAgent`` and falls back to scripted on failure.

Different behavioral modes control quality and adversarial behavior.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM tool-planning prompts
# ---------------------------------------------------------------------------

AWM_TOOL_PLANNING_SYSTEM_PROMPT = """\
You are a tool-calling agent operating inside a database-backed task environment.

Your job is to select which tools to call (and with what arguments) in order
to complete the assigned task efficiently and correctly.

Guidelines:
- Read before you write: prefer query/list operations before update/delete.
- Only use tools from the provided available-tools list.
- Arguments must match the parameter schemas described for each tool.
- Minimise unnecessary calls — efficiency matters.
- If you are unsure, prefer a conservative read-only action.
"""

AWM_TOOL_CALL_SCHEMA = """\
Respond with a JSON object matching this schema:
{
  "reasoning": "<brief explanation of your plan>",
  "tool_calls": [
    {"tool_name": "<name>", "arguments": {<arg_key>: <arg_value>, ...}},
    ...
  ]
}
Do NOT include any text outside the JSON object.
"""


class AWMAgent(BaseAgent):
    """Agent that interacts with AWM database-backed task environments.

    Config keys:
        mode: "diligent" | "lazy" | "adversarial" (default: "diligent")
        tool_call_count: Number of tool calls per task (default: 5)
        malformed_rate: Probability of sending malformed calls (default: 0.0)
        llm_planning: Enable LLM-based tool selection (default: False)
        llm_provider: LLM provider name (e.g. "anthropic", "openai")
        llm_model: Model identifier
        llm_api_key: API key (or read from env)
        llm_base_url: Override URL for provider
        llm_temperature: Sampling temperature (default: 0.3)
        llm_max_tokens: Max response tokens (default: 1024)
        llm_timeout: Request timeout seconds (default: 30.0)
        llm_max_retries: Retry count (default: 2)
        llm_system_prompt: Custom system prompt override
        llm_cost_tracking: Track LLM costs (default: True)
        llm_prompt_audit_path: JSONL path for prompt auditing
        llm_fallback_to_scripted: Fall back to scripted on LLM error (default: True)
        llm_max_calls_per_plan: Cap on tool calls per LLM plan (default: 10)
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.HONEST,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
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

        # LLM planning state
        self._llm_enabled: bool = cfg.get("llm_planning", False)
        self._llm_config_dict: Dict = {
            k: v for k, v in cfg.items() if k.startswith("llm_")
        }
        self._llm_fallback: bool = cfg.get("llm_fallback_to_scripted", True)
        self._llm_max_calls: int = cfg.get("llm_max_calls_per_plan", 10)
        self._llm_delegate: Any = None  # Lazy: LLMAgent instance
        self._tool_call_history: List[Dict] = []

    # ------------------------------------------------------------------
    # LLM delegate lifecycle
    # ------------------------------------------------------------------

    def _init_llm(self) -> Any:
        """Lazy-create LLMAgent delegate from config dict.

        Returns the delegate, or None on failure (disables LLM permanently).
        """
        if self._llm_delegate is not None:
            return self._llm_delegate

        try:
            from swarm.agents.llm_agent import LLMAgent
            from swarm.agents.llm_config import LLMConfig, LLMProvider

            cfg = self._llm_config_dict
            provider_str = cfg.get("llm_provider", "anthropic")
            provider = LLMProvider(provider_str)

            llm_config = LLMConfig(
                provider=provider,
                model=cfg.get("llm_model", "claude-sonnet-4-20250514"),
                api_key=cfg.get("llm_api_key"),
                base_url=cfg.get("llm_base_url"),
                temperature=cfg.get("llm_temperature", 0.3),
                max_tokens=cfg.get("llm_max_tokens", 1024),
                timeout=cfg.get("llm_timeout", 30.0),
                max_retries=cfg.get("llm_max_retries", 2),
                system_prompt=cfg.get("llm_system_prompt"),
                cost_tracking=cfg.get("llm_cost_tracking", True),
                prompt_audit_path=cfg.get("llm_prompt_audit_path"),
            )
            self._llm_delegate = LLMAgent(
                agent_id=f"{self.agent_id}_llm_delegate",
                llm_config=llm_config,
            )
            logger.info(
                "AWMAgent %s: LLM delegate initialized (provider=%s, model=%s)",
                self.agent_id,
                provider_str,
                llm_config.model,
            )
            return self._llm_delegate
        except Exception:
            logger.exception(
                "AWMAgent %s: failed to init LLM delegate — disabling LLM planning",
                self.agent_id,
            )
            self._llm_enabled = False
            return None

    # ------------------------------------------------------------------
    # LLM prompt construction
    # ------------------------------------------------------------------

    def _build_awm_tool_prompt(
        self, observation: Observation
    ) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) for an LLM tool-planning call."""
        cfg = self._llm_config_dict
        system_prompt = cfg.get("llm_system_prompt") or AWM_TOOL_PLANNING_SYSTEM_PROMPT

        # --- user prompt ---
        parts: List[str] = []

        # Task description
        task = observation.awm_task or {}
        parts.append(f"## Task\n{json.dumps(task, default=str)}")

        # Available tools with schemas
        tools = observation.awm_available_tools or []
        parts.append("## Available tools")
        for t in tools:
            parts.append(f"- **{t.get('name', '?')}**: {t.get('description', '')}")
            params = t.get("parameters") or t.get("params")
            if params:
                parts.append(f"  Parameters: {json.dumps(params, default=str)}")

        # Last result (if any)
        if observation.awm_last_result is not None:
            parts.append(
                f"## Last tool result\n```json\n{json.dumps(observation.awm_last_result, default=str)}\n```"
            )

        # Recent call history (last 5)
        if self._tool_call_history:
            recent = self._tool_call_history[-5:]
            parts.append("## Recent call history (newest last)")
            for h in recent:
                parts.append(f"- {h.get('tool_name', '?')}({json.dumps(h.get('arguments', {}), default=str)})")

        # Episode state
        if self.step_mode:
            parts.append(f"\nSteps remaining: {observation.awm_steps_remaining}")
            parts.append("Plan ONE tool call for the next step.")
        else:
            parts.append(
                f"\nPlan up to {self._llm_max_calls} tool calls to complete the task."
            )

        # Response format
        parts.append(f"\n{AWM_TOOL_CALL_SCHEMA}")

        user_prompt = "\n\n".join(parts)
        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    def _parse_tool_call_response(
        self, response_text: str, observation: Observation
    ) -> Optional[List[Dict]]:
        """Parse LLM response into validated tool call list.

        Returns None if parsing fails entirely.
        """
        delegate = self._llm_delegate
        if delegate is None:
            return None

        try:
            parsed = delegate._parse_action_response(response_text)
        except (ValueError, Exception):
            logger.warning(
                "AWMAgent %s: failed to parse LLM response JSON",
                self.agent_id,
            )
            return None

        raw_calls = parsed.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            logger.warning(
                "AWMAgent %s: tool_calls is not a list",
                self.agent_id,
            )
            return None

        # Validate and filter
        valid_tool_names = {
            t.get("name", "")
            for t in (observation.awm_available_tools or [])
            if t.get("name")
        }

        validated: List[Dict] = []
        for entry in raw_calls:
            if not isinstance(entry, dict):
                logger.warning("AWMAgent %s: skipping non-dict tool call entry", self.agent_id)
                continue
            tool_name = entry.get("tool_name", "")
            arguments = entry.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name:
                logger.warning("AWMAgent %s: skipping entry with invalid tool_name", self.agent_id)
                continue
            if tool_name not in valid_tool_names:
                logger.warning(
                    "AWMAgent %s: skipping unknown tool '%s'",
                    self.agent_id,
                    tool_name,
                )
                continue
            if not isinstance(arguments, dict):
                logger.warning("AWMAgent %s: coercing non-dict arguments to {}", self.agent_id)
                arguments = {}
            validated.append({"tool_name": tool_name, "arguments": arguments})

        # Cap
        if len(validated) > self._llm_max_calls:
            logger.info(
                "AWMAgent %s: capping tool calls from %d to %d",
                self.agent_id,
                len(validated),
                self._llm_max_calls,
            )
            validated = validated[: self._llm_max_calls]

        return validated if validated else None

    # ------------------------------------------------------------------
    # LLM planning call
    # ------------------------------------------------------------------

    def _plan_tool_calls_llm(
        self, observation: Observation
    ) -> Optional[List[Dict]]:
        """Use LLM delegate to plan tool calls.

        Returns None on any failure (so caller can fall back).
        """
        delegate = self._init_llm()
        if delegate is None:
            return None

        system_prompt, user_prompt = self._build_awm_tool_prompt(observation)

        try:
            response_text, input_tokens, output_tokens = delegate._call_llm_sync(
                system_prompt, user_prompt
            )
        except Exception:
            logger.exception(
                "AWMAgent %s: LLM call failed",
                self.agent_id,
            )
            return None

        # Audit
        delegate._audit_llm_exchange(
            kind="awm_tool_plan",
            observation=observation,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return self._parse_tool_call_response(response_text, observation)

    # ------------------------------------------------------------------
    # Planning dispatch
    # ------------------------------------------------------------------

    def _plan_tool_calls_scripted(self, observation: Observation) -> List[Dict]:
        """Original rule-based planning logic."""
        available = observation.awm_available_tools
        if not available:
            return []

        tool_names = [t.get("name", "") for t in available if t.get("name")]
        if not tool_names:
            return []

        calls: List[Dict] = []
        n_calls = self.tool_call_count

        if self.mode == "lazy":
            n_calls = max(1, n_calls // 3)
        elif self.mode == "adversarial":
            pass

        for _ in range(n_calls):
            if self._rng.random() < self.malformed_rate:
                calls.append({
                    "tool_name": "nonexistent_tool_xyz",
                    "arguments": {"invalid": True},
                })
            else:
                tool_name = self._rng.choice(tool_names)
                calls.append({
                    "tool_name": tool_name,
                    "arguments": self._generate_arguments(tool_name),
                })

        return calls

    def _plan_tool_calls(self, observation: Observation) -> List[Dict]:
        """Plan tool calls — dispatch to LLM or scripted path."""
        if self._llm_enabled:
            result = self._plan_tool_calls_llm(observation)
            if result is not None:
                return result
            # LLM failed
            if self._llm_fallback:
                logger.info(
                    "AWMAgent %s: LLM plan failed, falling back to scripted",
                    self.agent_id,
                )
                return self._plan_tool_calls_scripted(observation)
            return []
        return self._plan_tool_calls_scripted(observation)

    # ------------------------------------------------------------------
    # Action methods
    # ------------------------------------------------------------------

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
            self._tool_call_history = []

        # If plan exhausted or no steps remaining, finish
        if (
            self._plan_index >= len(self._current_plan)
            or observation.awm_steps_remaining <= 0
        ):
            return self.create_awm_finish_task_action()

        # Submit next tool call from plan
        call = self._current_plan[self._plan_index]
        self._plan_index += 1
        self._tool_call_history.append(call)
        return self.create_awm_tool_call_action(
            tool_name=call.get("tool_name", ""),
            arguments=call.get("arguments", {}),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    @property
    def llm_usage_stats(self) -> Optional[Dict[str, Any]]:
        """Expose LLM delegate usage stats, or None if no delegate."""
        if self._llm_delegate is not None:
            stats: Dict[str, Any] = self._llm_delegate.usage_stats.to_dict()
            return stats
        return None

    # ------------------------------------------------------------------
    # Interaction methods
    # ------------------------------------------------------------------

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
