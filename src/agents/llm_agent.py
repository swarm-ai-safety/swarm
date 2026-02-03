"""LLM-backed agent implementation."""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from src.agents.llm_config import LLMConfig, LLMProvider, LLMUsageStats, PersonaType
from src.agents.llm_prompts import build_accept_prompt, build_action_prompt
from src.models.agent import AgentType
from src.models.interaction import InteractionType

logger = logging.getLogger(__name__)


class LLMAgent(BaseAgent):
    """
    Agent backed by an LLM API call.

    This agent uses an LLM to decide actions based on observations.
    Supports Anthropic, OpenAI, and Ollama providers.
    """

    def __init__(
        self,
        agent_id: str,
        llm_config: LLMConfig,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize LLM agent.

        Args:
            agent_id: Unique identifier
            llm_config: LLM API configuration
            roles: List of roles this agent can fulfill
            config: Additional agent configuration
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,  # Default, behavior comes from LLM
            roles=roles,
            config=config,
        )

        self.llm_config = llm_config
        self.usage_stats = LLMUsageStats()

        # Resolve API key from environment if not provided
        self._api_key = llm_config.api_key or self._get_api_key_from_env()

        # Lazy-loaded clients
        self._anthropic_client = None
        self._openai_client = None

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        if self.llm_config.provider == LLMProvider.ANTHROPIC:
            return os.environ.get("ANTHROPIC_API_KEY")
        elif self.llm_config.provider == LLMProvider.OPENAI:
            return os.environ.get("OPENAI_API_KEY")
        return None

    def _get_anthropic_client(self):
        """Lazy-load Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=self.llm_config.timeout,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._anthropic_client

    def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            try:
                import openai
                self._openai_client = openai.OpenAI(
                    api_key=self._api_key,
                    timeout=self.llm_config.timeout,
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
        return self._openai_client

    async def _call_llm_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """
        Make an async LLM API call with retry logic.

        Args:
            system_prompt: System message
            user_prompt: User message

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            Exception: If all retries fail
        """
        last_error = None
        retries = 0

        for attempt in range(self.llm_config.max_retries + 1):
            try:
                if self.llm_config.provider == LLMProvider.ANTHROPIC:
                    return await self._call_anthropic_async(system_prompt, user_prompt)
                elif self.llm_config.provider == LLMProvider.OPENAI:
                    return await self._call_openai_async(system_prompt, user_prompt)
                elif self.llm_config.provider == LLMProvider.OLLAMA:
                    return await self._call_ollama_async(system_prompt, user_prompt)
                else:
                    raise ValueError(f"Unknown provider: {self.llm_config.provider}")

            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}): {e}"
                )

                if attempt < self.llm_config.max_retries:
                    delay = self.llm_config.retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        self.usage_stats.record_usage(
            model=self.llm_config.model,
            input_tokens=0,
            output_tokens=0,
            retries=retries,
            failed=True,
        )
        raise last_error or Exception("LLM call failed")

    async def _call_anthropic_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call Anthropic API."""
        client = self._get_anthropic_client()

        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        )

        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    async def _call_openai_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call OpenAI API."""
        client = self._get_openai_client()

        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        )

        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    async def _call_ollama_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call Ollama API."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx package not installed. "
                "Install with: pip install httpx"
            )

        base_url = self.llm_config.base_url or "http://localhost:11434"
        url = f"{base_url}/api/chat"

        async with httpx.AsyncClient(timeout=self.llm_config.timeout) as client:
            response = await client.post(
                url,
                json={
                    "model": self.llm_config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.llm_config.temperature,
                        "num_predict": self.llm_config.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        text = data["message"]["content"]
        # Ollama doesn't always provide token counts
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    def _call_llm_sync(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """
        Synchronous wrapper for LLM calls.

        Used by the sync act() method when no event loop is running.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._call_llm_async(system_prompt, user_prompt)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._call_llm_async(system_prompt, user_prompt)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self._call_llm_async(system_prompt, user_prompt)
            )

    def _parse_action_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into action dictionary.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed action dictionary
        """
        # Try to extract JSON from response
        # Handle markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    def _action_dict_to_action(self, action_dict: Dict[str, Any]) -> Action:
        """
        Convert parsed action dictionary to Action object.

        Args:
            action_dict: Parsed action dictionary from LLM

        Returns:
            Action object
        """
        action_type_str = action_dict.get("action_type", "NOOP").upper()
        params = action_dict.get("params", {})

        try:
            action_type = ActionType[action_type_str]
        except KeyError:
            logger.warning(f"Unknown action type: {action_type_str}, defaulting to NOOP")
            return self.create_noop_action()

        if action_type == ActionType.NOOP:
            return self.create_noop_action()

        elif action_type == ActionType.POST:
            return self.create_post_action(
                content=params.get("content", "")
            )

        elif action_type == ActionType.REPLY:
            return self.create_reply_action(
                post_id=params.get("post_id", ""),
                content=params.get("content", ""),
            )

        elif action_type == ActionType.VOTE:
            return self.create_vote_action(
                post_id=params.get("post_id", ""),
                direction=int(params.get("direction", 1)),
            )

        elif action_type == ActionType.PROPOSE_INTERACTION:
            interaction_type_str = params.get("interaction_type", "collaboration")
            try:
                interaction_type = InteractionType(interaction_type_str)
            except ValueError:
                interaction_type = InteractionType.COLLABORATION

            return self.create_propose_action(
                counterparty_id=params.get("counterparty_id", ""),
                interaction_type=interaction_type,
                content=params.get("content", ""),
                task_id=params.get("task_id"),
            )

        elif action_type == ActionType.CLAIM_TASK:
            return self.create_claim_task_action(
                task_id=params.get("task_id", "")
            )

        elif action_type == ActionType.SUBMIT_OUTPUT:
            return self.create_submit_output_action(
                task_id=params.get("task_id", ""),
                content=params.get("content", ""),
            )

        else:
            logger.warning(f"Unhandled action type: {action_type}, defaulting to NOOP")
            return self.create_noop_action()

    def act(self, observation: Observation) -> Action:
        """
        Decide on an action given the current observation.

        This is the synchronous interface required by BaseAgent.
        For async usage, use act_async() directly.

        Args:
            observation: Current view of the environment

        Returns:
            Action to take
        """
        try:
            system_prompt, user_prompt = build_action_prompt(
                persona=self.llm_config.persona,
                observation=observation,
                custom_system_prompt=self.llm_config.system_prompt,
                memory=self.get_memory(limit=10),
            )

            response, _, _ = self._call_llm_sync(system_prompt, user_prompt)

            action_dict = self._parse_action_response(response)

            # Store reasoning in memory
            if action_dict.get("reasoning"):
                self.remember({
                    "type": "action_reasoning",
                    "action_type": action_dict.get("action_type"),
                    "reasoning": action_dict["reasoning"],
                })

            return self._action_dict_to_action(action_dict)

        except Exception as e:
            logger.error(f"LLM action failed: {e}, returning NOOP")
            return self.create_noop_action()

    async def act_async(self, observation: Observation) -> Action:
        """
        Async version of act().

        Args:
            observation: Current view of the environment

        Returns:
            Action to take
        """
        try:
            system_prompt, user_prompt = build_action_prompt(
                persona=self.llm_config.persona,
                observation=observation,
                custom_system_prompt=self.llm_config.system_prompt,
                memory=self.get_memory(limit=10),
            )

            response, _, _ = await self._call_llm_async(system_prompt, user_prompt)

            action_dict = self._parse_action_response(response)

            # Store reasoning in memory
            if action_dict.get("reasoning"):
                self.remember({
                    "type": "action_reasoning",
                    "action_type": action_dict.get("action_type"),
                    "reasoning": action_dict["reasoning"],
                })

            return self._action_dict_to_action(action_dict)

        except Exception as e:
            logger.error(f"LLM action failed: {e}, returning NOOP")
            return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Decide whether to accept a proposed interaction.

        Args:
            proposal: The interaction proposal
            observation: Current observation

        Returns:
            True to accept, False to reject
        """
        try:
            proposal_dict = {
                "proposal_id": proposal.proposal_id,
                "initiator_id": proposal.initiator_id,
                "interaction_type": proposal.interaction_type.value,
                "content": proposal.content,
                "offered_transfer": proposal.offered_transfer,
            }

            system_prompt, user_prompt = build_accept_prompt(
                persona=self.llm_config.persona,
                proposal=proposal_dict,
                observation=observation,
                custom_system_prompt=self.llm_config.system_prompt,
                memory=self.get_memory(limit=10),
            )

            response, _, _ = self._call_llm_sync(system_prompt, user_prompt)

            decision_dict = self._parse_action_response(response)

            # Store reasoning
            if decision_dict.get("reasoning"):
                self.remember({
                    "type": "accept_decision",
                    "proposal_id": proposal.proposal_id,
                    "initiator_id": proposal.initiator_id,
                    "accepted": decision_dict.get("accept", False),
                    "reasoning": decision_dict["reasoning"],
                })

            return bool(decision_dict.get("accept", False))

        except Exception as e:
            logger.error(f"LLM accept_interaction failed: {e}, defaulting to False")
            return False

    async def accept_interaction_async(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Async version of accept_interaction().

        Args:
            proposal: The interaction proposal
            observation: Current observation

        Returns:
            True to accept, False to reject
        """
        try:
            proposal_dict = {
                "proposal_id": proposal.proposal_id,
                "initiator_id": proposal.initiator_id,
                "interaction_type": proposal.interaction_type.value,
                "content": proposal.content,
                "offered_transfer": proposal.offered_transfer,
            }

            system_prompt, user_prompt = build_accept_prompt(
                persona=self.llm_config.persona,
                proposal=proposal_dict,
                observation=observation,
                custom_system_prompt=self.llm_config.system_prompt,
                memory=self.get_memory(limit=10),
            )

            response, _, _ = await self._call_llm_async(system_prompt, user_prompt)

            decision_dict = self._parse_action_response(response)

            # Store reasoning
            if decision_dict.get("reasoning"):
                self.remember({
                    "type": "accept_decision",
                    "proposal_id": proposal.proposal_id,
                    "initiator_id": proposal.initiator_id,
                    "accepted": decision_dict.get("accept", False),
                    "reasoning": decision_dict["reasoning"],
                })

            return bool(decision_dict.get("accept", False))

        except Exception as e:
            logger.error(f"LLM accept_interaction failed: {e}, defaulting to False")
            return False

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """
        Create an interaction proposal for a counterparty.

        For LLM agents, this is typically handled by the act() method
        returning a PROPOSE_INTERACTION action. This method is provided
        for compatibility but delegates to act().

        Args:
            observation: Current observation
            counterparty_id: Target agent ID

        Returns:
            InteractionProposal or None if not proposing
        """
        # LLM agents propose interactions through the act() method
        # This is here for interface compatibility
        return None

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        return self.usage_stats.to_dict()

    def __repr__(self) -> str:
        return (
            f"LLMAgent(id={self.agent_id}, "
            f"provider={self.llm_config.provider.value}, "
            f"model={self.llm_config.model}, "
            f"persona={self.llm_config.persona.value})"
        )
