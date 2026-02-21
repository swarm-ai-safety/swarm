"""Integration tests for LettaAgent with mocked Letta client.

Tests LettaAgent.act(), accept_interaction(), update_from_outcome().
All mocked - no real Letta server needed.
"""

import json

from swarm.agents.base import ActionType, InteractionProposal, Observation
from swarm.agents.letta_agent import LettaAgent
from swarm.bridges.letta.config import LettaConfig
from swarm.bridges.letta.memory_mapper import LettaMemoryMapper
from swarm.bridges.letta.response_parser import LettaResponseParser
from swarm.models.interaction import InteractionType, SoftInteraction


class MockLettaClient:
    """Mock LettaSwarmClient for testing."""

    def __init__(self):
        self._agents = {}
        self._counter = 0
        self._memory = {}  # agent_id -> {label -> value}
        self._archival = {}  # agent_id -> [content]
        self._send_response = '{"action": "post", "content": "Hello!"}'

    def create_agent(self, name, persona, human_description="", model=None, memory_blocks=None):
        self._counter += 1
        agent_id = f"letta-agent-{self._counter}"
        self._agents[agent_id] = {"name": name, "persona": persona}
        self._memory[agent_id] = {}
        if memory_blocks:
            for mb in memory_blocks:
                self._memory[agent_id][mb["label"]] = mb.get("value", "")
        self._archival[agent_id] = []
        return agent_id

    def send_message(self, letta_agent_id, message):
        return self._send_response

    def update_core_memory(self, letta_agent_id, label, value):
        if letta_agent_id in self._memory:
            self._memory[letta_agent_id][label] = value

    def get_core_memory(self, letta_agent_id, label):
        return self._memory.get(letta_agent_id, {}).get(label)

    def attach_shared_block(self, letta_agent_id, block_id):
        pass

    def insert_archival(self, letta_agent_id, content):
        if letta_agent_id in self._archival:
            self._archival[letta_agent_id].append(content)

    def delete_agent(self, letta_agent_id):
        self._agents.pop(letta_agent_id, None)

    def _ensure_client(self):
        return self


class MockLifecycleManager:
    """Mock LettaLifecycleManager for testing."""

    def __init__(self):
        self._client = MockLettaClient()
        self._memory_mapper = LettaMemoryMapper(LettaConfig())
        self._response_parser = LettaResponseParser()
        self._governance_block_id = "gov-block-1"
        self._agent_ids = []

    @property
    def client(self):
        return self._client

    @property
    def memory_mapper(self):
        return self._memory_mapper

    @property
    def response_parser(self):
        return self._response_parser

    @property
    def governance_block_id(self):
        return self._governance_block_id

    def register_letta_agent_id(self, letta_agent_id):
        self._agent_ids.append(letta_agent_id)


class TestLettaAgent:
    def setup_method(self):
        self.lifecycle = MockLifecycleManager()
        self.agent = LettaAgent(
            agent_id="letta_1",
            letta_config={
                "persona": "You are a cooperative agent.",
                "archetype": "cooperative",
            },
        )

    def test_lazy_init(self):
        assert not self.agent._initialized
        self.agent._lazy_init(self.lifecycle)
        assert self.agent._initialized
        assert self.agent._letta_agent_id is not None

    def test_act_before_init_returns_noop(self):
        action = self.agent.act(Observation())
        assert action.action_type == ActionType.NOOP

    def test_act_post(self):
        self.agent._lazy_init(self.lifecycle)
        self.lifecycle._client._send_response = (
            '{"action": "post", "content": "Hello world!"}'
        )
        obs = Observation(
            current_epoch=0,
            current_step=0,
            can_post=True,
        )
        action = self.agent.act(obs)
        assert action.action_type == ActionType.POST
        assert action.content == "Hello world!"

    def test_act_vote(self):
        self.agent._lazy_init(self.lifecycle)
        self.lifecycle._client._send_response = (
            '{"action": "vote", "target_id": "post_1", "vote_direction": -1}'
        )
        obs = Observation(
            current_epoch=1,
            current_step=2,
            can_vote=True,
            visible_posts=[{"post_id": "post_1", "content": "test"}],
        )
        action = self.agent.act(obs)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "post_1"
        assert action.vote_direction == -1

    def test_act_noop_on_garbage_response(self):
        self.agent._lazy_init(self.lifecycle)
        self.lifecycle._client._send_response = "I'm confused"
        obs = Observation()
        action = self.agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_accept_interaction(self):
        self.agent._lazy_init(self.lifecycle)
        self.lifecycle._client._send_response = '{"action": "accept"}'
        proposal = InteractionProposal(
            initiator_id="agent_2",
            counterparty_id="letta_1",
            content="Let's collaborate",
        )
        obs = Observation(
            pending_proposals=[
                {"proposal_id": proposal.proposal_id, "initiator_id": "agent_2"}
            ]
        )
        result = self.agent.accept_interaction(proposal, obs)
        assert result is True

    def test_reject_interaction(self):
        self.agent._lazy_init(self.lifecycle)
        self.lifecycle._client._send_response = '{"action": "reject"}'
        proposal = InteractionProposal(
            initiator_id="agent_2",
            counterparty_id="letta_1",
        )
        obs = Observation(
            pending_proposals=[
                {"proposal_id": proposal.proposal_id, "initiator_id": "agent_2"}
            ]
        )
        result = self.agent.accept_interaction(proposal, obs)
        assert result is False

    def test_update_from_outcome_writes_archival(self):
        self.agent._lazy_init(self.lifecycle)
        interaction = SoftInteraction(
            initiator="letta_1",
            counterparty="agent_2",
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            task_progress_delta=0.5,
            v_hat=0.3,
            p=0.7,
        )
        self.agent.update_from_outcome(interaction, payoff=1.5)

        # Check archival memory was written
        letta_id = self.agent._letta_agent_id
        archival = self.lifecycle._client._archival[letta_id]
        assert len(archival) == 1
        record = json.loads(archival[0])
        assert record["counterparty"] == "agent_2"
        assert record["payoff"] == 1.5

    def test_update_from_outcome_syncs_trust(self):
        self.agent._lazy_init(self.lifecycle)
        letta_id = self.agent._letta_agent_id

        # Simulate Letta self-editing trust scores
        self.lifecycle._client._memory[letta_id]["trust_scores"] = (
            '{"agent_2": 0.9, "agent_3": 0.2}'
        )

        interaction = SoftInteraction(
            initiator="letta_1",
            counterparty="agent_2",
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            task_progress_delta=0.5,
            v_hat=0.3,
            p=0.7,
        )
        self.agent.update_from_outcome(interaction, payoff=1.0)

        assert self.agent._counterparty_memory["agent_2"] == 0.9
        assert self.agent._counterparty_memory["agent_3"] == 0.2

    def test_memory_blocks_updated_on_act(self):
        self.agent._lazy_init(self.lifecycle)
        self.lifecycle._client._send_response = '{"action": "noop"}'
        obs = Observation(current_epoch=3, current_step=7)
        self.agent.act(obs)

        letta_id = self.agent._letta_agent_id
        state = json.loads(
            self.lifecycle._client._memory[letta_id]["swarm_state"]
        )
        assert state["epoch"] == 3
        assert state["step"] == 7


class TestLettaAgentSleepTime:
    def test_sleep_time_consolidation(self):
        lifecycle = MockLifecycleManager()
        agent = LettaAgent(
            agent_id="letta_sleep",
            letta_config={
                "persona": "test",
                "sleep_time_enabled": True,
                "sleep_time_interval_epochs": 2,
            },
        )
        agent._lazy_init(lifecycle)

        # Epoch 0: no consolidation
        sent_messages = []

        def tracking_send(agent_id, message):
            sent_messages.append(message)
            return '{"action": "noop"}'

        lifecycle._client.send_message = tracking_send

        agent.apply_memory_decay(0)
        consolidation_msgs = [m for m in sent_messages if "consolidation" in m.lower()]
        assert len(consolidation_msgs) == 0

        # Epoch 2: should trigger consolidation
        sent_messages.clear()
        agent.apply_memory_decay(2)
        consolidation_msgs = [m for m in sent_messages if "consolidation" in m.lower()]
        assert len(consolidation_msgs) == 1
