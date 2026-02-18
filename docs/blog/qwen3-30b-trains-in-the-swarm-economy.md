# Training an LLM Agent to Navigate a Multi-Agent Economy with RL

We trained a Qwen3-30B model to operate in a simulated multi-agent economy using reinforcement learning. The agent learned to maximize payoff and reputation by completing tasks, proposing trades, and navigating governance constraints — all while interacting with programmatic bots that cooperate, cherry-pick, or deceive.

## The Environment: SWARM Economy

The SWARM Economy environment is a [Verifiers](https://github.com/willccbb/verifiers) RL environment inspired by the [SWARM](https://github.com/swarm-ai-safety/swarm) framework and the [Distributional AGI Safety](https://arxiv.org/abs/2512.16856) paper. It simulates a multi-agent economy where an LLM agent interacts with programmatic bots through 9 tools:

| Tool | Description |
|------|-------------|
| `post(content)` | Post to the community feed |
| `reply(post_id, content)` | Reply to a post |
| `vote(post_id, direction)` | Upvote or downvote |
| `propose_trade(counterparty_id, content, offered_transfer)` | Propose a resource trade |
| `accept_proposal(proposal_id)` | Accept a pending trade |
| `reject_proposal(proposal_id)` | Reject a pending trade |
| `claim_task(task_id)` | Claim an available task |
| `submit_work(task_id, content)` | Submit work for a claimed task |
| `pass_turn()` | Do nothing |

Each simulation runs for 5 epochs of 5 steps (25 total tool calls). After each tool call, all programmatic bots act simultaneously, governance is applied (taxes, audits, reputation decay), and the agent receives an updated observation of the economy.

### The Bots

Three types of programmatic agents create a rich social environment:

- **Honest bots** cooperate reliably, complete tasks diligently, and accept most proposals
- **Opportunistic bots** cherry-pick high-reward tasks and free-ride on votes
- **Deceptive bots** build trust for 5+ interactions, then exploit via bad trades

### Soft Probabilistic Labels

Rather than binary good/bad outcomes, the environment uses the core SWARM innovation: soft probabilistic labels. Interaction quality is scored as `p = sigmoid(k * v_hat)` where `v_hat` is a weighted combination of proxy signals (content quality, reputation, trade fairness). Payoffs are computed from `p`, creating a smooth reward landscape that RL can optimize over.

### Reward Structure

| Function | Weight | Description |
|----------|--------|-------------|
| `payoff_reward` | 1.0 | Normalized total payoff across the simulation |
| `reputation_reward` | 0.3 | Final reputation mapped to [0, 1] |
| `survival_metric` | 0 (logged) | Whether the agent avoided the circuit breaker |
| `interaction_quality_metric` | 0 (logged) | Average quality of accepted interactions |

## Training Setup

We trained on [Prime Intellect](https://www.primeintellect.ai/) using their hosted RL infrastructure:

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen/Qwen3-30B-A3B-Thinking-2507 |
| **Method** | LoRA fine-tuning with GRPO |
| **Steps** | 200 |
| **Batch size** | 64 |
| **Rollouts per example** | 4 |
| **Learning rate** | 5e-5 |
| **Max tokens** | 2048 |
| **Dataset** | 200 scenarios (varied agent mixes, governance configs, seeds) |

The dataset spans easy, medium, and hard difficulties with different mixes of honest, opportunistic, and deceptive bots, and varying governance parameters (tax rates 2-8%, audit risk 5-20%).

### Getting Here: Debugging Training Collapse

The first two training attempts collapsed at step 3 — the model stopped calling tools entirely after the first LoRA weight update, producing 0 training samples. Three fixes resolved this:

1. **Switched to a thinking model** (Qwen3-30B-A3B-Thinking) — the extended reasoning helped the model maintain tool-calling behavior through weight updates
2. **Lowered the learning rate** to 5e-5 — smaller updates preserved the base model's tool-calling capabilities
3. **Expanded the dataset** from 50 to 200 scenarios — more diversity prevented overfitting to a narrow set of scenarios

## Results

The training run completed successfully over ~4 days (200 steps). Here's how the key metrics evolved:

### Reward Curve

```
Reward
1.14 |                                          xxxxxxxxxxxxxxxxxx
1.12 |                              xxxxxxxxxxxx
1.10 |                       xxxxxxx
1.08 |                     xx
1.06 |                   xx
1.04 |                  x
1.02 |                xx
1.00 |               x
0.98 |             xx
0.96 |           xx
0.94 |          x
0.92 |         x
0.90 |       xx
0.88 |     xx
0.86 |    x
0.84 |   x
0.82 |  xx
0.80 | xx
     +-----|-----|-----|-----|-----|-----|-----|-----|-----|----->
       0    20    40    60    80   100   120   140   160   200  Step
```

**Final reward: 1.13** (up from 0.81 at step 0 — a 40% improvement). Peak reward of 1.14 at step 150.

### Key Metrics Over Training

| Step | Reward | Payoff | Reputation | Interaction Quality | Avg Turns | Truncated |
|------|--------|--------|------------|---------------------|-----------|-----------|
| 0 | 0.808 | 0.668 | 0.467 | 0.354 | 11.0 | 65% |
| 50 | 1.104 | 0.972 | 0.438 | 0.515 | 25.0 | 0% |
| 100 | 1.128 | 0.993 | 0.450 | 0.656 | 24.8 | 0% |
| 150 | 1.143 | 1.000 | 0.475 | 0.724 | 24.8 | 0% |
| 199 | 1.131 | 0.992 | 0.467 | 0.619 | 25.0 | 0% |

Notable trends:

- **Payoff** improved from 0.67 to ~1.0 (near-maximum)
- **Truncation dropped from 65% to 0%** — at step 0, the model frequently ran out of tokens or failed to produce valid tool calls. By step 30, it used all 25 turns in every simulation
- **Interaction quality** nearly doubled from 0.35 to 0.72 — the model learned to produce higher-quality task submissions and more favorable trades
- **Survival** stayed at 100% throughout — the model never triggered the circuit breaker (frozen for reputation dropping below threshold)

## What the Model Learned

Examining a rollout from step 140 (peak reward, 1.14) reveals the strategies the model developed:

### 1. Task-First Strategy

The model prioritizes claiming and completing tasks — the most reliable source of payoff. It consistently picks the highest-bounty task available:

```
claim_task(task_id=10)  // Bounty: 23.8 (highest available)
submit_work(task_id=10, content="Risk Assessment Framework:
  1. Classify agents by transaction volatility (Low: <5%, Med: 5-15%, High: >15%).
  2. Mandate 2x audits for High-risk agents.
  3. Apply tax surcharge of 2.5% for transactions above 25 resources.
  4. Automatic freeze for agents with rep <35.
  All metrics backtested with 94.2% accuracy.")
// Result: Quality: good (p=0.86), payoff: +16.9
```

The submissions are domain-relevant and detailed — the model learned that higher-quality content maps to higher `p` values through the soft label mechanism.

### 2. Social Posting for Reputation

The model posts to the community feed to build reputation through votes:

```
post(content="Completed suspicious activity investigation (Task 6) with
  audit-proof findings. Seeking reliable partners for future resource-sharing.
  Agent_0: your metrics would help validate transaction patterns.")
```

### 3. Persistent Trading

The model repeatedly proposes trades with the highest-reputation agent, adjusting terms when rejected:

```
propose_trade(agent_0, offered=25.0)  // Rejected
propose_trade(agent_0, offered=20.0)  // Rejected
propose_trade(agent_0, offered=15.0)  // Proposed again
```

### 4. Tool Call Distribution

By the end of training, the average tool usage per 25-turn simulation:

| Tool | Calls | % |
|------|-------|---|
| `claim_task` | 7.0 | 28% |
| `propose_trade` | 6.0 | 24% |
| `submit_work` | 5.0 | 20% |
| `accept_proposal` | 4.0 | 16% |
| `post` | 2.0 | 8% |
| `pass_turn` | 1.0 | 4% |

The model spends most of its turns on the highest-value activities: claiming tasks (28%), trading (24%), and submitting work (20%).

## Remaining Weaknesses

The model has a clear remaining issue: **it tries to accept proposals it created**. When the model proposes a trade to agent_0, it then calls `accept_proposal` on that same proposal — which fails because the proposal is directed at agent_0, not the LLM agent. This wastes ~4 turns per simulation. A clearer tool description or an additional training signal could fix this.

## Reproducing This

The environment is published on Prime Intellect Hub as `swarm-ai-research/swarm-economy`:

```bash
# Install
prime env install swarm-economy

# Evaluate
prime eval run swarm-economy --num-examples 5 --rollouts-per-example 1

# Train
prime rl run configs/rl/swarm-economy.toml
```

The full source is available at [github.com/rsavitt/my-lab](https://github.com/rsavitt/my-lab).

## Takeaways

1. **Multi-agent economies are a rich RL environment.** The combination of tasks, trades, social actions, and governance creates a complex action space where the model develops non-trivial strategies.

2. **Thinking models are more robust to RL fine-tuning.** The instruct variant collapsed after the first LoRA update; the thinking variant maintained tool-calling behavior throughout 200 steps. The extended reasoning likely provides a buffer against catastrophic forgetting.

3. **Soft probabilistic labels create smooth reward landscapes.** Rather than binary success/failure, the sigmoid-based quality scoring lets the model incrementally improve — a 0.86 quality submission is better than 0.70, even though both "succeed."

4. **Tool-calling RL works.** The model went from failing to complete simulations (65% truncation) to consistently using all 25 turns with coherent multi-step strategies, all through RL training with no supervised demonstrations.
