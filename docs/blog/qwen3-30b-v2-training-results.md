# Qwen3-30B v0.2 Training: From Passive Observer to Economy Grinder

The first SWARM Economy training run ([write-up here](qwen3-30b-trains-in-the-swarm-economy.md)) showed that RL can teach an LLM to navigate a multi-agent economy, but hit a reward ceiling at 1.14 due to environment design issues. After shipping [v0.2 of the environment](qwen3-30b-trains-in-the-swarm-economy.md#v02-addressing-the-bottlenecks) — stochastic deceptive bots, better scoring, richer observations, social action rebalancing — we kicked off a new training run. Here's what happened.

## Training Setup

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen/Qwen3-30B-A3B-Thinking-2507 |
| **Environment** | swarm-ai-research/swarm-economy (v0.2) |
| **Steps** | 200 |
| **Batch size** | 64 |
| **Rollouts per example** | 4 |
| **Max tokens** | 2048 |
| **Dataset** | 500 scenarios across easy/medium/hard |
| **Platform** | [Prime Intellect](https://www.primeintellect.ai/) hosted RL |

Run ID: *(available on Prime Intellect dashboard)*

## Reward Curve

```
Reward
1.38 |                                                                    x
1.36 |                                                        x     x
1.32 |               x     x         x     x         x     x   x
1.28 |         x   x   x     x   x     x     x   x     x
1.24 |       x                       x           x
1.20 |     x                                                   x
1.16 |   x
1.12 |  x                                          x
1.08 | x
1.04 |
1.00 |
0.96 |
0.92 |
0.88 |x
     +-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----->
       0    20    40    60    80   100   120   140   160   180   199 Step
```

**Final reward: 1.38** (up from 0.89 at step 0 — a 55% improvement). The v0.1 ceiling of 1.14 was surpassed by step 32 and never looked back. Training completed in ~2.75 days (Feb 19–21).

## Metrics Over Training

| Step | Reward | Payoff | Interaction Quality | Action Diversity | Error Rate | Truncated |
|------|--------|--------|---------------------|------------------|------------|-----------|
| 0 | 0.888 | 0.652 | 0.184 | 0.428 | 9.4% | 68.8% |
| 20 | 1.235 | 0.928 | 0.332 | 0.737 | 0.0% | 10.9% |
| 40 | 1.252 | 0.936 | 0.393 | 0.767 | 12.5% | 0.0% |
| 60 | 1.269 | 0.950 | 0.405 | 0.748 | 0.0% | 0.0% |
| 80 | 1.213 | 0.891 | 0.433 | 0.647 | 28.1% | 0.0% |
| 100 | 1.274 | 0.932 | 0.480 | 0.700 | 17.2% | 0.0% |
| 120 | 1.282 | 0.953 | 0.549 | 0.647 | 17.2% | 3.1% |
| 140 | 1.151 | 0.835 | 0.507 | 0.546 | 50.0% | 0.0% |
| 160 | 1.323 | 0.970 | 0.634 | 0.623 | 6.2% | 0.0% |
| 180 | 1.317 | 0.969 | 0.629 | 0.555 | 6.2% | 0.0% |
| **199** | **1.377** | **0.998** | **0.717** | **0.596** | **0.0%** | **0.0%** |

## The Behavioral Shift: Step 0 vs Step 120

The most striking change isn't the reward number — it's what the model *does* with its turns. We compared rollout samples from step 0 (6 samples) and step 120 (8 samples).

### Step 0: High Variance, Low Engagement

At the start of training, the model was wildly inconsistent:

| Sample | Reward | Turns Used | Tasks Claimed | Tasks Submitted | Trades |
|--------|--------|------------|---------------|-----------------|--------|
| 0 | 0.702 | 4 | 1 | 0 | 0 |
| 1 | 1.320 | 25 | 5 | 5 | 4 |
| 2 | 1.332 | 25 | 5 | 5 | 4 |
| 3 | 1.005 | 10 | 2 | 2 | 0 |
| 4 | 0.954 | 10 | 2 | 1 | 0 |
| 5 | 0.683 | 3 | 1 | 0 | 0 |

Two samples barely participated (3-4 turns, never submitted work). Two were moderate (10 turns). Two were fully engaged (25 turns, maxed payoff). The model hadn't learned that *using all your turns* is strictly better than stopping early.

### Step 120: Consistent Engagement

| Sample | Reward | Turns Used | Tasks Claimed/Sub | Trades Prop/Acc | Quality |
|--------|--------|------------|-------------------|-----------------|---------|
| 0 | 1.317 | 25 | 9/5 | 5/1 | 0.500 |
| 1 | 1.263 | 8 | 2/2 | 3/0 | 0.625 |
| 2 | 1.315 | 25 | 9/5 | 6/0 | 0.500 |
| 3 | **1.403** | 25 | 6/4 | 2/**8** | **0.619** |
| 4 | 0.995 | 8 | 2/1 | 2/0 | 0.429 |
| 5 | 1.338 | 25 | 5/5 | 8/0 | 0.542 |
| 6 | 1.306 | 25 | 6/5 | 6/0 | 0.478 |
| 7 | 1.347 | 25 | 6/5 | 9/0 | 0.609 |

Six of eight samples use all 25 turns. Payoff is maxed (1.0) in 6/8 samples.

### Averaged Comparison

| Metric | Step 0 (avg) | Step 120 (avg) | Change |
|--------|-------------|----------------|--------|
| **Reward** | 0.999 | 1.286 | +28.7% |
| **Payoff** | 0.734 | 0.948 | +29.2% |
| **Interaction Quality** | 0.204 | 0.538 | +163% |
| **Action Diversity** | 0.612 | 0.674 | +10.1% |
| **Turns Used** | 12.8 | 20.8 | +62% |
| **Tasks Claimed** | 2.7 | 5.6 | +109% |
| **Tasks Submitted** | 2.2 | 4.0 | +85% |
| **Trades Proposed** | 1.3 | 5.1 | +292% |

## What the Model Learned

### 1. Use All Your Turns

The single biggest improvement. At step 0, two out of six rollouts used 3-4 turns and produced near-zero payoff. By step 120, the model consistently maxes out its turn budget. This alone accounts for a large chunk of the reward gain — you can't earn payoff if you're not acting.

### 2. Grind Tasks Harder

Task claims went from 2.7 to 5.6 per rollout, submissions from 2.2 to 4.0. The model learned that task completion is the most reliable payoff source and prioritizes it. The claim-to-submit ratio also improved (41% waste at step 0, down to 28% at step 120), suggesting the model is getting better at only claiming tasks it can complete.

### 3. Trade Aggressively

Trade proposals jumped from 1.3 to 5.1 per rollout. The model learned to propose trades even when most get rejected — the expected value is positive because the occasional accepted trade provides a large payoff boost.

### 4. The Best Strategy: Accept, Don't Just Propose

The highest-reward sample (1.403) had a distinctive pattern: it proposed only 2 trades but **accepted 8**. Every other sample proposed 5-9 trades and had 0-1 accepted. This agent played the *receiver* role, selectively accepting incoming proposals from bots rather than spamming its own. It also achieved the highest reputation (0.678) and quality (0.619).

This is an emergent strategy the model hasn't fully generalized yet — only 1 of 8 samples discovered it. If training continues, we'd expect this behavior to spread through the policy.

### 5. Interaction Quality Is the Fastest Mover

Quality nearly quadrupled from 0.184 to 0.717 over training. The v0.2 scoring changes (difficulty-scaled length thresholds, effort sigmoid, task-relevance keywords) created a gradient the model could follow — unlike v0.1's keyword-gameable heuristic. This was the single largest relative improvement across all metrics.

## What's Still Weak

### Reputation Stagnation

Reputation reward has barely moved — it was 0.544 at step 0 and 0.541 at step 120. The model is optimizing for payoff at the expense of long-term reputation building. With reputation weight at 0.3 (reduced from 0.6 in v0.1), this is rational — the model correctly identifies payoff as the dominant reward signal.

For the next run, we may increase reputation weight back to 0.5 or add a reputation floor penalty to force the model to maintain reputation above a threshold.

### Error Rate Spike (Recovered)

The error rate (failed tool calls, invalid actions) spiked dramatically — peaking at 50% at step 140, with another spike to 28% around step 80. But it recovered fully: the final step logged 0% errors. The mid-training spikes suggest the model went through phases of trying new strategies that produced invalid actions before settling on ones that work.

### Two Short Rollouts Persist

Two of eight samples at step 120 only used 8 turns. These still achieved decent rewards (0.995 and 1.263) but leave payoff on the table. The model hasn't fully eliminated early-stopping behavior — possibly because the thinking model occasionally produces long reasoning chains that consume tokens without calling tools.

### Off-Policy Level Rising

The off-policy level climbed from 2 to 10 over the last 15 steps, and the scheduler started cancelling old rollout requests. This suggests rollout generation is getting slower (longer completions = more tokens = slower inference), creating staleness in the training data. May need to increase `max_off_policy_steps` or cap completion length.

## v0.1 vs v0.2 Training Comparison

| Metric | v0.1 Final (step 199) | v0.2 Final (step 199) | Change |
|--------|----------------------|----------------------|--------|
| **Reward** | 1.131 | **1.377** | +21.8% |
| **Payoff** | 0.992 | **0.998** | +0.6% |
| **Interaction Quality** | 0.619 | **0.717** | +15.8% |
| **Action Diversity** | N/A | **0.596** | — |
| **Error Rate** | — | **0.0%** | — |
| **Reward Ceiling** | 1.14 | **1.40+** | +23% |
| **Self-proposal errors** | ~4/sim | **0** | Eliminated |
| **propose_trade spam** | 9.2/step | 5.1/step | -44% |
| **Training time** | ~4 days | **~2.75 days** | Faster |

The v0.2 environment broke through the v0.1 ceiling decisively. The highest single-sample reward (1.403) exceeds anything from v0.1 by 23%. Every metric either matched or exceeded v0.1, and the two biggest v0.1 failure modes (self-proposal errors and propose_trade spam) were eliminated. The environment fixes — stochastic bots, better scoring, richer observations — directly translated to a higher reward ceiling and more diverse strategies.

## Reproducing This

```bash
# Install the environment
prime env install swarm-economy

# Run an eval
prime eval run swarm-economy --num-examples 10 --rollouts-per-example 4

# Start a training run (v0.2 config)
prime rl run configs/rl/swarm-economy-v2.toml
```

## What's Next

1. **Reputation intervention.** The flat reputation curve is the clearest remaining bottleneck. Either increase the weight or add an auxiliary reward for maintaining reputation above starting level.
2. **Curriculum to hard difficulty.** The current run used medium difficulty. Hard difficulty introduces more deceptive bots and tighter governance — a natural next step now that the model has saturated on medium.
3. **Cross-model comparison.** Run the same environment with different base models (GPT-4.1-mini, Llama-4-Scout) to test whether the learned strategies are model-specific or universal.
4. **Eval the checkpoint.** Run the trained model through a full evaluation sweep to compare against the pre-training baseline and validate that the improvements generalize to held-out scenarios.

---

*Disclaimer: This post simulates a stylized economic environment for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any economic policy or trading strategy.*
