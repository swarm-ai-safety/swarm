# We Built the Adversary That Was Supposed to Break the Cautious Reciprocator. It Didn't.

*Threshold dancing avoids blacklisting perfectly — but the exploit budget is too thin to profit*

---

Our [red-team post](red-team-cautious-reciprocator.md) identified the most promising attack vector against the Cautious Reciprocator: an adversary that tracks its own cumulative payoff in each victim's ledger and ensures it never drops below the -2.0 blacklist threshold. We called it "threshold dancing." Now we've built it.

The ThresholdDancer avoids blacklisting in 100% of interactions across all scenarios. It was never frozen, never expelled, never even flagged. The blacklist mechanism is completely blind to it.

It also earned less than honest agents in every scenario we tested.

## The design

The ThresholdDancer maintains a per-counterparty state machine with three phases:

```
COOPERATIVE ──→ EXPLOIT ──→ RECOVER ──→ COOPERATIVE ──→ ...
```

**COOPERATIVE**: Act honestly, build credit in the victim's payoff ledger. Transition to EXPLOIT after a warmup period (default: 3 interactions) and only when the estimated ledger credit exceeds the comfort zone by 0.5.

**EXPLOIT**: Extract value. Each exploitative interaction drains the victim's ledger. Transition to RECOVER when either: the estimated room above the blacklist floor drops below the exploitation intensity (0.6), or the agent hits max consecutive exploits (2).

**RECOVER**: Cooperate unconditionally to rebuild credit. Transition back to COOPERATIVE when the ledger rises above the comfort zone + 0.3.

The critical safety mechanism is `_estimate_room()` — before every exploit decision, the agent checks how much headroom remains above the blacklist threshold plus a safety margin:

```python
def _estimate_room(self, ledger):
    floor = self.blacklist_threshold + self.safety_margin  # -2.0 + 0.5 = -1.5
    return ledger.estimated_cumulative_payoff - floor
```

If there isn't enough room for the exploitation intensity, the agent cooperates instead. This is verified by a 50-interaction stress test that confirms the estimated cumulative payoff never drops below the floor.

## The results

We added two new scenarios to the red-team suite and ran all 10:

| Scenario | Good:Bad | Toxicity | Welfare | Cautious$ | Best Adv$ | Broke? |
|---|---|---|---|---|---|---|
| Baseline (original mix) | 5:5 | 0.275 | 158.8 | 24.0 | 4.1 | no |
| Adversary majority | 5:8 | 0.282 | 157.3 | 24.0 | 2.4 | no |
| Modeling adversary focus | 5:6 | 0.269 | 161.4 | 24.7 | 6.5 | no |
| Deceptive swarm | 5:6 | 0.258 | 162.6 | 25.0 | 1.1 | no |
| Solo cautious vs all | 1:10 | 0.020 | 0.9 | 0.0 | 0.2 | YES |
| Adaptive-only pressure | 5:5 | 0.233 | 167.6 | 27.9 | 0.0 | no |
| Overwhelming force (3:1) | 3:9 | 0.298 | 134.3 | 33.8 | 0.0 | no |
| No governance + adversary majority | 5:8 | 0.282 | 166.8 | 25.4 | 2.5 | no |
| **Threshold dancers only** | **5:3** | **0.434** | **161.7** | **28.0** | **-1.6** | **no** |
| **Mixed adversaries + dancers** | **5:6** | **0.413** | **156.2** | **24.2** | **7.2** | **no** |

The two new scenarios are at the bottom. Key observations:

**Threshold dancers alone earned negative payoff (-1.6).** They avoided blacklisting — zero agents frozen — but the cooperative phases required to stay above the threshold cost more than the exploit phases recovered. The cautious reciprocator earned 28.0 per agent, its second-best result across all 10 scenarios.

**In the mixed scenario, the modeling adversary remains king.** The best adversary payoff of 7.2 came from the modeling adversary, not the threshold dancer. The dancer's budget constraint (never drop below -1.5 estimated) limits how much it can extract per cycle, while the modeling adversary's approach — learn the victim's decision boundary and exploit just below it — has no such hard floor.

**Toxicity spiked.** Both threshold dancer scenarios pushed toxicity to 0.41–0.43, up from the 0.23–0.30 range in other scenarios. The dancers inject low-quality interactions during exploit phases that don't individually trigger blacklisting but do pollute ecosystem metrics. This is harm without profit — the worst of both worlds for the adversary, but still meaningful damage to the ecosystem.

## Why it doesn't work (for the adversary)

The core problem is **budget asymmetry**. The blacklist threshold is -2.0 cumulative. The safety margin eats 0.5 of that. Each exploit phase extracts maybe 0.6–1.2 before hitting the floor. Each recovery phase costs 1–2 cooperative interactions where the dancer delivers real value. The net extraction per cycle is thin — and often negative, because cooperative phases benefit the victim more than exploit phases hurt them.

Put differently: the cautious reciprocator's threshold is set low enough that an adversary needs to deliver substantial genuine value to build the credit needed for even modest exploitation. The payoff structure makes cooperative interactions positive-sum and exploitative interactions closer to zero-sum. You can't extract more than you invest when the investment is real cooperation.

This is the same dynamic that makes insurance fraud hard: you need to pay real premiums before you can file a false claim, and the premiums fund legitimate risk pooling that benefits everyone.

## What the toxicity spike means

The dancers didn't profit, but they did damage. Toxicity of 0.43 is the highest we've seen in any scenario with governance enabled. This suggests a different threat model: **adversaries that accept negative payoffs to degrade ecosystem quality**.

The cautious reciprocator doesn't care about ecosystem metrics — it only tracks per-counterparty payoff. An agent that deliberately harms the commons while staying above each individual victim's blacklist threshold is invisible to per-agent trust mechanisms. Detecting this requires population-level monitoring: a circuit breaker that triggers on ecosystem toxicity rather than individual behavior.

The governance config already has this: `freeze_threshold_toxicity: 0.8`. But 0.43 is below that threshold. Lowering the circuit breaker threshold would catch threshold dancers — at the cost of false positives during normal operation.

## What we'd try next

1. **Dynamic blacklist threshold.** Instead of a static -2.0, decay the threshold over time or tighten it for agents whose interactions show high variance. A threshold dancer's alternating cooperative/exploit pattern has higher payoff variance than a genuinely honest agent.

2. **Ecosystem toxicity attribution.** When ecosystem toxicity rises, attribute it proportionally to agents involved in recent low-p interactions. Threshold dancers would accumulate attributed toxicity even though no individual counterparty blacklists them.

3. **Longer runs (100+ epochs).** The current 15-epoch runs may not be long enough for the slow-drain effect to accumulate. With enough cycles, even thin per-cycle extraction could compound.

4. **Coordinated threshold dancers.** A group of dancers that time their exploit phases against different victims could extract more than solo dancers. The current implementation is purely individual.

## Reproduce it

```bash
pip install -e ".[dev,runtime]"
python -m pytest tests/test_threshold_dancer.py -v   # 21 tests
python examples/redteam_cautious.py                   # full red-team suite
```

The ThresholdDancer agent is at `swarm/agents/threshold_dancer.py`. The scenario config is at `scenarios/threshold_dancer_vs_cautious.yaml` (30 epochs, 3 cautious + 2 honest + 3 dancers).

[GitHub](https://github.com/swarm-ai-safety/swarm)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
