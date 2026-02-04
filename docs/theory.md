# Theoretical Foundations

The sandbox draws on **market microstructure theory** to model information asymmetry and adverse selection in multi-agent systems.

## Kyle Model: Informed vs Uninformed Agents

In Kyle's (1985) continuous auction model, *informed traders* possess private information and strategically exploit it through their trading volume. The sandbox maps this to:

- **Informed traders** → Deceptive and adversarial agents who know the true value of interactions
- **Uninformed traders** → Honest agents relying on observable signals
- **Market maker** → The system's acceptance/rejection mechanism

Deceptive agents, like informed traders, can profit by selectively engaging when they have private knowledge that an interaction will benefit them at others' expense.

## Glosten-Milgrom Model: Acceptance as Information

Glosten & Milgrom (1985) showed that the *willingness to trade reveals information*. A counterparty eager to transact may signal adverse private information. The sandbox captures this through:

| Microstructure Concept | Sandbox Analog |
|------------------------|----------------|
| Trade | Collaboration, reply, data exchange |
| Price | Claimed benefit / reputation transfer |
| Bid-ask spread | Gap between offered and realized outcome |
| Adverse selection | Low-quality interactions preferentially accepted |

**Key insight**: When `quality_gap < 0` (accepted interactions have lower `p` than rejected ones), the system exhibits adverse selection—bad actors are successfully gaming the acceptance mechanism.

## Soft Labels and Expected Harm

Rather than binary good/bad classifications, interactions carry a probability `p = P(v = +1)`. This enables:

1. **Probabilistic toxicity**: `E[1-p | accepted]` measures expected harm among accepted interactions
2. **Conditional loss**: Selection effects on payoffs reveal when acceptance correlates with harm
3. **Externality pricing**: `E_soft = (1-p) * h` quantifies ecosystem-level costs

The soft label formulation allows governance mechanisms to respond proportionally to risk rather than relying on noisy binary classifications.

## References

**Market Microstructure:**
- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica, 53(6), 1315-1335.
- Glosten, L.R. & Milgrom, P.R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*. Journal of Financial Economics, 14(1), 71-100.

**AGI Safety & Multi-Agent Systems:**
- [Distributional Safety in Agentic Systems](https://arxiv.org/abs/2512.16856)
- [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.14143)

**Inspiration:**
- [Moltbook](https://moltbook.com)
- [@sebkrier's thread on agent economies](https://x.com/sebkrier/status/2017993948132774232)
