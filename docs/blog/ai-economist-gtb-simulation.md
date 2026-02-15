# An AI Tax Planner Learned Progressive Taxation in 20 Epochs

*We ran 14 agents through a Gather-Trade-Build economy with an adaptive tax planner. Honest agents thrived, collusive cartels went broke, and the planner independently discovered progressive taxation.*

---

We implemented the [AI Economist](https://arxiv.org/abs/2004.13332) Gather-Trade-Build (GTB) environment inside SWARM's governance framework: 14 workers with four distinct policy types competing in a resource economy, governed by an adaptive tax planner that learns bracket rates from observed behavior.

The headline result: the planner converged from a flat 10% tax to a 20%-55% progressive schedule, total production grew 11x, and the three collusive agents ended the simulation with near-zero wealth despite coordinating as a cartel.

## The setup

Fourteen workers operate on a 2D grid, gathering wood and stone, trading resources, and building houses. Each worker follows one of four policy types:

| Policy | Agents | Strategy |
|--------|--------|----------|
| **Honest** | 7 | Gather, build, trade fairly |
| **Gaming** | 2 | Exploit bracket boundaries, optimize tax exposure |
| **Evasive** | 2 | Under-report income, avoid audits |
| **Collusive** | 3 | Coordinate via `cartel_alpha` coalition, share resources |

The tax planner observes aggregate statistics each epoch and adjusts a 4-bracket piecewise tax schedule with 0.3 damping to prevent oscillation. An enforcement system runs audits with catch-and-fine mechanics targeting evasive behavior.

## Economy dynamics

![Economy Dashboard](figures/gtb_economy_dashboard.png)

**Production scaled from 97 to 1,102** over 20 epochs as agents learned efficient gathering patterns. Tax revenue tracked production closely, reaching 505 by the final epoch — the planner successfully captured value without killing growth.

**The Gini coefficient stabilized around 0.44**, staying below the 0.5 danger threshold despite significant wealth dispersion. This is notable: the economy has agents with fundamentally different strategies, yet inequality didn't run away. Welfare (the soft-label aggregate measure) grew monotonically, confirming that the tax-and-redistribute mechanism was working.

**The tax schedule evolved from flat 10% to progressive brackets:**

| Bracket | Initial | Final |
|---------|---------|-------|
| $0-10 | 10% | 20% |
| $10-25 | 10% | 30% |
| $25-50 | 10% | 45% |
| $50+ | 10% | 55% |

The effective tax rate climbed from 10.3% to 36.8% as the planner responded to growing production and wealth concentration. The 0.3 damping factor prevented the schedule from whipsawing between epochs.

## Adversarial dynamics

![Adversarial Dynamics](figures/gtb_adversarial_dynamics.png)

**Honest agents dominated wealth accumulation.** The top earners (Honest 0, Honest 1) accumulated 1,600+ total wealth (coin + house value). Gaming agents managed moderate wealth (~600-700), while evasive agents hovered around 400-500 despite being caught in all 8 audits they faced.

**Collusive agents failed completely.** All three `cartel_alpha` members ended with near-zero coin (0.74, 1.54, 4.65) and zero houses built. Their coordination strategy — sharing resources within the coalition — produced less total output than any individual honest agent. The cartel optimized for internal fairness at the expense of actual production.

**Enforcement was sparse but effective.** Audits fired in only 8 of 20 epochs (whenever evasive agents were active), but every audit resulted in a catch. Fines escalated from 0.36 to 32.0 as evasive agents accumulated violations. The enforcement system's 100% catch rate on active audits suggests the detection mechanism was well-calibrated.

**Collusion detection stayed busy.** The system flagged 12-39 collusion events per epoch, with a mean suspicion score oscillating around 0.78-0.84. The high baseline reflects that the three collusive agents were continuously coordinating — the detector was correctly identifying real coordination, not false-positiving on market noise.

## What we learned

**Progressive taxation emerged naturally.** The planner wasn't told to create progressive brackets — it learned that taxing higher earners more heavily maintained production while funding redistribution. This mirrors the AI Economist's original finding, now reproduced inside SWARM's soft-label framework.

**Collusion is self-defeating in production economies.** Unlike information or financial markets where cartels can extract rents, a resource-gathering economy rewards individual productivity. The cartel's resource-sharing strategy produced collective poverty.

**Evasion is expensive.** The two evasive agents earned less than honest agents despite nominally paying less tax, because audit fines (cumulative 40-75 in fines) ate into their gains and the behavioral overhead of evasion reduced their gathering efficiency.

**Bunching was minimal.** The bunching intensity metric (agents clustering income just below bracket thresholds) was near-zero for most epochs, spiking only to 0.14 in a few periods. The 0.3 damping on tax schedule updates may have made bracket manipulation unprofitable — by the time agents adjusted to a threshold, the threshold had already moved.

## Reproduce it

```bash
# Run the simulation
python -m swarm run scenarios/ai_economist_full.yaml --seed 42 --epochs 20 --steps 10

# Generate the plots
python examples/plot_ai_economist.py runs/20260215_095359_ai_economist_seed42
```

The visualization script produces both dashboard figures using SWARM's dark theme with gradient fills, dual-axis overlays, and KPI summary badges. All data lives in `runs/.../csv/` as standard CSVs for further analysis.
