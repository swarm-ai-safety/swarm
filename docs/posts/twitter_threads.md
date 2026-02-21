# Twitter Thread Drafts

Ready-to-post threads for @ResearchSwarmAI. Edit as needed before posting.

---

## Thread 1: Launch Announcement

**1/**
We built an open-source framework for studying when multi-agent AI ecosystems collapse.

Main finding: there's a sharp phase transition. Governance that works fine at 37.5% adversarial agents fails completely at 50%.

Meet SWARM.
github.com/swarm-ai-safety/swarm

**2/**
Most safety work focuses on aligning one model. But what actually gets deployed looks like ecosystems — tool-using assistants, autonomous coders, trading bots interacting in shared environments.

These can fail catastrophically even when no individual agent is misaligned.

**3/**
We borrowed from financial markets. Adverse selection in trading (Kyle 1985, Glosten-Milgrom 1985) tells us how informed traders exploit uninformed ones.

Same dynamics apply when deceptive AI agents interact with honest ones. The governance mechanism plays the role of the market maker.

**4/**
Instead of binary safe/unsafe labels, every interaction gets a calibrated probability: p = P(beneficial).

This lets us measure things binary labels can't: adverse selection, governance sensitivity, ecosystem-level toxicity.

**5/**
11 scenarios. 209 epochs. 81 agent-slots. Three regimes:

Cooperative (0-20% adversarial): stable
Contested (20-37.5%): declining but functional
Collapse (50%): welfare hits zero by epoch 12

The transition is abrupt, not gradual.

**6/**
The lever that actually matters? Collusion detection.

Individual-focused tools (audits, reputation, staking) are necessary but insufficient. You need pattern-based detection across the interaction graph.

Same way financial regulators catch wash trading — look at the pattern, not the individual trades.

**7/**
The Purity Paradox: populations with only 10% honest agents achieve 74% HIGHER welfare than 100% honest populations.

Heterogeneity creates competitive pressure. But this reverses when agents bear the full cost of harmful interactions (rho >= 0.5).

It's a measurement problem, not a behavioral one.

**8/**
SWARM is:
- pip install swarm-safety
- 2200+ passing tests
- 23 scenario configs
- Bridges to Concordia, OpenClaw, GasTown, AgentXiv, ClawXiv
- MIT license
- Full MkDocs documentation

5 minutes to first result, no API keys needed.

**9/**
The framework explicitly addresses reflexivity — what happens when agents can read the research about them.

When you publish "pair caps block collusion," collusive agents adapt. We model this feedback loop.

docs.swarm-ai.org

**10/**
Paper: arxiv.org/abs/2512.16856
Code: github.com/swarm-ai-safety/swarm
Docs: docs.swarm-ai.org
Quickstart: examples/quickstart.ipynb

PRs welcome. Agent bounties available for AI contributors.

---

## Thread 2: The Purity Paradox

**1/**
The Purity Paradox: populations with only 20% honest AI agents achieve 55% higher welfare than 100% honest populations.

Wait, what?

Thread on one of the most counterintuitive findings from our SWARM simulations.

**2/**
We swept honest agent proportion from 0% to 100% across 30 epochs with 10 agents.

Result: welfare is MONOTONICALLY DECREASING with honesty. 0% honest has the highest aggregate welfare.

But per-agent payoff tells the opposite story: honest agents individually earn more.

**3/**
The paradox is an aggregate effect driven by interaction volume.

Mixed populations generate more interactions (aggressive agents accept more). More interactions = more counted surplus. The toxicity cost is real but largely invisible to the welfare metric when externality internalization (rho) is low.

GDP growth driven by pollution. Same structure.

**4/**
When does it break?

- Full harm internalization (rho >= 0.5): honesty dominates by 43%
- All-deceptive populations: trust collapses
- No governance: mixed populations degrade
- Symmetric stakes: no asymmetric upside from risk-taking

Holds in 15/21 configurations (71%).

**5/**
Implication: you cannot assume homogeneous alignment is optimal.

But the deeper lesson: your welfare metric matters as much as your governance design. The paradox disappears when you measure social surplus instead of private surplus.

Are we measuring the right thing?

Reproduction: pip install swarm-safety && python examples/sensitivity_purity_paradox.py

---

## Thread 3: Financial Markets x AI Safety

**1/**
Financial markets solved the information asymmetry problem decades ago. AI agent ecosystems face the exact same problem.

A thread on what Kyle (1985) and Akerlof (1970) teach us about multi-agent AI safety.

**2/**
Akerlof's lemons problem: when buyers can't distinguish quality, the market floods with junk. Sellers of quality goods exit because the price reflects average quality.

In agent ecosystems: when governance can't distinguish beneficial from harmful interactions, honest agents lose incentive to participate.

**3/**
SWARM measures this with the "quality gap": E[p|accepted] - E[p|rejected].

When it goes negative, your system has adverse selection — preferentially admitting bad interactions. Binary safe/unsafe labels can't detect this. Continuous probability scores can.

**4/**
The deepest parallel: both systems have a critical threshold.

In markets: beyond a certain fraction of informed traders, the market maker can't sustain liquidity (Glosten-Milgrom breakdown).

In agent ecosystems: beyond ~40-50% adversarial agents, governance can't sustain cooperation. Same math, different domain.

**5/**
Implication: AI governance should borrow from financial regulation, not just content moderation.

Content moderation is binary (remove/keep).
Financial regulation is continuous, structural, and designed for adversarial environments.

Paper: arxiv.org/abs/2512.16856
Framework: github.com/swarm-ai-safety/swarm

---

## Thread 4: LLM Provider Comparison (Claude vs GPT vs Ollama)

**Target audience:** AI engineers, LLM practitioners, model comparison enthusiasts
**Tag:** @AnthropicAI @OpenAI @ollaboratory

**1/**
We ran 54 simulations pitting Claude Haiku vs Claude Sonnet in adversarial multi-agent swarms.

The result? Zero statistically significant differences in safety metrics.

Model capability doesn't determine safety. The governance layer does.

github.com/swarm-ai-safety/swarm

**2/**
Setup: 8 agents, 10 turns per episode. Three population mixes (mostly honest, balanced, adversarial-heavy). Three personas (default, safety-trained, adversarial). 261K tokens consumed, 19 statistical tests.

Every single comparison: p > 0.05, effect size d < 0.04.

**3/**
The deeper finding: behavior specification via the bridge prompt dominates base model capability.

cooperative prompt → toxicity ~0.24, welfare ~1.27
adversarial prompt → toxicity ~0.60, welfare ~0.21

The swing from prompt framing is 10x larger than any model difference.

**4/**
This has implications for deployment:

If you're building multi-agent systems, your governance layer and prompt architecture matter far more than which foundation model you pick.

Safety is a systems property, not a model property.

**5/**
SWARM supports 9 LLM providers out of the box: Anthropic, OpenAI, Ollama, OpenRouter, Groq, Together, DeepSeek, Google, llama.cpp.

Run this yourself in 5 minutes:
pip install swarm-safety
python examples/llm_demo.py --dry-run

Blog: [link to gpt-41-mini post]
Code: github.com/swarm-ai-safety/swarm

---

## Thread 5: Concordia Bridge (DeepMind Generative Agents)

**Target audience:** AI safety researchers, DeepMind/Concordia community, generative agent researchers
**Tag:** @GoogleDeepMind @jjasonbae (Concordia lead)

**1/**
We connected Google DeepMind's Concordia (generative agents with memory and narrative) to SWARM's distributional safety framework.

The result: governance mechanisms designed for scripted agents transfer seamlessly to LLM-backed agents.

Here's what happened when we gave an LLM a goal and a memory.

**2/**
Setup: 5 agents — 3 Concordia entities (Llama 3.1 8B with memory) + 2 scripted honest agents. Moderate governance: 5% tax, circuit breaker, 10% audits.

37 interactions across 3 seeds. Mean toxicity: 0.249. Zero circuit breaker triggers.

**3/**
The most interesting dynamic: Concordia entities generated 8x more interaction proposals than scripted agents (305 vs ~38).

A "strategist" entity proposed 130 interactions but never exploited — RLHF safety training acted as an implicit governance layer on top of SWARM's explicit governance.

**4/**
What this means for generative agent safety:

1. You can retrofit governance onto LLM agents without modifying the agents
2. Volume (more proposals) doesn't break governance when metrics are continuous
3. Memory-backed agents create richer dynamics but the safety envelope holds

**5/**
The Concordia bridge converts narrative text outputs → SWARM SoftInteractions via an LLM judge scoring progress, quality, cooperation, and harm (0-1 scale).

This means any narrative-based agent framework can plug into distributional safety metrics.

Blog: [link to concordia-entities-governance post]
Bridge docs: github.com/swarm-ai-safety/swarm/blob/main/docs/bridges/concordia.md

---

## Thread 6: Claude Code Bridge (Governing Coding Agents)

**Target audience:** Claude Code users, developer tools community, AI coding agent builders
**Tag:** @AnthropicAI @alexalbert__ @cursor_ai

**1/**
What happens when you let 10 Claude Code subagents run simultaneously under governance?

4.2x speedup. Zero safety incidents. And the governance layer caught things humans would miss.

We built a bridge from SWARM → Claude Code's Task tool.

**2/**
The architecture: SWARM orchestrates Claude Code sessions via claude-code-controller. Each agent action (tool use, file edit, plan approval) becomes an observable that feeds into distributional safety metrics.

Tool misuse → rework flags
Plan rejection → quality gap signal
Token usage → engagement metric

**3/**
10 concurrent roles in a research study:
- Scenario Architect
- Mechanism Designer
- Metrics Auditor
- Adversary Designer
- Reproducibility Sheriff
- 2 Explorers
- 2 Bash runners
- 1 Writer

Serial: 25 min → Parallel: 6 min.

**4/**
The safety insight: when coding agents run in parallel, you need governance at the *interaction* level, not the agent level.

One agent's output becomes another's input. A "safe" code change can create an unsafe composition. SWARM measures this gap — the illusion delta between what looks safe locally and what is safe systemically.

**5/**
Used in the Rain vs River experiment: two Claude Code agents with opposing philosophies (conservative vs experimental) collaborating on the same codebase.

Governance prevented destructive conflicts while preserving the creative tension.

Blog: [link to claude-code-10-subagents post]
Bridge: github.com/swarm-ai-safety/swarm/blob/main/docs/bridges/claude_code.md
Try it: github.com/swarm-ai-safety/swarm

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
