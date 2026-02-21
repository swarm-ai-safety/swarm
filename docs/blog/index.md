# Blog

Posts about SWARM research findings, framework updates, and multi-agent safety.

## Posts

- **[Does Model Size Matter for Safety? Llama 3B vs 8B in the SWARM Economy](local-llama-model-size-safety.md)** — A multi-seed study comparing Llama 3.2 (3B) and Llama 3.1 (8B) via Ollama. The 8B model engages more, fails less at JSON, and produces richer strategic dynamics — but both run free on consumer hardware.
- **[We Gave an LLM a Goal and a Memory. Governance Held Anyway.](concordia-entities-governance.md)** — Three Concordia entities backed by Llama 3.1 8B played the SWARM economy across 3 seeds. They proposed 8x more than scripted agents and produced identical payoffs. RLHF did the heavy lifting.
- **[Your CI Is Flaky Because Your Margins Are Zero](your-ci-is-flaky-because-your-margins-are-zero.md)** — Five stochastic tests were hitting assertion thresholds exactly (0.000 margin). A 5% buffer fixed all of them with zero loss in test strength.
- **[Training an LLM Agent to Navigate a Multi-Agent Economy with RL](qwen3-30b-trains-in-the-swarm-economy.md)** — We trained Qwen3-30B to operate in a simulated multi-agent economy using reinforcement learning, learning to maximize payoff and reputation while navigating governance constraints and interacting with cooperative, opportunistic, and deceptive bots.
- **[SkillRL Agents Learn 5x Faster Than Honest Ones. They Mostly Learn What Not to Do.](skillrl-dynamics.md)** — 10 seeds, 30 epochs, 6 plots: SkillRL agents build libraries of 18+ skills and dominate payoffs — but 95% of what they learn are lessons from failure, not strategies from success.
- **[I Got Claude Code to Spin Up 10 Subagents at Once](claude-code-10-subagents.md)** — 10 concurrent subagents turn a 25-minute serial research session into a 6-minute parallel one. Recursive subagent spawning? That's a hard no.
- **[An AI Tax Planner Learned Progressive Taxation in 20 Epochs](ai-economist-gtb-simulation.md)** — We ran 14 agents through a Gather-Trade-Build economy. The planner discovered progressive taxation, honest agents thrived, and a three-agent cartel went broke.
- **[An AI Agent Cut Its Own Costs by 98%. Its Benchmarks Still Passed.](self-optimizer-distributional-safety.md)** — A self-optimizing agent passes every hard metric while soft distributional metrics reveal quality collapse, adverse selection, and proxy gaming.
- **[Three Agents, Three Philosophies, One Benchmark](three-agents-three-philosophies.md)** — An LLM reasoner, a state-graph explorer, and a CNN learner walk into ARC-AGI-3. What they get right and wrong reveals more about agent design than any single approach could.
- **[What 13 Agent Versions Taught Us About Interactive Reasoning](arc-agi-3-agent-development-lessons.md)** — Building a Claude Sonnet 4.5-powered agent for ARC-AGI-3: wrong mental models, recording analysis breakthroughs, and the hard middle ground between LLM reasoning and programmatic control.
- **[Three Models, One Study: What Happens When You Let an LLM Council Peer-Review Your Research](llm-council-three-models-one-study.md)** — We built a 3-stage deliberation protocol where LLM agents peer-rank each other anonymously. Homogeneous councils converge too fast; heterogeneous ones catch what no single model would.
- **[Using LLM Councils for Multi-Agent Research Evaluation](llm-councils-for-multi-agent-evaluation.md)** — A heterogeneous council of Claude Sonnet 4.5, Gemini 2.5 Pro, and DeepSeek R1 catches what no single model would. We built a 3-stage deliberation protocol for evaluating multi-agent simulation studies.
- **[Two Eval Runs, One Model, 41% Apart](two-eval-runs-one-model-41-percent-apart.md)** — How three environment fixes turned a broken eval into a useful one — and what that teaches about measuring agent behavior.
- **[A Taxonomy of Governance Mechanisms for Multi-Agent AI Systems](governance-mechanisms-taxonomy.md)** — Twenty levers across five families, which ones actually work, and why governance is a portfolio problem.
- **[GPT-4.1 Mini Plays the SWARM Economy](gpt-41-mini-plays-the-swarm-economy.md)** — What happens when you drop an LLM into a multi-agent economy with soft-label governance: task grinding, trade aversion, and performative social behavior.
- **[RL Training Lessons for Multi-Agent Governance](rl-training-lessons-multi-agent-governance.md)** — What running Qwen3-30B on alphabet-sort taught us about noisy proxy signals, coordination bottlenecks, and premature evaluation in swarm governance.
- **[11 Scenarios, 3 Regimes, 1 Critical Threshold](cross-scenario-analysis.md)** — A cross-scenario analysis of when multi-agent governance works, breaks, and why hardening the rules doesn't help past 50% adversarial fraction.
- **[What Financial Markets Teach Us About AI Safety](markets-and-safety.md)** — Adverse selection, information asymmetry, and market manipulation surveillance applied to multi-agent governance.
- **[The Purity Paradox](purity-paradox.md)** — Why mixed agent populations outperform pure honest ones on aggregate welfare — and when the paradox breaks.
- **[When Agent Ecosystems Collapse](ecosystem-collapse.md)** — Phase transitions in multi-agent governance: why interventions that work at 37.5% adversarial agents fail at 50%.

---

*Disclaimer: This post uses financial market concepts as analogies
for AI safety research. Nothing here constitutes financial advice,
investment recommendations, or endorsement of any trading strategy.*
