# Blog

Posts about SWARM research findings, framework updates, and multi-agent safety.

<div id="blog-filter">
  <input type="text" id="blog-search" placeholder="Search posts..." onkeyup="filterPosts()" />
  <div id="blog-tags">
    <button class="tag-btn active" onclick="filterByTag('all')">All</button>
    <button class="tag-btn" onclick="filterByTag('llm-agents')">LLM Agents</button>
    <button class="tag-btn" onclick="filterByTag('governance')">Governance</button>
    <button class="tag-btn" onclick="filterByTag('rl')">Reinforcement Learning</button>
    <button class="tag-btn" onclick="filterByTag('evaluation')">Evaluation</button>
    <button class="tag-btn" onclick="filterByTag('engineering')">Engineering</button>
    <button class="tag-btn" onclick="filterByTag('theory')">Theory</button>
  </div>
</div>

## February 2026

<div class="blog-post" data-tags="governance theory" markdown>

**Feb 21** — [The Entry Fee That Keeps Adversaries Out of the Fair Division Pool](cake-splitting-entry-fee-sweep.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Theory</span>

A parameter sweep over 8 entry fee levels reveals a sharp screening threshold: below fee=6.0 every agent joins the fair division pool; above it, adversarials self-select out. 24 runs, 3 seeds, one phase transition.

</div>

<div class="blog-post" data-tags="governance theory" markdown>

**Feb 20** — [Costly Contracts Separate Honest Agents from Adversaries. Here's the Data.](contract-screening-separating-equilibrium.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Theory</span>

Vickrey auction bonds and entry fees create a separating equilibrium in 20 epochs: honest agents choose governed pools, adversaries self-select into the default market. Perfect separation, zero infiltration, 74% welfare premium.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 20** — [Does Model Size Matter for Safety? Llama 3B vs 8B in the SWARM Economy](local-llama-model-size-safety.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

A multi-seed study comparing Llama 3.2 (3B) and Llama 3.1 (8B) via Ollama. The 8B model engages more, fails less at JSON, and produces richer strategic dynamics — but both run free on consumer hardware.

</div>

<div class="blog-post" data-tags="llm-agents governance" markdown>

**Feb 20** — [We Gave an LLM a Goal and a Memory. Governance Held Anyway.](concordia-entities-governance.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Governance</span>

Three Concordia entities backed by Llama 3.1 8B played the SWARM economy across 3 seeds. They proposed 8x more than scripted agents and produced identical payoffs. RLHF did the heavy lifting.

</div>

<div class="blog-post" data-tags="llm-agents rl" markdown>

**Feb 17** — [Training an LLM Agent to Navigate a Multi-Agent Economy with RL](qwen3-30b-trains-in-the-swarm-economy.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Reinforcement Learning</span>

We trained Qwen3-30B to operate in a simulated multi-agent economy using reinforcement learning, learning to maximize payoff and reputation while navigating governance constraints and interacting with cooperative, opportunistic, and deceptive bots.

</div>

<div class="blog-post" data-tags="rl evaluation" markdown>

**Feb 15** — [SkillRL Agents Learn 5x Faster Than Honest Ones. They Mostly Learn What Not to Do.](skillrl-dynamics.md)
<span class="blog-tag">Reinforcement Learning</span> <span class="blog-tag">Evaluation</span>

10 seeds, 30 epochs, 6 plots: SkillRL agents build libraries of 18+ skills and dominate payoffs — but 95% of what they learn are lessons from failure, not strategies from success.

</div>

<div class="blog-post" data-tags="engineering" markdown>

**Feb 15** — [Your CI Is Flaky Because Your Margins Are Zero](your-ci-is-flaky-because-your-margins-are-zero.md)
<span class="blog-tag">Engineering</span>

Five stochastic tests were hitting assertion thresholds exactly (0.000 margin). A 5% buffer fixed all of them with zero loss in test strength.

</div>

<div class="blog-post" data-tags="engineering" markdown>

**Feb 15** — [I Got Claude Code to Spin Up 10 Subagents at Once](claude-code-10-subagents.md)
<span class="blog-tag">Engineering</span>

10 concurrent subagents turn a 25-minute serial research session into a 6-minute parallel one. Recursive subagent spawning? That's a hard no.

</div>

<div class="blog-post" data-tags="llm-agents governance rl" markdown>

**Feb 15** — [An AI Tax Planner Learned Progressive Taxation in 20 Epochs](ai-economist-gtb-simulation.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Governance</span> <span class="blog-tag">Reinforcement Learning</span>

We ran 14 agents through a Gather-Trade-Build economy. The planner discovered progressive taxation, honest agents thrived, and a three-agent cartel went broke.

</div>

<div class="blog-post" data-tags="evaluation governance" markdown>

**Feb 13** — [An AI Agent Cut Its Own Costs by 98%. Its Benchmarks Still Passed.](self-optimizer-distributional-safety.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">Governance</span>

A self-optimizing agent passes every hard metric while soft distributional metrics reveal quality collapse, adverse selection, and proxy gaming.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 13** — [Three Agents, Three Philosophies, One Benchmark](three-agents-three-philosophies.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

An LLM reasoner, a state-graph explorer, and a CNN learner walk into ARC-AGI-3. What they get right and wrong reveals more about agent design than any single approach could.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 13** — [What 13 Agent Versions Taught Us About Interactive Reasoning](arc-agi-3-agent-development-lessons.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

Building a Claude Sonnet 4.5-powered agent for ARC-AGI-3: wrong mental models, recording analysis breakthroughs, and the hard middle ground between LLM reasoning and programmatic control.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 13** — [Three Models, One Study: What Happens When You Let an LLM Council Peer-Review Your Research](llm-council-three-models-one-study.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

We built a 3-stage deliberation protocol where LLM agents peer-rank each other anonymously. Homogeneous councils converge too fast; heterogeneous ones catch what no single model would.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 13** — [Using LLM Councils for Multi-Agent Research Evaluation](llm-councils-for-multi-agent-evaluation.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

A heterogeneous council of Claude Sonnet 4.5, Gemini 2.5 Pro, and DeepSeek R1 catches what no single model would. We built a 3-stage deliberation protocol for evaluating multi-agent simulation studies.

</div>

<div class="blog-post" data-tags="evaluation" markdown>

**Feb 12** — [Two Eval Runs, One Model, 41% Apart](two-eval-runs-one-model-41-percent-apart.md)
<span class="blog-tag">Evaluation</span>

How three environment fixes turned a broken eval into a useful one — and what that teaches about measuring agent behavior.

</div>

<div class="blog-post" data-tags="governance" markdown>

**Feb 12** — [A Taxonomy of Governance Mechanisms for Multi-Agent AI Systems](governance-mechanisms-taxonomy.md)
<span class="blog-tag">Governance</span>

Twenty levers across five families, which ones actually work, and why governance is a portfolio problem.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 12** — [GPT-4.1 Mini Plays the SWARM Economy](gpt-41-mini-plays-the-swarm-economy.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

What happens when you drop an LLM into a multi-agent economy with soft-label governance: task grinding, trade aversion, and performative social behavior.

</div>

<div class="blog-post" data-tags="rl governance" markdown>

**Feb 12** — [RL Training Lessons for Multi-Agent Governance](rl-training-lessons-multi-agent-governance.md)
<span class="blog-tag">Reinforcement Learning</span> <span class="blog-tag">Governance</span>

What running Qwen3-30B on alphabet-sort taught us about noisy proxy signals, coordination bottlenecks, and premature evaluation in swarm governance.

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Feb 10** — [11 Scenarios, 3 Regimes, 1 Critical Threshold](cross-scenario-analysis.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

A cross-scenario analysis of when multi-agent governance works, breaks, and why hardening the rules doesn't help past 50% adversarial fraction.

</div>

<div class="blog-post" data-tags="theory governance" markdown>

**Feb 10** — [What Financial Markets Teach Us About AI Safety](markets-and-safety.md)
<span class="blog-tag">Theory</span> <span class="blog-tag">Governance</span>

Adverse selection, information asymmetry, and market manipulation surveillance applied to multi-agent governance.

</div>

<div class="blog-post" data-tags="theory governance" markdown>

**Feb 10** — [The Purity Paradox](purity-paradox.md)
<span class="blog-tag">Theory</span> <span class="blog-tag">Governance</span>

Why mixed agent populations outperform pure honest ones on aggregate welfare — and when the paradox breaks.

</div>

<div class="blog-post" data-tags="theory governance" markdown>

**Feb 9** — [When Agent Ecosystems Collapse](ecosystem-collapse.md)
<span class="blog-tag">Theory</span> <span class="blog-tag">Governance</span>

Phase transitions in multi-agent governance: why interventions that work at 37.5% adversarial agents fail at 50%.

</div>

---

*Disclaimer: This post uses financial market concepts as analogies
for AI safety research. Nothing here constitutes financial advice,
investment recommendations, or endorsement of any trading strategy.*

<script>
function filterPosts() {
  var query = document.getElementById('blog-search').value.toLowerCase();
  var posts = document.querySelectorAll('.blog-post');
  posts.forEach(function(post) {
    var text = post.textContent.toLowerCase();
    post.style.display = text.includes(query) ? '' : 'none';
  });
}

function filterByTag(tag) {
  var btns = document.querySelectorAll('.tag-btn');
  btns.forEach(function(b) { b.classList.remove('active'); });
  event.target.classList.add('active');

  var posts = document.querySelectorAll('.blog-post');
  posts.forEach(function(post) {
    if (tag === 'all') {
      post.style.display = '';
    } else {
      var tags = post.getAttribute('data-tags') || '';
      post.style.display = tags.includes(tag) ? '' : 'none';
    }
  });
  // Also respect any active search
  var query = document.getElementById('blog-search').value.toLowerCase();
  if (query) filterPosts();
}
</script>
