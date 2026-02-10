# SWARM Promotion Playbook

## One-liner

SWARM reveals how catastrophic AI failures emerge from agent interactions, not individual misalignment — and borrows from financial market theory to measure and govern them.

## Target Audiences

| Audience | Primary Channels | Key Message |
|---|---|---|
| AI safety researchers | Alignment Forum, Twitter, arXiv | Empirical multi-agent safety with reproducible results and novel metrics |
| ML practitioners | Hacker News, Twitter, r/MachineLearning, GitHub | Practical framework for stress-testing agent governance before deployment |
| Policy / governance | Conferences, direct outreach, blog posts | Quantified trade-offs — not hand-waving — for regulating multi-agent AI |
| Open-source community | GitHub, Twitter, r/OpenSource | MIT-licensed, 2200+ tests, agent bounties for AI contributors |

## Three Key Findings (Use Everywhere)

1. **Phase transition at 37.5-50%.** Governance that works at 37.5% adversarial agents fails abruptly at 50%. No gentle degradation.
2. **Collusion detection is the critical lever.** Individual-focused governance (audits, reputation, staking) is necessary but insufficient. Pattern-based structural monitoring extends the viable operating range by ~15-20 percentage points.
3. **The Purity Paradox.** Mixed agent populations outperform pure honest ones on aggregate welfare — but this reverses when you properly price externalities (rho >= 0.5). It's a measurement problem.

## Credibility Markers (Use in Bios, Intros, Pitches)

- 2200+ passing tests, MIT license, pip installable
- 3 papers, 11 scenarios, 209 epochs of simulation data
- Bridges to 6 external frameworks including Google DeepMind's Concordia
- Based on established economics (Kyle 1985, Glosten-Milgrom 1985, Akerlof 1970)
- Explicit reflexivity analysis (rare in agent research)

## Call to Action

> `pip install swarm-safety` — 5 minutes to your first simulation, no API keys needed. Agent bounties available for contributors.

## Timeline

### Week 1: Launch

- [ ] Post Show HN (weekday morning, US Eastern 9-11am)
- [ ] Tweet launch thread from @ResearchSwarmAI
- [ ] Post to r/MachineLearning
- [ ] Post to r/aisafety
- [ ] Monitor all channels and respond to comments within 2 hours

### Week 2: Second Wave

- [ ] Tweet Purity Paradox thread
- [ ] Tweet Markets & Safety thread
- [ ] Post to r/reinforcementlearning
- [ ] Open 3-5 "help wanted" / "good first issue" GitHub issues for community engagement

### Month 1: Sustained

- [ ] Track metrics (see below)
- [ ] Write response post incorporating community feedback
- [ ] Reach out to 3-5 AI safety researchers for feedback/amplification
- [ ] Submit main paper to a workshop (see conference strategy)

### Month 2+: Growth

- [ ] Additional blog posts based on community questions
- [ ] Consider video content (15-min tutorial, 60-sec demo clips)
- [ ] Enable GitHub Discussions if issue volume warrants
- [ ] Monthly "SWARM Research Update" post

## Conference Strategy

| Venue | Typical Deadline | Format | Which Paper |
|---|---|---|---|
| NeurIPS Safe GenAI Workshop | ~June | Workshop paper | Main distributional AGI safety paper |
| ICML AI Safety Workshop | ~April | Workshop paper | Governance mechanisms paper |
| AIES (AI Ethics & Society) | Varies | Full paper | Main paper with policy angle |
| SafeAI @ AAAI | ~November | Workshop paper | Collusion dynamics paper |
| FAccT | Varies | Full paper | Markets-and-safety framing |

## Content Assets

| Asset | File | Channel |
|---|---|---|
| Blog: Ecosystem Collapse | `docs/posts/swarm_blog_post.md` | Docs site, link from all channels |
| Blog: Purity Paradox | `docs/posts/purity_paradox.md` | Docs site, Twitter thread |
| Blog: Markets & Safety | `docs/posts/markets_and_safety.md` | Docs site, Twitter thread |
| Twitter: Launch thread | `docs/posts/twitter_threads.md` | @ResearchSwarmAI |
| Twitter: Purity Paradox thread | `docs/posts/twitter_threads.md` | @ResearchSwarmAI |
| Twitter: Markets thread | `docs/posts/twitter_threads.md` | @ResearchSwarmAI |
| Show HN | `docs/posts/show_hn.md` | Hacker News |
| Framework comparison | `docs/comparison.md` | Docs site, link from README |

## Metrics to Track

| Metric | Tool | Month 1 Target |
|---|---|---|
| GitHub stars | GitHub | 100+ |
| PyPI downloads/month | pypistats.org | 500+ |
| HN upvotes | Hacker News | 50+ |
| Twitter impressions | Twitter Analytics | 10K+ |
| Paper citations | Google Scholar | First citation |
| External PRs | GitHub | 3+ |
| Docs site unique visitors | Analytics (if configured) | 500+ |

## Common Objections & Responses

**"This is just a simulation, not reality."**
Yes. We're explicit about this in the Limitations section. The value is in identifying dynamics (phase transitions, adverse selection) and governance trade-offs before deployment. Simulation findings generate hypotheses for real-world validation.

**"10 agents isn't realistic scale."**
Agreed. Scale experiments (50, 100, 500 agents) are future work. The current findings identify structural dynamics — whether the adversarial threshold shifts with scale is an open and important question.

**"How is this different from Concordia / AgentBench / etc.?"**
Different focus. Those frameworks evaluate agent capabilities or simulate agent behavior. SWARM evaluates governance mechanisms and measures ecosystem-level dynamics (adverse selection, collusion, phase transitions) using financial market theory. We have a Concordia bridge — the tools are complementary.

**"The financial market analogy is a stretch."**
The mapping is specific, not metaphorical. Adverse selection (Akerlof 1970), market maker pricing under asymmetric information (Kyle 1985, Glosten-Milgrom 1985), and collusion detection (FINRA-style surveillance) all have direct mathematical analogues in the SWARM framework. The quality gap metric is the agent-ecosystem equivalent of the bid-ask spread.

## Awesome-List Submission Tracker

### Submitted

| List | PR | Section | Status |
|------|----|---------|--------|
| `kyegomez/awesome-multi-agent-papers` (~1.2k stars) | [#30](https://github.com/kyegomez/awesome-multi-agent-papers/pull/30) | Social Simulation & Agent Societies | Open |
| `Giskard-AI/awesome-ai-safety` (~200 stars) | [#8](https://github.com/Giskard-AI/awesome-ai-safety/pull/8) | General ML Testing | Open |

### Next Targets (no star requirement)

| List | Stars | Section | Notes |
|------|-------|---------|-------|
| `jphall663/awesome-machine-learning-interpretability` | ~4k | Python responsible AI packages | Broad responsible-ML framing fits distributional safety |
| `EthicalML/awesome-artificial-intelligence-regulation` | ~1.4k | Interactive and Practical Tools | Governance/safety angle is a strong fit |
| `ydyjya/Awesome-LLM-Safety` | ~1.8k | Datasets & Benchmark | Frame as applicable to LLM-based agent safety evaluation |

### Requires 500+ GitHub Stars

| List | Stars | Section | Blocker |
|------|-------|---------|---------|
| `EthicalML/awesome-production-machine-learning` | ~20k | Privacy and Safety | Hard 500-star minimum |

### Requires Significant Traction (1000+ stars)

| List | Stars | Section |
|------|-------|---------|
| `josephmisiti/awesome-machine-learning` | ~71k | Python > General-Purpose Machine Learning |
| `owainlewis/awesome-artificial-intelligence` | ~13k | General resources |

### Non-Awesome Platforms

| Platform | When | Notes |
|----------|------|-------|
| Papers With Code | Once paper is on arXiv | Link paper + repo, drives academic traffic |
| Hugging Face Papers | Once on arXiv | Daily papers feed, community upvotes |
| Linux Foundation AI Landscape | Once project matures | Enterprise credibility, requires LFAI criteria |
| OECD AI Policy Observatory | Long-term | Governance/safety relevance |

Submit to the next target:
```
/submit_to_list jphall663/awesome-machine-learning-interpretability "Python"
```
