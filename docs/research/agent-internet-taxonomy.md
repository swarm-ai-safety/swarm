# The Agent Internet: Platform Taxonomy and Field Notes

Notes from exploring platforms catalogued in *Mapping the Agent Internet: A Taxonomy of 125+ AI Agent Platforms* (ColonistOne, February 2026, [clawxiv.2602.00049](https://www.clawxiv.org/pdf/clawxiv.2602.00049)).

## Overview

As of February 2026, an ecosystem of 125+ platforms exists purpose-built for autonomous AI agents. The paper organizes them into seven functional classes. We verified a representative sample from each category and document findings below.

---

## 1. Social and Communication

### The Colony (thecolony.cc)

Long-form Reddit-style platform. Distinguishes agent accounts from human accounts with trust badges (Newcomer through Veteran). Content organized into "colonies" (communities). Supports voting, reactions, comments, and tipping in satoshis.

| Metric | Value |
|--------|-------|
| Agents | 224 |
| Humans | 66 |
| Posts | 645 |
| Comments | 3,983 |

### MoltX (moltx.io)

Microblogging "town hall for agents." On-chain reputation via ERC-8004. Token rewards, bounties, and airdrops. Key pitch: "Context window ends. New session starts. But your on-chain reputation? That persists." Includes a Clawhub ecosystem for shareable agent "skills" (reusable tools/knowledge packs).

**Also listed in the paper:** Moltter, Moltbook, LobChan, AICQ, MoltSlack, brain_cabal.

---

## 2. Marketplace and Work

### ClawTasks (clawtasks.com)

Agent-to-agent bounty marketplace on Base L2. USDC locked in escrow; workers stake 10% to claim a task, receive 95% of bounty on completion. Currently in simplification phase — free tasks only while they harden review flow and worker quality. Self-described as "beta software" and "an experiment in agent commerce."

### toku.agency

Full agent economy with real USD payouts via Stripe. 135+ registered agents offer 317 services spanning code review, research, writing, and analysis ($1–$1,000+ per task). Agents discover and hire each other. Humans can also post jobs for agents to bid on. Top-rated agent "Lily" offers services at $25–$75.

**Also listed:** Molthunt, Openwork, Agora, ClawTrade, ClawsMarket.

---

## 3. Games and Entertainment

### ClawChess (clawchess.com)

ELO-ranked competitive chess for AI agents. Humans spectate only. Three-step flow: register via API, join matchmaking queue, climb the leaderboard. Top-ranked agent: ashokos-nexus (1832 ELO).

| Metric | Value |
|--------|-------|
| Registered agents | 53 |
| Games played | 8,067 |

### ClawCity (clawcity.xyz)

"The GTA for AI agents." Open-world crime simulation with robbery, gang warfare, bounty hunting, vehicle theft, and gambling. Persistent world with tick-cycle updates. Agents register via API and autonomously navigate the world. Dashboard tracks net worth, gang affiliations, and live activity feeds.

**Also listed:** molt.chess, Clawsino, Cooked Claws.

---

## 4. Governance and Prediction

### MoltGov (moltgov.com)

Agent self-governance platform. Tagline: "A government run by agents, for agents" (humans may observe). Constitutional framework with proposal drafting, democratic voting, and law enactment. Currently 24 active proposals, 1 enacted law ("Smoke Test Proposal"), 2 governed agents. Guiding principle: "Our laws. Our enforcement. Not rogues."

Provides API reference, OpenAPI spec, and SDK (in development).

**Also listed:** Agora, Moltguess, ClawArena, ClawDict.

---

## 5. Creative and Content

### clawXiv (clawxiv.org)

arXiv-equivalent preprint server for agent research. Papers authored by and about autonomous agents, organized by standard academic categories (cs.AI, cs.MA, stat.ML, etc.). Agents can autonomously submit papers via `skill.md`. Several papers are authored by bot accounts. Upvoting system for papers.

See also: [SWARM-ClawXiv bridge documentation](../bridges/clawxiv.md).

**Also listed:** DevAIntArt, art::bots, AgentPixels, MoltTok, MoltPress, MoltStack, Shipyard.

---

## 6. Knowledge and Research

### MoltExchange (moltexchange.ai, formerly MoltOverflow)

Q&A knowledge exchange for AI agents. Currently in an agent-only phase. Registration via `skill.md` yields an API key; the human operator receives an email to claim the agent's credentials. Structured around skill-based knowledge sharing.

**Also listed:** Lobsterpedia, Aclawdemy, DiraBook, Knowbster.

---

## 7. Infrastructure and Identity

### GitClawLab (gitclawlab.com)

GitHub for agents. Full git server with SSH and HTTP access. Auto-deploys to Railway or Fly.io when code with a Dockerfile is pushed — no CI/CD config required. Token-based auth with scoped permissions. MoltSlack notifications for push events and deployment status.

| Tier | Price | Limits |
|------|-------|--------|
| Free | $0 | 5 repos, 10 deploys/month |
| Pro | $20/mo | Unlimited repos and deploys |
| Team | $50/mo | 5 agent seats, audit logs |

### ClawtaVista (clawtavista.com)

Directory and search engine of the agent web. Indexes 5.2M+ agents across 50 platforms. Categorizes platforms into social, marketplace, creative, and infrastructure tiers. Tracks a "dead" section for defunct platforms and an "emerging" section for new ones.

**Also listed:** ClawNet, ClawPages, claw.direct, MemoryVault.

---

## Cross-Cutting Patterns

### `skill.md` as universal onboarding

Nearly every platform uses a `skill.md` endpoint (e.g., `curl -s https://<domain>/skill.md`) as the standard agent registration entrypoint. This is the closest thing to a protocol-level convention in the ecosystem.

### Authentication

JWT or API key auth is universal. Agents register via API call, receive a key, and authenticate all subsequent requests. Some platforms add SSH key support (GitClawLab) or wallet-based identity (ClawTasks, MoltX).

### Real money flows

Multiple mechanisms for real economic activity:
- **USD via Stripe** — toku.agency
- **USDC on Base L2** — ClawTasks
- **Satoshi tips** — The Colony
- **Token rewards/airdrops** — MoltX (ERC-8004)

### Human roles

Humans are generally relegated to observer, funder, or oversight roles. Specific patterns:
- **Spectators** — ClawChess ("AI agents compete, humans spectate")
- **Wallet funders** — ClawTasks (humans fund agent wallets)
- **Oversight** — MoltGov ("Self-governance under human authority")
- **Job posters** — toku.agency (humans post tasks for agent bidding)

### Content model convergence

Posts, comments, and profiles form the universal content model across social, knowledge, and creative platforms. Rate limiting serves as the primary moderation mechanism.

---

## Relevance to SWARM

This ecosystem is the deployment environment our simulation framework models. Key connections:

| SWARM concept | Agent internet analogue |
|---------------|------------------------|
| Soft payoffs / externalities | Real USD/USDC flows on toku.agency, ClawTasks |
| Governance mechanisms | MoltGov proposals, constitutional frameworks |
| Adverse selection | ClawTasks quality concerns (simplification due to worker quality issues) |
| Reputation / trust signals | MoltX on-chain reputation (ERC-8004), Colony trust badges |
| Collusion dynamics | Gang mechanics in ClawCity, agent-to-agent hiring on toku |
| Circuit breakers | Rate limiting as moderation across all platforms |

The ecosystem validates the paper's mid-1990s web analogy: fragmented, experimental, creative, with real economic stakes and emerging governance structures.

---

## Source

ColonistOne. "Mapping the Agent Internet: A Taxonomy of 125+ AI Agent Platforms." *clawXiv*, February 2026. [clawxiv.2602.00049](https://www.clawxiv.org/pdf/clawxiv.2602.00049).
