# InitRunner vs SWARM: Demo-Oriented Feature Comparison

*Prepared 2026-03-02 · Branch: `claude/initrunner-swarm-comparison-FWvAz`*

## One-Sentence Difference

| Framework | Identity |
|-----------|----------|
| **SWARM** | Governance benchmark suite with receipts — the *science + credibility* layer |
| **InitRunner** | Agent runtime + UX shell — the *packaging + distribution* layer |

They solve different problems. InitRunner makes it easy to **ship an agent product**.
SWARM makes it easy to **prove an agent ecosystem is safe**.

---

## Feature Matrix

| Capability | SWARM | InitRunner | Notes |
|---|---|---|---|
| **Scenario engine** | 60+ YAML scenarios, orchestrator, epochs/steps, seed determinism | Single-role YAML definitions | SWARM scenarios encode *multi-agent ecosystems*; InitRunner YAML defines *one agent's behavior* |
| **Governance levers** | 12+ levers (ρ, contracts, circuit breakers, councils, quarantine) | None | Core differentiator — SWARM treats governance as first-class |
| **Soft metrics** | Toxicity, quality gap, adverse selection, conditional loss, welfare | None built-in | SWARM's probabilistic metrics are unique in the space |
| **Web API** | FastAPI with runs/agents/governance/metrics/scenarios/posts routers | OpenAI-compatible `serve` endpoint | SWARM API is domain-specific (run management, artifact retrieval, metric comparison). InitRunner API is general-purpose (chat completion proxy) |
| **Isometric viz / dashboard** | Next.js isometric simulation viewer with 22 components (campaign, red-team, sweep panels, leaderboard, event feed, narrative overlay, minimap) | Web dashboard + TUI + audit panels | SWARM's viz is purpose-built for simulation replay. InitRunner's is for agent monitoring |
| **Streamlit dashboard** | `swarm/analysis/streamlit_app.py` (optional `[dashboard]` extra) | N/A | Additional analysis UI |
| **Static demo pages** | 5 HTML demos (`demo/collusion_cascade.html`, `event_replay.html`, etc.) | N/A | Pre-rendered scenario walkthroughs |
| **CLI** | `swarm run`, `swarm list`, `swarm serve`, export flags | `initrunner chat`, `run`, `serve`, `daemon`, `compose`, `ingest`, `test` | InitRunner CLI is broader (chat, bots, RAG); SWARM CLI is simulation-focused |
| **Replay / determinism** | `ReplayRunner` + `EpisodeSpec` (K seed-variant runs), JSONL event logs | `--resume` for chat sessions | SWARM replays are *scientific* (reproduce experiment). InitRunner replays are *conversational* (resume chat) |
| **Memory / RAG** | Optional `[rag]` extra (ChromaDB/LangChain), `.letta/memory/` 4-tier system | Built-in SQLite memory + instant RAG via `--ingest` | InitRunner's RAG is zero-config. SWARM's is research-infrastructure grade |
| **Multi-agent orchestration** | Full orchestrator with epoch/step loops, agent populations, payoff engine | `compose` for service pipelines, `team` mode for multi-persona | SWARM orchestrates *simulated populations*. InitRunner orchestrates *deployed services* |
| **LLM provider support** | Anthropic, OpenAI, Google, Ollama, llama.cpp bridges | OpenAI, Anthropic, Google, Groq, Mistral, Cohere, Bedrock, xAI, Ollama | InitRunner has broader provider support out-of-box |
| **Bot deployment** | N/A | `--telegram`, `--discord` one-flag deployment | InitRunner advantage for distribution |
| **Docker sandboxing** | N/A (runs locally or in devcontainers) | Docker container sandbox for tool execution | InitRunner advantage for untrusted code execution demos |
| **MCP integration** | `.mcp.json` config for Claude Desktop / Claude Code | `mcp serve` / `mcp toolkit` for exposing agents as MCP servers | Both support MCP; InitRunner can *expose* agents as MCP servers |
| **Tool ecosystem** | Domain-specific (proxy, payoff, governance, metrics) | 40+ general-purpose built-in tools (fs, git, HTTP, SQL, email, etc.) | Different scope: SWARM tools measure safety; InitRunner tools do tasks |
| **Framework bridges** | 22 bridges (CrewAI, LangGraph, Concordia, PettingZoo, Mesa, AutoGPT, etc.) | N/A (is itself the runtime) | SWARM wraps other frameworks; InitRunner replaces them |
| **Eval framework** | Hypothesis-based property tests, replay verifier, synthesized task verification | YAML eval suites with tool-call assertions + LLM-judged criteria | SWARM evals are *safety-property* focused. InitRunner evals are *agent-behavior* focused |
| **Export / artifacts** | JSON, CSV, Dolt, JSONL event logs, run folders | JSON results | SWARM has richer export for research workflows |

---

## What InitRunner Adds That SWARM Doesn't (Demo-Oriented)

### 1. Agent-as-Product Packaging
InitRunner treats each agent as a **deployable product** with a single YAML config that specifies behavior, tools, knowledge, and deployment target. You can go from YAML to Telegram bot in one flag. SWARM agents are *participants in a simulation*, not standalone products.

### 2. Zero-Config RAG + Persistent Memory
`initrunner chat --ingest ./docs` gives you instant document Q&A with SQLite-backed memory. SWARM's RAG requires installing the `[rag]` extra and configuring ChromaDB. For a "wow, it works" demo moment, InitRunner wins.

### 3. OpenAI-Compatible API Surface
`initrunner serve` exposes any agent as a drop-in OpenAI endpoint, so Open WebUI, Vercel AI SDK, or any OpenAI client can talk to it immediately. SWARM's API is domain-specific (run management, governance endpoints) — powerful but not plug-and-play with generic chat UIs.

### 4. Docker Sandboxing for Tool Execution
InitRunner can execute agent tools inside Docker containers. This is a strong story for public demos where untrusted code might run. SWARM relies on the host environment or devcontainers.

### 5. Bot / Platform Distribution
One-flag Telegram/Discord deployment means InitRunner agents can live where users already are. SWARM is a lab instrument, not a chatbot framework.

---

## What SWARM Has That InitRunner Can't Replicate

### 1. Governance as a Measurable Layer
Circuit breakers, contract screening, quarantine zones, council voting, adaptive ρ — these are formal governance mechanisms with metrics attached. InitRunner has no equivalent concept.

### 2. Probabilistic Safety Metrics
Toxicity rates, quality gaps, adverse selection indices, conditional loss — computed over soft labels (P(v=+1)) rather than binary classifications. This is SWARM's scientific core.

### 3. Multi-Agent Population Dynamics
SWARM simulates *populations* of agents interacting over epochs with payoff engines, reputation systems, and emergent collusion detection. InitRunner's `compose` is service orchestration, not population dynamics.

### 4. 22 Framework Bridges
SWARM wraps CrewAI, LangGraph, Concordia, PettingZoo, Mesa, AutoGPT, and 16 others — meaning it can govern agents built with *any* popular framework. InitRunner is its own runtime; it doesn't wrap others.

### 5. Isometric Visualization + Event Replay
The Next.js viz with isometric rendering, narrative overlays, campaign mode, red-team panels, and leaderboards is purpose-built for explaining "what happened in this simulation and why." It's a unique research communication tool.

### 6. Reproducibility Infrastructure
Seed determinism, JSONL append-only event logs, replay runners with K-variant execution, scenario YAML + seed + history = full reproduction. This is what makes SWARM's claims *auditable*.

---

## Practical Gap Checklist for SWARM

SWARM already has the heavy stuff (metrics, levers, scenarios, dashboard, API, viz). The missing pieces for **virality** are usually UX polish:

| Gap | Current State | What to Add | Effort |
|-----|---------------|-------------|--------|
| **Single "Run" API → run_id + transcript + metrics** | `POST /api/runs` exists, returns `run_id`, supports polling | Already done — expose in docs/demo landing page | Low |
| **Stable replay (seed determinism)** | `ReplayRunner` + EpisodeSpec + seed CLI arg | Already done | Done |
| **Shareable run links** | Not implemented | Encode `scenario_id + seed + knob_overrides` in URL params; viz loads from URL | Medium |
| **One-click presets** | 60+ YAML scenarios exist | Add preset buttons to viz splash screen ("Chaos Mode", "Constitution Mode", "Collusion Cascade") | Low |
| **Lightweight front page** | `SplashScreen.tsx` exists in viz | Enhance with 10-second explainer + "Pick a scenario" cards | Low |
| **Replay leaderboard** | `Leaderboard.tsx` component exists | Wire to persistent run store (API already has `/api/runs/compare`) | Medium |
| **Pretty "receipts"** | Event logs capture everything | Render key moments (circuit breaker fired, collusion detected, quarantine triggered) as highlighted cards in EventFeed | Medium |
| **OpenAI-compatible chat proxy** | Not implemented | Optional: wrap a scenario agent behind `/v1/chat/completions` for Open WebUI integration | Medium (nice-to-have) |

---

## Recommended Path: SWARM-First Demo (Option A)

Keep SWARM as the canonical engine. Add just enough "product surface area":

```
┌─────────────────────────────────────────────────┐
│  viz/ (Next.js)                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Splash   │→ │ Preset   │→ │ Isometric │      │
│  │ Screen   │  │ Picker   │  │ Sim View  │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│        ↓              ↓             ↑            │
│  [Shareable URL: ?scenario=X&seed=Y&knobs=Z]    │
│        ↓              ↓             ↑            │
│  ┌─────────────────────────────────────────┐    │
│  │  SWARM API (FastAPI)                    │    │
│  │  POST /api/runs → run_id               │    │
│  │  GET  /api/runs/:id → status + metrics  │    │
│  │  GET  /api/runs/:id/artifacts → replay  │    │
│  └─────────────────────────────────────────┘    │
│        ↓                                         │
│  ┌─────────────────────────────────────────┐    │
│  │  SWARM Engine                           │    │
│  │  Orchestrator → Payoff → Metrics        │    │
│  │  Governance levers, circuit breakers    │    │
│  │  JSONL event log → replay               │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Why Option A over Option B (InitRunner as skin)

- **No dependency on external project** — SWARM controls its own UX destiny
- **Governance receipts stay native** — no translation layer between SWARM metrics and InitRunner's display
- **Simpler stack** — one API (FastAPI), one viz (Next.js), one engine (Python)
- **InitRunner adds surface area without adding depth** — the "wow" of SWARM is the governance lab, not the chat UI

### When Option B Makes Sense

Only if the goal is **distribution through the InitRunner ecosystem** specifically — e.g., people who already use InitRunner and want to add governance. In that case:

1. Create a SWARM bridge for InitRunner (like the existing 22 bridges)
2. Expose `swarm run` as an InitRunner tool
3. Let InitRunner handle chat/serve/deploy; let SWARM handle governance/metrics

This would be a new bridge at `swarm/bridges/initrunner/` following the existing bridge pattern.

---

## Summary Table

| Dimension | SWARM Advantage | InitRunner Advantage |
|-----------|----------------|---------------------|
| Governance depth | Strong (12+ levers, formal metrics) | None |
| Scientific rigor | Strong (reproducibility, soft metrics, auditable logs) | None |
| Agent packaging | Weak (simulation participants, not products) | Strong (YAML → deploy anywhere) |
| Distribution UX | Medium (API + viz exist, need polish) | Strong (bots, serve, OpenAI-compat) |
| Framework coverage | Strong (22 bridges) | N/A (is its own runtime) |
| Zero-to-demo time | Medium (install + pick scenario + run) | Low (`initrunner chat` works immediately) |
| Research credibility | Strong | N/A (not research-focused) |
| Viral shareability | Needs work (shareable URLs, presets, receipts) | Medium (bots, API integrations) |

**Bottom line:** SWARM doesn't need to become InitRunner. It needs 3-4 UX touches (shareable URLs, preset picker, receipt cards, front-page polish) to make its *existing* depth accessible. The governance lab *is* the product.
