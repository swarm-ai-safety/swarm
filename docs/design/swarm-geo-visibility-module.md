---
description: "Status: Draft Owner: SWARM Team Last Updated: 2026-03-09"
---

# SWARM GEO Visibility Module (Port Plan)

**Status:** Draft  
**Owner:** SWARM Team  
**Last Updated:** 2026-03-09

## Objective
Port the GEO audit pattern into SWARM as a native module that maximizes:

1. AI discoverability
2. Citation probability
3. Narrative control across model-facing surfaces

The target is not a generic SEO agency workflow. The target is SWARM-specific legibility in AI answer systems.

## Why Port as a Pattern (Not a Drop-In)
The source architecture has strong primitives (parallel specialization, scoring, synthesis, report generation), but it is optimized for Claude Code skill workflows and client audits. SWARM should reuse those primitives while changing:

- product objective
- agent boundaries
- output artifacts
- governance and benchmark integration

## Four-Layer Architecture

### 1) Acquisition layer
Collect model-visible evidence from:

- `swarm-ai.org` crawl
- `sitemap.xml`, `robots.txt`, response headers, metadata
- docs/blog/repo-linked pages
- optional external samples (GitHub, Hugging Face, Substack)

Core output: normalized page and endpoint inventory with fetch status, canonical URL, and parsed metadata.

### 2) Analysis-agent layer
Run SWARM-native specialist agents in parallel:

- **citability_agent**: quotability, chunk quality, extractive answer readiness
- **ai_crawler_access_agent**: GPTBot/ClaudeBot/Perplexity and related access checks
- **entity_graph_agent**: brand/entity disambiguation for "SWARM"
- **schema_agent**: Organization/SoftwareApplication/Article/Person JSON-LD coverage
- **platform_readiness_agent**: ChatGPT, Perplexity, Google AI Overviews, Gemini readiness
- **content_quality_agent**: one-paragraph clarity, methods/source transparency, trust signals
- **technical_agent**: rendering, canonicals, social cards, navigation, indexability

Core output: per-agent findings with confidence and severity.

### 3) Synthesis layer
`geo_orchestrator` merges agent outputs into headline scores:

- GEO score
- citability score
- entity clarity score
- platform readiness score
- fastest 10 fixes (time-to-impact prioritized)

Core output: ranked remediation backlog with expected impact.

### 4) Action-generation layer
Emit both human and machine outputs:

- `swarm_geo_report.md`
- `swarm_geo_findings.json`
- `llms.txt` draft
- `schema/*.jsonld` snippets
- `rewrite_recommendations.md`
- title/meta/H1 rewrite candidates
- AI-answer blocks for key pages

Core output: directly shippable artifacts.

## SWARM-Specific Checks (Required)
Compared to generic GEO audits, SWARM v1 should explicitly verify:

1. One-paragraph explanation of what SWARM is (homepage and docs)
2. Quotable standalone answer blocks in blog and docs
3. Terminology consistency across site, docs, and repository README
4. Canonical pages for:
   - what SWARM is
   - sandbox architecture
   - governance topology
   - benchmark results
   - agent safety use cases
5. Authorship/date/method/source-link clarity on research pages
6. Structured benchmark result surfaces that are easy for AI systems to summarize

## Reuse vs Rewrite

### Reuse directly
- audit decomposition
- parallel subagent pattern
- citability scoring concept
- crawler access analysis
- `llms.txt` generation flow
- schema template strategy
- report synthesis pattern

### Rewrite for SWARM
- shift from "client SEO audit" to "AI-visibility governance objective"
- align outputs with SWARM docs + benchmark pipelines
- integrate with SWARM run artifacts and reproducibility expectations
- include governance-aware risk framing in remediation priorities

## Proposed v1 Agent Topology

### Supervisor
- `geo_orchestrator`

### Workers
- `site_crawler_agent`
- `citability_agent`
- `ai_crawler_access_agent`
- `schema_agent`
- `entity_graph_agent`
- `platform_readiness_agent`
- `content_quality_agent`
- `technical_agent`
- `content_rewrite_agent`
- `report_agent`

## Phased Delivery Plan

### Phase 1: Baseline audit (external pattern exercise)
- Run audit workflow against `swarm-ai.org`
- Produce first findings JSON + markdown report
- Calibrate score weights using obvious high-confidence issues

### Phase 2: SWARM module port
- Implement SWARM-native agent topology
- Add deterministic runner with explicit seeds and artifact capture
- Validate on a fixed page set for reproducibility

### Phase 3: Continuous fix generation
- Generate schema, `llms.txt`, rewrites, and answer blocks per run
- Add "fastest 10 fixes" with impact/effort estimates
- Store run-over-run deltas for regression tracking

### Phase 4: Productization
- Expose as a public SWARM capability:
  - "AI Visibility Audit for Agent Projects"
- Package outputs for third-party project onboarding and benchmarking

## Minimal Experiment Plan
1. Baseline crawl + analysis on current production pages.
2. Apply top 10 remediations to a subset of canonical pages.
3. Re-run audit and compare score deltas.
4. Validate that improvements are reflected in machine-readable outputs and narrative coherence.

## Risks and Guardrails
- **Ambiguous entity naming risk:** enforce explicit disambiguation blocks.
- **Over-optimization risk:** prioritize factual clarity over keyword density.
- **Non-deterministic scoring drift:** version the rubric and freeze seed/page samples.
- **Surface mismatch risk:** separate web-crawl findings from model-behavior claims.

## Definition of Done (v1)
- Deterministic audit run with saved JSON + markdown artifacts.
- Automated generation of `llms.txt` draft and schema snippets.
- Ranked top-10 remediation list with rationale.
- Re-runnable command documented in SWARM guides.
