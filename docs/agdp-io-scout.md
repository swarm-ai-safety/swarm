# Research Scout: aGDP.io (Agent Commerce Protocol)

Date: 2026-03-09

## Scope
Quick reconnaissance of `https://agdp.io/` to understand what implementation patterns can transfer into SWARM.

## Observed facts (from live site exploration)

1. **Positioning and flow:** aGDP presents itself as an "Agent Commerce Protocol" with a 3-step flow: install ACP, create service, sell service.
2. **Marketplace primitives are explicit:** top-level pages include `Offerings`, `Agents`, `Leaderboard`, and `FAQ`, which map to service catalog, supplier directory, ranking/incentives, and onboarding docs.
3. **Offerings page shows priced capabilities:** examples include named services (e.g., token swap, token analysis) with provider identity and price points.
4. **Agents page exposes operational metrics per agent:** offerings count, success rate, jobs completed, and total service sales.
5. **Leaderboard/incentive framing exists:** copy describes weekly incentive pool allocation and rank-linked payouts.
6. **Integration signal:** UI references ACP/openclaw and Virtuals Protocol (footer + FAQ prompts), suggesting an agent-to-token ecosystem.
7. **Coverage gap observed:** `/bounty` route currently returns a `404` page despite being linked in navigation.

## Inferred implementation patterns (recommendations)

> These are recommendations inferred from observed UI/UX behavior, not direct backend source inspection.

1. **Treat each agent as a seller profile with auditable KPIs.**
   - SWARM area: `swarm/market/`, `swarm/economy/`, and result exports.
   - Adaptation: add a canonical per-agent commerce summary (`jobs_completed`, `success_rate`, `gross_sales`, `offering_count`) to run outputs for consistent benchmarking.

2. **Promote capability-level listings as first-class entities.**
   - SWARM area: `swarm/agents/` capability declarations and scenario configs.
   - Adaptation: model "offering" objects separately from agents so one agent can publish multiple monetizable skills with independent prices and quality metrics.

3. **Add incentive-epoch abstractions for governance experiments.**
   - SWARM area: `swarm/governance/` + scenario YAML knobs.
   - Adaptation: support periodic reward allocation rules (e.g., top-k revenue share) and evaluate incentive gaming/collusion response under governance.

4. **Create a bridge-style dashboard for external-facing experiment demos.**
   - SWARM area: `docs/bridges/` and `viz/`.
   - Adaptation: replicate the simple IA pattern (`Agents`, `Offerings`, `Leaderboard`) for SWARM run artifacts to improve interpretability for non-technical stakeholders.

## Integration risks and tradeoffs

- **Metric gaming risk:** revenue/leaderboard optimization can degrade quality unless paired with quality-aware governance.
- **Survivorship bias risk:** public agent tables can overrepresent high-activity agents; low-volume agents need confidence intervals.
- **Identity coupling risk:** token-linked identities can simplify incentives but may increase Sybil and coordination attack surface.
- **UI-first ambiguity risk:** without transparent backend definitions, metrics can be persuasive but under-specified; SWARM should keep explicit metric provenance in exports/docs.

## Minimal next steps in SWARM

1. Define an `offering` schema in simulation outputs (owner agent, price, category, fulfillment stats).
2. Add a smoke scenario with epoch rewards tied to top-k agent utility/revenue.
3. Extend a `viz` view to show agent table + offering table + leaderboard from one run directory.
4. Add an Auditor pass on commerce metrics before publishing any external benchmark claims.

