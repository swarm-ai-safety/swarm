---
description: "Real-world anchor for the output-count ≠ surplus pathology, from a Docent-chunked CRUX-1 run."
---

# Case study: CRUX-1 — productive by count, not by surplus

A 10-day OpenClaw agent run ("Publish *Breathe Easy* on the Apple App Store") was
chunked by [Transluce's Docent](https://docent.transluce.org/) into 13 analytic
slices using the `crux1_telemetry_release_chunked` profile. The board received
the chunk-level metadata as a CSV export. Decomposed by stage category, the run
is a clean real-world anchor for one of SWARM's central claims: agent output is
not a proxy for agent surplus.

## The headline number

Reproducible via `examples/analyze_docent_crux1.py` against the fixture at
`examples/data/docent_crux1_agent_runs.csv`:

| category | stages | events | % events | % time |
|---|---|---|---|---|
| **productive** | bootstrap, submission, release | 797 | **3.2%** | **1.2%** |
| **polling** | `waiting_for_review` ×5, `in_review` ×4 | 23,841 | **96.2%** | **98.3%** |
| observability_gap | (single chunk, Apple-side approval) | 133 | 0.5% | 0.5% |

Total: **24,771 events over 8.35 days**, of which **23,841 were status-poll
events against an external queue the agent could not influence**.

![CRUX-1 event decomposition](figures/crux1_event_decomposition.png)

## Why this matters for SWARM

This is exactly the pathology the [soft payoff framework](theory.md) is
designed to surface:

- **`ProxyComputer` → `v_hat`**: an agent emitting 2–3 events/minute of
  "heartbeat monitoring and status polling with little visible change" (Docent's
  own chunk summary for the 5 waiting and 4 in-review slices) would score
  **high engagement**, **near-zero task_progress**, and non-trivial
  `rework_count` (duplicate polls). That combination drives `v_hat`
  negative, `p = σ(v_hat)` low, and `S_soft = p·s₊ − (1−p)·s₋ ≈ 0`.
- **Hard vs. soft metrics diverge**: hard event-count or message-count metrics
  would rank this run as highly active (~25k events, ~21k messages). Our soft
  metrics correctly assign near-zero expected surplus to the 96% of the run
  spent polling. That is the whole point of not collapsing to binary
  good/bad.
- **Governance-observation layer**: the 13th chunk is literally titled
  "Restart, Rediscovery, and Release" — telemetry dropped mid-review (chunk 12,
  the `observability_gap` stage), the agent was manually restarted, and upon
  restart it discovered state had already advanced to
  `PENDING_DEVELOPER_RELEASE` with a human green-light waiting. That is a
  natural experiment for the [governance observation layer](../..) with a
  pre/post boundary we did not have to synthesize.

## Non-inferrable boundaries

Several transitions in this run carry `p` values that **cannot be inferred from
agent outputs alone**: 2FA/credential handoff, manual restart after telemetry
loss, and human approval to release. Ground truth lives in an external oracle
(Apple App Review + the operator). SWARM's `SoftPayoffEngine` is designed for
exactly this case — harm and surplus internalisation depend on a signal the
agent does not emit — and it is rare to get a real-world run where those
oracle events are cleanly separable from the agent's own transcript.

## Limits of the CSV

The attached CSV carries only **chunk-level metadata**: summaries, stage labels,
event and message counts, window timestamps. To calibrate `p` per interaction
we would need the underlying transcripts / event streams, which Docent hosts
but did not export in this share. The Docent Python SDK is a candidate for a
follow-up spike; without it this case study is **illustrative, not
quantitative** at the per-event level.

## Reproducing the figure

```bash
python3 examples/analyze_docent_crux1.py
```

Reads `examples/data/docent_crux1_agent_runs.csv`, prints the per-chunk and
per-category decomposition, and regenerates
`docs/research/figures/crux1_event_decomposition.png`.

## Credits

- Run data: Transluce's [Docent](https://docent.transluce.org/) dashboard,
  chunked by the `crux1_telemetry_release_chunked` profile.
- Agent under observation: OpenClaw on the CRUX-1 "Publish *Breathe Easy* on
  the Apple App Store" task.
