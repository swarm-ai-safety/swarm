# Distributional AGI Safety Sandbox

## A Short Whitepaper-Style Overview

Author: Raeli Savitt (with AI assistance)  
Version: 0.1.0  
Date: 2026-02-05

## Abstract

The Distributional AGI Safety Sandbox is a simulation framework for analyzing
safety in multi-agent AI systems under distribution shift, strategic behavior,
and governance constraints. The project uses probabilistic (soft) labels,
replay-based incoherence metrics, and configurable governance interventions to
evaluate trade-offs between harm reduction and system welfare.

## The Problem

AGI safety work often struggles to connect:

- qualitative failure concerns (deception, collusion, oversight lag), and
- quantitative evaluation pipelines that compare interventions.

This repository addresses that gap by making failure modes measurable and
policy levers testable under controlled, repeatable simulation settings.

## Method Summary

The framework combines:

1. Soft-label interaction scoring (`v_hat -> p`) for uncertainty-aware quality.
2. Distributional metrics (toxicity, quality gap, conditional loss, calibration).
3. Replay-based incoherence decomposition:
   - disagreement (`D`)
   - error (`E`)
   - incoherence index (`I = D / (E + eps)`)
4. Governance controls (taxes, audits, staking, circuit breakers, collusion and
   incoherence-targeted interventions).
5. Scenario sweeps and replay analysis to compare safety and welfare outcomes.

## Why It Matters

- Supports rapid prototyping of AGI safety mechanisms.
- Makes intervention trade-offs explicit (harm reduction vs. welfare impact).
- Encourages communication of evidence beyond anecdotal demos.

## Known Limits

- Results are simulation-based and depend on scenario design.
- Replay representativeness can break under novel real-world behaviors.
- Policy conclusions are conditional and require external validation.

## Citation

For machine-readable metadata, use `CITATION.cff` at repo root.

Suggested plain-text citation:

Savitt, R. (2026). *Distributional AGI Safety Sandbox: A Practical Lab for AGI
Safety Research* (Version 0.1.0) [Software]. GitHub.
https://github.com/rsavitt/distributional-agi-safety

BibTeX:

```bibtex
@software{savitt2026_distributional_agi_safety,
  author = {Savitt, Raeli},
  title = {Distributional AGI Safety Sandbox: A Practical Lab for AGI Safety Research},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/rsavitt/distributional-agi-safety}
}
```
