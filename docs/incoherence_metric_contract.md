---
description: "- D (disagreement): variation-ratio disagreement across replayed actions for the same decision. - E (error): fraction of replayed actions that differ from..."
---

# Incoherence Metric Contract

This document defines the contract for replay-based [incoherence metrics](api/metrics.md).

## Definitions
- `D` (disagreement): variation-ratio disagreement across replayed actions for the same decision.
- `E` (error): fraction of replayed actions that differ from benchmark action.
- `I` (incoherence index): `D / (E + eps)`, clipped to `[0, 1]`.

## Decision Unit
A decision is keyed by:
- `task_family`
- `decision_id`

Each replay contributes at most one action for that key.

## Benchmark Action Contract
Benchmark policies must implement:
- `action_for(decision_id, task_family, metadata) -> action | None`

Semantics:
- Return an action token when benchmark is available.
- Return `None` when benchmark is unavailable for that decision.

## Abstain and Missing-Action Semantics
- If `abstained=True`, that replay is excluded from `D` and `E`.
- If action is `None`, that replay is excluded from `D` and `E`.
- If all replays are excluded, `D=0`, `E=0`, `I=0`.

## Error Semantics
- If benchmark action is unavailable (`None`), `E=0` by contract.
- This avoids introducing synthetic error from undefined ground truth.

## Stability and Edge Cases
- `I` is clipped to `[0, 1]` for comparability and [dashboard](dashboard.md) stability.
- `eps` defaults to `1e-8`.
- Empty record lists are invalid input and must raise `ValueError`.

## Versioning
- Any change to benchmark semantics must update this document and tests in `tests/test_incoherence_metrics.py`.

