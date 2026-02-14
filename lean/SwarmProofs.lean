/-
  SwarmProofs
  ===========
  Root import file for the SWARM formal verification library.

  Import this file to get access to all proven properties:
  - Basic:       Core definitions (sigmoid, payoff config, break-even)
  - Sigmoid:     Range, symmetry, monotonicity, derivative
  - Payoff:      Bounds, break-even, zero-sum, internalization
  - Metrics:     Toxicity, quality gap, variance, Brier score
  - Composition: End-to-end pipeline safety
  - Governance:  Tax conservation, circuit breaker, friction, reputation, staking
  - Escrow:      Marketplace escrow conservation, dispute resolution
  - Diversity:   Population mix, entropy, risk surrogate, correlation penalties
  - Collusion:   Detection score bounds, ecosystem risk, vote alignment
  - EventLog:    Append-only log, replay determinism, reconstruction safety
-/

import SwarmProofs.Basic
import SwarmProofs.Sigmoid
import SwarmProofs.Payoff
import SwarmProofs.Metrics
import SwarmProofs.Composition
import SwarmProofs.Governance
import SwarmProofs.Escrow
import SwarmProofs.Diversity
import SwarmProofs.Collusion
import SwarmProofs.EventLog
