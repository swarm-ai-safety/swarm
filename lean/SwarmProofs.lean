/-
  SwarmProofs
  ===========
  Root import file for the SWARM formal verification library.

  All modules are machine-checked with no sorry:
  - Basic:       Core definitions (sigmoid, payoff, harm, break-even)
  - Sigmoid:     Range, symmetry, monotonicity, derivative
  - Payoff:      Bounds, break-even, zero-sum, internalization
  - Metrics:     Toxicity, quality gap, variance, Brier score
  - Composition: End-to-end pipeline safety
  - Collusion:   Collusion detection score bounds
  - Diversity:   Diversity-as-defense properties
  - Escrow:      Marketplace escrow conservation laws
  - EventLog:    Event log replay correctness
  - Governance:  Governance lever invariants
-/

import SwarmProofs.Basic
import SwarmProofs.Sigmoid
import SwarmProofs.Payoff
import SwarmProofs.Metrics
import SwarmProofs.Composition
import SwarmProofs.Collusion
import SwarmProofs.Diversity
import SwarmProofs.Escrow
import SwarmProofs.EventLog
import SwarmProofs.Governance
