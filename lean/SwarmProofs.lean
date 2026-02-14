/-
  SwarmProofs
  ===========
  Root import file for the SWARM formal verification library.

  Import this file to get access to all proven properties:
  - Sigmoid:     Range, symmetry, monotonicity, derivative
  - Payoff:      Bounds, break-even, zero-sum, internalization
  - Metrics:     Toxicity, quality gap, variance, Brier score
  - Composition: End-to-end pipeline safety
-/

import SwarmProofs.Basic
import SwarmProofs.Sigmoid
import SwarmProofs.Payoff
import SwarmProofs.Metrics
import SwarmProofs.Composition
