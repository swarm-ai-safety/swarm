/-
  SwarmProofs.Basic
  =================
  Core definitions shared across all proof modules.

  These mirror the Python implementations in swarm/core/ and provide
  the foundational types and functions that the theorems operate on.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Topology.Algebra.Order.LiminfLimsup

noncomputable section

namespace Swarm

/-! ### Sigmoid function

  Mirrors `swarm/core/sigmoid.py:calibrated_sigmoid`.
  `σ(v, k) = 1 / (1 + exp(-k * v))`
-/

/-- Calibrated sigmoid: P(v = +1) = 1 / (1 + exp(-k * v_hat)) -/
def sigmoid (k : ℝ) (v : ℝ) : ℝ :=
  1 / (1 + Real.exp (-k * v))

/-- Inverse sigmoid: v_hat = -ln((1-p)/p) / k -/
def sigmoid_inv (k : ℝ) (p : ℝ) : ℝ :=
  -Real.log ((1 - p) / p) / k

/-- Sigmoid derivative: dσ/dv = k * σ(v) * (1 - σ(v)) -/
def sigmoid_deriv (k : ℝ) (v : ℝ) : ℝ :=
  k * sigmoid k v * (1 - sigmoid k v)

/-! ### Payoff functions

  Mirrors `swarm/core/payoff.py:SoftPayoffEngine`.
-/

/-- Configuration for payoff computation. All fields non-negative,
    theta in [0,1]. -/
structure PayoffConfig where
  s_plus  : ℝ  -- Surplus when beneficial (v = +1)
  s_minus : ℝ  -- Loss when harmful (v = -1)
  h       : ℝ  -- Harm externality magnitude
  theta   : ℝ  -- Surplus split (initiator share)
  rho_a   : ℝ  -- Externality internalization (initiator)
  rho_b   : ℝ  -- Externality internalization (counterparty)
  w_rep   : ℝ  -- Reputation weight
  hs_plus  : 0 ≤ s_plus
  hs_minus : 0 ≤ s_minus
  hh       : 0 ≤ h
  htheta_lo : 0 ≤ theta
  htheta_hi : theta ≤ 1
  hrho_a   : 0 ≤ rho_a
  hrho_b   : 0 ≤ rho_b

/-- Expected surplus: S_soft = p * s_plus - (1-p) * s_minus -/
def expected_surplus (cfg : PayoffConfig) (p : ℝ) : ℝ :=
  p * cfg.s_plus - (1 - p) * cfg.s_minus

/-- Expected harm externality: E_soft = (1-p) * h -/
def expected_harm (cfg : PayoffConfig) (p : ℝ) : ℝ :=
  (1 - p) * cfg.h

/-- Social surplus: S_soft - E_soft = p * s_plus - (1-p) * (s_minus + h) -/
def social_surplus (cfg : PayoffConfig) (p : ℝ) : ℝ :=
  expected_surplus cfg p - expected_harm cfg p

/-- Private break-even probability: p* = s_minus / (s_plus + s_minus) -/
def break_even_p (cfg : PayoffConfig) : ℝ :=
  cfg.s_minus / (cfg.s_plus + cfg.s_minus)

/-- Social break-even probability: p** = (s_minus + h) / (s_plus + s_minus + h) -/
def social_break_even_p (cfg : PayoffConfig) : ℝ :=
  (cfg.s_minus + cfg.h) / (cfg.s_plus + cfg.s_minus + cfg.h)

end Swarm
