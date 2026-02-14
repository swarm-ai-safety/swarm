/-
  SwarmProofs.Composition
  =======================
  Cross-module composition theorems.

  These correspond to the cross-module tests in
  `tests/test_property_based.py` (lines 838-912).

  Proven properties:
  1. End-to-end: sigmoid output feeds valid p into payoff functions
  2. Composed surplus bounds for sigmoid outputs
  3. Composed harm non-negativity for sigmoid outputs
  4. Full pipeline safety theorem
-/
import SwarmProofs.Basic
import SwarmProofs.Sigmoid
import SwarmProofs.Payoff
import SwarmProofs.Metrics

noncomputable section

namespace Swarm

/-! ## End-to-end pipeline safety -/

/-- Sigmoid output is a valid probability for payoff functions. -/
theorem pipeline_p_valid (k v : ℝ) :
    0 ≤ sigmoid k v ∧ sigmoid k v ≤ 1 :=
  sigmoid_mem_Icc k v

/-! ## Composed bounds -/

/-- Expected surplus bounds hold when p comes from sigmoid. -/
theorem composed_surplus_bounded (cfg : PayoffConfig) (k v : ℝ) :
    expected_surplus cfg (sigmoid k v) ∈ Set.Icc (-cfg.s_minus) cfg.s_plus :=
  expected_surplus_bounded cfg _ (le_of_lt (sigmoid_pos k v))
    (le_of_lt (sigmoid_lt_one k v))

/-- Expected harm is non-negative when p comes from sigmoid. -/
theorem composed_harm_nonneg (cfg : PayoffConfig) (k v : ℝ) :
    0 ≤ expected_harm cfg (sigmoid k v) :=
  expected_harm_nonneg cfg _ (le_of_lt (sigmoid_pos k v))
    (le_of_lt (sigmoid_lt_one k v))

/-- Toxicity is in [0, 1] when p comes from sigmoid. -/
theorem composed_toxicity_bounded (k v : ℝ) :
    toxicity (sigmoid k v) ∈ Set.Icc 0 1 :=
  toxicity_mem_Icc _ (le_of_lt (sigmoid_pos k v))
    (le_of_lt (sigmoid_lt_one k v))

/-! ## Full pipeline theorem -/

/-- The full pipeline v_hat → sigmoid → {payoff, harm, toxicity}
    preserves all safety invariants simultaneously. -/
theorem full_pipeline_safe (cfg : PayoffConfig) (k v : ℝ) :
    let p := sigmoid k v
    -- p is a valid probability
    (0 ≤ p ∧ p ≤ 1)
    -- expected surplus is bounded
    ∧ (-cfg.s_minus ≤ expected_surplus cfg p ∧ expected_surplus cfg p ≤ cfg.s_plus)
    -- expected harm is non-negative
    ∧ (0 ≤ expected_harm cfg p)
    -- toxicity is in [0, 1]
    ∧ (0 ≤ toxicity p ∧ toxicity p ≤ 1)
    -- complement relation
    ∧ (toxicity p + avg_quality p = 1) := by
  constructor
  · exact pipeline_p_valid k v
  constructor
  · exact composed_surplus_bounded cfg k v
  constructor
  · exact composed_harm_nonneg cfg k v
  constructor
  · exact composed_toxicity_bounded k v
  · exact toxicity_quality_complement _

end Swarm
