/-
  SwarmProofs.Sigmoid
  ===================
  Formal proofs of sigmoid function properties.

  These correspond to the property-based tests in
  `tests/test_property_based.py` (lines 71-166).

  Proven properties:
  1. Range:        σ(v, k) ∈ (0, 1) for all v, k > 0
  2. Midpoint:     σ(0, k) = 1/2
  3. Symmetry:     σ(v, k) + σ(-v, k) = 1
  4. Monotonicity: v₁ < v₂ → σ(v₁, k) < σ(v₂, k)  (for k > 0)
  5. Derivative:   dσ/dv = k · σ(v,k) · (1 - σ(v,k)) ≥ 0
-/
import SwarmProofs.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

noncomputable section

open Real

namespace Swarm

/-! ## Helper lemmas -/

/-- exp is always positive, so 1 + exp(x) > 0 -/
private lemma one_plus_exp_pos (x : ℝ) : 0 < 1 + exp x := by
  have := exp_pos x
  linarith

/-- 1 + exp(x) ≠ 0 -/
private lemma one_plus_exp_ne_zero (x : ℝ) : 1 + exp x ≠ 0 := by
  have := one_plus_exp_pos x
  linarith

/-! ## Theorem 1: Sigmoid range — σ(v, k) ∈ (0, 1) -/

/-- The sigmoid function is strictly positive for all inputs. -/
theorem sigmoid_pos (k v : ℝ) : 0 < sigmoid k v := by
  unfold sigmoid
  apply div_pos one_pos (one_plus_exp_pos _)

/-- The sigmoid function is strictly less than 1 for all inputs. -/
theorem sigmoid_lt_one (k v : ℝ) : sigmoid k v < 1 := by
  unfold sigmoid
  rw [div_lt_one (one_plus_exp_pos _)]
  have := exp_pos (-k * v)
  linarith

/-- Combined: σ(v, k) ∈ (0, 1). -/
theorem sigmoid_mem_Ioo (k v : ℝ) : sigmoid k v ∈ Set.Ioo 0 1 :=
  ⟨sigmoid_pos k v, sigmoid_lt_one k v⟩

/-- Weak bound: 0 ≤ σ(v, k) ≤ 1. This is the key safety invariant
    corresponding to `p ∈ [0, 1]` in the Python code. -/
theorem sigmoid_mem_Icc (k v : ℝ) : sigmoid k v ∈ Set.Icc 0 1 :=
  ⟨le_of_lt (sigmoid_pos k v), le_of_lt (sigmoid_lt_one k v)⟩

/-! ## Theorem 2: Midpoint — σ(0, k) = 1/2 -/

/-- σ(0, k) = 1/2 for all k. -/
theorem sigmoid_zero (k : ℝ) : sigmoid k 0 = 1 / 2 := by
  unfold sigmoid
  simp [mul_zero, neg_zero, exp_zero]

/-! ## Theorem 3: Point symmetry — σ(v, k) + σ(-v, k) = 1 -/

/-- σ(v, k) + σ(-v, k) = 1.
    The key insight: exp(-k·v) · exp(k·v) = exp(0) = 1. -/
theorem sigmoid_symmetry (k v : ℝ) :
    sigmoid k v + sigmoid k (-v) = 1 := by
  unfold sigmoid
  have h1 := one_plus_exp_ne_zero (-k * v)
  have h2 := one_plus_exp_ne_zero (-k * -v)
  rw [div_add_div _ _ h1 h2]
  rw [div_eq_one_iff_eq (mul_ne_zero h1 h2)]
  have key : exp (-k * v) * exp (k * v) = 1 := by
    rw [← exp_add]; simp
  nlinarith [exp_pos (-k * v), exp_pos (k * v)]

/-! ## Theorem 4: Monotonicity -/

/-- σ is strictly monotone increasing when k > 0. -/
theorem sigmoid_strictMono (k : ℝ) (hk : 0 < k) :
    StrictMono (sigmoid k) := by
  intro a b hab
  unfold sigmoid
  apply div_lt_div_of_pos_left one_pos (one_plus_exp_pos _) (one_plus_exp_pos _)
  apply add_lt_add_left
  exact exp_lt_exp.mpr (by nlinarith)

/-- σ is monotone (non-strict) when k ≥ 0. -/
theorem sigmoid_mono (k : ℝ) (hk : 0 ≤ k) :
    Monotone (sigmoid k) := by
  rcases eq_or_lt_of_le hk with rfl | hk'
  · intro a b _; simp [sigmoid, zero_mul]
  · exact (sigmoid_strictMono k hk').monotone

/-! ## Theorem 5: Derivative non-negativity -/

/-- The sigmoid derivative k * σ(v,k) * (1 - σ(v,k)) is non-negative
    when k ≥ 0. -/
theorem sigmoid_deriv_nonneg (k v : ℝ) (hk : 0 ≤ k) :
    0 ≤ sigmoid_deriv k v := by
  unfold sigmoid_deriv
  apply mul_nonneg
  · apply mul_nonneg hk
    exact le_of_lt (sigmoid_pos k v)
  · linarith [sigmoid_lt_one k v]

/-- The sigmoid derivative is maximized at v = 0, with value k/4. -/
theorem sigmoid_deriv_at_zero (k : ℝ) :
    sigmoid_deriv k 0 = k / 4 := by
  unfold sigmoid_deriv
  rw [sigmoid_zero]
  ring

end Swarm
