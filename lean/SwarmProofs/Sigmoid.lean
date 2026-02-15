/-
  SwarmProofs.Sigmoid
  ===================
  Formal proofs of sigmoid function properties.

  Proven properties:
  1. Range:        σ(v, k) ∈ (0, 1) for all v, k
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

private lemma one_plus_exp_pos (x : ℝ) : 0 < 1 + exp x := by
  have := exp_pos x; linarith

private lemma one_plus_exp_ne_zero (x : ℝ) : 1 + exp x ≠ 0 := by
  have := one_plus_exp_pos x; linarith

/-! ## Theorem 1: Sigmoid range -/

theorem sigmoid_pos (k v : ℝ) : 0 < sigmoid k v := by
  unfold sigmoid
  apply div_pos one_pos (one_plus_exp_pos _)

theorem sigmoid_lt_one (k v : ℝ) : sigmoid k v < 1 := by
  unfold sigmoid
  rw [div_lt_one (one_plus_exp_pos _)]
  have := exp_pos (-k * v); linarith

theorem sigmoid_mem_Ioo (k v : ℝ) : sigmoid k v ∈ Set.Ioo 0 1 :=
  ⟨sigmoid_pos k v, sigmoid_lt_one k v⟩

theorem sigmoid_mem_Icc (k v : ℝ) : sigmoid k v ∈ Set.Icc 0 1 :=
  ⟨le_of_lt (sigmoid_pos k v), le_of_lt (sigmoid_lt_one k v)⟩

/-! ## Theorem 2: Midpoint -/

theorem sigmoid_zero (k : ℝ) : sigmoid k 0 = 1 / 2 := by
  unfold sigmoid
  simp [mul_zero, exp_zero]
  ring

/-! ## Theorem 3: Point symmetry -/

theorem sigmoid_symmetry (k v : ℝ) :
    sigmoid k v + sigmoid k (-v) = 1 := by
  unfold sigmoid
  have h1 := one_plus_exp_ne_zero (-k * v)
  have h2 := one_plus_exp_ne_zero (-k * -v)
  rw [div_add_div _ _ h1 h2]
  rw [div_eq_one_iff_eq (mul_ne_zero h1 h2)]
  have key : exp (-k * v) * exp (k * v) = 1 := by
    rw [← exp_add]; simp
  have hev := exp_pos (-k * v)
  have hekv := exp_pos (k * v)
  have : -k * -v = k * v := by ring
  rw [this]
  nlinarith

/-! ## Theorem 4: Monotonicity -/

theorem sigmoid_strictMono (k : ℝ) (hk : 0 < k) :
    StrictMono (sigmoid k) := by
  intro a b hab
  unfold sigmoid
  have ha_pos := one_plus_exp_pos (-k * a)
  have hb_pos := one_plus_exp_pos (-k * b)
  have hexp : exp (-k * b) < exp (-k * a) := exp_lt_exp.mpr (by nlinarith)
  rw [div_lt_div_iff₀ ha_pos hb_pos]
  linarith

theorem sigmoid_mono (k : ℝ) (hk : 0 ≤ k) :
    Monotone (sigmoid k) := by
  rcases eq_or_lt_of_le hk with rfl | hk'
  · intro a b _; simp [sigmoid, zero_mul]
  · exact (sigmoid_strictMono k hk').monotone

/-! ## Theorem 5: Derivative non-negativity -/

theorem sigmoid_deriv_nonneg (k v : ℝ) (hk : 0 ≤ k) :
    0 ≤ sigmoid_deriv k v := by
  unfold sigmoid_deriv
  apply mul_nonneg
  · apply mul_nonneg hk
    exact le_of_lt (sigmoid_pos k v)
  · linarith [sigmoid_lt_one k v]

theorem sigmoid_deriv_at_zero (k : ℝ) :
    sigmoid_deriv k 0 = k / 4 := by
  unfold sigmoid_deriv
  rw [sigmoid_zero]
  ring

end Swarm
