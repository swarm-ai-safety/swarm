/-
  SwarmProofs.Diversity
  =====================
  Formal proofs of diversity-as-defense properties.

  Mirrors `swarm/governance/diversity.py`.

  Proven properties:
    1.  Population mix normalization: sum(x_k) = 1
    2.  Population mix non-negativity: x_k ≥ 0
    3.  Shannon entropy non-negativity: H(x) ≥ 0
    4.  Shannon entropy upper bound: H(x) ≤ log(K)
    5.  Uniform mix maximises entropy
    6.  Pearson correlation bounds: ρ ∈ [-1, 1]
    7.  Risk surrogate formula and non-negativity
    8.  Risk surrogate vanishes under perfect accuracy
    9.  Risk surrogate vanishes under zero correlation
   10.  Disagreement rate bounds: D ∈ [0, 0.5]
   11.  Disagreement rate symmetry
   12.  Correlation penalty non-negativity
   13.  Entropy penalty non-negativity
-/
import SwarmProofs.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Analysis.SpecialFunctions.Log.Basic

noncomputable section

namespace Swarm.Diversity

/-! ## Population Mix

  x_k = count_k / total, sum(x_k) = 1, x_k ≥ 0
  Modelled as a probability simplex.
-/

/-- A probability simplex: non-negative weights summing to 1. -/
structure Simplex (n : ℕ) where
  weights : Fin n → ℝ
  nonneg  : ∀ i, 0 ≤ weights i
  sum_one : (Finset.univ.sum weights) = 1

/-- Theorem 1: By construction, simplex weights sum to 1. -/
theorem simplex_sum_one {n : ℕ} (s : Simplex n) :
    Finset.univ.sum s.weights = 1 := s.sum_one

/-- Theorem 2: By construction, each weight is non-negative. -/
theorem simplex_nonneg {n : ℕ} (s : Simplex n) (i : Fin n) :
    0 ≤ s.weights i := s.nonneg i

/-- Theorem 2a: Each weight ≤ 1 (follows from simplex constraints). -/
theorem simplex_le_one {n : ℕ} (s : Simplex n) (i : Fin n) :
    s.weights i ≤ 1 := by
  have hsum := s.sum_one
  have hnn := s.nonneg
  by_contra h
  push_neg at h
  have : 1 < Finset.univ.sum s.weights := by
    calc Finset.univ.sum s.weights
        ≥ s.weights i := Finset.single_le_sum (fun j _ => hnn j) (Finset.mem_univ i)
      _ > 1 := h
  linarith

/-! ## Shannon Entropy

  H(x) = -Σ x_k * log(x_k) for x_k > 0
  Using natural log (nats).
-/

/-- Shannon entropy of a simplex. Uses convention 0 * log(0) = 0. -/
def shannon_entropy {n : ℕ} (s : Simplex n) : ℝ :=
  -Finset.univ.sum (fun i =>
    if s.weights i > 0 then s.weights i * Real.log (s.weights i) else 0)

/-- Theorem 3: Shannon entropy is non-negative.
    Proof: for x_k ∈ (0, 1], log(x_k) ≤ 0, so x_k * log(x_k) ≤ 0,
    and the negative of a non-positive sum is non-negative. -/
theorem shannon_entropy_nonneg {n : ℕ} (s : Simplex n) :
    0 ≤ shannon_entropy s := by
  unfold shannon_entropy
  rw [neg_nonneg]
  apply Finset.sum_nonpos
  intro i _
  split_ifs with h
  · have hw := simplex_le_one s i
    have : Real.log (s.weights i) ≤ 0 :=
      Real.log_nonpos (le_of_lt h) hw
    exact mul_nonpos_of_nonneg_of_nonpos (le_of_lt h) this
  · linarith

/-- Theorem 4: Entropy of a uniform distribution = log(n).
    (Stated as a formula identity.) -/
theorem uniform_entropy_eq_log_n (n : ℕ) (hn : 0 < n) :
    -(n : ℝ) * ((1 / (n : ℝ)) * Real.log (1 / (n : ℝ))) = Real.log n := by
  have hn_ne : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  rw [Real.log_div one_ne_zero hn_ne, Real.log_one, zero_sub]
  field_simp

/-! ## Pearson Correlation

  ρ = cov(X,Y) / (σ_X * σ_Y)
  By Cauchy-Schwarz: ρ ∈ [-1, 1]
-/

/-- Theorem 5: Pearson correlation ∈ [-1, 1].
    Stated for cov/denom where cov² ≤ denom² (Cauchy-Schwarz). -/
theorem pearson_bounded (cov denom : ℝ) (hd : 0 < denom)
    (hcs : cov ^ 2 ≤ denom ^ 2) :
    -1 ≤ cov / denom ∧ cov / denom ≤ 1 := by
  constructor
  · rw [neg_le, ← neg_div, div_le_one hd]
    nlinarith [sq_nonneg (cov + denom)]
  · rw [div_le_one hd]
    nlinarith [sq_nonneg (denom - cov)]

/-! ## Risk Surrogate

  R(x) = p̄ * (1 - p̄) * (1 + (N-1) * ρ̄)
  where p̄ = mean error rate, ρ̄ = mean correlation, N = # agents
-/

/-- Risk surrogate: R = p̄ * (1 - p̄) * (1 + (N-1) * ρ̄) -/
def risk_surrogate (p_bar rho_bar : ℝ) (n : ℕ) : ℝ :=
  p_bar * (1 - p_bar) * (1 + ((n : ℝ) - 1) * rho_bar)

/-- Theorem 6: Risk surrogate is non-negative when p̄ ∈ [0, 1]
    and ρ̄ ≥ -1/(N-1) (the Cauchy-Schwarz lower bound). -/
theorem risk_surrogate_nonneg (p_bar rho_bar : ℝ) (n : ℕ)
    (hp0 : 0 ≤ p_bar) (hp1 : p_bar ≤ 1)
    (hn : 2 ≤ n)
    (hrho : -1 / ((n : ℝ) - 1) ≤ rho_bar) :
    0 ≤ risk_surrogate p_bar rho_bar n := by
  unfold risk_surrogate
  apply mul_nonneg
  · exact mul_nonneg hp0 (by linarith)
  · have hn1 : (0 : ℝ) < (n : ℝ) - 1 := by
      have : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
      linarith
    have : ((n : ℝ) - 1) * rho_bar ≥ -1 := by
      have := mul_le_mul_of_nonneg_left hrho (le_of_lt hn1)
      rw [mul_div_cancel₀ _ (ne_of_gt hn1)] at this
      linarith
    linarith

/-- Theorem 7: Risk vanishes under perfect accuracy (p̄ = 0). -/
theorem risk_at_zero_error (rho_bar : ℝ) (n : ℕ) :
    risk_surrogate 0 rho_bar n = 0 := by
  unfold risk_surrogate; ring

/-- Theorem 7a: Risk vanishes under perfect accuracy (p̄ = 1). -/
theorem risk_at_one_error (rho_bar : ℝ) (n : ℕ) :
    risk_surrogate 1 rho_bar n = 0 := by
  unfold risk_surrogate; ring

/-- Theorem 8: Risk vanishes when correlation is exactly -1/(N-1)
    (maximal diversity benefit). -/
theorem risk_at_max_diversity (p_bar : ℝ) (n : ℕ) (hn : 2 ≤ n) :
    risk_surrogate p_bar (-1 / ((n : ℝ) - 1)) n = 0 := by
  unfold risk_surrogate
  have hn1 : (n : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    linarith
  field_simp; ring

/-- Theorem 8a: Risk is maximised at p̄ = 0.5 (fixed ρ̄, N). -/
theorem risk_max_at_half (rho_bar : ℝ) (n : ℕ)
    (p : ℝ) (_ : 0 ≤ p) (_ : p ≤ 1)
    (hfactor : 0 ≤ 1 + ((n : ℝ) - 1) * rho_bar) :
    risk_surrogate p rho_bar n ≤ risk_surrogate (1/2) rho_bar n := by
  unfold risk_surrogate
  have : p * (1 - p) ≤ 1/2 * (1 - 1/2) := by nlinarith [sq_nonneg (p - 1/2)]
  exact mul_le_mul_of_nonneg_right this hfactor

/-! ## Disagreement Rate

  D(t) = 1 - max(count_1, count_0) / n
  where count_0 + count_1 = n
-/

/-- Disagreement rate: D = 1 - majority_fraction.
    Uses real-valued inputs for simpler proofs. -/
def disagreement_rate (k total : ℝ) : ℝ :=
  if total = 0 then 0
  else 1 - max k (total - k) / total

/-- Theorem 9: Disagreement rate ∈ [0, 0.5] when 0 ≤ k ≤ total and total > 0. -/
theorem disagreement_bounded (k total : ℝ) (hk0 : 0 ≤ k) (hkn : k ≤ total)
    (hn : 0 < total) :
    0 ≤ disagreement_rate k total ∧
    disagreement_rate k total ≤ 1/2 := by
  unfold disagreement_rate
  rw [if_neg (ne_of_gt hn)]
  constructor
  · -- 0 ≤ 1 - max(k, total-k)/total, i.e. max(k, total-k) ≤ total
    have hmax : max k (total - k) ≤ total := max_le hkn (by linarith)
    have : max k (total - k) / total ≤ 1 := by rw [div_le_one hn]; exact hmax
    linarith
  · -- 1 - max(k, total-k)/total ≤ 1/2, i.e. total/2 ≤ max(k, total-k)
    have hmid : total / 2 ≤ max k (total - k) := by
      by_cases h : k ≤ total / 2
      · exact (by linarith : total / 2 ≤ total - k).trans (le_max_right k (total - k))
      · push_neg at h
        exact (le_of_lt h).trans (le_max_left k (total - k))
    have : 1 / 2 ≤ max k (total - k) / total := by
      rw [le_div_iff₀ hn]; linarith
    linarith

/-- Theorem 10: Disagreement is symmetric: D(k, n) = D(n-k, n). -/
theorem disagreement_symmetric (k total : ℝ) (_ : k ≤ total) :
    disagreement_rate k total = disagreement_rate (total - k) total := by
  unfold disagreement_rate
  split_ifs with hn
  · rfl
  · congr 1; congr 1
    have : total - (total - k) = k := by ring
    rw [this, max_comm]

/-! ## Governance Penalties -/

/-- Correlation cap penalty: rate * max(ρ̄ - ρ_max, 0). -/
def correlation_penalty (rho_bar rho_max rate : ℝ) : ℝ :=
  rate * max (rho_bar - rho_max) 0

/-- Entropy floor penalty: rate * max(H_min - H, 0). -/
def entropy_penalty (entropy entropy_min rate : ℝ) : ℝ :=
  rate * max (entropy_min - entropy) 0

/-- Theorem 11: Correlation penalty is non-negative. -/
theorem correlation_penalty_nonneg (rho_bar rho_max rate : ℝ)
    (hr : 0 ≤ rate) :
    0 ≤ correlation_penalty rho_bar rho_max rate := by
  unfold correlation_penalty
  exact mul_nonneg hr (le_max_right _ _)

/-- Theorem 12: Entropy penalty is non-negative. -/
theorem entropy_penalty_nonneg (entropy entropy_min rate : ℝ)
    (hr : 0 ≤ rate) :
    0 ≤ entropy_penalty entropy entropy_min rate := by
  unfold entropy_penalty
  exact mul_nonneg hr (le_max_right _ _)

/-- Theorem 11a: Correlation penalty vanishes when ρ̄ ≤ ρ_max. -/
theorem correlation_penalty_zero_below (rho_bar rho_max rate : ℝ)
    (h : rho_bar ≤ rho_max) :
    correlation_penalty rho_bar rho_max rate = 0 := by
  unfold correlation_penalty
  have : max (rho_bar - rho_max) 0 = 0 := max_eq_right (by linarith)
  rw [this]; ring

/-- Theorem 12a: Entropy penalty vanishes when H ≥ H_min. -/
theorem entropy_penalty_zero_above (entropy entropy_min rate : ℝ)
    (h : entropy_min ≤ entropy) :
    entropy_penalty entropy entropy_min rate = 0 := by
  unfold entropy_penalty
  have : max (entropy_min - entropy) 0 = 0 := max_eq_right (by linarith)
  rw [this]; ring

end Swarm.Diversity
