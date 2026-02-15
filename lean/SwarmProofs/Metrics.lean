/-
  SwarmProofs.Metrics
  ===================
  Formal proofs of metric properties.

  These correspond to the property-based tests in
  `tests/test_property_based.py` (lines 614-801) and
  unit tests in `tests/test_metrics.py`.

  Proven properties:
  1. Toxicity range:      toxicity(p) ∈ [0, 1]  for p ∈ [0, 1]
  2. Complement relation: toxicity + quality = 1
  3. Quality gap bounds:  quality_gap ∈ [-1, 1]
  4. Variance bounds:     Var[p] ∈ [0, 0.25]  for p ∈ [0, 1]
  5. Brier score range:   Brier ∈ [0, 1]
-/
import SwarmProofs.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

noncomputable section

namespace Swarm

/-! ## Definitions -/

/-- Toxicity of a single interaction: 1 - p. -/
def toxicity (p : ℝ) : ℝ := 1 - p

/-- Average quality (expected p). -/
def avg_quality (p : ℝ) : ℝ := p

/-! ## Theorem 1: Toxicity range -/

/-- Toxicity is in [0, 1] when p ∈ [0, 1]. -/
theorem toxicity_mem_Icc (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    toxicity p ∈ Set.Icc 0 1 := by
  constructor <;> (unfold toxicity; linarith)

/-- Toxicity is non-negative for p ∈ [0, 1]. -/
theorem toxicity_nonneg (p : ℝ) (_ : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ toxicity p := by unfold toxicity; linarith

/-- Toxicity is at most 1 for p ∈ [0, 1]. -/
theorem toxicity_le_one (p : ℝ) (hp0 : 0 ≤ p) (_ : p ≤ 1) :
    toxicity p ≤ 1 := by unfold toxicity; linarith

/-! ## Theorem 2: Complement relation -/

/-- toxicity(p) + quality(p) = 1. -/
theorem toxicity_quality_complement (p : ℝ) :
    toxicity p + avg_quality p = 1 := by
  unfold toxicity avg_quality; ring

/-! ## Theorem 3: Quality gap bounds -/

/-- Quality gap between two probs in [0,1] is bounded in [-1, 1]. -/
theorem quality_gap_bounded (p_acc p_rej : ℝ)
    (ha0 : 0 ≤ p_acc) (ha1 : p_acc ≤ 1)
    (hr0 : 0 ≤ p_rej) (hr1 : p_rej ≤ 1) :
    p_acc - p_rej ∈ Set.Icc (-1 : ℝ) 1 := by
  constructor <;> linarith

/-- Negative quality gap ↔ adverse selection. -/
theorem adverse_selection_iff (p_acc p_rej : ℝ) :
    p_acc - p_rej < 0 ↔ p_acc < p_rej := by
  constructor <;> (intro h; linarith)

/-! ## Theorem 4: Variance bounds -/

/-- For p ∈ [0, 1], the Bernoulli variance p*(1-p) ∈ [0, 1/4]. -/
theorem bernoulli_variance_bound (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ p * (1 - p) ∧ p * (1 - p) ≤ 1 / 4 := by
  constructor
  · exact mul_nonneg hp0 (by linarith)
  · nlinarith [sq_nonneg (p - 1/2)]

/-- Maximum variance 1/4 at p = 1/2. -/
theorem bernoulli_variance_max_at_half :
    (1 / 2 : ℝ) * (1 - 1 / 2) = 1 / 4 := by norm_num

/-- Variance zero at boundaries. -/
theorem bernoulli_variance_zero_at_zero :
    (0 : ℝ) * (1 - 0) = 0 := by ring

theorem bernoulli_variance_zero_at_one :
    (1 : ℝ) * (1 - 1) = 0 := by ring

/-- Squared deviation from any point in [0,1] is ≤ 1. -/
theorem squared_deviation_bound (x μ : ℝ)
    (hx0 : 0 ≤ x) (hx1 : x ≤ 1) (hμ0 : 0 ≤ μ) (hμ1 : μ ≤ 1) :
    (x - μ) ^ 2 ≤ 1 := by nlinarith [sq_nonneg (x - μ)]

/-! ## Theorem 5: Brier score range -/

/-- Brier score (p - v)² ∈ [0, 1] for p ∈ [0,1], v ∈ {0,1}. -/
theorem brier_score_bounded (p v : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hv : v = 0 ∨ v = 1) :
    0 ≤ (p - v) ^ 2 ∧ (p - v) ^ 2 ≤ 1 := by
  constructor
  · exact sq_nonneg _
  · rcases hv with rfl | rfl
    · nlinarith [sq_nonneg p]
    · nlinarith [sq_nonneg (p - 1)]

/-- Perfect prediction: Brier = 0. -/
theorem brier_score_perfect (v : ℝ) : (v - v) ^ 2 = 0 := by ring

/-- Random guessing (p = 0.5): Brier = 0.25. -/
theorem brier_score_random_guess :
    ((1/2 : ℝ) - 0) ^ 2 = 1 / 4 ∧ ((1/2 : ℝ) - 1) ^ 2 = 1 / 4 := by
  constructor <;> ring

end Swarm
