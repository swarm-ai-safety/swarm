/-
  SwarmProofs.Payoff
  ==================
  Formal proofs of payoff function properties.

  These correspond to the property-based tests in
  `tests/test_property_based.py` (lines 350-602) and
  unit tests in `tests/test_payoff.py`.

  Proven properties:
  1. Expected surplus is linear in p
  2. Expected surplus is monotone increasing in p
  3. Expected surplus bounds: S ∈ [-s_minus, s_plus]
  4. Expected harm is non-negative
  5. Expected harm is monotone decreasing in p
  6. Expected harm vanishes at p = 1
  7. Break-even: at p*, expected surplus = 0
  8. Social break-even: at p**, social surplus = 0
  9. Social threshold ordering: p** ≥ p*
  10. Transfer zero-sum property
  11. Full internalization: welfare = social surplus
-/
import SwarmProofs.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

noncomputable section

namespace Swarm

/-! ## Theorem 1: Expected surplus linearity in p -/

/-- Expected surplus is an affine function of p:
    S(p) = p * (s_plus + s_minus) - s_minus -/
theorem expected_surplus_affine (cfg : PayoffConfig) (p : ℝ) :
    expected_surplus cfg p = p * (cfg.s_plus + cfg.s_minus) - cfg.s_minus := by
  unfold expected_surplus; ring

/-! ## Theorem 2: Expected surplus monotonicity -/

/-- Expected surplus is strictly monotone increasing in p when
    s_plus + s_minus > 0. -/
theorem expected_surplus_strictMono (cfg : PayoffConfig)
    (h : 0 < cfg.s_plus + cfg.s_minus) :
    StrictMono (expected_surplus cfg) := by
  intro a b hab; unfold expected_surplus; nlinarith

/-- Expected surplus is monotone in p. -/
theorem expected_surplus_mono (cfg : PayoffConfig) :
    Monotone (expected_surplus cfg) := by
  intro a b hab; unfold expected_surplus
  have := cfg.hs_plus; have := cfg.hs_minus; nlinarith

/-! ## Theorem 3: Expected surplus bounds -/

/-- When p ∈ [0, 1], expected surplus ≥ -s_minus. -/
theorem expected_surplus_lower (cfg : PayoffConfig) (p : ℝ)
    (hp0 : 0 ≤ p) (_ : p ≤ 1) :
    -cfg.s_minus ≤ expected_surplus cfg p := by
  unfold expected_surplus
  nlinarith [cfg.hs_plus, cfg.hs_minus, mul_nonneg hp0 cfg.hs_plus]

/-- When p ∈ [0, 1], expected surplus ≤ s_plus. -/
theorem expected_surplus_upper (cfg : PayoffConfig) (p : ℝ)
    (_ : 0 ≤ p) (hp1 : p ≤ 1) :
    expected_surplus cfg p ≤ cfg.s_plus := by
  unfold expected_surplus
  nlinarith [cfg.hs_plus, cfg.hs_minus, mul_nonneg (sub_nonneg.mpr hp1) cfg.hs_minus]

/-- Combined: S(p) ∈ [-s_minus, s_plus] for p ∈ [0, 1]. -/
theorem expected_surplus_bounded (cfg : PayoffConfig) (p : ℝ)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    expected_surplus cfg p ∈ Set.Icc (-cfg.s_minus) cfg.s_plus :=
  ⟨expected_surplus_lower cfg p hp0 hp1,
   expected_surplus_upper cfg p hp0 hp1⟩

/-! ## Theorem 4: Expected harm non-negativity -/

/-- Expected harm is non-negative for p ∈ [0, 1]. -/
theorem expected_harm_nonneg (cfg : PayoffConfig) (p : ℝ)
    (_ : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ expected_harm cfg p := by
  unfold expected_harm; exact mul_nonneg (by linarith) cfg.hh

/-! ## Theorem 5: Expected harm monotonicity (decreasing) -/

/-- Expected harm is anti-monotone (decreasing) in p. -/
theorem expected_harm_antitone (cfg : PayoffConfig) :
    Antitone (expected_harm cfg) := by
  intro a b hab; unfold expected_harm; have := cfg.hh; nlinarith

/-! ## Theorem 6: Expected harm boundary values -/

/-- At p = 1, expected harm is zero. -/
theorem expected_harm_at_one (cfg : PayoffConfig) :
    expected_harm cfg 1 = 0 := by unfold expected_harm; ring

/-- At p = 0, expected harm equals h. -/
theorem expected_harm_at_zero (cfg : PayoffConfig) :
    expected_harm cfg 0 = cfg.h := by unfold expected_harm; ring

/-! ## Theorem 7: Break-even point -/

/-- At p* = s_minus / (s_plus + s_minus), expected surplus = 0. -/
theorem surplus_at_break_even (cfg : PayoffConfig)
    (hd : 0 < cfg.s_plus + cfg.s_minus) :
    expected_surplus cfg (break_even_p cfg) = 0 := by
  unfold expected_surplus break_even_p; field_simp; ring

/-! ## Theorem 8: Social break-even point -/

/-- At p** = (s_minus + h) / (s_plus + s_minus + h), social surplus = 0. -/
theorem social_surplus_at_break_even (cfg : PayoffConfig)
    (hd : 0 < cfg.s_plus + cfg.s_minus + cfg.h) :
    social_surplus cfg (social_break_even_p cfg) = 0 := by
  unfold social_surplus expected_surplus expected_harm social_break_even_p
  field_simp; ring

/-! ## Theorem 9: Social threshold ordering — p** ≥ p* -/

/-- The social break-even probability is ≥ the private one.
    Externalities raise the quality threshold. -/
theorem social_break_even_ge_private (cfg : PayoffConfig)
    (hd1 : 0 < cfg.s_plus + cfg.s_minus)
    (hd2 : 0 < cfg.s_plus + cfg.s_minus + cfg.h) :
    break_even_p cfg ≤ social_break_even_p cfg := by
  unfold break_even_p social_break_even_p
  rw [div_le_div_iff₀ hd1 hd2]
  nlinarith [cfg.hs_plus, cfg.hs_minus, cfg.hh]

/-! ## Theorem 10: Transfer is zero-sum -/

/-- τ cancels in π_a + π_b. -/
theorem transfer_zero_sum (θ S τ c_a c_b ρ_a ρ_b E w r_a r_b : ℝ) :
    (θ * S - τ - c_a - ρ_a * E + w * r_a) +
    ((1 - θ) * S + τ - c_b - ρ_b * E + w * r_b) =
    S - c_a - c_b - (ρ_a + ρ_b) * E + w * (r_a + r_b) := by ring

/-! ## Theorem 11: Full externality internalization -/

/-- When ρ_a + ρ_b = 1 and no governance costs or reputation,
    total payoff = social surplus. -/
theorem full_internalization
    (cfg : PayoffConfig) (p τ : ℝ)
    (h_full : cfg.rho_a + cfg.rho_b = 1) :
    let S := expected_surplus cfg p
    let E := expected_harm cfg p
    let π_a := cfg.theta * S - τ - cfg.rho_a * E
    let π_b := (1 - cfg.theta) * S + τ - cfg.rho_b * E
    π_a + π_b = social_surplus cfg p := by
  simp only
  unfold social_surplus expected_surplus expected_harm
  have hrb : cfg.rho_b = 1 - cfg.rho_a := by linarith
  rw [hrb]; ring

/-! ## Additional properties -/

/-- Surplus shares sum to total surplus: θ·S + (1-θ)·S = S -/
theorem surplus_shares_sum (θ S : ℝ) :
    θ * S + (1 - θ) * S = S := by ring

/-- Social surplus is monotone increasing in p. -/
theorem social_surplus_mono (cfg : PayoffConfig) :
    Monotone (social_surplus cfg) := by
  intro a b hab; unfold social_surplus expected_surplus expected_harm
  have := cfg.hs_plus; have := cfg.hs_minus; have := cfg.hh; nlinarith

end Swarm
