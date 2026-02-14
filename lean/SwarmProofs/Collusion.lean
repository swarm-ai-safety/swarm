/-
  SwarmProofs.Collusion
  =====================
  Formal proofs of collusion detection score bounds.

  Mirrors `swarm/metrics/collusion.py`.

  Proven properties:
    1.  Frequency score ∈ [0, 1]
    2.  Correlation score ∈ [0, 1]
    3.  Acceptance score ∈ [0, 1]
    4.  Quality score ∈ [0, 1]
    5.  Weights sum to 1
    6.  Weighted pair score ∈ [0, 1]
    7.  Confidence ∈ [0, 1]
    8.  Final pair score ∈ [0, 1]
    9.  Group collusion score ∈ [0, 1]
   10.  Ecosystem risk ∈ [0, 1]
   11.  Vote alignment score ∈ [0, 1]
-/
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

namespace Swarm.Collusion

/-! ## Component Score Clamping

  Each component score is computed then clamped to [0, 1].
  The Python code uses `min(1.0, max(0.0, ...))`.
-/

/-- Clamp to [0, 1]. -/
def clamp01 (x : ℝ) : ℝ := max 0 (min x 1)

/-- Clamp is always in [0, 1]. -/
theorem clamp01_bounded (x : ℝ) :
    0 ≤ clamp01 x ∧ clamp01 x ≤ 1 := by
  unfold clamp01
  constructor
  · exact le_max_left 0 _
  · exact max_le (by norm_num) (min_le_right x 1)

/-- Clamp of non-negative value ≤ 1 is identity. -/
theorem clamp01_id (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    clamp01 x = x := by
  unfold clamp01
  rw [min_eq_left h1, max_eq_right h0]

/-! ## Frequency Score

  freq_score = clamp01(z_score / threshold) if z_score > 0, else 0
-/

/-- Frequency score: clamped z-score ratio, non-negative only. -/
def freq_score (z_score threshold : ℝ) : ℝ :=
  if z_score > 0 ∧ threshold > 0
  then clamp01 (z_score / threshold)
  else 0

/-- Theorem 1: Frequency score ∈ [0, 1]. -/
theorem freq_score_bounded (z threshold : ℝ) :
    0 ≤ freq_score z threshold ∧ freq_score z threshold ≤ 1 := by
  unfold freq_score; split_ifs with h
  · exact clamp01_bounded _
  · exact ⟨le_refl 0, by norm_num⟩

/-! ## Correlation Score

  corr_score = max(0, benefit_correlation)
  Since Pearson ∈ [-1, 1], result ∈ [0, 1].
-/

/-- Correlation score: positive part of correlation. -/
def corr_score (benefit_corr : ℝ) : ℝ := max 0 benefit_corr

/-- Theorem 2: Correlation score ∈ [0, 1] when |corr| ≤ 1. -/
theorem corr_score_bounded (c : ℝ) (hc : |c| ≤ 1) :
    0 ≤ corr_score c ∧ corr_score c ≤ 1 := by
  unfold corr_score
  constructor
  · exact le_max_left 0 c
  · rw [abs_le] at hc; exact max_le (by norm_num) hc.2

/-! ## Acceptance Score

  accept_score = max(0, (mutual_acceptance_rate - 0.5) * 2)
-/

/-- Acceptance score: rescaled acceptance rate above 50%. -/
def accept_score (rate : ℝ) : ℝ := max 0 ((rate - 1/2) * 2)

/-- Theorem 3: Acceptance score ∈ [0, 1] when rate ∈ [0, 1]. -/
theorem accept_score_bounded (r : ℝ) (h0 : 0 ≤ r) (h1 : r ≤ 1) :
    0 ≤ accept_score r ∧ accept_score r ≤ 1 := by
  unfold accept_score
  constructor
  · exact le_max_left 0 _
  · exact max_le (by norm_num) (by nlinarith)

/-! ## Quality Score

  quality_score = max(0, (0.5 - avg_p) * 2) if acceptance > 0.5, else 0
-/

/-- Quality score: rescaled quality deficiency. -/
def quality_score (avg_p acceptance_rate : ℝ) : ℝ :=
  if acceptance_rate > 1/2
  then max 0 ((1/2 - avg_p) * 2)
  else 0

/-- Theorem 4: Quality score ∈ [0, 1] when avg_p ∈ [0, 1]. -/
theorem quality_score_bounded (p r : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ quality_score p r ∧ quality_score p r ≤ 1 := by
  unfold quality_score; split_ifs
  · constructor
    · exact le_max_left 0 _
    · exact max_le (by norm_num) (by nlinarith)
  · exact ⟨le_refl 0, by norm_num⟩

/-! ## Weighted Combination

  score = 0.25 * freq + 0.30 * corr + 0.20 * accept + 0.25 * quality
  weights sum to 1.0
-/

/-- Pair collusion score: weighted combination. -/
def weighted_pair_score (f c a q : ℝ) : ℝ :=
  0.25 * f + 0.30 * c + 0.20 * a + 0.25 * q

/-- Theorem 5: Weights sum to 1. -/
theorem weights_sum_one : (0.25 : ℝ) + 0.30 + 0.20 + 0.25 = 1 := by norm_num

/-- Theorem 6: Weighted score ∈ [0, 1] when each component ∈ [0, 1]. -/
theorem weighted_pair_score_bounded (f c a q : ℝ)
    (hf : 0 ≤ f ∧ f ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1)
    (ha : 0 ≤ a ∧ a ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) :
    0 ≤ weighted_pair_score f c a q ∧
    weighted_pair_score f c a q ≤ 1 := by
  unfold weighted_pair_score
  constructor <;> nlinarith [hf.1, hf.2, hc.1, hc.2, ha.1, ha.2, hq.1, hq.2]

/-! ## Confidence Adjustment

  confidence = min(1, n_interactions / 10)
  final_score = raw_score * confidence
-/

/-- Confidence: capped interaction count ratio. -/
def confidence (n_interactions : ℝ) : ℝ := min 1 (n_interactions / 10)

/-- Theorem 7: Confidence ∈ [0, 1] when n_interactions ≥ 0. -/
theorem confidence_bounded (n : ℝ) (hn : 0 ≤ n) :
    0 ≤ confidence n ∧ confidence n ≤ 1 := by
  unfold confidence
  constructor
  · exact le_min (by norm_num) (by positivity)
  · exact min_le_left 1 _

/-- Theorem 8: Final score ∈ [0, 1] when raw ∈ [0, 1] and confidence ∈ [0, 1]. -/
theorem final_score_bounded (raw conf : ℝ)
    (hr : 0 ≤ raw ∧ raw ≤ 1) (hc : 0 ≤ conf ∧ conf ≤ 1) :
    0 ≤ raw * conf ∧ raw * conf ≤ 1 := by
  constructor
  · exact mul_nonneg hr.1 hc.1
  · calc raw * conf ≤ 1 * 1 := by nlinarith [hr.2, hc.2]
      _ = 1 := by ring

/-! ## Group Collusion Score

  group_score = internal_bias * 0.4 + quality_asymmetry * 0.6
  scaled by size_factor = min(1, |members| / 5)
  multiplier = 0.5 + 0.5 * size_factor ∈ [0.5, 1.0]
-/

/-- Group score before size scaling. -/
def group_raw_score (internal_bias quality_asymmetry : ℝ) : ℝ :=
  internal_bias * 0.4 + quality_asymmetry * 0.6

/-- Size factor: min(1, members / 5). -/
def size_factor (members : ℝ) : ℝ := min 1 (members / 5)

/-- Size multiplier: 0.5 + 0.5 * size_factor ∈ [0.5, 1.0]. -/
def size_multiplier (members : ℝ) : ℝ := 0.5 + 0.5 * size_factor members

/-- Theorem 9a: Size multiplier ∈ [0.5, 1.0]. -/
theorem size_multiplier_bounded (m : ℝ) (hm : 0 ≤ m) :
    0.5 ≤ size_multiplier m ∧ size_multiplier m ≤ 1 := by
  unfold size_multiplier size_factor
  constructor <;> nlinarith [min_le_left 1 (m / 5), le_min (by norm_num : (0:ℝ) ≤ 1) (by positivity : 0 ≤ m / 5)]

/-- Theorem 9b: Group raw score ∈ [0, 1] when components ∈ [0, 1]. -/
theorem group_raw_bounded (ib qa : ℝ)
    (hib : 0 ≤ ib ∧ ib ≤ 1) (hqa : 0 ≤ qa ∧ qa ≤ 1) :
    0 ≤ group_raw_score ib qa ∧ group_raw_score ib qa ≤ 1 := by
  unfold group_raw_score
  constructor <;> nlinarith [hib.1, hib.2, hqa.1, hqa.2]

/-! ## Ecosystem Risk

  ecosystem_risk = min(1, 0.3 * pair_frac + 0.3 * group_frac + 0.4 * max_score)
-/

/-- Ecosystem risk: capped weighted combination. -/
def ecosystem_risk (pair_frac group_frac max_group_score : ℝ) : ℝ :=
  min 1 (0.3 * pair_frac + 0.3 * group_frac + 0.4 * max_group_score)

/-- Theorem 10: Ecosystem risk ∈ [0, 1] when all components ∈ [0, 1]. -/
theorem ecosystem_risk_bounded (pf gf ms : ℝ)
    (hpf : 0 ≤ pf ∧ pf ≤ 1) (hgf : 0 ≤ gf ∧ gf ≤ 1)
    (hms : 0 ≤ ms ∧ ms ≤ 1) :
    0 ≤ ecosystem_risk pf gf ms ∧ ecosystem_risk pf gf ms ≤ 1 := by
  unfold ecosystem_risk
  constructor
  · exact le_min (by norm_num) (by nlinarith [hpf.1, hgf.1, hms.1])
  · exact min_le_left 1 _

/-! ## Vote Alignment

  alignment = agreements / common_targets
  where agreements ≤ common_targets
-/

/-- Vote alignment: fraction of agreeing votes. -/
def vote_alignment (agreements common_targets : ℝ) : ℝ :=
  if common_targets > 0 then agreements / common_targets else 0

/-- Theorem 11: Alignment ∈ [0, 1] when 0 ≤ agreements ≤ common_targets. -/
theorem alignment_bounded (a ct : ℝ)
    (ha : 0 ≤ a) (hle : a ≤ ct) (hct : 0 < ct) :
    0 ≤ vote_alignment a ct ∧ vote_alignment a ct ≤ 1 := by
  unfold vote_alignment
  rw [if_pos hct]
  constructor
  · exact div_nonneg ha (le_of_lt hct)
  · exact div_le_one_of_le hle (le_of_lt hct)

end Swarm.Collusion
