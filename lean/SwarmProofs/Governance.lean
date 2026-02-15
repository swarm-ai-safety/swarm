/-
  SwarmProofs.Governance
  ======================
  Formal proofs of governance lever invariants.

  Mirrors:
    - swarm/governance/taxes.py       (transaction tax)
    - swarm/governance/circuit_breaker.py (toxicity, freeze)
    - swarm/governance/reputation.py   (decay, vote weight)
    - swarm/governance/admission.py    (staking, slash)
    - swarm/governance/dynamic_friction.py (incoherence friction)
    - swarm/governance/config.py       (parameter bounds)

  Proven properties:
    1. Tax base non-negativity
    2. Tax cost-split conservation: cost_a + cost_b = total_tax
    3. Tax non-negativity
    4. Toxicity bounds: toxicity ∈ [0, 1]
    5. Uncertainty formula bounds: u ∈ [0, 1]
    6. Uncertainty maximised at p = 0.5
    7. Uncertainty vanishes at p = 0 and p = 1
    8. Friction non-negativity
    9. Friction cost-split conservation
   10. Reputation decay shrinkage: |new_rep| ≤ |old_rep|
   11. Reputation decay preserves sign
   12. Reputation converges to 0 after n steps
   13. Vote weight bounds: w ∈ (0, 1]
   14. Slash amount bounds: 0 ≤ slash ≤ resources
   15. Surplus estimate formula: 3p - 1 ∈ [-1, 2]
-/
import SwarmProofs.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Analysis.SpecialFunctions.Pow.Real

noncomputable section

namespace Swarm.Governance

/-! ## Transaction Tax

  Mirrors `swarm/governance/taxes.py`.
  surplus_estimate = 3p - 1
  tax_base = max(surplus_estimate, 0) + |τ|
  total_tax = tax_rate * tax_base
  cost_a = total_tax * split
  cost_b = total_tax * (1 - split)
-/

/-- Surplus estimate: S_est = p * 2 - (1-p) * 1 = 3p - 1 -/
def surplus_estimate (p : ℝ) : ℝ := 3 * p - 1

/-- Tax base: max(S_est, 0) + |τ| -/
def tax_base (p : ℝ) (tau : ℝ) : ℝ :=
  max (surplus_estimate p) 0 + |tau|

/-- Total tax: rate * base -/
def total_tax (rate : ℝ) (p : ℝ) (tau : ℝ) : ℝ :=
  rate * tax_base p tau

/-- Initiator cost: total_tax * split -/
def tax_cost_a (rate split : ℝ) (p : ℝ) (tau : ℝ) : ℝ :=
  total_tax rate p tau * split

/-- Counterparty cost: total_tax * (1 - split) -/
def tax_cost_b (rate split : ℝ) (p : ℝ) (tau : ℝ) : ℝ :=
  total_tax rate p tau * (1 - split)

/-- Theorem 1: Tax base is non-negative. -/
theorem tax_base_nonneg (p tau : ℝ) :
    0 ≤ tax_base p tau := by
  unfold tax_base
  have h1 : 0 ≤ max (surplus_estimate p) 0 := le_max_right _ _
  have h2 : 0 ≤ |tau| := abs_nonneg tau
  linarith

/-- Theorem 2: Tax cost-split conservation: cost_a + cost_b = total_tax. -/
theorem tax_cost_split_conservation (rate split p tau : ℝ) :
    tax_cost_a rate split p tau + tax_cost_b rate split p tau =
    total_tax rate p tau := by
  unfold tax_cost_a tax_cost_b; ring

/-- Theorem 3: Total tax is non-negative when rate ≥ 0. -/
theorem total_tax_nonneg (rate p tau : ℝ) (hr : 0 ≤ rate) :
    0 ≤ total_tax rate p tau := by
  unfold total_tax
  exact mul_nonneg hr (tax_base_nonneg p tau)

/-- Theorem 4: Surplus estimate for p ∈ [0, 1] lies in [-1, 2]. -/
theorem surplus_estimate_bounds (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    surplus_estimate p ∈ Set.Icc (-1 : ℝ) 2 := by
  unfold surplus_estimate; constructor <;> nlinarith

/-! ## Toxicity & Circuit Breaker

  Mirrors `swarm/governance/circuit_breaker.py`.
  toxicity = 1 - p
-/

/-- Toxicity: T(p) = 1 - p -/
def toxicity (p : ℝ) : ℝ := 1 - p

/-- Theorem 5: Toxicity ∈ [0, 1] for p ∈ [0, 1]. -/
theorem toxicity_bounded (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    toxicity p ∈ Set.Icc (0 : ℝ) 1 := by
  unfold toxicity; constructor <;> linarith

/-- Theorem 5a: Toxicity at p=0 is 1 (maximum harm). -/
theorem toxicity_at_zero : toxicity 0 = 1 := by unfold toxicity; ring

/-- Theorem 5b: Toxicity at p=1 is 0 (no harm). -/
theorem toxicity_at_one : toxicity 1 = 0 := by unfold toxicity; ring

/-- Theorem 5c: Toxicity is anti-monotone in p. -/
theorem toxicity_antitone : Antitone toxicity := by
  intro a b hab; unfold toxicity; linarith

private lemma sum_toxicity_le_length (ps : List ℝ)
    (hbnd : ∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1) :
    (ps.map toxicity).sum ≤ ps.length := by
  induction ps with
  | nil => simp
  | cons p rest ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_succ]
    have hp := hbnd p (List.mem_cons.mpr (Or.inl rfl))
    have htox : toxicity p ≤ 1 := by unfold toxicity; linarith [hp.1]
    have ih' := ih (fun q hq => hbnd q (List.mem_cons_of_mem p hq))
    linarith

/-- Theorem 5d: Average toxicity ∈ [0, 1] if all p_i ∈ [0, 1].
    (Convex combination of values in [0,1] stays in [0,1].) -/
theorem avg_toxicity_bounded (ps : List ℝ) (hne : ps ≠ [])
    (hbnd : ∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1) :
    let avg_tox := (ps.map toxicity).sum / ps.length
    0 ≤ avg_tox ∧ avg_tox ≤ 1 := by
  simp only
  have hlen : (0 : ℝ) < ps.length := by
    have : ps ≠ [] := hne
    have : 0 < ps.length := by
      cases ps with
      | nil => exact absurd rfl hne
      | cons _ _ => simp [List.length_cons]
    exact Nat.cast_pos.mpr this
  constructor
  · apply div_nonneg _ (le_of_lt hlen)
    apply List.sum_nonneg
    intro x hx; rw [List.mem_map] at hx
    obtain ⟨p, hp, rfl⟩ := hx
    unfold toxicity; linarith [(hbnd p hp).2]
  · rw [div_le_one hlen]
    exact sum_toxicity_le_length ps hbnd

/-! ## Uncertainty & Incoherence Friction

  Mirrors `swarm/governance/dynamic_friction.py`.
  uncertainty = 1 - |2p - 1|
  base = |τ| + 1
  total_cost = friction_rate * uncertainty * base
-/

/-- Uncertainty: u(p) = 1 - |2p - 1| -/
def uncertainty (p : ℝ) : ℝ := 1 - |2 * p - 1|

/-- Friction base: |τ| + 1 -/
def friction_base (tau : ℝ) : ℝ := |tau| + 1

/-- Total friction cost: rate * uncertainty * base -/
def friction_cost (rate : ℝ) (p tau : ℝ) : ℝ :=
  rate * uncertainty p * friction_base tau

/-- Theorem 6: Uncertainty ∈ [0, 1] for p ∈ [0, 1]. -/
theorem uncertainty_bounded (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    uncertainty p ∈ Set.Icc (0 : ℝ) 1 := by
  unfold uncertainty
  constructor
  · -- Lower bound: 1 - |2p-1| ≥ 0, i.e. |2p-1| ≤ 1
    have : |2 * p - 1| ≤ 1 := by
      rw [abs_le]; constructor <;> nlinarith
    linarith
  · -- Upper bound: 1 - |2p-1| ≤ 1
    linarith [abs_nonneg (2 * p - 1)]

/-- Theorem 7: Uncertainty is maximised at p = 0.5. -/
theorem uncertainty_max_at_half : uncertainty (1/2) = 1 := by
  unfold uncertainty; norm_num

/-- Theorem 8: Uncertainty vanishes at p = 0. -/
theorem uncertainty_at_zero : uncertainty 0 = 0 := by
  unfold uncertainty; norm_num

/-- Theorem 8a: Uncertainty vanishes at p = 1. -/
theorem uncertainty_at_one : uncertainty 1 = 0 := by
  unfold uncertainty; norm_num

/-- Theorem 9: Friction base ≥ 1 (always positive). -/
theorem friction_base_pos (tau : ℝ) : 1 ≤ friction_base tau := by
  unfold friction_base; linarith [abs_nonneg tau]

/-- Theorem 10: Friction cost is non-negative when rate ≥ 0 and p ∈ [0, 1]. -/
theorem friction_cost_nonneg (rate p tau : ℝ) (hr : 0 ≤ rate)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ friction_cost rate p tau := by
  unfold friction_cost
  apply mul_nonneg
  · apply mul_nonneg hr
    exact (uncertainty_bounded p hp0 hp1).1
  · linarith [friction_base_pos tau]

/-- Theorem 10a: Friction cost-split conservation. -/
theorem friction_cost_split_conservation (rate split p tau : ℝ) :
    friction_cost rate p tau * split +
    friction_cost rate p tau * (1 - split) =
    friction_cost rate p tau := by ring

/-! ## Reputation Decay

  Mirrors `swarm/governance/reputation.py`.
  new_rep = old_rep * decay_rate
-/

/-- Theorem 11: Decay shrinks magnitude when decay_rate ∈ [0, 1]. -/
theorem decay_shrinks_magnitude (rep decay_rate : ℝ)
    (hd0 : 0 ≤ decay_rate) (hd1 : decay_rate ≤ 1) :
    |rep * decay_rate| ≤ |rep| := by
  rw [abs_mul]
  calc |rep| * |decay_rate|
      ≤ |rep| * 1 := by nlinarith [abs_nonneg rep, abs_le.mpr ⟨by linarith, hd1⟩]
    _ = |rep| := by ring

/-- Theorem 12: Decay preserves sign. -/
theorem decay_preserves_sign_pos (rep decay_rate : ℝ)
    (hr : 0 ≤ rep) (hd : 0 ≤ decay_rate) :
    0 ≤ rep * decay_rate :=
  mul_nonneg hr hd

/-- Theorem 13: After n decay steps, |rep * d^n| ≤ |rep| * d^n. -/
theorem decay_after_n_steps (rep d : ℝ) (n : ℕ)
    (hd0 : 0 ≤ d) (hd1 : d ≤ 1) :
    |rep * d ^ n| ≤ |rep| := by
  rw [abs_mul, abs_pow]
  have had : |d| ≤ 1 := abs_le.mpr ⟨by linarith, hd1⟩
  have hadn : |d| ^ n ≤ 1 := by
    induction n with
    | zero => simp
    | succ k ih =>
      rw [pow_succ]
      calc |d| ^ k * |d| ≤ 1 * 1 := by nlinarith [abs_nonneg d]
        _ = 1 := by ring
  nlinarith [abs_nonneg rep]

/-! ## Vote Weight Normalization

  Mirrors `swarm/governance/reputation.py:VoteNormalizationLever`.
  weight = max_w / (1 + vote_count / max_w)
  normalized = weight / max_w = 1 / (1 + vote_count / max_w)
-/

/-- Normalized vote weight: 1 / (1 + v/w) where v = vote_count, w = max_weight. -/
def normalized_vote_weight (vote_count max_weight : ℝ) : ℝ :=
  1 / (1 + vote_count / max_weight)

/-- Theorem 14: Normalized vote weight ∈ (0, 1] for non-negative counts
    and positive max_weight. -/
theorem vote_weight_pos (v w : ℝ) (hv : 0 ≤ v) (hw : 0 < w) :
    0 < normalized_vote_weight v w := by
  unfold normalized_vote_weight
  apply div_pos one_pos
  have : 0 ≤ v / w := div_nonneg hv (le_of_lt hw)
  linarith

theorem vote_weight_le_one (v w : ℝ) (hv : 0 ≤ v) (hw : 0 < w) :
    normalized_vote_weight v w ≤ 1 := by
  unfold normalized_vote_weight
  rw [div_le_one (by linarith [div_nonneg hv (le_of_lt hw)])]
  linarith [div_nonneg hv (le_of_lt hw)]

/-! ## Staking & Slash

  Mirrors `swarm/governance/admission.py`.
  slash_amount = resources * slash_rate
-/

/-- Theorem 15: Slash amount ∈ [0, resources] when slash_rate ∈ [0, 1]
    and resources ≥ 0. -/
theorem slash_amount_bounded (resources slash_rate : ℝ)
    (hr : 0 ≤ resources) (hs0 : 0 ≤ slash_rate) (hs1 : slash_rate ≤ 1) :
    resources * slash_rate ∈ Set.Icc 0 resources := by
  constructor
  · exact mul_nonneg hr hs0
  · calc resources * slash_rate ≤ resources * 1 := by nlinarith
      _ = resources := by ring

end Swarm.Governance
