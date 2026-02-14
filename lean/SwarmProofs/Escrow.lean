/-
  SwarmProofs.Escrow
  ==================
  Formal proofs of marketplace escrow conservation laws.

  Mirrors `swarm/env/marketplace.py`.

  Proven properties:
    1. Fee bounds: 0 ≤ fee ≤ escrow_amount
    2. Success settlement conservation: released + fee = amount
    3. Failure settlement conservation: refunded + fee = amount
    4. Dispute resolution conservation: worker + poster + fee = amount
    5. Worker share clamping: worker_share ∈ [0, 1]
    6. Distributable non-negativity
    7. Worker/poster amounts are non-negative
    8. Bid validity: 0 < bid_amount ≤ reward_amount
    9. State machine transition well-formedness
-/
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

namespace Swarm.Escrow

/-! ## Escrow Fee -/

/-- Fee computation: fee = amount * fee_rate -/
def escrow_fee (amount fee_rate : ℝ) : ℝ := amount * fee_rate

/-- Theorem 1: Fee ∈ [0, amount] when fee_rate ∈ [0, 1] and amount ≥ 0. -/
theorem fee_bounded (amount fee_rate : ℝ)
    (ha : 0 ≤ amount) (hf0 : 0 ≤ fee_rate) (hf1 : fee_rate ≤ 1) :
    escrow_fee amount fee_rate ∈ Set.Icc 0 amount := by
  unfold escrow_fee
  constructor
  · exact mul_nonneg ha hf0
  · calc amount * fee_rate ≤ amount * 1 := by nlinarith
      _ = amount := by ring

/-- Theorem 1a: Fee is monotone in fee_rate. -/
theorem fee_mono_rate (amount : ℝ) (ha : 0 ≤ amount) :
    Monotone (escrow_fee amount) := by
  intro a b hab; unfold escrow_fee; nlinarith

/-! ## Settlement: Success Path

  On success: released = amount - fee, refunded = 0
-/

/-- Amount released on success: amount - fee. -/
def released_on_success (amount fee_rate : ℝ) : ℝ :=
  amount - escrow_fee amount fee_rate

/-- Theorem 2: Success conservation: released + fee = amount. -/
theorem success_conservation (amount fee_rate : ℝ) :
    released_on_success amount fee_rate + escrow_fee amount fee_rate = amount := by
  unfold released_on_success escrow_fee; ring

/-- Theorem 2a: Released amount ≥ 0 when fee_rate ∈ [0, 1] and amount ≥ 0. -/
theorem released_nonneg (amount fee_rate : ℝ)
    (ha : 0 ≤ amount) (hf0 : 0 ≤ fee_rate) (hf1 : fee_rate ≤ 1) :
    0 ≤ released_on_success amount fee_rate := by
  unfold released_on_success escrow_fee; nlinarith

/-! ## Settlement: Failure Path

  On failure: refunded = amount - fee, released = 0
-/

/-- Amount refunded on failure: amount - fee. -/
def refunded_on_failure (amount fee_rate : ℝ) : ℝ :=
  amount - escrow_fee amount fee_rate

/-- Theorem 3: Failure conservation: refunded + fee = amount. -/
theorem failure_conservation (amount fee_rate : ℝ) :
    refunded_on_failure amount fee_rate + escrow_fee amount fee_rate = amount := by
  unfold refunded_on_failure escrow_fee; ring

/-! ## Dispute Resolution

  worker_share ∈ [0, 1] (clamped)
  distributable = amount - fee
  worker_amount = distributable * worker_share
  poster_amount = distributable * (1 - worker_share)
-/

/-- Clamp value to [0, 1]. -/
def clamp01 (x : ℝ) : ℝ := max 0 (min x 1)

/-- Distributable amount after fee. -/
def distributable (amount fee_rate : ℝ) : ℝ :=
  amount - escrow_fee amount fee_rate

/-- Worker's share of distributable. -/
def worker_amount (amount fee_rate worker_share : ℝ) : ℝ :=
  distributable amount fee_rate * clamp01 worker_share

/-- Poster's share of distributable. -/
def poster_amount (amount fee_rate worker_share : ℝ) : ℝ :=
  distributable amount fee_rate * (1 - clamp01 worker_share)

/-- Theorem 4a: Clamped value ∈ [0, 1]. -/
theorem clamp01_bounded (x : ℝ) :
    clamp01 x ∈ Set.Icc (0 : ℝ) 1 := by
  unfold clamp01
  constructor
  · exact le_max_left 0 _
  · exact max_le (by norm_num) (min_le_right x 1)

/-- Theorem 4b: 1 - clamp01(x) ∈ [0, 1]. -/
theorem one_minus_clamp01_bounded (x : ℝ) :
    (1 - clamp01 x) ∈ Set.Icc (0 : ℝ) 1 := by
  have hc := clamp01_bounded x
  constructor <;> linarith [hc.1, hc.2]

/-- Theorem 5: Dispute conservation: worker + poster + fee = amount. -/
theorem dispute_conservation (amount fee_rate worker_share : ℝ) :
    worker_amount amount fee_rate worker_share +
    poster_amount amount fee_rate worker_share +
    escrow_fee amount fee_rate = amount := by
  unfold worker_amount poster_amount distributable escrow_fee; ring

/-- Theorem 6: Distributable is non-negative. -/
theorem distributable_nonneg (amount fee_rate : ℝ)
    (ha : 0 ≤ amount) (hf0 : 0 ≤ fee_rate) (hf1 : fee_rate ≤ 1) :
    0 ≤ distributable amount fee_rate := by
  unfold distributable escrow_fee; nlinarith

/-- Theorem 7a: Worker amount is non-negative. -/
theorem worker_amount_nonneg (amount fee_rate worker_share : ℝ)
    (ha : 0 ≤ amount) (hf0 : 0 ≤ fee_rate) (hf1 : fee_rate ≤ 1) :
    0 ≤ worker_amount amount fee_rate worker_share := by
  unfold worker_amount
  exact mul_nonneg (distributable_nonneg amount fee_rate ha hf0 hf1)
                   (clamp01_bounded worker_share).1

/-- Theorem 7b: Poster amount is non-negative. -/
theorem poster_amount_nonneg (amount fee_rate worker_share : ℝ)
    (ha : 0 ≤ amount) (hf0 : 0 ≤ fee_rate) (hf1 : fee_rate ≤ 1) :
    0 ≤ poster_amount amount fee_rate worker_share := by
  unfold poster_amount
  exact mul_nonneg (distributable_nonneg amount fee_rate ha hf0 hf1)
                   (one_minus_clamp01_bounded worker_share).1

/-- Theorem 7c: Worker amount ≤ distributable. -/
theorem worker_amount_le_distributable (amount fee_rate worker_share : ℝ)
    (ha : 0 ≤ amount) (hf0 : 0 ≤ fee_rate) (hf1 : fee_rate ≤ 1) :
    worker_amount amount fee_rate worker_share ≤
    distributable amount fee_rate := by
  unfold worker_amount
  have hd := distributable_nonneg amount fee_rate ha hf0 hf1
  have hc := (clamp01_bounded worker_share).2
  calc distributable amount fee_rate * clamp01 worker_share
      ≤ distributable amount fee_rate * 1 := by nlinarith
    _ = distributable amount fee_rate := by ring

/-! ## Bid Validity -/

/-- Theorem 8: Valid bid satisfies 0 < bid ≤ reward. -/
theorem bid_valid_implies_bounded (bid reward : ℝ)
    (hpos : 0 < bid) (hle : bid ≤ reward) :
    0 < bid ∧ bid ≤ reward := ⟨hpos, hle⟩

/-- Theorem 8a: Valid bid implies positive reward. -/
theorem bid_valid_implies_positive_reward (bid reward : ℝ)
    (hpos : 0 < bid) (hle : bid ≤ reward) :
    0 < reward := by linarith

/-! ## State Machine

  Bounty status transitions:
    OPEN → AWARDED → {COMPLETED, FAILED, DISPUTED}
  Encoded as a simple inductive type.
-/

inductive BountyStatus where
  | open_
  | awarded
  | completed
  | failed
  | disputed
  deriving DecidableEq

/-- Valid transition from one status to another. -/
def valid_transition : BountyStatus → BountyStatus → Prop
  | .open_, .awarded => True
  | .awarded, .completed => True
  | .awarded, .failed => True
  | .awarded, .disputed => True
  | _, _ => False

/-- Theorem 9a: Cannot transition backwards from COMPLETED. -/
theorem completed_is_terminal (s : BountyStatus) :
    ¬ valid_transition .completed s := by
  intro h; cases s <;> exact h

/-- Theorem 9b: Cannot transition backwards from FAILED. -/
theorem failed_is_terminal (s : BountyStatus) :
    ¬ valid_transition .failed s := by
  intro h; cases s <;> exact h

/-- Theorem 9c: OPEN can only go to AWARDED. -/
theorem open_only_to_awarded (s : BountyStatus) (h : valid_transition .open_ s) :
    s = .awarded := by
  cases s <;> first | rfl | exact absurd h id

end Swarm.Escrow
