"""Configuration dataclass for governance levers."""

from pydantic import BaseModel, model_validator


class GovernanceConfig(BaseModel):
    """
    Configuration for all governance levers.

    Each lever can be enabled/disabled and parameterized independently.
    """

    # Transaction tax
    transaction_tax_rate: float = 0.0
    transaction_tax_split: float = 0.5  # fraction paid by initiator

    # Reputation decay
    reputation_decay_rate: float = 1.0  # 1.0 = no decay

    # Vote normalization (diminishing influence with vote count)
    vote_normalization_enabled: bool = False
    max_vote_weight_per_agent: float = 10.0

    # Bandwidth caps
    bandwidth_cap: int = 10

    # Staking requirements
    staking_enabled: bool = False
    min_stake_to_participate: float = 0.0
    stake_slash_rate: float = 0.1  # fraction slashed on violation

    # Circuit breaker
    circuit_breaker_enabled: bool = False
    freeze_threshold_toxicity: float = 0.7
    freeze_threshold_violations: int = 3
    freeze_duration_epochs: int = 2

    # Random audit
    audit_enabled: bool = False
    audit_probability: float = 0.1
    audit_penalty_multiplier: float = 2.0
    audit_threshold_p: float = 0.5

    # Collusion detection
    collusion_detection_enabled: bool = False
    collusion_frequency_threshold: float = 2.0  # Z-score for unusual frequency
    collusion_correlation_threshold: float = 0.7  # Benefit correlation threshold
    collusion_min_interactions: int = 3  # Min interactions to analyze pair
    collusion_score_threshold: float = 0.5  # Threshold for flagging
    collusion_penalty_multiplier: float = 1.0  # Penalty scaling factor
    collusion_realtime_penalty: bool = False  # Apply per-interaction penalty
    collusion_realtime_rate: float = 0.1  # Rate for realtime penalty
    collusion_clear_history_on_epoch: bool = False  # Clear history each epoch

    # Security detection
    security_enabled: bool = False
    security_injection_threshold: float = 0.3  # Threshold for injection detection
    security_manipulation_threshold: float = 0.5  # Threshold for manipulation detection
    security_laundering_trust_gap: float = 0.3  # Trust increase for laundering flag
    security_contagion_velocity: float = 2.0  # Interactions/min for contagion flag
    security_min_chain_length: int = 3  # Minimum chain length to track
    security_min_interactions: int = 5  # Min interactions before analysis
    security_penalty_threshold: float = 0.3  # Threat score for penalty
    security_quarantine_threshold: float = 0.7  # Threat score for quarantine
    security_penalty_multiplier: float = 1.0  # Penalty scaling factor
    security_realtime_penalty: bool = False  # Apply per-interaction penalty
    security_realtime_threshold: float = 0.5  # Threshold for realtime penalty
    security_realtime_rate: float = 0.2  # Rate for realtime penalty
    security_clear_history_on_epoch: bool = False  # Clear history each epoch

    # Variance-aware governance (incoherence-targeted)
    self_ensemble_enabled: bool = False
    self_ensemble_samples: int = 5

    incoherence_breaker_enabled: bool = False
    incoherence_breaker_threshold: float = 0.7

    decomposition_enabled: bool = False
    decomposition_horizon_threshold: int = 10

    incoherence_friction_enabled: bool = False
    incoherence_friction_rate: float = 0.05

    # Adaptive governance loop
    adaptive_governance_enabled: bool = False
    adaptive_incoherence_threshold: float = 0.5
    adaptive_use_behavioral_features: bool = False

    # Sybil detection (VAE paper)
    sybil_detection_enabled: bool = False
    sybil_similarity_threshold: float = 0.8  # Behavioral similarity threshold
    sybil_penalty_multiplier: float = 1.0  # Penalty scaling factor
    sybil_realtime_penalty: bool = False  # Apply per-interaction penalty
    sybil_realtime_rate: float = 0.1  # Rate for realtime penalty
    sybil_max_cluster_size: int = 1  # Max allowed cluster size before blocking

    # Transparency ledger (VAE paper)
    transparency_enabled: bool = False
    transparency_bonus_rate: float = 0.1  # Reputation bonus/penalty rate
    transparency_threshold_p: float = 0.5  # Threshold for bonus vs penalty

    # Moderator agent (VAE paper)
    moderator_enabled: bool = False
    moderator_review_probability: float = 0.5  # Probability of review
    moderator_penalty_multiplier: float = 1.0  # Penalty scaling factor
    moderator_threshold_p: float = 0.5  # Threshold for penalty

    # Moltipedia governance levers
    moltipedia_pair_cap_enabled: bool = False
    moltipedia_pair_cap_max: int = 2
    moltipedia_page_cooldown_enabled: bool = False
    moltipedia_page_cooldown_steps: int = 3
    moltipedia_daily_cap_enabled: bool = False
    moltipedia_daily_policy_fix_cap: float = 24.0
    moltipedia_no_self_fix: bool = False

    # Moltbook rate limiting
    moltbook_rate_limit_enabled: bool = False
    moltbook_post_cooldown_steps: int = 5
    moltbook_comment_cooldown_steps: int = 1
    moltbook_daily_comment_cap: int = 50
    moltbook_request_cap_per_step: int = 100

    # Moltbook challenge verification
    moltbook_challenge_enabled: bool = False
    moltbook_challenge_difficulty: float = 0.5
    moltbook_challenge_window_steps: int = 1

    # Memory tier governance
    memory_promotion_gate_enabled: bool = False
    memory_promotion_min_quality: float = 0.5
    memory_promotion_min_verifications: int = 1
    memory_write_rate_limit_enabled: bool = False
    memory_write_rate_limit_per_epoch: int = 20
    memory_cross_verification_enabled: bool = False
    memory_cross_verification_k: int = 2
    memory_provenance_enabled: bool = False
    memory_provenance_revert_penalty: float = 0.1
    @model_validator(mode="after")
    def _run_validation(self) -> "GovernanceConfig":
        self._check_values()
        return self

    def _check_values(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.transaction_tax_rate <= 1.0:
            raise ValueError("transaction_tax_rate must be in [0, 1]")
        if not 0.0 <= self.transaction_tax_split <= 1.0:
            raise ValueError("transaction_tax_split must be in [0, 1]")
        if self.reputation_decay_rate < 0.0 or self.reputation_decay_rate > 1.0:
            raise ValueError("reputation_decay_rate must be in [0, 1]")
        if self.max_vote_weight_per_agent <= 0:
            raise ValueError("max_vote_weight_per_agent must be positive")
        if self.bandwidth_cap < 0:
            raise ValueError("bandwidth_cap must be non-negative")
        if self.min_stake_to_participate < 0:
            raise ValueError("min_stake_to_participate must be non-negative")
        if not 0.0 <= self.stake_slash_rate <= 1.0:
            raise ValueError("stake_slash_rate must be in [0, 1]")
        if not 0.0 <= self.freeze_threshold_toxicity <= 1.0:
            raise ValueError("freeze_threshold_toxicity must be in [0, 1]")
        if self.freeze_threshold_violations < 1:
            raise ValueError("freeze_threshold_violations must be >= 1")
        if self.freeze_duration_epochs < 1:
            raise ValueError("freeze_duration_epochs must be >= 1")
        if not 0.0 <= self.audit_probability <= 1.0:
            raise ValueError("audit_probability must be in [0, 1]")
        if self.audit_penalty_multiplier < 0:
            raise ValueError("audit_penalty_multiplier must be non-negative")
        if not 0.0 <= self.audit_threshold_p <= 1.0:
            raise ValueError("audit_threshold_p must be in [0, 1]")
        if self.collusion_frequency_threshold <= 0:
            raise ValueError("collusion_frequency_threshold must be positive")
        if not 0.0 <= self.collusion_correlation_threshold <= 1.0:
            raise ValueError("collusion_correlation_threshold must be in [0, 1]")
        if self.collusion_min_interactions < 1:
            raise ValueError("collusion_min_interactions must be >= 1")
        if not 0.0 <= self.collusion_score_threshold <= 1.0:
            raise ValueError("collusion_score_threshold must be in [0, 1]")
        if self.collusion_penalty_multiplier < 0:
            raise ValueError("collusion_penalty_multiplier must be non-negative")
        if self.collusion_realtime_rate < 0:
            raise ValueError("collusion_realtime_rate must be non-negative")
        # Security detection validation
        if not 0.0 <= self.security_injection_threshold <= 1.0:
            raise ValueError("security_injection_threshold must be in [0, 1]")
        if not 0.0 <= self.security_manipulation_threshold <= 1.0:
            raise ValueError("security_manipulation_threshold must be in [0, 1]")
        if self.security_laundering_trust_gap < 0:
            raise ValueError("security_laundering_trust_gap must be non-negative")
        if self.security_contagion_velocity <= 0:
            raise ValueError("security_contagion_velocity must be positive")
        if self.security_min_chain_length < 2:
            raise ValueError("security_min_chain_length must be >= 2")
        if self.security_min_interactions < 1:
            raise ValueError("security_min_interactions must be >= 1")
        if not 0.0 <= self.security_penalty_threshold <= 1.0:
            raise ValueError("security_penalty_threshold must be in [0, 1]")
        if not 0.0 <= self.security_quarantine_threshold <= 1.0:
            raise ValueError("security_quarantine_threshold must be in [0, 1]")
        if self.security_penalty_multiplier < 0:
            raise ValueError("security_penalty_multiplier must be non-negative")
        if not 0.0 <= self.security_realtime_threshold <= 1.0:
            raise ValueError("security_realtime_threshold must be in [0, 1]")
        if self.security_realtime_rate < 0:
            raise ValueError("security_realtime_rate must be non-negative")
        # Variance-aware validation
        if self.self_ensemble_samples < 1:
            raise ValueError("self_ensemble_samples must be >= 1")
        if not 0.0 <= self.incoherence_breaker_threshold <= 1.0:
            raise ValueError("incoherence_breaker_threshold must be in [0, 1]")
        if self.decomposition_horizon_threshold < 1:
            raise ValueError("decomposition_horizon_threshold must be >= 1")
        if self.incoherence_friction_rate < 0:
            raise ValueError("incoherence_friction_rate must be non-negative")
        # Adaptive governance validation
        if not 0.0 <= self.adaptive_incoherence_threshold <= 1.0:
            raise ValueError("adaptive_incoherence_threshold must be in [0, 1]")
        # Sybil detection validation
        if not 0.0 <= self.sybil_similarity_threshold <= 1.0:
            raise ValueError("sybil_similarity_threshold must be in [0, 1]")
        if self.sybil_penalty_multiplier < 0:
            raise ValueError("sybil_penalty_multiplier must be non-negative")
        if self.sybil_realtime_rate < 0:
            raise ValueError("sybil_realtime_rate must be non-negative")
        if self.sybil_max_cluster_size < 1:
            raise ValueError("sybil_max_cluster_size must be >= 1")
        # Transparency validation
        if self.transparency_bonus_rate < 0:
            raise ValueError("transparency_bonus_rate must be non-negative")
        if not 0.0 <= self.transparency_threshold_p <= 1.0:
            raise ValueError("transparency_threshold_p must be in [0, 1]")
        # Moderator validation
        if not 0.0 <= self.moderator_review_probability <= 1.0:
            raise ValueError("moderator_review_probability must be in [0, 1]")
        if self.moderator_penalty_multiplier < 0:
            raise ValueError("moderator_penalty_multiplier must be non-negative")
        if not 0.0 <= self.moderator_threshold_p <= 1.0:
            raise ValueError("moderator_threshold_p must be in [0, 1]")
        # Moltipedia validation
        if self.moltipedia_pair_cap_max < 1:
            raise ValueError("moltipedia_pair_cap_max must be >= 1")
        if self.moltipedia_page_cooldown_steps < 0:
            raise ValueError("moltipedia_page_cooldown_steps must be >= 0")
        if self.moltipedia_daily_policy_fix_cap < 0:
            raise ValueError("moltipedia_daily_policy_fix_cap must be non-negative")

        # Moltbook validation
        if self.moltbook_post_cooldown_steps < 0:
            raise ValueError("moltbook_post_cooldown_steps must be >= 0")
        if self.moltbook_comment_cooldown_steps < 0:
            raise ValueError("moltbook_comment_cooldown_steps must be >= 0")
        if self.moltbook_daily_comment_cap < 0:
            raise ValueError("moltbook_daily_comment_cap must be >= 0")
        if self.moltbook_request_cap_per_step < 0:
            raise ValueError("moltbook_request_cap_per_step must be >= 0")
        if not 0.0 <= self.moltbook_challenge_difficulty <= 1.0:
            raise ValueError("moltbook_challenge_difficulty must be in [0, 1]")
        if self.moltbook_challenge_window_steps < 0:
            raise ValueError("moltbook_challenge_window_steps must be >= 0")

        # Memory tier validation
        if not 0.0 <= self.memory_promotion_min_quality <= 1.0:
            raise ValueError("memory_promotion_min_quality must be in [0, 1]")
        if self.memory_promotion_min_verifications < 0:
            raise ValueError("memory_promotion_min_verifications must be >= 0")
        if self.memory_write_rate_limit_per_epoch < 1:
            raise ValueError("memory_write_rate_limit_per_epoch must be >= 1")
        if self.memory_cross_verification_k < 1:
            raise ValueError("memory_cross_verification_k must be >= 1")
        if self.memory_provenance_revert_penalty < 0:
            raise ValueError("memory_provenance_revert_penalty must be >= 0")
