"""Configuration dataclass for governance levers."""

from dataclasses import dataclass


@dataclass
class GovernanceConfig:
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

    def validate(self) -> None:
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
