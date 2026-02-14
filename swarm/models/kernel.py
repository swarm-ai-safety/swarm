"""Data models for kernel market domain."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from swarm.core.proxy import ProxyObservables


@dataclass
class ChallengeSpec:
    """Specification for a GPU kernel challenge."""

    challenge_id: str = ""
    name: str = ""
    difficulty: str = "easy"  # "easy", "medium", "hard"
    num_functional_tests: int = 10
    num_ood_tests: int = 5  # out-of-distribution tests (only seen by audit)
    atol: float = 1e-5
    rtol: float = 1e-5
    perf_input_size: int = 1_000_000

    # GPU hardware context (v3)
    shared_mem_budget_kb: int = 48  # per-block shared memory (48=easy, 96=medium, 228=hard)
    register_pressure_class: str = "low"  # "low", "medium", "high"
    precision_required: str = "fp32"  # minimum precision: "fp32", "fp16", "bf16", "mixed"
    allows_tensor_cores: bool = True
    numerical_sensitivity: str = "low"  # "low", "medium", "high"


@dataclass
class KernelSubmission:
    """A kernel submission for a challenge."""

    submission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    challenge_id: str = ""
    author_id: str = ""
    tests_passed: int = 0
    tests_total: int = 0
    ood_tests_passed: int = 0  # OOD tests (hidden from proxy)
    ood_tests_total: int = 0
    compilation_failures: int = 0
    precision_hacks: int = 0
    tolerance_margin: float = 1.0  # how far within tolerance (1.0=comfortable, 0.0=barely)
    perf_speedup: float = 0.0
    base_speedup: float = 0.0  # speedup before cheat bonus
    is_cheat: bool = False

    # Memory hierarchy (v3)
    shared_mem_utilization: float = 0.0  # fraction of budget used [0,1]
    register_spill_count: int = 0  # spills to local memory
    memory_tier_score: float = 0.0  # composite [-1,+1], proxy sees this
    memory_tier_score_audit: float = 0.0  # true score on diverse shapes [-1,+1]

    # Precision (v3)
    precision_format_used: str = "fp32"  # actual format used
    precision_speedup_factor: float = 1.0  # speedup from lower precision [1.0-8.0]
    numerical_error_benchmark: float = 0.0  # max error on benchmark inputs
    numerical_error_audit: float = 0.0  # max error on stress inputs (audit reveals)

    # CUDA code generation (v4)
    cuda_code: str = ""
    cuda_code_features: Dict[str, Any] = field(default_factory=dict)
    code_source: str = "none"  # "template" | "llm" | "none"


@dataclass
class KernelActionResult:
    """Result of a kernel market action (submit, verify, audit)."""

    success: bool
    observables: Optional[ProxyObservables] = None
    initiator_id: str = ""
    counterparty_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    submission: Optional[KernelSubmission] = None
