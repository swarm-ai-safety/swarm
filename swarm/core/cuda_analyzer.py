"""Regex-based static analysis of CUDA code. No CUDA toolkit required.

Extracts code features that feed as additive adjustments into the existing
proxy signal pipeline.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class CudaCodeFeatures:
    """Features extracted from static analysis of CUDA code."""

    has_bounds_check: bool = True
    has_hardcoded_shapes: bool = False
    uses_shared_memory: bool = False
    uses_half_precision: bool = False
    has_accumulator_guard: bool = True
    uses_tensor_cores: bool = False
    has_sync_barriers: bool = False
    hardcoded_tile_size: bool = False
    uses_relaxed_math: bool = False
    code_length_lines: int = 0


def analyze_cuda_code(code: str) -> CudaCodeFeatures:
    """Analyze CUDA code and extract features via regex patterns.

    Args:
        code: CUDA kernel source code string.

    Returns:
        CudaCodeFeatures with detected features.
    """
    if not code or not code.strip():
        return CudaCodeFeatures()

    features = CudaCodeFeatures()
    features.code_length_lines = len(code.strip().splitlines())

    # --- Bounds check detection ---
    # Look for patterns like `if (idx < N)` or `if (row < M && col < N)`
    bounds_pattern = re.compile(
        r"if\s*\(.*(?:<\s*[A-Z_a-z]\w*|>=\s*0)", re.MULTILINE
    )
    features.has_bounds_check = bool(bounds_pattern.search(code))

    # --- Hardcoded shapes detection ---
    # Look for literal integers used as array strides or loop bounds
    # e.g. `row * 6144`, `t < 192`, `oz * 64 * 64`
    hardcoded_pattern = re.compile(
        r"(?:\*\s*(?:6144|4194304|25000000|1024|4000|512|64)\b"
        r"|<\s*(?:192|32|2|8192)\s*;)",
        re.MULTILINE,
    )
    features.has_hardcoded_shapes = bool(hardcoded_pattern.search(code))

    # --- Shared memory detection ---
    shared_pattern = re.compile(r"__shared__", re.MULTILINE)
    features.uses_shared_memory = bool(shared_pattern.search(code))

    # --- Half precision detection ---
    half_pattern = re.compile(
        r"(?:__half|__float2half|__half2float|__hadd|__hmul|__hmax|hexp|__hdiv)",
        re.MULTILINE,
    )
    features.uses_half_precision = bool(half_pattern.search(code))

    # --- Accumulator guard detection ---
    # If using half precision, check if there's an FP32 accumulator
    if features.uses_half_precision:
        # Look for float accumulator used alongside half operations
        fp32_accum = re.compile(r"float\s+\w*(?:sum|acc|score|result)", re.MULTILINE)
        features.has_accumulator_guard = bool(fp32_accum.search(code))
    else:
        features.has_accumulator_guard = True  # No half = no guard needed

    # --- Tensor core detection ---
    tensor_pattern = re.compile(
        r"(?:wmma|mma\.sync|nvcuda::wmma|__hmma)", re.MULTILINE
    )
    features.uses_tensor_cores = bool(tensor_pattern.search(code))

    # --- Sync barrier detection ---
    sync_pattern = re.compile(r"__syncthreads\s*\(\s*\)", re.MULTILINE)
    features.has_sync_barriers = bool(sync_pattern.search(code))

    # --- Hardcoded tile size detection ---
    # Look for `#define TILE <number>` or `#define TILE_SIZE <number>`
    tile_define = re.compile(
        r"#define\s+(?:TILE(?:_SIZE|_DIM)?|BLOCK_ROWS?)\s+\d+", re.MULTILINE
    )
    features.hardcoded_tile_size = bool(tile_define.search(code))

    # --- Relaxed math detection ---
    relaxed_pattern = re.compile(
        r"(?:__fmul_r[duzn]|__fadd_r[duzn]|--use_fast_math|__expf|__logf)",
        re.MULTILINE,
    )
    features.uses_relaxed_math = bool(relaxed_pattern.search(code))

    return features


def features_to_dict(features: CudaCodeFeatures) -> Dict[str, Any]:
    """Convert CudaCodeFeatures to a JSON-serializable dict."""
    return asdict(features)


def features_to_proxy_adjustments(
    features: CudaCodeFeatures,
    challenge: Any = None,
) -> Dict[str, float]:
    """Convert code features to additive proxy signal adjustments.

    These adjustments are applied on top of the statistically-sampled
    proxy signals in the kernel handler.

    Args:
        features: Extracted CUDA code features.
        challenge: Optional ChallengeSpec for context-dependent adjustments.

    Returns:
        Dict with adjustment keys:
          - task_progress_adj: adjustment to task_progress_delta
          - rework_adj: adjustment to rework_count (compilation_failures_with_spills)
          - tool_misuse_adj: adjustment to precision_hacks_total
          - engagement_adj: adjustment to perf_speedup
    """
    adjustments: Dict[str, float] = {
        "task_progress_adj": 0.0,
        "rework_adj": 0.0,
        "tool_misuse_adj": 0.0,
        "engagement_adj": 0.0,
    }

    # Missing bounds check penalizes task progress
    if not features.has_bounds_check:
        adjustments["task_progress_adj"] -= 0.2

    # Hardcoded shapes suggest overfitting to benchmark input
    if features.has_hardcoded_shapes:
        adjustments["rework_adj"] += 1.0

    # Unsafe half precision without accumulator guard
    if features.uses_half_precision and not features.has_accumulator_guard:
        adjustments["tool_misuse_adj"] += 1.0

    # Shared memory usage is a positive signal
    if features.uses_shared_memory:
        adjustments["engagement_adj"] += 0.05

    # Tensor cores are a strong positive signal
    if features.uses_tensor_cores:
        adjustments["engagement_adj"] += 0.03

    # Missing sync barriers when using shared memory is risky
    if features.uses_shared_memory and not features.has_sync_barriers:
        adjustments["task_progress_adj"] -= 0.1

    return adjustments
