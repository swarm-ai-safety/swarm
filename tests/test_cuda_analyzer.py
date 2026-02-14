"""Tests for CUDA code static analyzer."""

import pytest

from swarm.core.cuda_analyzer import (
    CudaCodeFeatures,
    analyze_cuda_code,
    features_to_dict,
    features_to_proxy_adjustments,
)


class TestBoundsCheckDetection:
    def test_bounds_check_present(self):
        code = """
        __global__ void kernel(float* A, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                A[idx] = 1.0f;
            }
        }
        """
        features = analyze_cuda_code(code)
        assert features.has_bounds_check is True

    def test_bounds_check_missing(self):
        code = """
        __global__ void kernel(float* A, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            A[idx] = 1.0f;
        }
        """
        features = analyze_cuda_code(code)
        assert features.has_bounds_check is False


class TestHardcodedShapeDetection:
    def test_hardcoded_stride(self):
        code = """
        __global__ void kernel(float* A) {
            int row = blockIdx.y * 32 + threadIdx.y;
            A[row * 6144 + threadIdx.x] = 1.0f;
        }
        """
        features = analyze_cuda_code(code)
        assert features.has_hardcoded_shapes is True

    def test_no_hardcoded_shapes(self):
        code = """
        __global__ void kernel(float* A, int N, int M) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                A[idx] = 1.0f;
            }
        }
        """
        features = analyze_cuda_code(code)
        assert features.has_hardcoded_shapes is False

    def test_hardcoded_loop_bound(self):
        code = """
        __global__ void kernel(float* A) {
            for (int t = 0; t < 192; t++) {
                A[t] = 1.0f;
            }
        }
        """
        features = analyze_cuda_code(code)
        assert features.has_hardcoded_shapes is True


class TestSharedMemoryDetection:
    def test_shared_memory_present(self):
        code = """
        __global__ void kernel(float* A) {
            __shared__ float sdata[256];
            sdata[threadIdx.x] = A[threadIdx.x];
            __syncthreads();
        }
        """
        features = analyze_cuda_code(code)
        assert features.uses_shared_memory is True
        assert features.has_sync_barriers is True

    def test_no_shared_memory(self):
        code = """
        __global__ void kernel(float* A, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) A[idx] = 1.0f;
        }
        """
        features = analyze_cuda_code(code)
        assert features.uses_shared_memory is False


class TestHalfPrecisionDetection:
    def test_half_precision_without_guard(self):
        code = """
        __global__ void kernel(const float* A, float* B) {
            __shared__ __half sdata[256];
            __half val = __float2half(A[threadIdx.x]);
            __half result = __hadd(val, val);
            B[threadIdx.x] = __half2float(result);
        }
        """
        features = analyze_cuda_code(code)
        assert features.uses_half_precision is True
        assert features.has_accumulator_guard is False

    def test_half_precision_with_guard(self):
        code = """
        __global__ void kernel(const float* A, float* B) {
            __half val = __float2half(A[threadIdx.x]);
            float sum = 0.0f;
            sum += __half2float(val);
            B[threadIdx.x] = sum;
        }
        """
        features = analyze_cuda_code(code)
        assert features.uses_half_precision is True
        assert features.has_accumulator_guard is True

    def test_no_half_precision(self):
        code = """
        __global__ void kernel(float* A) {
            float sum = 0.0f;
            A[0] = sum;
        }
        """
        features = analyze_cuda_code(code)
        assert features.uses_half_precision is False
        assert features.has_accumulator_guard is True  # No half = guard not needed


class TestEmptyCodeDefaults:
    def test_empty_string(self):
        features = analyze_cuda_code("")
        assert features.has_bounds_check is True
        assert features.has_hardcoded_shapes is False
        assert features.uses_shared_memory is False
        assert features.code_length_lines == 0

    def test_whitespace_only(self):
        features = analyze_cuda_code("   \n\n  ")
        assert features.code_length_lines == 0


class TestSyncBarriers:
    def test_shared_memory_without_sync(self):
        code = """
        __global__ void kernel(float* A) {
            __shared__ float sdata[256];
            sdata[threadIdx.x] = A[threadIdx.x];
            // missing sync barrier
            A[threadIdx.x] = sdata[threadIdx.x + 1];
        }
        """
        features = analyze_cuda_code(code)
        assert features.uses_shared_memory is True
        assert features.has_sync_barriers is False


class TestCodeLength:
    def test_line_count(self):
        code = "line1\nline2\nline3\n"
        features = analyze_cuda_code(code)
        assert features.code_length_lines == 3


class TestProxyAdjustments:
    def test_missing_bounds_check_penalizes_progress(self):
        features = CudaCodeFeatures(has_bounds_check=False)
        adj = features_to_proxy_adjustments(features)
        assert adj["task_progress_adj"] == pytest.approx(-0.2)

    def test_hardcoded_shapes_increase_rework(self):
        features = CudaCodeFeatures(has_hardcoded_shapes=True)
        adj = features_to_proxy_adjustments(features)
        assert adj["rework_adj"] == pytest.approx(1.0)

    def test_unsafe_half_increases_misuse(self):
        features = CudaCodeFeatures(
            uses_half_precision=True,
            has_accumulator_guard=False,
        )
        adj = features_to_proxy_adjustments(features)
        assert adj["tool_misuse_adj"] == pytest.approx(1.0)

    def test_shared_memory_boosts_engagement(self):
        features = CudaCodeFeatures(uses_shared_memory=True, has_sync_barriers=True)
        adj = features_to_proxy_adjustments(features)
        assert adj["engagement_adj"] == pytest.approx(0.05)

    def test_defaults_zero_adjustments(self):
        features = CudaCodeFeatures()
        adj = features_to_proxy_adjustments(features)
        assert adj["task_progress_adj"] == 0.0
        assert adj["rework_adj"] == 0.0
        assert adj["tool_misuse_adj"] == 0.0
        assert adj["engagement_adj"] == 0.0

    def test_missing_sync_with_shared_memory(self):
        features = CudaCodeFeatures(
            uses_shared_memory=True, has_sync_barriers=False
        )
        adj = features_to_proxy_adjustments(features)
        assert adj["task_progress_adj"] == pytest.approx(-0.1)
        assert adj["engagement_adj"] == pytest.approx(0.05)

    def test_tensor_cores_boost_engagement(self):
        features = CudaCodeFeatures(uses_tensor_cores=True)
        adj = features_to_proxy_adjustments(features)
        assert adj["engagement_adj"] == pytest.approx(0.03)


class TestFeaturesToDict:
    def test_serializable(self):
        features = CudaCodeFeatures(uses_shared_memory=True, code_length_lines=42)
        d = features_to_dict(features)
        assert isinstance(d, dict)
        assert d["uses_shared_memory"] is True
        assert d["code_length_lines"] == 42
