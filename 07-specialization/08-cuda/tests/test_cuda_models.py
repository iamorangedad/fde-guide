import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from coalesce import (  # noqa: E402
    column_indices,
    consecutive_float_addresses,
    count_segments,
    read_conflict_degree,
)
from occupancy import DeviceLimits, estimate_occupancy  # noqa: E402
from reference_kernels import (  # noqa: E402
    add_float4,
    matmul_reference,
    matmul_tiled_sim,
    reduce_naive_block,
    reduce_sequential_block,
    reduce_warp_block,
)
from roofline import attainable, gemm_flops, ridge_point, square_gemm_compulsory_ai  # noqa: E402


class CoalesceTest(unittest.TestCase):
    def test_aligned_float_warp_uses_four_segments(self):
        self.assertEqual(count_segments(consecutive_float_addresses(0)), 4)

    def test_misaligned_float_warp_uses_five_segments(self):
        self.assertEqual(count_segments(consecutive_float_addresses(4)), 5)

    def test_stride_32_floats_are_not_coalesced(self):
        addresses = [i * 32 * 4 for i in range(32)]
        self.assertEqual(count_segments(addresses), 32)

    def test_shared_broadcast_and_padding(self):
        self.assertEqual(read_conflict_degree([7] * 32), 1)
        self.assertEqual(read_conflict_degree(list(range(32))), 1)
        self.assertEqual(read_conflict_degree(column_indices(0, 32, 32)), 32)
        self.assertEqual(read_conflict_degree(column_indices(0, 32, 33)), 1)


class OccupancyTest(unittest.TestCase):
    def setUp(self):
        self.limits = DeviceLimits(
            max_threads_per_sm=2048,
            max_threads_per_block=1024,
            regs_per_sm=65536,
            max_regs_per_block=65536,
            smem_bytes_per_sm=65536,
            max_smem_bytes_per_block=49152,
            max_blocks_per_sm=32,
        )

    def test_register_and_smem_bind_together(self):
        est = estimate_occupancy(self.limits, 256, 64, 16384)
        self.assertEqual(est.active_blocks, 4)
        self.assertAlmostEqual(est.occupancy, 0.5)
        self.assertIn("registers", est.limiter)
        self.assertIn("shared_memory", est.limiter)

    def test_block_too_wide_cannot_launch(self):
        est = estimate_occupancy(self.limits, 2048, 8, 0)
        self.assertEqual(est.active_blocks, 0)
        self.assertEqual(est.limiter, "threads_per_block")

    def test_partial_warp_consumes_a_full_warp_slot(self):
        est = estimate_occupancy(self.limits, 100, 0, 0)
        self.assertEqual(est.warps_per_block, 4)
        self.assertEqual(est.active_blocks, 16)
        self.assertAlmostEqual(est.occupancy, 1.0)

    def test_smem_over_block_limit(self):
        est = estimate_occupancy(self.limits, 256, 8, 60000)
        self.assertEqual(est.active_blocks, 0)
        self.assertEqual(est.limiter, "shared_memory")


class RooflineTest(unittest.TestCase):
    def test_ridge_and_square_gemm_bounds(self):
        peak = 1e14
        bandwidth = 2e12
        self.assertAlmostEqual(ridge_point(peak, bandwidth), 50.0)
        wide = attainable(square_gemm_compulsory_ai(1024, 2), peak, bandwidth)
        narrow = attainable(square_gemm_compulsory_ai(32, 2), peak, bandwidth)
        self.assertAlmostEqual(square_gemm_compulsory_ai(1024, 2), 1024 / 3)
        self.assertEqual(wide.limited_by, "compute")
        self.assertEqual(wide.attainable_flops, peak)
        self.assertEqual(narrow.limited_by, "bandwidth")
        self.assertAlmostEqual(narrow.attainable_flops, (32 / 3) * bandwidth)
        self.assertEqual(gemm_flops(2, 3, 4), 48)


class ReferenceKernelTest(unittest.TestCase):
    def test_vector_and_reductions(self):
        self.assertEqual(add_float4([1, 2, 3, 4], [10, 20, 30, 40]), [11, 22, 33, 44])
        ones = [1.0] * 8
        self.assertEqual(reduce_naive_block(ones), 8)
        self.assertEqual(reduce_sequential_block(ones), 8)
        self.assertEqual(reduce_warp_block([1.0] * 32), 32)
        self.assertEqual(reduce_warp_block([1.0] * 64), 64)

    def test_tiled_matmul_matches_reference_on_ragged_shape(self):
        a = [[float((r + k) % 5) for k in range(19)] for r in range(18)]
        b = [[float((k + 2 * c) % 4) for c in range(20)] for k in range(19)]
        self.assertEqual(matmul_tiled_sim(a, b, tile=16), matmul_reference(a, b))


class CudaSourceContractTest(unittest.TestCase):
    def test_examples_check_errors_warmup_and_reference(self):
        examples = ROOT / "examples"
        for name in ("vector_add.cu", "reduction.cu", "matmul_tiled.cu"):
            text = (examples / name).read_text(encoding="utf-8")
            for needle in ("CUDA_CHECK", "cudaEvent", "warmup", "correct"):
                self.assertIn(needle, text, name)
            self.assertIn("不是", text)


if __name__ == "__main__":
    unittest.main()
