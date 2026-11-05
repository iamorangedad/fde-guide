import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))

from capacity import cost_from_inputs, max_cached_tokens, remaining_bytes, weight_bytes  # noqa: E402
from kv_cache import kv_cache_bytes, paged_blocks, paged_waste_fraction  # noqa: E402
from latency import output_throughput, percentile_type7, request_latency  # noqa: E402
from loadsim import Request, simulate_continuous, simulate_serial, simulate_static  # noqa: E402
from speculative import expected_tokens, modeled_speedup  # noqa: E402


def _pair() -> list[Request]:
    return [
        Request("A", 0, 10, 2, "alpha"),
        Request("B", 5, 10, 2, "beta"),
    ]


class KvAndLatencyTest(unittest.TestCase):
    def test_gqa_cache_is_512_mib_for_the_worked_shape(self):
        report = kv_cache_bytes(32, 8, 128, 4096, batch=1, bytes_per_elem=2)
        self.assertEqual(report["bytes_per_token_per_sequence"], 131072)
        self.assertEqual(report["total_bytes"], 536_870_912)
        mha = kv_cache_bytes(32, 32, 128, 4096, batch=1, bytes_per_elem=2)
        self.assertEqual(mha["total_bytes"], report["total_bytes"] * 4)

    def test_paged_waste_is_only_the_tail(self):
        self.assertEqual(paged_blocks(4096, 16), 256)
        self.assertEqual(paged_waste_fraction(4096, 16), 0.0)
        self.assertAlmostEqual(paged_waste_fraction(17, 16), 15 / 32)

    def test_latency_definitions(self):
        self.assertEqual(request_latency(30, 5, 1), 30)
        self.assertEqual(request_latency(30, 5, 4), 45)
        self.assertEqual(percentile_type7([10, 20, 30, 40], 50), 25)
        self.assertAlmostEqual(output_throughput(90, 3), 30)


class SpeculativeAndCapacityTest(unittest.TestCase):
    def test_expected_tokens_limits(self):
        self.assertAlmostEqual(expected_tokens(0.0, 4), 1.0)
        self.assertAlmostEqual(expected_tokens(1.0, 4), 5.0)
        self.assertAlmostEqual(expected_tokens(0.5, 1), (1 - 0.5**2) / 0.5)
        # 草稿并不免费:这个比值完全由输入决定,不是测量。
        self.assertAlmostEqual(modeled_speedup(1.0, 3, t_draft=1.0, t_target=4.0), 16 / 7)

    def test_capacity_budget(self):
        weights = weight_bytes(7_000_000_000, 2)
        left = remaining_bytes(24 * 1024**3, weights, 2 * 1024**3)
        per_token = 131072
        tokens = max_cached_tokens(left, per_token)
        self.assertEqual(int(left), 9_622_320_128)
        self.assertEqual(tokens, 73412)
        bill = cost_from_inputs(2, 1800, price_per_gpu_hour=1)
        self.assertAlmostEqual(bill["gpu_hours"], 1.0)
        self.assertAlmostEqual(bill["cost"], 1.0)


class LoadSimTest(unittest.TestCase):
    def test_hand_computed_policies(self):
        reqs = _pair()
        serial = {row["rid"]: row for row in simulate_serial(reqs, 4)["rows"]}
        self.assertEqual(serial["A"]["latency"], 14)
        self.assertEqual(serial["A"]["ttft"], 10)
        self.assertEqual(serial["B"]["latency"], 23)
        self.assertEqual(serial["B"]["ttft"], 19)

        static = {row["rid"]: row for row in simulate_static(reqs, 4, group_size=2)["rows"]}
        self.assertEqual(static["A"]["latency"], 29)
        self.assertEqual(static["A"]["ttft"], 15)
        self.assertEqual(static["B"]["latency"], 24)
        self.assertEqual(static["B"]["ttft"], 20)

        cont = {row["rid"]: row for row in simulate_continuous(reqs, 4, max_batch=2)["rows"]}
        self.assertEqual(cont["A"]["latency"], 24)
        self.assertEqual(cont["A"]["ttft"], 10)
        self.assertEqual(cont["B"]["latency"], 19)
        self.assertEqual(cont["B"]["ttft"], 15)
        self.assertEqual(simulate_serial(reqs, 4)["exact_match_rate"], 1.0)

    def test_static_straggler_holds_the_short_request(self):
        reqs = [
            Request("A", 0, 10, 2, "alpha"),
            Request("B", 5, 10, 4, "beta"),
        ]
        static = {row["rid"]: row for row in simulate_static(reqs, 4, 2)["rows"]}
        self.assertEqual(static["A"]["latency"], 37)
        self.assertEqual(static["B"]["latency"], 32)

    def test_single_token_request_finishes_at_prefill(self):
        reqs = [Request("A", 3, 10, 1, "only")]
        row = simulate_continuous(reqs, 4, 1)["rows"][0]
        self.assertEqual(row["latency"], 10)
        self.assertEqual(row["ttft"], 10)


if __name__ == "__main__":
    unittest.main()
