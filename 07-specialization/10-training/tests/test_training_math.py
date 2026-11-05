import json
import sys
import tempfile
import unittest
from pathlib import Path

PY = Path(__file__).resolve().parents[1] / "py"
sys.path.insert(0, str(PY))

from experiment_log import REQUIRED_KEYS, build_record, sha256_file  # noqa: E402
from grad_accum import unweighted_average, weighted_average  # noqa: E402
from memory_accounting import (  # noqa: E402
    activation_bytes,
    adam_fp32_bytes,
    adam_mixed_precision_bytes,
    checkpoint_boundary_bytes,
    gib,
)
from packing import block_causal_mask, pack_first_fit, padding_waste_fraction  # noqa: E402


class MemoryTest(unittest.TestCase):
    def test_adam_totals_match_the_16_byte_identity(self):
        params = 7_000_000_000
        fp32 = adam_fp32_bytes(params)
        mixed = adam_mixed_precision_bytes(params)
        self.assertEqual(fp32["total"], 16 * params)
        self.assertEqual(mixed["total"], 16 * params)
        self.assertEqual(mixed["low_precision_parameters"], 2 * params)
        self.assertAlmostEqual(gib(mixed["total"]), 112_000_000_000 / (1024**3))
        self.assertNotEqual(
            fp32["parameters"] + fp32["gradients"],
            mixed["low_precision_parameters"] + mixed["low_precision_gradients"],
        )

    def test_activation_switch_and_checkpoint_boundary(self):
        plain = activation_bytes(1, 4, 8, 2, 2, saved_tensors_per_layer=1)
        scores = activation_bytes(
            1, 4, 8, 2, 2, saved_tensors_per_layer=1, materialized_scores=True, heads=2
        )
        self.assertGreater(scores["total"], plain["total"])
        self.assertEqual(scores["scores"], 2 * 1 * 2 * 4 * 4 * 2)
        self.assertEqual(checkpoint_boundary_bytes(1, 4, 8, segments=2, bytes_per_elem=2), plain["hidden_like"])


class GradAndPackTest(unittest.TestCase):
    def test_unequal_tokens_need_weights(self):
        micros = [[1.0, 0.0], [0.0, 3.0]]
        weighted = weighted_average(micros, [2, 6])
        plain = unweighted_average(micros)
        self.assertEqual(weighted, [0.25, 2.25])
        self.assertNotEqual(weighted, plain)

    def test_padding_and_block_mask(self):
        self.assertAlmostEqual(padding_waste_fraction([2, 2, 8]), 0.5)
        packed = pack_first_fit([[1, 2, 3], [4], [5, 6]], max_len=4)
        self.assertEqual(sum(bucket["length"] for bucket in packed), 6)
        segments = [(0, 3), (3, 4)]
        mask = block_causal_mask(4, segments)
        self.assertTrue(mask[2][0])
        self.assertFalse(mask[2][3])
        self.assertFalse(mask[3][0])
        self.assertTrue(mask[3][3])


class LogTest(unittest.TestCase):
    def test_record_has_required_fields_and_stable_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text('{"a":1}\n', encoding="utf-8")
            first = sha256_file(path)
            record = build_record({"lr": 1e-4, "note": "illustrative"}, 7, path, Path(__file__).resolve().parents[3])
            self.assertEqual(record["data_sha256"], first)
            self.assertTrue(record["git_sha"])
            for key in REQUIRED_KEYS:
                self.assertIn(key, record)
            json.dumps(record)


if __name__ == "__main__":
    unittest.main()
