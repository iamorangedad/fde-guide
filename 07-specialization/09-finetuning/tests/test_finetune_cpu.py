import sys
import unittest
from pathlib import Path

EX = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EX))

from chat_template import CharTokenizer, example_ids, render_prompt, render_training  # noqa: E402
from data_quality import (  # noqa: E402
    assert_no_exact_leak,
    exact_leaks,
    jaccard,
    near_duplicate_pairs,
    split_rows,
)
from lora_math import (  # noqa: E402
    apply_lora,
    attention_lora_count,
    lora_parameter_count,
    lora_scaling,
    mlp_lora_count,
    trainable_fraction,
)
from loss_mask import masked_mean_nll, supervised_token_count  # noqa: E402
from quant_toy import (  # noqa: E402
    dequantize_uniform,
    max_abs_error,
    quantize_uniform,
    storage_bytes,
    theoretical_max_error,
)
from synthetic_sop import LEAKY_COPY, ITEMS, read_jsonl, records, write_jsonl  # noqa: E402


class LoraMathTest(unittest.TestCase):
    def test_square_attention_count(self):
        per = lora_parameter_count(4096, 4096, 8)
        self.assertEqual(per, 65536)
        total = attention_lora_count(32, 4096, 8, projections=4)
        full = 32 * 4 * 4096 * 4096
        self.assertEqual(total, 8_388_608)
        self.assertAlmostEqual(trainable_fraction(total, full), 0.00390625)

    def test_mlp_and_scaling_and_merge(self):
        self.assertEqual(mlp_lora_count(32, 4096, 11008, 8, matrices=3), 11_599_872)
        self.assertAlmostEqual(lora_scaling(16, 8), 2.0)
        self.assertAlmostEqual(lora_scaling(16, 16, use_rslora=True), 4.0)
        merged = apply_lora(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.25]],
            scaling=2.0,
        )
        # B @ A = [[0.5, 0], [0, 0.25]], times 2 added to W.
        self.assertEqual(merged, [[2.0, 2.0], [3.0, 4.5]])


class LossAndTemplateTest(unittest.TestCase):
    def test_prompt_is_prefix_and_loss_ignores_it(self):
        row = records()[0]
        tok = CharTokenizer.from_texts(
            [render_training(row["system"], row["user"], row["assistant"])]
        )
        encoded = example_ids(tok, row["system"], row["user"], row["assistant"])
        self.assertTrue(render_training(row["system"], row["user"], row["assistant"]).startswith(render_prompt(row["system"], row["user"])))
        self.assertTrue(all(v == -100 for v in encoded["labels"][: len(encoded["prompt_ids"])]))
        self.assertEqual(
            supervised_token_count(encoded["labels"]),
            len(encoded["input_ids"]) - len(encoded["prompt_ids"]),
        )
        # 两个 logit 行相等时,被屏蔽的提示token不改变平均损失。
        logits = [[0.0, 0.0] for _ in encoded["input_ids"]]
        # 词表只有真实 tokenizer 那么大,这里改用手工短序列。
        short_logits = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        masked = [-100, -100, 1]
        unmasked = [-100, 1, 1]
        self.assertAlmostEqual(masked_mean_nll(short_logits, masked), masked_mean_nll([[9, 9], [0.0, 1.0], [0.0, 1.0]], masked))
        self.assertGreater(masked_mean_nll(short_logits, unmasked), 0)


class DataTest(unittest.TestCase):
    def test_split_is_disjoint_and_deterministic(self):
        rows = records()
        a = split_rows(rows, (0.7, 0.15, 0.15), seed=0)
        b = split_rows(rows, (0.7, 0.15, 0.15), seed=0)
        self.assertEqual([r["id"] for r in a["train"]], [r["id"] for r in b["train"]])
        ids = [r["id"] for split in a.values() for r in split]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(rows))
        assert_no_exact_leak(a, field="user")

    def test_exact_and_near_duplicates(self):
        leaks = exact_leaks([ITEMS[0]["user"]], [LEAKY_COPY["user"], "完全不同的问题内容"])
        self.assertEqual(leaks, [LEAKY_COPY["user"]])
        self.assertGreater(jaccard(ITEMS[0]["assistant"], LEAKY_COPY["assistant"]), 0.99)
        pairs = near_duplicate_pairs(
            [ITEMS[0]["assistant"], ITEMS[0]["assistant"] + "。"], threshold=0.8
        )
        self.assertEqual(len(pairs), 1)

    def test_jsonl_roundtrip(self):
        path = Path(__file__).resolve().parents[1] / "data" / "sop.jsonl"
        write_jsonl(path)
        self.assertEqual(read_jsonl(path), records())


class QuantTest(unittest.TestCase):
    def test_grid_points_are_exact_and_error_is_bounded(self):
        values = [0.0, 0.5, 1.0, 1.5]
        packed = quantize_uniform(values, bits=2, group_size=4)
        self.assertEqual(dequantize_uniform(packed), values)
        noisy = [0.1, 0.2, 0.4, 0.9]
        packed_noisy = quantize_uniform(noisy, bits=4, group_size=4)
        bound = theoretical_max_error(noisy, bits=4, group_size=4)
        self.assertLessEqual(max_abs_error(noisy, packed_noisy), bound + 1e-9)

    def test_storage_identity(self):
        report = storage_bytes(8, bits=4, group_size=64)
        self.assertEqual(report["data_bytes"], 4)
        self.assertEqual(report["metadata_bytes"], 4)
        self.assertEqual(report["fp16_bytes"], 16)


if __name__ == "__main__":
    unittest.main()
