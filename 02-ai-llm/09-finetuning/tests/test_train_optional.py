"""有 torch、transformers、peft 时才跑一步真实训练。没有这些包则跳过。"""

import math
import sys
import unittest
from pathlib import Path

EX = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EX))


class OptionalTrainTest(unittest.TestCase):
    def test_one_step_self_check(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"可选依赖未安装: {exc}")
        from train_lora import run

        report = run(steps=1, batch_size=2, lr=1e-3, seed=0, self_check=True)
        self.assertEqual(report["trainable"], 2048)
        self.assertTrue(report["merged_matches"])
        self.assertTrue(math.isfinite(report["train_loss_last"]))
        self.assertTrue(math.isfinite(report["val_loss"]))
        self.assertGreater(report["vocab_size"], 5)
        self.assertLess(report["longest_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
