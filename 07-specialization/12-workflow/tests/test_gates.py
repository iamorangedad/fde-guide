import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))

from gates import evaluate_gates, validate_experiment_plan  # noqa: E402


def _plan(**overrides):
    plan = {
        "hypothesis": "只屏蔽提示 token 会提高引用格式命中,且不改变检索。",
        "independent_variable": "prompt_loss_mask",
        "controlled_factors": ["data", "seed", "steps", "rank", "learning_rate"],
        "metrics": ["citation_exact_match", "general_probe_exact_match"],
        "gates": [
            {"name": "citation_exact_match", "op": ">=", "threshold": 0.9},
            {"name": "general_probe_drop", "op": "<=", "threshold": 0.05},
        ],
        "rollback": "金丝雀的忠实度或 p95 TTFT 越出门限则切回上一版本。",
    }
    plan.update(overrides)
    return plan


class GateTest(unittest.TestCase):
    def test_plan_and_gates(self):
        self.assertEqual(validate_experiment_plan(_plan()), [])
        failures = evaluate_gates(
            {"citation_exact_match": 0.91, "general_probe_drop": 0.02},
            _plan()["gates"],
        )
        self.assertEqual(failures, [])
        blocked = evaluate_gates(
            {"citation_exact_match": 0.5, "general_probe_drop": 0.2},
            _plan()["gates"],
        )
        self.assertEqual(len(blocked), 2)

    def test_two_variables_are_rejected(self):
        errors = validate_experiment_plan(
            _plan(independent_variable=["lr", "rank"], changed_together=["lr", "rank"])
        )
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
