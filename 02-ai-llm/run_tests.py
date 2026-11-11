"""运行专项部分的 CPU 测试。不启动 GPU,不下载模型。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    suites = [
        ROOT / "02-ai-llm/02-cuda/tests/test_cuda_models.py",
        ROOT / "02-ai-llm/03-finetuning/tests/test_finetune_cpu.py",
        ROOT / "02-ai-llm/04-training/tests/test_training_math.py",
        ROOT / "02-ai-llm/05-inference/tests/test_inference_math.py",
        ROOT / "02-ai-llm/06-workflow/tests/test_gates.py",
        ROOT / "02-ai-llm/textbook/examples/test_worked_examples.py",
        ROOT / "tests/test_markdown_links.py",
    ]
    failed = False
    for suite in suites:
        print(f"== {suite.relative_to(ROOT)}", flush=True)
        result = subprocess.run([sys.executable, str(suite)], cwd=ROOT)
        failed = failed or result.returncode != 0
    optional = ROOT / "02-ai-llm/03-finetuning/tests/test_train_optional.py"
    if optional.exists():
        print(f"== {optional.relative_to(ROOT)} (缺少 torch/transformers/peft 时跳过)", flush=True)
        result = subprocess.run([sys.executable, str(optional)], cwd=ROOT)
        failed = failed or result.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
