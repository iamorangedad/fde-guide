"""运行第 3 章的 CPU 演算测试和仓库链接检查。不启动 GPU,不下载模型。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    suites = [
        ROOT / "02-ai-llm/examples/test_worked_examples.py",
        ROOT / "tests/test_markdown_links.py",
    ]
    failed = False
    for suite in suites:
        print(f"== {suite.relative_to(ROOT)}", flush=True)
        result = subprocess.run([sys.executable, str(suite)], cwd=ROOT)
        failed = failed or result.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
