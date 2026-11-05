"""检查仓库内 Markdown 相对链接,并要求章节页带有返回主页。"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"```.*?```", re.DOTALL)


def visible_text(text: str) -> str:
    return FENCE.sub("", text)


class MarkdownLinkTest(unittest.TestCase):
    def test_local_links_resolve(self):
        missing = []
        for path in ROOT.rglob("*.md"):
            if any(part.startswith(".") for part in path.parts):
                continue
            text = visible_text(path.read_text(encoding="utf-8"))
            for target in LINK.findall(text):
                target = target.strip()
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                file_part, _, _anchor = target.partition("#")
                if not file_part:
                    continue
                resolved = (path.parent / file_part).resolve()
                if not resolved.exists():
                    missing.append(f"{path.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_chapter_pages_link_home(self):
        problems = []
        for path in ROOT.rglob("*.md"):
            if path.name == "README.md" and path.parent == ROOT:
                continue
            if any(part.startswith(".") for part in path.parts):
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            head = "\n".join(lines[:25])
            tail = "\n".join(lines[-15:])
            if "返回主页" not in head or "返回主页" not in tail:
                problems.append(str(path.relative_to(ROOT)))
            if "](README.md)" in head and path.parent != ROOT:
                # 章节里的 README.md 可能是本章目录,主页必须指向仓库根。
                pass
            home_targets = re.findall(r"返回主页\]\(([^)]+)\)", head + "\n" + tail)
            for target in home_targets:
                resolved = (path.parent / target).resolve()
                if resolved != (ROOT / "README.md").resolve():
                    problems.append(f"{path.relative_to(ROOT)} home -> {target}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
