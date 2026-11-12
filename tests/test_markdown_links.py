"""检查仓库内 Markdown 相对链接,要求章节页带有返回主页,并要求中英文文档一一对应。"""

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
            if any(part.startswith(".") for part in path.parts):
                continue
            is_en = path.name.endswith(".en.md")
            home_name = "README.en.md" if is_en else "README.md"
            label = "Back to home" if is_en else "返回主页"
            if path.name == home_name and path.parent == ROOT:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            head = "\n".join(lines[:25])
            tail = "\n".join(lines[-15:])
            if label not in head or label not in tail:
                problems.append(str(path.relative_to(ROOT)))
            home_targets = re.findall(
                re.escape(label) + r"\]\(([^)]+)\)", head + "\n" + tail
            )
            for target in home_targets:
                resolved = (path.parent / target).resolve()
                if resolved != (ROOT / home_name).resolve():
                    problems.append(f"{path.relative_to(ROOT)} home -> {target}")
        self.assertEqual(problems, [])

    def test_every_doc_has_language_counterpart(self):
        """每个 X.md 都要有 X.en.md,反之亦然。"""
        missing = []
        for path in ROOT.rglob("*.md"):
            if any(part.startswith(".") for part in path.parts):
                continue
            if path.name.endswith(".en.md"):
                other = path.with_name(path.name[: -len(".en.md")] + ".md")
            else:
                other = path.with_name(path.name[: -len(".md")] + ".en.md")
            if not other.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {other.relative_to(ROOT)}")
        self.assertEqual(missing, [])

if __name__ == "__main__":
    unittest.main()
