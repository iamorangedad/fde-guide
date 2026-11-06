"""一次实验至少要能被别人按同一命令重跑到同一配置。

日志记录配置、种子、数据哈希和库版本。它不记录虚构的指标提升。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_KEYS = (
    "created_at",
    "seed",
    "config",
    "data_sha256",
    "python",
    "command",
    "library_versions",
    "git_sha",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def library_versions(names: tuple[str, ...] = ("torch", "transformers", "peft")) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            found[name] = None
            continue
        found[name] = getattr(module, "__version__", "unknown")
    return found


def git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def build_record(config: dict, seed: int, data_path: Path, cwd: Path) -> dict:
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": config,
        "data_sha256": sha256_file(data_path),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "library_versions": library_versions(),
        "git_sha": git_sha(cwd),
    }
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise RuntimeError(f"实验日志缺少 {missing}")
    return record


def write_record(record: dict, path: Path) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
