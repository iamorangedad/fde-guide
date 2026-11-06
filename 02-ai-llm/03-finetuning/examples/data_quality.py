"""数据集划分、精确泄漏和粗粒度近重复。纯标准库,不访问网络。"""

from __future__ import annotations

import random
import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().casefold()
    return re.sub(r"\s+", "", text)


def exact_leaks(train_texts: list[str], eval_texts: list[str]) -> list[str]:
    train_norm = {normalize_text(text) for text in train_texts}
    return [text for text in eval_texts if normalize_text(text) in train_norm]


def char_bigrams(text: str) -> set[str]:
    norm = normalize_text(text)
    if len(norm) < 2:
        return {norm} if norm else set()
    return {norm[i : i + 2] for i in range(len(norm) - 1)}


def jaccard(a: str, b: str) -> float:
    left, right = char_bigrams(a), char_bigrams(b)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def near_duplicate_pairs(
    texts: list[str], threshold: float = 0.8, ids: list[str] | None = None
) -> list[tuple[str, str, float]]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 [0, 1]")
    names = ids if ids is not None else [str(i) for i in range(len(texts))]
    if len(names) != len(texts):
        raise ValueError("ids 与 texts 长度必须相同")
    found = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = jaccard(texts[i], texts[j])
            if score >= threshold:
                found.append((names[i], names[j], score))
    return found


def split_rows(rows: list[dict], ratios: tuple[float, float, float], seed: int) -> dict[str, list[dict]]:
    """按比例打乱后切成互斥的 train/val/test。同一行不会同时出现在两份里。"""
    if len(rows) < 3:
        raise ValueError("至少需要 3 条才能分成 train/val/test")
    if any(part <= 0 for part in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios 必须为正且和为 1")
    order = list(rows)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_train = min(n - 2, max(1, int(n * ratios[0])))
    n_val = min(n - n_train - 1, max(1, int(n * ratios[1])))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("划分后有空集,请增加数据或调整比例")
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def assert_no_exact_leak(splits: dict[str, list[dict]], field: str = "user") -> None:
    leaks = exact_leaks(
        [row[field] for row in splits["train"]],
        [row[field] for row in splits["val"]] + [row[field] for row in splits["test"]],
    )
    if leaks:
        raise ValueError(f"训练与验证/测试存在精确泄漏: {leaks[:3]}")
