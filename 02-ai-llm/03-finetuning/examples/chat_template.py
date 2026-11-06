"""离线教学用的字符分词器和固定聊天模板。

这不是生产分词器。生产训练必须使用目标模型自带的 tokenizer 和
apply_chat_template;训练字符串和推理字符串只要差一个控制符,模型就会
在错误的条件分布上学习。本模块把模板收成唯一实现,并检查"提示是完整
序列的前缀"这一不变量。

训练序列:
    <|system|>\\n{system}\\n<|user|>\\n{user}\\n<|assistant|>\\n{assistant}<|eot|>
推理时只渲染到 <|assistant|>\\n,不包含答案,也不包含 <|eot|>。
损失只落在 assistant 正文和 <|eot|> 上。
"""

from __future__ import annotations

import json
from pathlib import Path

from loss_mask import IGNORE_INDEX, causal_labels

SPECIALS = ("<|system|>", "<|user|>", "<|assistant|>", "<|eot|>", "<|pad|>")


def render_training(system: str, user: str, assistant: str) -> str:
    return (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n{assistant}<|eot|>"
    )


def render_prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


class CharTokenizer:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("token id 必须唯一")
        self.pad_id = self.token_to_id["<|pad|>"]
        self.eot_id = self.token_to_id["<|eot|>"]

    @property
    def vocab_size(self) -> int:
        return max(self.id_to_token) + 1

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        chars: set[str] = set()
        for text in texts:
            chars.update(_literal_chars(text))
        tokens = list(SPECIALS) + sorted(chars)
        return cls({tok: i for i, tok in enumerate(tokens)})

    def encode(self, text: str) -> list[int]:
        specials = sorted(SPECIALS, key=len, reverse=True)
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for special in specials:
                if text.startswith(special, i):
                    ids.append(self.token_to_id[special])
                    i += len(special)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            if ch not in self.token_to_id:
                raise KeyError(f"词表中没有字符 {ch!r}")
            ids.append(self.token_to_id[ch])
            i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        parts = []
        for idx in ids:
            if idx == self.pad_id:
                continue
            parts.append(self.id_to_token[idx])
        return "".join(parts)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def _literal_chars(text: str) -> set[str]:
    specials = sorted(SPECIALS, key=len, reverse=True)
    chars: set[str] = set()
    i = 0
    while i < len(text):
        for special in specials:
            if text.startswith(special, i):
                i += len(special)
                break
        else:
            chars.add(text[i])
            i += 1
    return chars


def example_ids(tokenizer: CharTokenizer, system: str, user: str, assistant: str) -> dict[str, list[int]]:
    full = render_training(system, user, assistant)
    prompt = render_prompt(system, user)
    if not full.startswith(prompt):
        raise RuntimeError("提示必须是训练序列的前缀")
    input_ids = tokenizer.encode(full)
    prompt_ids = tokenizer.encode(prompt)
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("分词后提示不再是前缀,模板或特殊符号切分有误")
    labels = causal_labels(input_ids, prompt_len=len(prompt_ids))
    return {"input_ids": input_ids, "prompt_ids": prompt_ids, "labels": labels}


def pad_batch(rows: list[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids, labels, attention = [], [], []
    for row in rows:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [IGNORE_INDEX] * pad)
        attention.append([1] * len(row["input_ids"]) + [0] * pad)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}
