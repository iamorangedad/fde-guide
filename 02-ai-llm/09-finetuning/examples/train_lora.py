"""在随机初始化的微型 GPT-2 上跑通 Hugging Face + PEFT LoRA。

这个脚本不下载模型,不访问付费 API,也不需要 GPU。它证明的是训练配方:
聊天模板、提示损失屏蔽、只更新 LoRA 参数、验证集损失、以及 merge 后
前向与未合并适配器一致。

超参数是为了让 CPU 在数步内跑完而选的教学配置,不是生产推荐,也不是基准。
打印出的损失只描述这次随机初始化和这份虚构数据,不能拿去比较模型质量。

依赖(CPU 冒烟时安装过;没有这些包时脚本会退出码 2,标准库测试仍可运行):
    torch、transformers、peft
见同目录 requirements.txt 中的版本。QLoRA / bitsandbytes 不在这条路径里。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from chat_template import CharTokenizer, example_ids, pad_batch, render_training
from data_quality import assert_no_exact_leak, split_rows
from synthetic_sop import records


def _require_ml():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        print(
            "缺少 torch、transformers 或 peft,无法运行微调脚本。\n"
            "这是依赖缺失,不是算法失败。CPU 环境可参考 requirements.txt;\n"
            "无网络或无 GPU 时,请只运行标准库单元测试。\n"
            f"原始错误: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return torch, LoraConfig, get_peft_model, GPT2Config, GPT2LMHeadModel


def build_tokenizer(rows: list[dict]) -> CharTokenizer:
    texts = [render_training(row["system"], row["user"], row["assistant"]) for row in rows]
    return CharTokenizer.from_texts(texts)


def encode_split(tokenizer: CharTokenizer, rows: list[dict]) -> list[dict]:
    return [
        example_ids(tokenizer, row["system"], row["user"], row["assistant"]) for row in rows
    ]


def batches(torch, encoded: list[dict], pad_id: int, batch_size: int):
    out = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        padded = pad_batch(chunk, pad_id)
        out.append(
            {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in padded.items()
            }
        )
    return out


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def frozen_snapshot(model) -> dict[str, list[float]]:
    snap = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            snap[name] = param.detach().flatten()[:8].float().cpu().tolist()
    return snap


def run(steps: int, batch_size: int, lr: float, seed: int, self_check: bool) -> dict:
    torch, LoraConfig, get_peft_model, GPT2Config, GPT2LMHeadModel = _require_ml()
    random.seed(seed)
    torch.manual_seed(seed)

    rows = records()
    splits = split_rows(rows, (0.7, 0.15, 0.15), seed)
    assert_no_exact_leak(splits, field="user")
    tokenizer = build_tokenizer(rows)
    train_rows = encode_split(tokenizer, splits["train"])
    val_rows = encode_split(tokenizer, splits["val"])
    longest = max(len(row["input_ids"]) for row in train_rows + val_rows)
    n_positions = max(128, longest + 8)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_inner=128,
        bos_token_id=tokenizer.eot_id,
        eos_token_id=tokenizer.eot_id,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        pad_token_id=tokenizer.pad_id,
    )
    base = GPT2LMHeadModel(config)
    # Transformers 5.22 在 loss_type 为空时会警告并回退到 ForCausalLMLoss。
    # 显式指定后,损失仍是带 ignore_index=-100 的因果语言建模损失。
    base.loss_type = "ForCausalLM"
    base.config.use_cache = False
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    model = get_peft_model(base, lora)
    trainable, total = count_params(model)
    expected = config.n_layer * lora.r * (config.n_embd + 3 * config.n_embd)
    if trainable != expected:
        raise RuntimeError(f"可训练参数 {trainable} 与公式 {expected} 不一致")

    device = torch.device("cpu")
    model.to(device)
    train_batches = batches(torch, train_rows, tokenizer.pad_id, batch_size)
    val_batches = batches(torch, val_rows, tokenizer.pad_id, batch_size)
    if not train_batches or not val_batches:
        raise RuntimeError("划分后没有批次")
    if not (train_batches[0]["labels"] == -100).any():
        raise RuntimeError("标签里没有提示屏蔽")

    frozen_before = frozen_snapshot(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    train_loss = None
    for step in range(steps):
        batch = train_batches[step % len(train_batches)]
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError("训练损失不是有限数")
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach())

    frozen_after = frozen_snapshot(model)
    if frozen_before != frozen_after:
        raise RuntimeError("冻结权重在优化步之后发生变化")

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            val_losses.append(float(model(**batch).loss.detach()))
    val_loss = sum(val_losses) / len(val_losses)

    merged_ok = None
    if self_check:
        probe = train_batches[0]["input_ids"][:1].to(device)
        probe_mask = train_batches[0]["attention_mask"][:1].to(device)
        with torch.no_grad():
            before = model(input_ids=probe, attention_mask=probe_mask).logits
            merged = model.merge_and_unload()
            after = merged(input_ids=probe, attention_mask=probe_mask).logits
        merged_ok = bool(torch.allclose(before, after, atol=1e-4, rtol=1e-4))
        if not merged_ok:
            raise RuntimeError("merge_and_unload 后的 logits 与未合并适配器不一致")

    report = {
        "steps": steps,
        "train_loss_last": train_loss,
        "val_loss": val_loss,
        "trainable": trainable,
        "total": total,
        "vocab_size": tokenizer.vocab_size,
        "longest_tokens": longest,
        "merged_matches": merged_ok,
        "note": "教学运行的损失,不是质量基准,也不是生产超参数。",
    }
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="微型 GPT-2 上的离线 LoRA 教学脚本")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    print("说明:这是虚构数据上的教学配置,损失数值不能当作基准或模型质量结论。")
    run(args.steps, args.batch_size, args.lr, args.seed, args.self_check)


if __name__ == "__main__":
    main()
