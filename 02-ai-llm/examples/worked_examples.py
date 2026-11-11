"""CPU-only worked examples for Chapter 3.

No model downloads, no CUDA. Run:
  python3 02-ai-llm/examples/worked_examples.py
  python3 02-ai-llm/examples/test_worked_examples.py
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def clm_uniform_nll(vocab_size: int) -> float:
    """Expected NLL (nats) for uniform predictive distribution."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return math.log(vocab_size)


def masked_vs_full_mean_nll(
    prompt_n: int,
    answer_n: int,
    prompt_nll: float,
    answer_nll: float,
) -> Tuple[float, float]:
    """Return (mean_if_all_tokens, mean_answer_only)."""
    if prompt_n < 0 or answer_n <= 0:
        raise ValueError("need non-negative prompt_n and positive answer_n")
    total = prompt_n * prompt_nll + answer_n * answer_nll
    full = total / (prompt_n + answer_n)
    ans = (answer_n * answer_nll) / answer_n
    return full, ans


def lora_param_count(d_in: int, d_out: int, r: int) -> int:
    return r * (d_in + d_out)


def attention_lora_count(n_layer: int, d: int, r: int, projections: int = 4) -> int:
    """Count LoRA params for `projections` square d×d matrices per layer."""
    return n_layer * projections * lora_param_count(d, d, r)


def adam_static_bytes(num_params: int, bytes_per_param: int = 16) -> int:
    return num_params * bytes_per_param


def kv_cache_bytes(
    n_layer: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    elem_bytes: int = 2,
) -> int:
    per_token = n_layer * 2 * n_kv_heads * head_dim * elem_bytes
    return per_token * seq_len * batch


def request_latency_ms(ttft_ms: float, tpot_ms: float, n_tokens: int) -> float:
    if n_tokens < 1:
        raise ValueError("n_tokens must be >= 1")
    return ttft_ms + (n_tokens - 1) * tpot_ms


def grpo_advantages(rewards: Sequence[float], eps: float = 0.0) -> List[float]:
    if not rewards:
        raise ValueError("empty rewards")
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std <= eps:
        raise ValueError("reward std too small; skip group in real training")
    return [(r - mean) / std for r in rewards]


def dpo_margin_demo(beta: float, log_ratio_win: float, log_ratio_lose: float) -> float:
    """Toy implicit reward gap beta * (log_ratio_win - log_ratio_lose). Not full DPO loss."""
    return beta * (log_ratio_win - log_ratio_lose)


def roofline_bound(intensity: float, peak_flops: float, peak_bw_bytes: float) -> float:
    """Return min(pi, beta * I) upper bound in FLOP/s for given intensity."""
    return min(peak_flops, peak_bw_bytes * intensity)


def matmul_arithmetic_intensity_n(n: int, elem_bytes: int = 4) -> float:
    """Rough I = 2n^3 / (3 n^2 elem_bytes) assuming A,B,C each streamed once."""
    flops = 2 * (n ** 3)
    bytes_moved = 3 * (n ** 2) * elem_bytes
    return flops / bytes_moved


def toy_attention_row0() -> Tuple[List[float], List[float]]:
    """Return (softmax weights row0, output row0) for the 2x2 textbook example."""
    scale = 1.0 / math.sqrt(2.0)
    # QK^T = I, scaled = scale * I
    logits = [scale, 0.0]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    weights = [e / z for e in exps]
    # V = [[1,2],[3,4]]
    out0 = [
        weights[0] * 1 + weights[1] * 3,
        weights[0] * 2 + weights[1] * 4,
    ]
    return weights, out0


def causal_ce_nll_from_logits(logit_row: Sequence[float], label: int) -> float:
    m = max(logit_row)
    exps = [math.exp(x - m) for x in logit_row]
    z = sum(exps)
    return -(logit_row[label] - m - math.log(z))


def main() -> None:
    print("clm_uniform_nll(50257) =", clm_uniform_nll(50257))
    print("masked_vs_full =", masked_vs_full_mean_nll(30, 2, 0.1, 2.0))
    print("lora =", lora_param_count(1024, 2048, 16), "full=", 1024 * 2048)
    print("attn_lora =", attention_lora_count(32, 4096, 8))
    print("adam_7b_gib =", adam_static_bytes(7_000_000_000) / 2**30)
    print("kv =", kv_cache_bytes(32, 8, 128, 4096))
    print("latency =", request_latency_ms(200, 20, 50))
    print("grpo =", grpo_advantages([1.0, 0.0, 1.0, 0.0]))
    print("dpo_margin =", dpo_margin_demo(0.1, 2.0, -1.0))
    print("I_matmul_1024 =", matmul_arithmetic_intensity_n(1024))
    peak = 10e12
    bw = 200e9
    print("bound_I1 =", roofline_bound(1.0, peak, bw))
    print("bound_I100 =", roofline_bound(100.0, peak, bw))
    print("toy_attn =", toy_attention_row0())
    print("ce_nll =", causal_ce_nll_from_logits([0.0, 1.0], 1))


if __name__ == "__main__":
    main()
