"""Tests for CPU worked examples."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worked_examples import (
    adam_static_bytes,
    attention_lora_count,
    causal_ce_nll_from_logits,
    clm_uniform_nll,
    dpo_margin_demo,
    grpo_advantages,
    kv_cache_bytes,
    lora_param_count,
    masked_vs_full_mean_nll,
    matmul_arithmetic_intensity_n,
    request_latency_ms,
    roofline_bound,
    toy_attention_row0,
)


def assert_close(a: float, b: float, tol: float = 1e-9) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} (tol={tol})")


def test_clm_uniform() -> None:
    assert_close(clm_uniform_nll(50257), math.log(50257))


def test_mask_mean() -> None:
    full, ans = masked_vs_full_mean_nll(30, 2, 0.1, 2.0)
    assert_close(full, 7.0 / 32.0)
    assert_close(ans, 2.0)


def test_lora() -> None:
    assert lora_param_count(1024, 2048, 16) == 49152
    assert attention_lora_count(32, 4096, 8) == 8_388_608


def test_adam_kv_latency() -> None:
    assert adam_static_bytes(7_000_000_000) == 112_000_000_000
    assert kv_cache_bytes(32, 8, 128, 4096) == 536_870_912
    assert kv_cache_bytes(32, 32, 128, 4096) == 4 * 536_870_912
    assert_close(request_latency_ms(200, 20, 50), 1180.0)


def test_grpo_dpo() -> None:
    adv = grpo_advantages([1.0, 0.0, 1.0, 0.0])
    assert adv == [1.0, -1.0, 1.0, -1.0]
    assert_close(dpo_margin_demo(0.1, 2.0, -1.0), 0.3)


def test_roofline_attn_ce() -> None:
    assert_close(matmul_arithmetic_intensity_n(1024), 1024 / 6)
    peak, bw = 10e12, 200e9
    assert_close(roofline_bound(1.0, peak, bw), bw * 1.0)
    assert_close(roofline_bound(100.0, peak, bw), peak)
    w, out = toy_attention_row0()
    assert len(w) == 2 and abs(sum(w) - 1.0) < 1e-12
    assert_close(causal_ce_nll_from_logits([0.0, 1.0], 1), -(1.0 - math.log(math.exp(0)+math.exp(1))))


def main() -> None:
    test_clm_uniform()
    test_mask_mean()
    test_lora()
    test_adam_kv_latency()
    test_grpo_dpo()
    test_roofline_attn_ce()
    print("ALL_PASSED")


if __name__ == "__main__":
    main()
