# 第 11 章 推理优化

> [🏠 返回主页](../../README.md) · [返回专项](../README.md)

本章是专项。服务一个模型的 FDE 需要能回答:第一个 token 要多久、每个后续 token 要多久、尾延迟是多少、KV cache 会占多少字节、以及这些数在什么负载下会互相牺牲。本章的数字来自定义和排队模型,不来自一次未做的 GPU 压测。

## 先修

第 8 章的算术强度,第 10 章的字节账。不要求先部署过 vLLM。

## 学习目标

- 区分预填和解码,并推导 KV cache 的字节
- 用 TTFT、TPOT、p50、p95 和吞吐写服务等级,而不是只用平均延迟
- 说明静态组批、连续组批、分页缓存、量化和投机解码各解决哪一种浪费
- 用一个可重复的模拟器练习"延迟、正确性、质量门",并知道它不是 vLLM

## 地图

1. [预填、解码与 KV cache](01-prefill-decode-kv.md)
2. [组批、量化与投机解码](02-batching-quant-speculative.md)
3. [容量、剖析与可重复负载试验](03-serving-profiling-loadtest.md)

## 代码

| 文件 | 作用 |
|---|---|
| [py/kv_cache.py](py/kv_cache.py) | KV 字节和末块浪费 |
| [py/latency.py](py/latency.py) | TTFT、TPOT、type-7 分位数、吞吐 |
| [py/speculative.py](py/speculative.py) | 期望 token 数和模型化比值 |
| [py/capacity.py](py/capacity.py) | 权重、剩余显存、单价作为输入的费用 |
| [py/loadsim.py](py/loadsim.py) | 串行、静态组批、阻塞式连续组批的排队模型 |

```bash
python3 07-specialization/11-inference/tests/test_inference_math.py
```

## 小结

预填可以吃算力,解码常常吃带宽。缓存的字节按层数、KV 头数和序列长度相乘。组批提高的是设备吞吐,尾延迟要单独设门。任何加速比都必须附带硬件、软件、负载和正确性结果;本章不提供这样的实测。

## 关键术语

预填、解码、KV cache、TTFT、TPOT、连续组批、分页注意力、投机解码

## 延伸阅读

- GQA — https://arxiv.org/abs/2305.13245
- PagedAttention — https://arxiv.org/abs/2309.06180
- Orca 论文 PDF — https://www.usenix.org/system/files/osdi22-yu.pdf

---

[🏠 返回主页](../../README.md)
