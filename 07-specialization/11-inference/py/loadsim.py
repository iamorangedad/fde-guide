"""可重复的排队教学模拟。它不是 vLLM,也不是 GPU 压测。

三种策略都使用调用方给定的 prefill_time 和 decode_step:

- serial:一次只服务一个请求。预填产生第一个 token,之后每个剩余 token 花一个 decode_step。
- static:按到达顺序分组。组开始于"工人空闲且组内全部到达"。组内预填串行,
  然后按组内最大剩余 token 数解码。短请求要等到整组结束才返回。
- continuous:有空位且已到达就准入。新请求的预填独占工人,此期间其他活动请求等待。
  这是一个故意写明的悲观假设,用来说明"预填堵住解码"时连续组批不一定帮助队首。
  真实系统用分块预填避免这一点;本模拟器没有把分块预填实现成第三种测量。

输出文本就是请求上的 expected。模拟器不生成语言,因此精确匹配只检验
"完成的请求是否带回了预定字符串",用来把正确性门和延迟门分开。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    rid: str
    arrival: float
    prefill_time: float
    output_tokens: int
    expected: str

    def __post_init__(self) -> None:
        if self.prefill_time < 0 or self.output_tokens < 1 or self.arrival < 0:
            raise ValueError(f"{self.rid} 的时间或 token 数非法")


def _finish_row(req: Request, finish: float, ttft: float, output: str) -> dict:
    return {
        "rid": req.rid,
        "latency": finish - req.arrival,
        "ttft": ttft,
        "output": output,
        "exact": output == req.expected,
    }


def _summarize(rows: list[dict], reqs: list[Request], finish_time: float) -> dict:
    if not rows:
        raise ValueError("没有完成的请求")
    first = min(r.arrival for r in reqs)
    wall = finish_time - first
    tokens = sum(r.output_tokens for r in reqs)
    return {
        "rows": rows,
        "exact_match_rate": sum(1 for row in rows if row["exact"]) / len(rows),
        "throughput_output_tokens_per_s": tokens / wall if wall > 0 else float("inf"),
        "makespan": wall,
    }


def simulate_serial(reqs: list[Request], decode_step: float) -> dict:
    if decode_step < 0:
        raise ValueError("decode_step 不能为负")
    ordered = sorted(reqs, key=lambda r: (r.arrival, r.rid))
    now = 0.0
    rows = []
    for req in ordered:
        start = max(now, req.arrival)
        first_token_at = start + req.prefill_time
        finish = first_token_at + (req.output_tokens - 1) * decode_step
        rows.append(_finish_row(req, finish, first_token_at - req.arrival, req.expected))
        now = finish
    return _summarize(rows, ordered, now if ordered else 0.0)


def simulate_static(reqs: list[Request], decode_step: float, group_size: int) -> dict:
    if decode_step < 0 or group_size < 1:
        raise ValueError("参数非法")
    ordered = sorted(reqs, key=lambda r: (r.arrival, r.rid))
    now = 0.0
    rows = []
    index = 0
    while index < len(ordered):
        group = ordered[index : index + group_size]
        index += len(group)
        start = max(now, max(req.arrival for req in group))
        cursor = start
        first_at = {}
        for req in group:
            cursor += req.prefill_time
            first_at[req.rid] = cursor
        extra = max(req.output_tokens - 1 for req in group)
        finish = cursor + extra * decode_step
        for req in group:
            rows.append(_finish_row(req, finish, first_at[req.rid] - req.arrival, req.expected))
        now = finish
    return _summarize(rows, ordered, now if ordered else 0.0)


def simulate_continuous(reqs: list[Request], decode_step: float, max_batch: int) -> dict:
    if decode_step < 0 or max_batch < 1:
        raise ValueError("参数非法")
    pending = sorted(reqs, key=lambda r: (r.arrival, r.rid))
    active: list[dict] = []
    rows: list[dict] = []
    now = pending[0].arrival if pending else 0.0
    index = 0

    def admit() -> None:
        nonlocal index, now
        while index < len(pending) and len(active) < max_batch and pending[index].arrival <= now:
            req = pending[index]
            index += 1
            now += req.prefill_time
            left = req.output_tokens - 1
            if left == 0:
                rows.append(_finish_row(req, now, now - req.arrival, req.expected))
            else:
                active.append({"req": req, "left": left, "ttft": now - req.arrival})

    while index < len(pending) or active:
        if not active and index < len(pending):
            now = max(now, pending[index].arrival)
        admit()
        if not active:
            continue
        if decode_step == 0 and all(item["left"] > 0 for item in active):
            raise ValueError("decode_step 为 0 且仍有剩余 token,时间不会前进")
        now += decode_step
        still = []
        for item in active:
            item["left"] -= 1
            if item["left"] == 0:
                req = item["req"]
                rows.append(_finish_row(req, now, item["ttft"], req.expected))
            else:
                still.append(item)
        active = still
    finish = max((row["latency"] + next(r.arrival for r in pending if r.rid == row["rid"])) for row in rows)
    return _summarize(rows, pending, finish)
