"""虚构的"澄海仓储"制度问答。没有真实客户、个人或机密。

数据只给离线教学脚本提供监督信号。它不证明模型学会了仓储制度,
也不建议用微调去背诵会变更的制度文本。会变更且必须引用原文的知识
应走检索;微调更适合稳定的回答格式。见第 3.3 节和第 3.6 节。
"""

from __future__ import annotations

import json
from pathlib import Path

SYSTEM = "你是澄海仓储的制度助手。回答必须写明制度编号。制度没有规定的内容,直接说不知道。"

ITEMS: list[dict[str, str]] = [
    {"id": "SOP-01", "user": "托盘标签上必须有哪些字段?", "assistant": "根据澄海仓储制度 SOP-01,托盘标签必须同时包含 SKU、批次号、数量和入库日期。"},
    {"id": "SOP-02", "user": "没有批次号的货物可以收吗?", "assistant": "根据澄海仓储制度 SOP-02,没有批次号的货物必须拒收,不能先入库再补单。"},
    {"id": "SOP-03", "user": "冷藏区温度范围是多少?", "assistant": "根据澄海仓储制度 SOP-03,冷藏区温度保持在 2 到 8 摄氏度。"},
    {"id": "SOP-04", "user": "冷冻区的温度上限是多少?", "assistant": "根据澄海仓储制度 SOP-04,冷冻区温度不得高于零下 18 摄氏度。"},
    {"id": "SOP-05", "user": "常温货要避开什么?", "assistant": "根据澄海仓储制度 SOP-05,常温货必须避光,并且不能紧贴外墙堆放。"},
    {"id": "SOP-06", "user": "危险品可以和食品放在同一区吗?", "assistant": "根据澄海仓储制度 SOP-06,危险品与食品不得放在同一区域。"},
    {"id": "SOP-07", "user": "发现外箱破损时第一步做什么?", "assistant": "根据澄海仓储制度 SOP-07,发现外箱破损要先拍照,再冻结对应库位,等待主管确认。"},
    {"id": "SOP-08", "user": "高值 SKU 拣选后谁来复核?", "assistant": "根据澄海仓储制度 SOP-08,高值 SKU 拣选后必须由第二个人复核。"},
    {"id": "SOP-09", "user": "交接班要核对什么单据?", "assistant": "根据澄海仓储制度 SOP-09,交接班必须核对未完成的拣选单。"},
    {"id": "SOP-10", "user": "仓内叉车限速是多少?", "assistant": "根据澄海仓储制度 SOP-10,仓内叉车限速为每小时 5 公里。"},
    {"id": "SOP-11", "user": "退货可以在暂存区放多久?", "assistant": "根据澄海仓储制度 SOP-11,退货在暂存区不得超过 48 小时。"},
    {"id": "SOP-12", "user": "过期品放到什么颜色的库位?", "assistant": "根据澄海仓储制度 SOP-12,过期品必须隔离到红色库位。"},
    {"id": "SOP-13", "user": "盘点差了几件要升级主管?", "assistant": "根据澄海仓储制度 SOP-13,同一 SKU 盘点差异达到或超过 1 件就要升级主管。"},
    {"id": "SOP-14", "user": "夜间多久巡检一次冷库温度?", "assistant": "根据澄海仓储制度 SOP-14,夜间值守每两小时巡检一次冷库温度并记录。"},
    {"id": "SOP-15", "user": "外来司机可以进拣选区吗?", "assistant": "根据澄海仓储制度 SOP-15,外来司机不得进入拣选区。"},
    {"id": "SOP-16", "user": "标签污损了可以手写更正吗?", "assistant": "根据澄海仓储制度 SOP-16,标签污损必须重打,禁止手写覆盖。"},
    {"id": "SOP-17", "user": "出库复核要扫描哪些信息?", "assistant": "根据澄海仓储制度 SOP-17,出库复核必须扫描 SKU 和批次号。"},
    {"id": "SOP-18", "user": "紧急停机按钮装在哪里?", "assistant": "根据澄海仓储制度 SOP-18,紧急停机按钮装在每条巷道入口。"},
    {"id": "SOP-19", "user": "未知液体泄漏时能先用水冲吗?", "assistant": "根据澄海仓储制度 SOP-19,未知液体泄漏要先疏散再上报,不要用水冲洗。"},
    {"id": "SOP-20", "user": "安全制度和效率制度冲突时听哪个?", "assistant": "根据澄海仓储制度 SOP-20,制度冲突时安全类优先于效率类。"},
]

# 故意写出的近重复,只用于泄漏检测练习,不进入清洁划分。
LEAKY_COPY = {
    "id": "SOP-01-COPY",
    "user": "托盘标签上必须有哪些字段?",
    "assistant": "根据澄海仓储制度 SOP-01,托盘标签必须同时包含 SKU、批次号、数量和入库日期。",
}


def records() -> list[dict[str, str]]:
    return [{**item, "system": SYSTEM} for item in ITEMS]


def write_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in records()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "data" / "sop.jsonl"
    write_jsonl(destination)
    print(destination)
