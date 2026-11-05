"""发布门和实验计划的结构检查。门失败就停止,不把未测的加速写进结论。"""

from __future__ import annotations

OPS = {
    ">=": lambda actual, threshold: actual >= threshold,
    "<=": lambda actual, threshold: actual <= threshold,
    ">": lambda actual, threshold: actual > threshold,
    "<": lambda actual, threshold: actual < threshold,
}

PLAN_FIELDS = (
    "hypothesis",
    "independent_variable",
    "controlled_factors",
    "metrics",
    "gates",
    "rollback",
)


def evaluate_gates(observed: dict[str, float], criteria: list[dict]) -> list[str]:
    failures = []
    for item in criteria:
        name = item["name"]
        op = item["op"]
        threshold = item["threshold"]
        if op not in OPS:
            failures.append(f"{name}: 不支持的比较 {op}")
            continue
        if name not in observed:
            failures.append(f"{name}: 缺少观测值")
            continue
        if not OPS[op](observed[name], threshold):
            failures.append(f"{name}: 观测 {observed[name]} 不满足 {op} {threshold}")
    return failures


def validate_experiment_plan(plan: dict) -> list[str]:
    """一次只允许一个自变量。对照因素、指标、门和回滚都必须写出来。"""
    errors = []
    for field in PLAN_FIELDS:
        value = plan.get(field)
        if value is None or value == "" or value == [] or value == {}:
            errors.append(f"缺少 {field}")
    variable = plan.get("independent_variable")
    if isinstance(variable, list):
        errors.append("independent_variable 必须是一个变量,不能是列表")
    changed = plan.get("changed_together", [])
    if isinstance(changed, list) and len(changed) > 1:
        errors.append("同一次实验不能同时改变多个因素")
    gates = plan.get("gates")
    if isinstance(gates, list):
        for item in gates:
            if not isinstance(item, dict) or not {"name", "op", "threshold"} <= set(item):
                errors.append("每条 gate 需要 name、op、threshold")
                break
    return errors
