"""风险矩阵：认知面 × 行为面 -> 处置优先级。

规格（CLAUDE.md 第 7 节）：
    认知面 \\ 行为面   低          中          高
    假                告知误区    尽快劝阻    尽快劝阻
    部分属实/无法核查  无需处理        提醒留意    提醒留意
    真                无需处理        无需处理        提醒留意

认知面以文章中最严重的一条主张判定为准。
纯逻辑，可单测。
"""
from __future__ import annotations

from typing import Any, Iterable

from .models import SEVERITY, CognitiveLabel, RiskDecision

# 兜底矩阵：config.yaml 缺失时使用，保证系统不会因为配置缺项而崩
DEFAULT_MATRIX: dict[str, dict[str, str]] = {
    "假": {"低": "告知误区", "中": "尽快劝阻", "高": "尽快劝阻"},
    "部分属实": {"低": "无需处理", "中": "提醒留意", "高": "提醒留意"},
    "无法核查": {"低": "无需处理", "中": "提醒留意", "高": "提醒留意"},
    "真": {"低": "无需处理", "中": "无需处理", "高": "提醒留意"},
}


def worst_label(labels: Iterable[str]) -> CognitiveLabel:
    """取最严重的认知面判定。空列表视为无法核查（保守）。"""
    labels = list(labels)
    if not labels:
        return "无法核查"
    return max(labels, key=lambda x: SEVERITY.get(x, 0))  # type: ignore[return-value]


def decide(cognitive_label: str, behavior_level: str, matrix_cfg: dict[str, Any] | None = None) -> RiskDecision:
    """查表得处置优先级。配置缺项时回退到 DEFAULT_MATRIX。"""
    matrix = (matrix_cfg or {}).get("matrix") or DEFAULT_MATRIX
    row = matrix.get(cognitive_label) or DEFAULT_MATRIX.get(cognitive_label, {})
    priority = row.get(behavior_level) or DEFAULT_MATRIX.get(cognitive_label, {}).get(behavior_level, "提醒留意")
    reason = f"认知面={cognitive_label}，行为面={behavior_level} → {priority}"
    return RiskDecision(
        priority=priority,  # type: ignore[arg-type]
        cognitive_label=cognitive_label,  # type: ignore[arg-type]
        behavior_level=behavior_level,  # type: ignore[arg-type]
        reason=reason,
    )
