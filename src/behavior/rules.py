"""行为面纯逻辑：由命中类型推导风险等级。

不调用模型、不依赖网络，因此可以完全单测。
规则来自 config.yaml 的 behavior.risk_rules，按顺序匹配，先命中先返回。
"""
from __future__ import annotations

from typing import Any, Sequence

from ..models import BehaviorHit, BehaviorResult


def effective_hits(hits: Sequence[BehaviorHit], confidence_threshold: float = 0.5) -> list[BehaviorHit]:
    """低于置信度阈值的命中不计入——模型偶发的弱命中不该抬高风险等级。"""
    return [h for h in hits if h.confidence >= confidence_threshold]


def risk_level_from_hits(hits: Sequence[str], rules: list[dict[str, Any]]) -> str:
    """按规则表推导风险等级。

    规则形态：
      - {level: 高, any_of: [医疗行为建议, 金钱相关]}   命中其中任一即匹配
      - {level: 中, min_hits: 2}                        命中类型数 >= 2 即匹配
    """
    hit_set = set(hits)
    n = len(hit_set)
    for rule in rules:
        if "any_of" in rule and hit_set & set(rule["any_of"]):
            return rule["level"]
        if "min_hits" in rule and n >= int(rule["min_hits"]):
            return rule["level"]
    return "低"


def screen_result(hits: Sequence[BehaviorHit], behavior_cfg: dict[str, Any]) -> BehaviorResult:
    """把模型给出的原始命中，套用规则得到最终 BehaviorResult。"""
    kept = effective_hits(hits, float(behavior_cfg.get("confidence_threshold", 0.5)))
    rules = behavior_cfg.get("risk_rules") or []
    types = sorted({h.type for h in kept})
    level = risk_level_from_hits(types, rules)
    return BehaviorResult(risk_level=level, behaviors=list(kept))  # type: ignore[arg-type]


def behavior_level_from_types(types: Sequence[str], behavior_cfg: dict[str, Any]) -> str:
    """评测时用：只有标注的行为类型，没有置信度信息。"""
    rules = behavior_cfg.get("risk_rules") or []
    return risk_level_from_hits(list(types), rules)
