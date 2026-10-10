"""M3 行为面滤网：识别文章的传播行为特征。

与认知面**完全独立**——一篇完全属实的文章照样可以含"转给家里人"或"点击链接购买"。
这条独立性是双滤网设计的前提：认知面回答"内容对不对"，行为面回答"它在怎么推着读者走"。

职责划分（重要）：
- **模型**只报"在原文里看到了哪几类行为"，附原文片段与置信度
- **规则**（`src/behavior/rules.py` + config 的 `behavior.risk_rules`）负责把命中算成风险等级

刻意不让模型输出 risk_level：否则就有两个真值来源，不一致时无从裁决；
而且等级规则要能在不重新调用模型的前提下调整（换阈值、做消融）。

成本记账由 `LLMClient` 统一负责（写 runs 表）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import doc_max_chars
from ..models import BehaviorResult, BehaviorScreenOut, Document
from ..prompts import load_prompt, render
from .rules import screen_result

log = logging.getLogger("screen.behavior")

PROMPT_NAME = "behavior_screen"


@dataclass
class BehaviorScreen:
    """行为面滤网。client 由调用方注入，便于测试时换假模型。"""

    cfg: dict[str, Any]
    client: Any
    max_chars: int = 6000

    @classmethod
    def build(cls, cfg: dict[str, Any], client: Any = None, store: Any = None,
              run_id: str = "adhoc") -> "BehaviorScreen":
        if client is None:
            from ..llm import build_client
            client = build_client(cfg, stage="behavior_screen", store=store, run_id=run_id)
        return cls(cfg=cfg, client=client, max_chars=doc_max_chars(cfg, "behavior"))

    @property
    def prompt_version(self) -> str:
        try:
            return load_prompt(PROMPT_NAME, self.cfg)[2]
        except Exception:
            return "unknown"

    def screen_hits(self, text: str, title: str = "") -> BehaviorScreenOut:
        """只调模型，不套规则——原始命中，供调试与消融观察用。"""
        system, user_tpl, version = load_prompt(PROMPT_NAME, self.cfg)
        prompt = render(user_tpl, title=title or "（无标题）",
                        text=text[: self.max_chars], max_chars=self.max_chars)
        data = self.client.complete(prompt, system, BehaviorScreenOut,
                                    stage="behavior_screen", prompt_version=version)
        return BehaviorScreenOut.model_validate(data)

    def screen(self, text: str, title: str = "") -> BehaviorResult:
        """调模型 + 套规则，产出最终行为面结论。"""
        raw = self.screen_hits(text, title)
        return screen_result(raw.behaviors, self.cfg.get("behavior", {}) or {})

    def screen_document(self, doc: Document) -> BehaviorResult:
        return self.screen(doc.text, doc.title)


def screen(text: str, title: str = "", cfg: dict[str, Any] | None = None,
           client: Any = None) -> BehaviorResult:
    """便捷入口：临时构造一个 screen 跑一次。"""
    if cfg is None:
        from ..config import load_config
        cfg = load_config()
    return BehaviorScreen.build(cfg, client=client).screen(text, title)
