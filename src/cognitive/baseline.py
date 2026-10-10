"""M2 baseline：单个大模型直接判断文章整体事实性。

这是消融实验的**参照基准**（CLAUDE.md 消融表里的 `baseline` 行）。
刻意保持最小：不检索、不评分、不辩论、不看行为面。后续每个模块的增量价值，
都要拿它来对照——所以它一旦"加料"，整张消融表就失去意义。

成本记账由 `LLMClient` 统一负责（写 runs 表），本模块只负责构造提示词与解析结果。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..models import BaselineVerdict, Document
from ..prompts import load_prompt, render

log = logging.getLogger("screen.baseline")

PROMPT_NAME = "baseline_judge"


def resolve_max_chars(cfg: dict[str, Any]) -> int:
    """正文截断上限。

    走全链路统一规则（`config.doc_max_chars`）：`cognitive.baseline.max_chars` 为 0/缺省时
    继承 `ingest.max_chars`——刻意共用同一个上限：baseline 与链路吃同样多的正文，
    消融出来的差值才是模块本身的贡献，而不是"谁被喂得少"。
    """
    from ..config import doc_max_chars
    return doc_max_chars(cfg, "cognitive.baseline")


@dataclass
class BaselineJudge:
    """一次调用出结果。client 由调用方注入，便于测试时换假模型。"""

    cfg: dict[str, Any]
    client: Any
    max_chars: int = 6000

    @classmethod
    def build(cls, cfg: dict[str, Any], client: Any = None, store: Any = None,
              run_id: str = "adhoc") -> "BaselineJudge":
        if client is None:
            from ..llm import build_client
            client = build_client(cfg, stage="baseline", store=store, run_id=run_id)
        return cls(cfg=cfg, client=client, max_chars=resolve_max_chars(cfg))

    @property
    def prompt_version(self) -> str:
        try:
            return load_prompt(PROMPT_NAME, self.cfg)[2]
        except Exception:
            return "unknown"

    def judge(self, text: str, title: str = "") -> BaselineVerdict:
        system, user_tpl, version = load_prompt(PROMPT_NAME, self.cfg)
        prompt = render(user_tpl, title=title or "（无标题）",
                        text=text[: self.max_chars], max_chars=self.max_chars)
        data = self.client.complete(prompt, system, BaselineVerdict,
                                    stage="baseline", prompt_version=version)
        return BaselineVerdict.model_validate(data)

    def judge_document(self, doc: Document) -> BaselineVerdict:
        return self.judge(doc.text, doc.title)


def judge(text: str, title: str = "", cfg: dict[str, Any] | None = None,
          client: Any = None) -> BaselineVerdict:
    """便捷入口：临时构造一个 judge 跑一次。"""
    if cfg is None:
        from ..config import load_config
        cfg = load_config()
    return BaselineJudge.build(cfg, client=client).judge(text, title)
