"""「发给爸妈的话」生成（报告 4.5「代际提醒话术生成模式」）。

产品定位（D51）：子女在小程序里核查文章，系统给出真伪结论和一段可以直接发给爸妈的话。
输入核查结论、建议档位、权威证据与命中的风险话术，输出一段 ≤300 字的话。

模型只负责写正文；三条硬约束由代码兜底，不交给模型自觉：
1. 长度：RebuttalOut 的 schema 校验拦超长，llm.py 带着错误信息让模型重写一次
2. 来源：sources 里不在证据清单中的机构名一律剔除——辟谣稿绝不能编造出处
3. 结尾：统一追加「具体用药请遵医嘱。」（CLAUDE.md 开发原则第 7 条：系统不给医疗建议）

何时调用模型：结论为「真」且行为风险为「低」时内容可信、也没有推人行动的话术，
不调用模型，直接给固定的一句 SAFE_MESSAGE。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..models import REBUTTAL_ENDING, BehaviorHit, Evidence, Rebuttal, RebuttalOut
from ..prompts import load_prompt, render

PROMPT_NAME = "rebuttal"

# 「真 × 低」：不花模型调用，给一句固定的话，子女照样有东西可以发给爸妈
SAFE_MESSAGE = ("这篇我帮您查了一下，没发现不靠谱的说法，也没有劝人停药、买东西的内容，可以放心看。"
                + REBUTTAL_ENDING)


def needs_rebuttal(label: str, behavior_level: str) -> bool:
    return not (label == "真" and behavior_level == "低")


def evidence_for_rebuttal(label: str, evs: Sequence[Evidence], used: Sequence[int] = (),
                          limit: int = 3) -> list[Evidence]:
    """辟谣稿能引用哪些证据。

    「无法核查」= 判定时没有可用证据，一条都不给——否则模型会拿检索到的无关证据凑出处
    （v1 实测：无糖饮料一文的辟谣稿转述了讲喝水的证据）。
    其他结论只给判定时**实际用到**的证据（1 基编号）；判定没标编号时才退回检索前几条。
    """
    if label == "无法核查":
        return []
    picked = [evs[i - 1] for i in dict.fromkeys(used) if 1 <= i <= len(evs)]
    return (picked or list(evs))[:limit]


def format_evidence_for_rebuttal(evs: Sequence[Evidence], max_chars: int = 300) -> str:
    if not evs:
        return "（未检索到相关的权威证据）"
    return "\n".join(f"- {e.source_name or e.domain}：{(e.snippet or '')[:max_chars]}"
                     for e in evs)


def format_behaviors(hits: Sequence[BehaviorHit]) -> str:
    if not hits:
        return "（未发现）"
    return "\n".join(f"- {h.type}：「{h.evidence}」" for h in hits)


def finalize(out: RebuttalOut, evs: Sequence[Evidence]) -> Rebuttal:
    """代码兜底：剔除证据里没有的来源，统一加结尾。"""
    known = {e.source_name for e in evs if e.source_name} | {e.domain for e in evs if e.domain}
    sources = [s for s in dict.fromkeys(out.sources) if s in known]
    text = out.text.strip()
    if not text.endswith(REBUTTAL_ENDING):
        sep = "" if text.endswith(("。", "！", "？", "”")) else "。"
        text = f"{text}{sep}{REBUTTAL_ENDING}"
    return Rebuttal(text=text, sources=sources)


@dataclass
class RebuttalWriter:
    cfg: dict[str, Any]
    client: Any
    max_chars: int = 1500           # 喂给模型的正文节选长度：写提醒不需要全文

    @classmethod
    def build(cls, cfg: dict[str, Any], client: Any = None, store: Any = None,
              run_id: str = "adhoc") -> "RebuttalWriter":
        if client is None:
            from ..llm import build_client
            client = build_client(cfg, stage="report", store=store, run_id=run_id)
        return cls(cfg=cfg, client=client)

    def write(self, *, title: str, text: str, label: str, reasoning: str = "", priority: str = "",
              evidence: Sequence[Evidence] = (), used_evidence: Sequence[int] = (),
              behaviors: Sequence[BehaviorHit] = ()) -> Rebuttal:
        system, user_tpl, version = load_prompt(PROMPT_NAME, self.cfg)
        usable = evidence_for_rebuttal(label, evidence, used_evidence)
        prompt = render(
            user_tpl, title=title or "（无标题）", text=text[: self.max_chars],
            label=label, reasoning=reasoning or "（无）", priority=priority or "（未给出）",
            behaviors=format_behaviors(behaviors),
            evidence=format_evidence_for_rebuttal(usable),
        )
        data = self.client.complete(prompt, system, RebuttalOut,
                                    stage="rebuttal", prompt_version=version)
        # 来源核对只认「交给模型的那几条」：没给模型的证据，模型不可能据此说话
        return finalize(RebuttalOut.model_validate(data), usable)
