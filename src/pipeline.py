"""一键核查主流程：输入一篇文章，输出完整核查报告（CLAUDE.md「输入与输出」）。

    文章 ─┬─ 行为面滤网（话术识别 + 规则算风险等级）──────────────┐
          └─ 认知面滤网（检索 → 信源评分 → 快判 → 必要时辩论）──┴→ 风险矩阵 → 辟谣稿

两道滤网互不依赖，用两个线程并行跑（演示时等待时间约等于较慢的那一路）。
异常处理遵循报告 2.5：单点失败不中断整体流程，失败原因写进 report.notes，不编造结论。

评测走 eval/eval.py（逐条打分、做消融）；本模块面向演示与实际使用，固定用 `full` 配置。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .config import Config, load_config
from .models import (
    BehaviorResult, CheckReport, CognitiveVerdict, Document, Evidence, EvidenceVerdict,
)
from .risk_matrix import decide


def _full_overrides(cfg: Config) -> dict[str, Any]:
    full = (cfg.get_path("eval.configs", {}) or {}).get("full") or {}
    return dict(full) or {"retrieve": True, "source_score": True, "debate": True,
                          "review": True, "behavior": True}


class Checker:
    """组装好的核查器。各组件可注入，便于离线测试；默认按 config 构造真实组件。"""

    def __init__(self, cfg: Config | None = None, *, behavior: Any = None, judge: Any = None,
                 writer: Any = None):
        self.cfg = cfg or load_config()
        self.cfg["_overrides"] = _full_overrides(self.cfg)
        if behavior is None:
            from .behavior.screen import BehaviorScreen
            behavior = BehaviorScreen.build(self.cfg)
        if judge is None:
            from .cognitive.debate import DebateJudge
            judge = DebateJudge.build(self.cfg)
            if self.cfg["_overrides"].get("fallback"):
                from .cognitive.fallback import FallbackJudge
                judge = FallbackJudge.build(self.cfg, judge)
        if writer is None:
            from .report.rebuttal import RebuttalWriter
            writer = RebuttalWriter.build(self.cfg)
        self.behavior, self.judge, self.writer = behavior, judge, writer

    # ---------- 两道滤网 ----------
    def _behavior(self, doc: Document) -> tuple[BehaviorResult, str]:
        from .llm import LLMError
        try:
            return self.behavior.screen(doc.text, doc.title), ""
        except LLMError as exc:
            # 行为面失败不影响认知面；按「未命中」计，但必须留痕，不能装作真的没命中
            return BehaviorResult(risk_level="低", behaviors=[]), f"话术识别失败，按未命中处理：{exc}"

    def _cognitive(self, doc: Document) -> tuple[EvidenceVerdict, list[Evidence], dict, str]:
        from .llm import LLMError
        try:
            verdict, evs = self.judge.judge(doc.text, doc.title)
            return verdict, evs, dict(getattr(self.judge, "last_debate_trace", {}) or {}), ""
        except LLMError as exc:
            v = EvidenceVerdict(label="无法核查", confidence=0.0, reasoning="认知面调用失败")
            return v, [], {}, f"真伪判定失败，记为无法核查：{exc}"

    # ---------- 入口 ----------
    def check(self, doc: Document, progress: Callable[[str], None] | None = None) -> CheckReport:
        from .llm import LLMError

        say = progress or (lambda _msg: None)
        t0 = time.perf_counter()
        notes: list[str] = []

        say("正在识别话术、检索权威证据……")
        with ThreadPoolExecutor(max_workers=2) as pool:
            fb = pool.submit(self._behavior, doc)
            fc = pool.submit(self._cognitive, doc)
            bres, bnote = fb.result()
            verdict, evs, trace, cnote = fc.result()
        notes += [n for n in (bnote, cnote) if n]

        say("正在综合研判……")
        decision = decide(verdict.label, bres.risk_level, self.cfg.get("risk_matrix", {}))
        cv = CognitiveVerdict(
            claim_id=0, claim_text=doc.title or doc.text[:60], label=verdict.label,
            confidence=verdict.confidence, reasoning=verdict.reasoning, evidence=evs,
            used_evidence=list(verdict.used_evidence),
        )

        rebuttal_text, rebuttal_sources = "", []
        from .report.rebuttal import SAFE_MESSAGE, needs_rebuttal
        if not needs_rebuttal(verdict.label, bres.risk_level):
            rebuttal_text = SAFE_MESSAGE
        else:
            say("正在生成发给爸妈的话……")
            try:
                r = self.writer.write(title=doc.title, text=doc.text, label=verdict.label,
                                      reasoning=verdict.reasoning, priority=decision.priority,
                                      evidence=evs,
                                      used_evidence=verdict.used_evidence,
                                      behaviors=bres.behaviors)
                rebuttal_text, rebuttal_sources = r.text, r.sources
            except LLMError as exc:
                notes.append(f"「发给爸妈的话」生成失败：{exc}")

        return CheckReport(
            doc_id=doc.id, title=doc.title, behavior=bres, verdicts=[cv], decision=decision,
            rebuttal=rebuttal_text, rebuttal_sources=rebuttal_sources, config_name="full",
            prompt_versions=self._prompt_versions(), debate_trace=trace, notes=notes,
            latency_s=round(time.perf_counter() - t0, 2),
        )

    def _prompt_versions(self) -> dict[str, str]:
        from .cognitive.debate import PROMPTS
        from .prompts import load_prompt
        out = {}
        for n in (*PROMPTS, "behavior_screen", "rebuttal"):
            try:
                out[n] = load_prompt(n, self.cfg)[2]
            except Exception:                                   # noqa: BLE001
                out[n] = "unknown"
        return out


def check_text(text: str, title: str = "", cfg: Config | None = None, **kw: Any) -> CheckReport:
    """便捷入口：直接核查一段正文。"""
    import hashlib
    doc = Document(id="paste-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10],
                   source_type="paste", title=title, text=text)
    return Checker(cfg).check(doc, **kw)
