"""证据不可用时的兜底（D41；参考 CRAG 的「检索结果不可用即改道」、Mallen 等的自适应检索）。

D38 发现：知识库没覆盖的样本，检索回来的证据答非所问，证据判断如实给出「无法核查」——
「真→无法核查」从 3 条涨到 30 条（27 条不在库里），整体准确率反而低于裸模型。

做法：包住任何一个基于证据的判断器（+retrieve / +source / +debate 都适用）。
证据判断给出「无法核查」或指不出用了哪条证据 = 证据不可用（CRAG 的 Incorrect），
此时改用裸模型（baseline）的判断，并且：
  - 理由前缀写明「无权威证据，依据模型自身知识」——不把模型记忆冒充成有据可查的结论
  - 置信度封顶（默认 0.6）——没有证据支撑的结论不该显得很有把握
  - 证据链清空——这些证据没有支撑结论，不能出现在证据链里误导读者

不需要额外的相关性判断调用：「无法核查」本身就是证据判断给出的不可用信号；
也不需要在 dev 上调阈值，所以不存在拿评测集调参的问题。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Evidence, EvidenceVerdict

FALLBACK_PREFIX = "【无权威证据，依据模型自身知识】"


def evidence_unusable(v: EvidenceVerdict) -> bool:
    return v.label == "无法核查" or not v.used_evidence


@dataclass
class FallbackJudge:
    inner: Any                       # EvidenceJudge / SourceScoredJudge / DebateJudge
    baseline: Any                    # BaselineJudge
    confidence_cap: float = 0.6
    last_fallback: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: dict[str, Any], inner: Any, store: Any = None,
              run_id: str = "adhoc") -> "FallbackJudge":
        from .baseline import BaselineJudge
        fcfg = (cfg.get("cognitive", {}).get("fallback", {}) or {})
        return cls(inner=inner, baseline=BaselineJudge.build(cfg, store=store, run_id=run_id),
                   confidence_cap=float(fcfg.get("confidence_cap", 0.6)))

    # 评测与主流程从判断器上读这些属性，原样转给内层
    @property
    def client(self) -> Any:
        return self.inner.client

    @property
    def clients(self) -> list[Any]:
        inner = list(getattr(self.inner, "clients", None) or [self.inner.client])
        return inner + [self.baseline.client]

    @property
    def mode(self) -> str:
        return getattr(self.inner, "mode", "")

    @property
    def last_trace(self) -> dict[str, Any]:
        t = dict(getattr(self.inner, "last_trace", {}) or {})
        t["fallback_used"] = bool(self.last_fallback.get("used"))
        return t

    @property
    def last_debate_trace(self) -> dict[str, Any]:
        t = dict(getattr(self.inner, "last_debate_trace", {}) or {})
        if self.last_fallback:
            t["fallback"] = dict(self.last_fallback)
        return t

    def judge(self, text: str, title: str = "") -> tuple[EvidenceVerdict, list[Evidence]]:
        verdict, evs = self.inner.judge(text, title)
        if not evidence_unusable(verdict):
            self.last_fallback = {"used": False}
            return verdict, evs
        b = self.baseline.judge(text, title)
        self.last_fallback = {
            "used": True, "evidence_verdict": verdict.model_dump(mode="json"),
            "baseline_label": b.label, "baseline_confidence": b.confidence,
        }
        return EvidenceVerdict(
            label=b.label, confidence=min(float(b.confidence), self.confidence_cap),
            reasoning=(FALLBACK_PREFIX + (b.reasoning or ""))[:200], used_evidence=[],
        ), []
