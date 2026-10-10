"""M4/M5 信源评分。

`source_score` 不调用模型：它是可解释的规则层。证据的 `tier` 来自权威来源配置，
再用域名 / 来源名关键词做兜底修正，最后把可信度写回 `Evidence.score`。

为什么不让 LLM 打分：信源可信度是比赛报告里要能解释、能人工调整的规则；
让模型临场打分既贵又不可复现，还会把"证据内容是否相关"和"来源是否可靠"混在一起。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import doc_max_chars
from ..models import Evidence, EvidenceVerdict
from ..prompts import load_prompt, render
from .evidence_judge import EvidenceJudge


def _norm_domain(s: str) -> str:
    return (s or "").lower().strip().removeprefix("www.")


@dataclass
class SourceScorer:
    """规则化信源评分器。"""

    cfg: dict[str, Any]

    @property
    def scfg(self) -> dict[str, Any]:
        return (self.cfg.get("cognitive", {}).get("source_score", {}) or {})

    def score(self, ev: Evidence) -> float:
        """计算单条证据来源可信度，范围 0~1。"""
        tier_scores = {int(k): float(v) for k, v in (self.scfg.get("tier_scores") or {}).items()}
        score = tier_scores.get(int(ev.tier or 4), float(self.scfg.get("default_score", 0.35)))

        hay = " ".join([
            _norm_domain(ev.domain), _norm_domain(ev.url), ev.source_name or "", ev.title or "",
        ])
        for rule in self.scfg.get("keyword_rules") or []:
            keys = [str(x).lower() for x in (rule.get("match") or [])]
            if keys and any(k in hay for k in keys):
                score = max(score, float(rule.get("score", score)))

        # authority_domains 里的 tier 是最稳定的配置源：如果域名显式登记，按登记 tier 修正。
        rcfg = (self.cfg.get("cognitive", {}).get("retrieve", {}) or {})
        domain = _norm_domain(ev.domain or ev.url)
        for src in rcfg.get("authority_domains") or []:
            d = _norm_domain(src.get("domain", ""))
            if d and d in domain:
                score = max(score, tier_scores.get(int(src.get("tier") or 4), score))
        return max(0.0, min(1.0, round(score, 4)))

    def apply(self, evs: list[Evidence]) -> list[Evidence]:
        for ev in evs:
            ev.score = self.score(ev)
        return evs


@dataclass
class SourceScoredJudge:
    """检索 + 信源评分 + 证据判断。

    `+source` 要和 `+retrieve` 共用同一套检索与判定逻辑，只多一层可解释的信源分。
    因此这里复用 EvidenceJudge.retrieve()，但在送入 evidence_judge 前把 score 写进证据文本。
    """

    cfg: dict[str, Any]
    inner: EvidenceJudge
    scorer: SourceScorer
    max_chars: int = 6000
    last_trace: dict[str, Any] | None = None

    @classmethod
    def build(cls, cfg: dict[str, Any], store: Any = None, run_id: str = "adhoc",
              mode: str | None = None) -> "SourceScoredJudge":
        inner = EvidenceJudge.build(cfg, store=store, run_id=run_id, mode=mode)
        return cls(cfg=cfg, inner=inner, scorer=SourceScorer(cfg),
                   max_chars=doc_max_chars(cfg, "cognitive.baseline"))

    @property
    def client(self) -> Any:
        return self.inner.client

    @property
    def clients(self) -> list[Any]:
        return [self.inner.client]

    @property
    def mode(self) -> str:
        return self.inner.mode

    def judge(self, text: str, title: str = "") -> tuple[EvidenceVerdict, list[Evidence]]:
        evs = self.scorer.apply(self.inner.retrieve(title or text[:60]))
        self.last_trace = dict(self.inner.last_trace or {})
        if evs:
            self.last_trace["avg_source_score"] = round(sum(e.score for e in evs) / len(evs), 4)
            self.last_trace["min_source_score"] = min(e.score for e in evs)
        system, user_tpl, version = load_prompt("evidence_judge", self.cfg)
        prompt = render(
            user_tpl,
            title=title or "（无标题）",
            text=text[: self.max_chars],
            evidence=format_scored_evidence(evs),
            n_evidence=len(evs),
            max_chars=self.max_chars,
        )
        data = self.client.complete(prompt, system, EvidenceVerdict,
                                    stage="evidence_judge", prompt_version=f"{version}+source")
        return EvidenceVerdict.model_validate(data), evs


def format_scored_evidence(evs: list[Evidence], max_chars: int = 1500) -> str:
    """渲染带信源分的证据清单。"""
    if not evs:
        return "（未检索到相关证据）"
    lines: list[str] = []
    for i, e in enumerate(evs, 1):
        head = " · ".join(x for x in [
            e.source_name or e.domain,
            e.domain,
            f"source_score={e.score:.2f}",
            e.published_at.strftime("%Y-%m-%d") if e.published_at else "",
        ] if x)
        lines.append(f"[{i}] {head}\n    {(e.snippet or '')[:max_chars]}")
    return "\n".join(lines)
