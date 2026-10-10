"""M4 主流程：检索证据 -> 结合证据判断事实性。

这是消融表里 `+retrieve` 及四行 `retrieve-*` 子消融共用的执行体。
**四行只变检索方式，提示词与判据完全一致**——这样分数差异才能归因到检索本身。

与 M2 baseline 的关键区别：baseline 是裸模型凭记忆判断；这里把权威来源的
段落片段喂进去，让模型有据可依。M2 实测暴露的问题（`部分属实` 一条都判不对、
塌缩成二分类）正是要靠这一步来修。

**关于证据与自身知识冲突**：提示词明确要求"以证据为准"。理由不是"证据永远对"，
而是这个系统的对外承诺是"结论有据可查"——模型的记忆可能过时，而权威来源
的表述至少可以被核对。宁可引错可追溯的证据，也不要给一个无从查证的判断。
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import doc_max_chars
from ..kb.search import hybrid_search, retrieval_config
from ..models import Document, Evidence, EvidenceVerdict
from ..prompts import load_prompt, render

log = logging.getLogger("screen.evidence_judge")

PROMPT_NAME = "evidence_judge"

# 检索模式 -> hybrid 三开关。四行子消融靠它区分。
RETRIEVE_MODES: dict[str, dict[str, bool]] = {
    "bm25": {"bm25": True, "vector": False, "rerank": False},
    "vector": {"bm25": False, "vector": True, "rerank": False},
    "hybrid": {"bm25": True, "vector": True, "rerank": False},
    "hybrid-rerank": {"bm25": True, "vector": True, "rerank": True},
}


class RewriteCache:
    """查询改写缓存。

    **为什么必须有**：改写是一次 LLM 调用，即使 temperature=0，实测两次调用
    也会给出不同的关键词（服务端的批处理与路由不是严格确定的）。如果四行子消融
    各改一次，它们检索用的查询就不一样了——那么分数差异里混进了"改写随机性"，
    这张子消融表就不再是"只变检索方式"的对照实验。

    缓存之后：同一主张只改写一次，四行共用，差异只能来自检索本身。
    顺带把改写成本从 4 次降到 1 次。
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.data: dict[str, list[str]] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:                                # noqa: BLE001
                self.data = {}

    @staticmethod
    def key(claim: str, version: str = "") -> str:
        # 提示词版本进 key：换了改写提示词，旧缓存必须失效
        return f"{version}|{claim.strip()}"

    def get(self, claim: str, version: str = "") -> list[str] | None:
        v = self.data.get(self.key(claim, version))
        if v:
            self.hits += 1
        return v

    def put(self, claim: str, queries: list[str], version: str = "") -> None:
        self.misses += 1
        self.data[self.key(claim, version)] = queries
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                                 encoding="utf-8")


# 政府机构名：从标题里认出具体是哪个部门（标题常见「赤峰市卫生健康委员会_工作动态」）
_GOV_ORG = re.compile(
    r"([一-龥]{2,16}?(?:卫生健康委员会|卫生健康局|卫健委|疾病预防控制中心|疾病预防控制局|"
    r"医疗保障局|药品监督管理局|市场监督管理局|市场监管局|人民政府))")


def display_source_name(source_name: str, url: str, title: str = "") -> str:
    """证据来源的展示名。

    建库时 authority_domains 里 `gov.cn` 被命名为「中国政府网」，但它匹配的是**所有** gov.cn
    子域名——省市卫健委、医保局、市场监管局都在里面。辟谣稿若写「中国政府网说……」就是张冠李戴。
    这里按网址纠正：只有 www.gov.cn 才叫中国政府网；其余优先用标题里认出的机构名，
    认不出就叫「政府部门官方网站」。其他来源原样返回。
    """
    host = urllib.parse.urlparse(url).netloc.lower() if url else ""
    if source_name != "中国政府网" or host in ("www.gov.cn", "gov.cn"):
        return source_name
    m = _GOV_ORG.search(title or "")
    return m.group(1) if m else "政府部门官方网站"


def apply_mode(cfg: dict[str, Any], mode: str) -> None:
    """把检索模式写进 config，供 hybrid_search 读取。"""
    sw = RETRIEVE_MODES.get(mode)
    if sw is None:
        raise KeyError(f"未知检索模式 {mode}；可选：{list(RETRIEVE_MODES)}")
    r = cfg.setdefault("cognitive", {}).setdefault("retrieve", {})
    r.setdefault("hybrid", {}).update(sw)


def format_evidence(evs: list[Evidence], max_chars: int = 1500) -> str:
    """把证据列表渲染成提示词里的编号清单。

    编号从 1 开始（模型更容易引用），但 Evidence.id 保持 0 基——两者差 1，
    转换只在这一处做，避免下游到处 ±1。
    """
    if not evs:
        return "（未检索到相关证据）"
    lines: list[str] = []
    for i, e in enumerate(evs, 1):
        head = " · ".join(x for x in [e.source_name or e.domain, e.domain,
                                      e.published_at.strftime("%Y-%m-%d")
                                      if e.published_at else ""] if x)
        lines.append(f"[{i}] {head}\n    {(e.snippet or '')[:max_chars]}")
    return "\n".join(lines)


@dataclass
class EvidenceJudge:
    """检索 + 证据判断。client 由调用方注入，便于测试。"""

    cfg: dict[str, Any]
    client: Any
    max_chars: int = 6000
    embedder: Any = None
    reranker: Any = None
    cache: RewriteCache | None = None
    last_trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: dict[str, Any], client: Any = None, store: Any = None,
              run_id: str = "adhoc", mode: str | None = None,
              cache: RewriteCache | None = None) -> "EvidenceJudge":
        if client is None:
            from ..llm import build_client
            client = build_client(cfg, stage="baseline", store=store, run_id=run_id)
        if mode:
            apply_mode(cfg, mode)
        if cache is None:
            rc = (cfg.get("cognitive", {}).get("retrieve", {}) or {})
            if rc.get("rewrite_cache", True):
                root = cfg.get("_root") or "."
                cache = RewriteCache(Path(root) / "data" / "cache" / "rewrites.json")
        return cls(cfg=cfg, client=client, max_chars=doc_max_chars(cfg, "cognitive.baseline"),
                   cache=cache)

    def _queries(self, claim: str) -> list[str]:
        """改写查询，命中缓存则直接复用（保证四行子消融用同一组查询）。"""
        from .retrieve import rewrite_queries

        rcfg = (self.cfg.get("cognitive", {}).get("retrieve", {}) or {})
        version = str(rcfg.get("query_rewrite_prompt_version", "v1"))
        if self.cache is not None:
            hit = self.cache.get(claim, version)
            if hit:
                return hit
        queries = rewrite_queries(claim, self.cfg) or [claim]
        if self.cache is not None:
            self.cache.put(claim, queries, version)
        return queries

    @property
    def prompt_version(self) -> str:
        try:
            return load_prompt(PROMPT_NAME, self.cfg)[2]
        except Exception:
            return "unknown"

    @property
    def mode(self) -> str:
        rc = retrieval_config(self.cfg)
        for name, sw in RETRIEVE_MODES.items():
            if all(rc[k] == v for k, v in sw.items()):
                return name
        return "custom"

    # ---------- 检索 ----------
    def retrieve(self, claim: str) -> list[Evidence]:
        """按当前配置检索，产出带 chunk_id / 相关度分 / 命中路的 Evidence 列表。

        **先做查询改写**：整句主张直接检索，BM25 实测一律 0 命中
        （"降压药吃上就停不下来了？老中医教你一个偏方"这种串匹配不到任何段落）。
        四种检索模式共用同一组改写结果，否则 retrieve-bm25 量的是"忘了改写"
        而不是"BM25 好不好"——那样这张子消融表就没有意义了。
        """
        from .retrieve import rewrite_queries

        queries = self._queries(claim)
        out = hybrid_search(self.cfg, queries, embedder=self.embedder,
                            reranker=self.reranker, rerank_query=claim)
        self.last_trace = dict(out.get("trace") or {})
        self.last_trace["rewrite_cache_hit"] = bool(
            self.cache is not None and self.cache.hits and not self.cache.misses)
        ev_max = int((self.cfg.get("cognitive", {}).get("retrieve", {}) or {})
                     .get("evidence_max_chars", 1500))
        evs: list[Evidence] = []
        for i, row in enumerate(out["results"]):
            matched = []
            if "bm25_rank" in row:
                matched.append("bm25")
            if "vec_rank" in row:
                matched.append("vector")
            evs.append(Evidence(
                id=i, claim_id=0,
                title=row.get("title") or "",
                url=row.get("url") or "",
                source_name=display_source_name(row.get("source_name") or "",
                                                row.get("url") or "", row.get("title") or ""),
                domain=row.get("source") or "",
                snippet=(row.get("text") or "")[:ev_max],
                tier=int(row.get("tier") or 4),
                channel="kb",
                chunk_id=row.get("chunk_id") or "",
                relevance=float(row.get("rerank_score", row.get("rrf_score",
                                row.get("vec_score", row.get("bm25_score", 0.0))))),
                matched_by="both" if len(matched) > 1 else (matched[0] if matched else ""),
            ))
        return evs

    # ---------- 判断 ----------
    def judge(self, text: str, title: str = "") -> tuple[EvidenceVerdict, list[Evidence]]:
        evs = self.retrieve(title or text[:60])
        system, user_tpl, version = load_prompt(PROMPT_NAME, self.cfg)
        prompt = render(
            user_tpl,
            title=title or "（无标题）",
            text=text[: self.max_chars],
            evidence=format_evidence(evs),
            n_evidence=len(evs),
            max_chars=self.max_chars,
        )
        data = self.client.complete(prompt, system, EvidenceVerdict,
                                    stage="evidence_judge", prompt_version=version)
        return EvidenceVerdict.model_validate(data), evs

    def judge_document(self, doc: Document) -> tuple[EvidenceVerdict, list[Evidence]]:
        return self.judge(doc.text, doc.title)
