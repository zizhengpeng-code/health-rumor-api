"""重排：对候选块做 cross-encoder 精排，取 Top-K。

初检（BM25 / 向量）是**双塔**式的——query 和文档各自编码，快但精度有限。
重排用 cross-encoder 把 (query, 文档) 拼在一起打分，精度高得多，代价是慢，
所以只对候选 Top-N 做，而不是全库。

`backend` 在 config.yaml 切换：
- `api`   —— DashScope 的 rerank 端点。实测 `gte-rerank-v2` 可用（见 D29）。
- `local` —— bge-reranker-v2-m3 本地推理。
- `none`  —— 不重排，直接返回初检顺序（`retrieve-hybrid` 消融行用这个）。

**降级策略**：重排失败（限流/模型没开通）不抛异常，而是**原样返回初检结果**并记一条警告。
理由：重排是"锦上添花"，初检已经拿到了候选；因为精排挂了就让整条检索链失败，
是把可用的证据也一起丢掉。但降级必须**可见**——`RerankResult.degraded` 会带出去。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

log = logging.getLogger("kb.rerank")


class RerankError(RuntimeError):
    pass


def rerank_config(cfg: dict[str, Any]) -> dict[str, Any]:
    r = (cfg.get("kb", {}) or {}).get("rerank", {}) or {}
    return {
        "backend": str(r.get("backend", "api")).lower(),
        "model": r.get("model") or os.getenv("RERANK_MODEL", "gte-rerank-v2"),
        "base_url": r.get("base_url") or os.getenv("RERANK_BASE_URL", ""),
        "api_key": os.getenv("RERANK_API_KEY", ""),
        "timeout": float(r.get("timeout_seconds", 60)),
        "max_attempts": int(r.get("max_attempts", 3)),
        "max_doc_chars": int(r.get("max_doc_chars", 1200)),   # 送进模型的文档截断
        "local_path": r.get("local_path", ""),
        "device": r.get("device", "cpu"),
    }


@dataclass
class RerankResult:
    """重排结果。order 是**下标序列**，指向传入的 candidates。"""

    order: list[int]
    scores: list[float] = field(default_factory=list)
    degraded: bool = False
    reason: str = ""

    def top(self, n: int) -> list[int]:
        return self.order[:n]


@dataclass
class Reranker:
    backend: str
    model: str
    base_url: str = ""
    api_key: str = ""
    timeout: float = 60.0
    max_attempts: int = 3
    max_doc_chars: int = 1200
    local_path: str = ""
    device: str = "cpu"
    calls: int = 0
    degraded_calls: int = 0
    _local: Any = field(default=None, repr=False)

    @classmethod
    def build(cls, cfg: dict[str, Any]) -> "Reranker":
        return cls(**rerank_config(cfg))

    @property
    def enabled(self) -> bool:
        return self.backend not in ("none", "", "off")

    def rerank(self, query: str, docs: Sequence[str],
               top_n: int | None = None) -> RerankResult:
        """返回按相关度降序的下标序列。失败时降级为原始顺序。"""
        n = len(docs)
        if n == 0:
            return RerankResult(order=[])
        if not self.enabled:
            return RerankResult(order=list(range(n)), degraded=False, reason="backend=none")

        trimmed = [(d or "")[: self.max_doc_chars] for d in docs]
        try:
            if self.backend == "api":
                order, scores = self._api(query, trimmed)
            elif self.backend == "local":
                order, scores = self._local_rerank(query, trimmed)
            else:
                raise RerankError(f"未知的 rerank.backend：{self.backend}")
        except Exception as exc:                            # noqa: BLE001
            # 精排挂了不该拖垮整条检索链——初检候选还在，原样返回
            self.degraded_calls += 1
            reason = f"{type(exc).__name__}: {exc}"
            log.warning("重排降级为初检顺序：%s", reason)
            return RerankResult(order=list(range(n)), degraded=True, reason=reason)

        if top_n:
            order = order[:top_n]
        return RerankResult(order=order, scores=scores)

    def _api(self, query: str, docs: list[str]) -> tuple[list[int], list[float]]:
        import httpx

        if not self.base_url or not self.api_key:
            raise RerankError("缺少 RERANK_BASE_URL / RERANK_API_KEY（见 .env）")
        url = self.base_url.rstrip("/") + "/api/v1/services/rerank/text-rerank/text-rerank"
        payload = {"model": self.model,
                   "input": {"query": query, "documents": docs},
                   "parameters": {"top_n": len(docs), "return_documents": False}}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        last = ""
        for attempt in range(self.max_attempts):
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code == 200:
                    results = (r.json().get("output") or {}).get("results") or []
                    results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
                    self.calls += 1
                    return ([int(x["index"]) for x in results],
                            [float(x.get("relevance_score", 0.0)) for x in results])
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 401, 403, 404):
                    break
            except Exception as exc:                        # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_attempts - 1:
                time.sleep(1.5 * (attempt + 1))
        raise RerankError(f"重排失败（{self.model}）：{last}")

    def _local_rerank(self, query: str, docs: list[str]) -> tuple[list[int], list[float]]:
        if self._local is None:
            try:
                from FlagEmbedding import FlagReranker  # type: ignore
            except ImportError as exc:                     # pragma: no cover
                raise RerankError(
                    "本地重排需要 FlagEmbedding：pip install FlagEmbedding。"
                    "或把 kb.rerank.backend 切回 api / none。"
                ) from exc
            from .embed import load_local_model
            self._local = load_local_model(FlagReranker, self.local_path or self.model, self.device)
        pairs = [[query, d] for d in docs]
        scores = self._local.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        self.calls += 1
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return order, [float(scores[i]) for i in order]

    def stats(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model, "calls": self.calls,
                "degraded_calls": self.degraded_calls}
