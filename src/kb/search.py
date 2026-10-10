"""块级检索：BM25 与向量两路，可分别开关，RRF 融合，可选重排。

三件事分开，互不耦合：
1. `bm25_search`   —— FTS5 trigram，关键词精确匹配。专有名词、药名、机构名靠它。
2. `vector_search` —— 稠密向量，语义相近但用词不同的表述靠它
                      （实测"血压偏高"召回不到"高血压"的相关段落，就是这类问题）。
3. `fuse_rrf`       —— Reciprocal Rank Fusion。**按名次融合，不按分数**——
                      BM25 的分数是无界的负值，余弦相似度在 [-1,1]，两者不可比，
                      直接加权求和需要调一个没法解释的权重。RRF 只用名次，免调参。

两路各自可开关，对应规格要求的四行子消融：
  retrieve-bm25 / retrieve-vector / retrieve-hybrid / retrieve-hybrid-rerank
"""
from __future__ import annotations

import logging
from typing import Any

from .chunk import is_usable_article
from .embed import Embedder
from .rerank import Reranker
from .store import KnowledgeBase

log = logging.getLogger("kb.search")

# RRF 常数。60 是原论文推荐值：它决定"名次靠后但两路都命中"能拿到多少分。
RRF_K = 60

# 向量矩阵缓存：4088 块 × 1024 维 ≈ 17MB，每次查询都重载太浪费
_VEC_CACHE: dict[tuple[str, str], tuple[list[str], Any]] = {}


def retrieval_config(cfg: dict[str, Any]) -> dict[str, Any]:
    r = (cfg.get("cognitive", {}) or {}).get("retrieve", {}) or {}
    h = r.get("hybrid", {}) or {}
    return {
        "bm25": bool(h.get("bm25", True)),
        "vector": bool(h.get("vector", True)),
        "rerank": bool(h.get("rerank", False)),
        "candidates": int(h.get("candidates", 20)),   # 初检候选数（进重排的量）
        "top_k": int(r.get("top_k", 5)),              # 最终证据条数
        "rrf_k": int(h.get("rrf_k", RRF_K)),
    }


def load_matrix(db_path: str, model_tag: str) -> tuple[list[str], Any]:
    """载入并缓存归一化向量矩阵。"""
    key = (str(db_path), model_tag)
    if key not in _VEC_CACHE:
        with KnowledgeBase(db_path) as kb:
            _VEC_CACHE[key] = kb.load_vectors(model_tag)
    return _VEC_CACHE[key]


def clear_matrix_cache() -> None:
    _VEC_CACHE.clear()


# ---------------------------------------------------------------------------
# 单路检索
# ---------------------------------------------------------------------------
def bm25_search(cfg: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
    path = (cfg.get("paths", {}) or {}).get("kb_path")
    if not path or not query.strip():
        return []
    with KnowledgeBase(path) as kb:
        rows = kb.search_chunks(query, top_k=top_k)
    # bm25() 返回负值，越小越相关 -> 转成"越大越好"的分数，方便统一比较
    for i, r in enumerate(rows):
        r["bm25_rank"] = i
        r["bm25_score"] = -float(r.get("rank") or 0.0)
    return rows


def vector_search(cfg: dict[str, Any], query: str, top_k: int,
                  embedder: Embedder | None = None) -> list[dict[str, Any]]:
    import numpy as np

    from .embed import EmbedError

    path = (cfg.get("paths", {}) or {}).get("kb_path")
    if not path or not query.strip():
        return []
    emb = embedder or Embedder.build(cfg)
    if not emb.enabled:
        return []
    ids, mat = load_matrix(path, emb.model_tag)
    if not ids or mat.shape[0] == 0:
        log.debug("向量库为空，跳过向量检索")
        return []

    # 查询编码失败（配额耗尽/断网）不该让整条检索链崩掉——BM25 那路还在。
    # 降级为"本轮没有向量结果"，由 trace 记录，调用方能看到。
    try:
        qv = np.asarray(emb.encode_one(query), dtype=np.float32)
    except EmbedError as exc:
        log.warning("查询向量化失败，本轮降级为仅 BM25：%s", exc)
        # 记在 embedder 上，由 hybrid_search 带进 trace——"向量路没结果"和
        # "向量路挂了"是两回事，不区分就没法解释消融分数的变化。
        emb.last_error = str(exc)[:200]
        return []
    if qv.size == 0 or qv.shape[0] != mat.shape[1]:
        log.warning("查询向量维度 %s 与库 %s 不一致，跳过向量检索", qv.shape, mat.shape)
        return []
    n = float(np.linalg.norm(qv)) or 1.0
    sims = mat @ (qv / n)

    k = min(top_k, len(ids))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]

    with KnowledgeBase(path) as kb:
        meta = kb.get_chunks_by_ids([ids[i] for i in top])
    out: list[dict[str, Any]] = []
    for rank, i in enumerate(top):
        cid = ids[i]
        row = meta.get(cid)
        if not row:
            continue
        row = dict(row)
        row["vec_rank"] = rank
        row["vec_score"] = float(sims[i])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------
def fuse_rrf(lists: list[list[dict[str, Any]]], key: str = "chunk_id",
             k: int = RRF_K) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank)。

    只用名次，不用分数——BM25 与余弦相似度的量纲不可比，硬加权要调一个
    说不清理由的权重。RRF 免调参，且对两路的分数尺度差异免疫。
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict[str, Any]] = {}
    for lst in lists:
        for rank, row in enumerate(lst):
            cid = row.get(key)
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in payload:
                payload[cid] = dict(row)
                continue
            # 同一个块被多路召回时，把另一路的 `*_rank` / `*_score` **并进**同一条记录。
            # 早期版本这里是 setdefault（只留先见到的那一行），结果 BM25 排在前、
            # 向量排在后，凡是两路都命中的块都只带 bm25_rank ——
            # `matched_by` 于是永远算不出 "both"。
            # 而"两路都命中"恰恰是融合里最可信的那部分，也正是 hybrid 该赢过单路的原因。
            # 丢掉它，消融表就只能说"hybrid 分高"，说不出高在哪。
            for kk, vv in row.items():
                if kk.endswith(("_rank", "_score")) and kk not in payload[cid]:
                    payload[cid][kk] = vv
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[dict[str, Any]] = []
    for cid, sc in ordered:
        row = dict(payload[cid])
        row["rrf_score"] = sc
        out.append(row)
    return out


def hybrid_search(cfg: dict[str, Any], queries: str | list[str],
                  embedder: Embedder | None = None,
                  reranker: Reranker | None = None,
                  rerank_query: str | None = None) -> dict[str, Any]:
    """完整检索：初检（BM25 / 向量）-> 融合 -> 重排 -> Top-K。

    `queries` 接受**多个改写后的查询**。这一点很关键：整句主张直接检索，
    BM25 实测一律 0 命中（"降压药吃上就停不下来了？老中医教你一个偏方"这种串，
    trigram 匹配不到任何段落）。所以必须先做查询改写，而且**四种检索模式共用
    同一组改写结果**——否则 retrieve-bm25 量的是"忘了改写"，不是"BM25 好不好"。

    `rerank_query` 是送进重排器的查询，默认取第一个（即原始主张）。
    重排用**完整主张**而不是关键词：cross-encoder 能吃下完整上下文，
    用关键词反而丢信息。

    返回 {"results": [...], "trace": {...}}。trace 记录每一路各命中多少、
    重排有没有降级——消融分析要靠它解释"为什么这一行分数变了"。
    """
    qs = [queries] if isinstance(queries, str) else [q for q in queries if q and q.strip()]
    rc = retrieval_config(cfg)
    n_cand = rc["candidates"]
    trace: dict[str, Any] = {"queries": qs, "bm25": False, "vector": False,
                             "rerank": False, "rerank_degraded": False,
                             "rerank_moved_top1": False, "rerank_displaced": 0,
                             "n_bm25": 0, "n_vector": 0, "n_fused": 0}
    if not qs:
        return {"results": [], "trace": trace}

    lists: list[list[dict[str, Any]]] = []
    if rc["bm25"]:
        for q in qs:
            b = bm25_search(cfg, q, n_cand)
            trace["n_bm25"] += len(b)
            lists.append(b)
        trace["bm25"] = True
    if rc["vector"]:
        emb = embedder or Embedder.build(cfg)
        for q in qs:
            v = vector_search(cfg, q, n_cand, embedder=emb)
            trace["n_vector"] += len(v)
            lists.append(v)
        trace["vector"] = True
        # 向量路"挂了"与"没命中"必须区分：前者会让 hybrid 退化成纯 BM25，
        # 不说明的话，消融表上看到的就是一个无法解释的分数。
        if getattr(emb, "last_error", ""):
            trace["vector_degraded"] = True
            trace["vector_reason"] = emb.last_error

    lists = [x for x in lists if x]
    if not lists:
        return {"results": [], "trace": trace}
    if len(lists) == 1:
        fused = list(lists[0])                    # 单路：不必融合，保留原始名次
    else:
        fused = fuse_rrf(lists, k=rc["rrf_k"])
    trace["n_fused"] = len(fused)

    if rc["rerank"] and fused:
        rr = reranker or Reranker.build(cfg)
        pool = fused[:n_cand]
        res = rr.rerank(rerank_query or qs[0], [(r.get("text") or "") for r in pool],
                        top_n=rc["top_k"])
        trace["rerank"] = True
        trace["rerank_degraded"] = res.degraded
        if res.reason:
            trace["rerank_reason"] = res.reason
        picked = [dict(pool[i]) for i in res.order]
        for rank, (row, sc) in enumerate(zip(picked, res.scores or [0.0] * len(picked))):
            row["rerank_rank"] = rank
            row["rerank_score"] = sc
        # 重排**不改变候选集大小**，只改顺序。所以光看 "n_fused / 平均证据数"
        # 的话，hybrid 与 hybrid-rerank 两行会一模一样——分不出重排到底干没干活。
        # 这两个数就是用来回答"重排真的动了它吗"的：Top-1 是否换人、有几条挪了位置。
        pre = [r.get("chunk_id") for r in pool[: rc["top_k"]]]
        post = [r.get("chunk_id") for r in picked]
        trace["rerank_moved_top1"] = bool(pre and post and pre[0] != post[0])
        trace["rerank_displaced"] = sum(1 for a, b in zip(pre, post) if a != b)
        fused = picked
    else:
        fused = fused[: rc["top_k"]]

    return {"results": fused[: rc["top_k"]], "trace": trace}
