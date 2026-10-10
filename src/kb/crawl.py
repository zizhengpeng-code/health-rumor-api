"""知识库构建：从权威来源抓取科普与辟谣文章，冻结成本地快照。

流程（分两段，慢的部分与快的部分分开）：

    种子关键词 ──[URL 发现，慢]──> URL 列表 ──[正文抓取，可并发，快]──> 本地库 ──> freeze()

URL 发现走两个通道：
  1. **DeepSeek 官方联网搜索**（限域查询）——覆盖面最广，实测一次能拿回 10+ 条
     辟谣平台/疾控中心的真实文章链接，不需要逐个站点写适配器。
  2. **站点适配器**（浏览器站内搜索）——科普中国这种 JS 渲染站只有这条能走。

正文抓取**不需要浏览器**：实测权威站的文章页都是服务端渲染的，httpx 直接拿，
所以可以并发。这一步是把整体耗时从小时级压到分钟级的关键。

发现的随机性不影响可复现——因为**库会被冻结成带版本号的快照**，
评测只认快照（见 src/kb/store.py 的 freeze()）。

    python scripts/build_kb.py --keywords 20 --per-source 8
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

from ..cognitive.retrieve import (
    SITE_ADAPTERS, BrowserSession, SearchHit, _domain, _fetch,
)
from ..cognitive.websearch import search_site
from .store import KnowledgeBase

log = logging.getLogger("screen.kb.crawl")

# 种子关键词：覆盖 CLAUDE.md 里的 topic 分类，并补上家庭群高频谣言主题
SEED_KEYWORDS: list[str] = [
    # 慢病
    "高血压", "糖尿病", "冠心病", "血脂", "脑卒中", "痛风", "降压药",
    # 用药
    "抗生素", "阿司匹林", "他汀", "保健品 药物", "擅自停药", "用药误区", "中药 偏方",
    # 饮食
    "隔夜水", "隔夜菜", "亚硝酸盐", "碱性食物", "骨头汤 补钙", "维生素C 感冒",
    "食物相克", "喝醋 血管", "生吃 寄生虫", "微波炉 致癌", "味精",
    # 癌症
    "癌症 谣言", "抗癌食物", "化疗", "酸碱体质", "饿死癌细胞", "防癌体检",
    # 养生器械与养生骗局
    "理疗仪", "量子 保健", "养生 骗局", "磁疗", "按摩椅 治病", "足浴盆",
    # 通用辟谣
    "健康谣言", "朋友圈谣言", "老人 谣言", "养生误区",
]

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|nav|footer|header)\b.*?</\1>", re.S | re.I)


def extract_article(html: str) -> tuple[str, str]:
    """从 HTML 抽 (标题, 正文)。优先 trafilatura，失败退回朴素清洗。"""
    title, text = "", ""
    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        md = trafilatura.extract_metadata(html)
        if md is not None:
            title = (getattr(md, "title", "") or "").strip()
    except Exception as exc:
        log.debug("trafilatura 抽取失败：%s", exc)

    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        title = _TAG.sub("", m.group(1)).strip() if m else ""
    if not text:
        body = _SCRIPT.sub(" ", html)
        body = re.sub(r"<br\s*/?>|</p>|</div>", "\n", body, flags=re.I)
        text = re.sub(r"\n{2,}", "\n", _TAG.sub("", body)).strip()
    return title, text


def fetch_article(url: str, cfg: dict[str, Any], min_chars: int = 200) -> tuple[str, str] | None:
    """抓一篇正文。太短（导航页/404）直接丢弃。"""
    try:
        html = _fetch(url, cfg)
    except Exception as exc:
        log.debug("抓取失败 %s：%s", url, exc)
        return None
    title, text = extract_article(html)
    if len(text) < min_chars:
        return None
    max_chars = int((cfg.get("kb", {}) or {}).get("max_chars", 8000))
    return title, text[:max_chars]


def discover(keyword: str, domain: str, cfg: dict[str, Any],
             per_source: int = 8, session: BrowserSession | None = None) -> list[SearchHit]:
    """发现某个域下与关键词相关的文章 URL。站点适配器优先，否则用联网搜索。"""
    if domain in SITE_ADAPTERS:
        try:
            hits = SITE_ADAPTERS[domain](keyword, cfg, per_source, session)
            if hits:
                return hits
        except Exception as exc:
            log.warning("站点适配器失败 %s：%s", domain, exc)
    return search_site(keyword, domain, cfg)[:per_source]


def crawl(cfg: dict[str, Any], keywords: Iterable[str] | None = None,
          per_source: int = 8, workers: int = 8,
          db_path: str | None = None, verbose: bool = True) -> dict[str, Any]:
    """跑一轮完整建库。返回统计信息。"""
    kb_cfg = cfg.get("kb", {}) or {}
    keywords = list(keywords if keywords is not None else
                    (kb_cfg.get("seed_keywords") or SEED_KEYWORDS))
    rcfg = cfg.get("cognitive", {}).get("retrieve", {}) or {}
    domains = [d.get("domain") for d in (rcfg.get("authority_domains") or []) if d.get("domain")]
    name_of = {d.get("domain"): d.get("name", "") for d in (rcfg.get("authority_domains") or [])}
    tier_of = {d.get("domain"): int(d.get("tier", 1)) for d in (rcfg.get("authority_domains") or [])}

    db = db_path or (cfg.get("paths", {}) or {}).get("kb_path") or "data/kb.db"
    kb = KnowledgeBase(db)

    # ---- 第一段：URL 发现（慢，浏览器/联网搜索，串行）----
    pending: dict[str, str] = {}          # url -> seed_keyword
    if verbose:
        print(f"发现阶段：{len(keywords)} 个关键词 × {len(domains)} 个权威域")
    need_browser = any(d in SITE_ADAPTERS for d in domains)
    session = BrowserSession(cfg) if need_browser else None
    try:
        for i, kw in enumerate(keywords, 1):
            for dom in domains:
                for h in discover(kw, dom, cfg, per_source, session):
                    if h.url.startswith("http"):
                        pending.setdefault(h.url, kw)
            if verbose and i % 5 == 0:
                print(f"  [{i}/{len(keywords)}] 已发现 {len(pending)} 条 URL")
    finally:
        if session is not None:
            session.close()

    # ---- 第二段：正文抓取（快，纯 HTTP，并发）----
    if verbose:
        print(f"抓取阶段：{len(pending)} 条 URL，{workers} 并发")
    docs: list[dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_article, u, cfg): (u, kw) for u, kw in pending.items()}
        for fut in as_completed(futs):
            url, kw = futs[fut]
            done += 1
            try:
                got = fut.result()
            except Exception:
                got = None
            if not got:
                continue
            title, text = got
            dom = _domain(url)
            src = next((d for d in domains if d in dom), dom)
            docs.append({
                "source": src, "source_name": name_of.get(src, src), "tier": tier_of.get(src, 3),
                "title": title or url, "url": url, "text": text, "seed_keyword": kw,
            })
            if verbose and done % 25 == 0:
                print(f"  [{done}/{len(pending)}] 成功入库 {len(docs)} 篇")

    added = kb.add_many(docs)
    kb.commit()
    snap = kb.freeze()
    total = kb.doc_count()
    kb.close()

    result = {"discovered": len(pending), "fetched": len(docs), "added": added,
              "total_docs": total, "snapshot": snap}
    if verbose:
        print("\n建库完成")
        print(f"  发现 URL {result['discovered']} 条，成功抓取 {result['fetched']} 篇，"
              f"新增 {added} 篇，库内共 {total} 篇")
        print(f"  快照版本 {snap['kb_version']}  指纹 {snap['content_hash']}")
        print(f"  来源分布 {snap['by_source']}")
    return result
