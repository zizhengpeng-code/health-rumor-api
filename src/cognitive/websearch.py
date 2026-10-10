"""DeepSeek 官方联网搜索通道。

实测（2026-10-02）：DeepSeek 在 **Anthropic 兼容端点**上提供**服务端运行**的
`web_search` 工具，OpenAI 格式的 `/chat/completions` 上**没有**这个能力。

    POST https://api.deepseek.com/anthropic/v1/messages
    headers: x-api-key, anthropic-version: 2023-06-01
    body: {"model": "deepseek-flash", "messages": [...],
           "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]}

返回里会出现 `server_tool_use`（模型自己生成的查询词）与 `web_search_tool_result`
（真实结果，含标题与 URL）。实测首个结果常是政府源，质量不差。

两个用途：
1. **兜底通道**：本地知识库与站内搜索都没命中时，用它补。
2. **建库的 URL 发现**：比逐个站点写适配器覆盖面更广。库冻结后发现的随机性不影响复现。

⚠️ 它不可复现（今天和下月返回不同），因此**绝不作为主通道**，
且每条证据都会标记 `channel="websearch"`，评测时单独统计占比。
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .retrieve import SearchHit, _domain

log = logging.getLogger("screen.websearch")

ANTHROPIC_ENDPOINT = "/anthropic/v1/messages"
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


class WebSearchUnavailable(RuntimeError):
    """接口不支持该工具，或凭据不可用。"""


def web_search(query: str, cfg: dict[str, Any], max_uses: int = 3,
               timeout: float = 90.0) -> list[SearchHit]:
    """调 DeepSeek 官方联网搜索，返回 (标题, URL) 列表。

    失败一律返回空列表——兜底通道挂了不该让整条链路崩掉。
    """
    wcfg = (cfg.get("cognitive", {}).get("retrieve", {}).get("websearch", {}) or {})
    if not wcfg.get("enabled", True):
        return []
    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    # OpenAI 兼容地址 -> Anthropic 兼容地址
    if base.endswith("/v1"):
        base = base[:-3]
    url = base.rstrip("/") + ANTHROPIC_ENDPOINT
    key = os.environ.get("LLM_API_KEY", "")
    model = wcfg.get("model") or os.environ.get("LLM_MODEL", "deepseek-flash")
    if not key:
        raise WebSearchUnavailable("未设置 LLM_API_KEY")

    payload = {
        "model": model,
        "max_tokens": int(wcfg.get("max_tokens", 2048)),
        "messages": [{"role": "user", "content": query}],
        "tools": [{**WEB_SEARCH_TOOL, "max_uses": max_uses}],
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    try:
        import httpx
        r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code in (401, 402, 403):
            # 余额不足/鉴权失败不是"没搜到"：静默返回空会让批量任务把每个关键词都当成已完成（D43）
            raise WebSearchUnavailable(f"联网搜索不可用（HTTP {r.status_code}）：{r.text[:120]}")
        if r.status_code != 200:
            log.warning("联网搜索返回 %s：%s", r.status_code, r.text[:200])
            return []
        data = r.json()
    except WebSearchUnavailable:
        raise
    except Exception as exc:
        log.warning("联网搜索调用失败：%s", exc)
        return []
    return parse_web_search(data)


def parse_web_search(data: dict[str, Any]) -> list[SearchHit]:
    """从 Anthropic 格式响应里抽出 web_search_tool_result 的结果。"""
    out: list[SearchHit] = []
    seen: set[str] = set()
    for block in data.get("content") or []:
        if block.get("type") != "web_search_tool_result":
            continue
        for item in block.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "web_search_result":
                continue
            u = item.get("url") or ""
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(SearchHit(title=(item.get("title") or "").strip(),
                                 url=u, domain=_domain(u)))
    return out


def web_search_queries(data: dict[str, Any]) -> list[str]:
    """把模型自己生成的查询词抽出来——便于审计它到底搜了什么。"""
    return [b.get("input", {}).get("query", "")
            for b in (data.get("content") or [])
            if b.get("type") == "server_tool_use" and b.get("name") == "web_search"]


def search_site(keyword: str, domain: str, cfg: dict[str, Any],
                max_uses: int = 3) -> list[SearchHit]:
    """限定域名的搜索，用于建库时做 URL 发现。结果按域名过滤。"""
    q = (f"请联网搜索「{keyword}」相关的权威健康科普或辟谣内容，"
         f"只返回来自 {domain} 的页面，给出标题和完整链接。")
    hits = web_search(q, cfg, max_uses=max_uses)
    return [h for h in hits if domain in h.domain]
