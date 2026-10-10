"""retrieve —— 权威健康来源检索。

实测结论（详见 docs/decisions.md，均为本机实跑验证）：

1. **Bing CN**：可直连，但**完全无视 `site:` 限定**，返回百科/自媒体噪声，不可用。
2. **Sogou**：支持 `site:`，且结果 HTML 的 `data-url` 属性里直接是真实 URL
   （不需要解析 /link?url= 跳转）。但**反爬激进**：连续十几次请求后整个 IP 被
   跳转到 /antispider/，Cookie 与真实浏览器都救不回来，只能靠节流规避。
3. **权威站站内搜索**：科普中国 `kepuchina.cn/search/index?keyword=` 是 JS 渲染的，
   纯 HTTP 只能拿到空壳；**用 Playwright 驱动本机 Chrome 可以正常拿到结果**，
   且结果质量远高于通用搜索引擎（关键词「隔夜水」返回 16 条对口科普文章）。
4. 关键前提：**必须做查询改写**。把整句主张（"喝醋软化血管"）直接当查询词会 0 结果，
   拆成关键词（"隔夜水"）才有结果。

因此本模块的检索策略：
    - 优先走**站点适配器**（浏览器站内搜索），质量最高
    - 没有适配器的域走**通用引擎 + site:** （HTTP，默认 Sogou，带节流）
    - 全部失败就返回空列表，由上层如实标注"无法核查"，**绝不编造证据**
"""
from __future__ import annotations

import html as html_mod
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..models import Claim, Evidence

log = logging.getLogger("screen.retrieve")

SOGOU = "https://www.sogou.com/web"
BING = "https://cn.bing.com/search"
KEPUCHINA_SEARCH = "https://www.kepuchina.cn/search/index"

# Sogou 结果容器：真实 URL 与标题都在 data-* 属性里
_RE_DATA_BLOCK = re.compile(r'<div\b[^>]*?\bdata-url="([^"]+)"[^>]*?>', re.S)
_RE_DATA_TITLE = re.compile(r'\bdata-title="([^"]*)"')
_RE_H3 = re.compile(r'<h3[^>]*class="[^"]*vr-title[^"]*"[^>]*>(.*?)</h3>', re.S)
_RE_A_TITLE = re.compile(r'<a\b[^>]*>(.*?)</a>', re.S)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)

# 标点集：用字符集而非正则——中文引号在正则字符类里极易因转义/引号配对出错
_PUNCT_CHARS = set(
    "，。！？、；：\u201c\u201d\u2018\u2019（）()《》【】[]…—·|/\\"
    "~!@#$%^&*-_+=<>,.?;:\"'`\n\t\r "
)

# 查询改写用的停用词：这些词出现在查询里只会稀释关键词
_STOPWORDS = (
    "能不能", "可以", "不可以", "真的", "到底", "是不是", "有没有", "会", "能", "可以",
    "吗", "呢", "吧", "的", "了", "是", "在", "和", "与", "及", "等", "请", "大家",
    "千万", "一定", "必须", "赶紧", "马上", "转发", "转给", "提醒", "注意",
)


@dataclass
class SearchHit:
    title: str
    url: str
    source_name: str = ""
    domain: str = ""
    snippet: str = ""
    published_at: str | None = None
    tier: int = 4


# ---------------------------------------------------------------------------
# 查询改写
# ---------------------------------------------------------------------------
def rewrite_query_rule(text: str, max_len: int = 18) -> str:
    """规则法查询改写（兜底用）。

    实测教训：整句"喝醋软化血管"在科普中国站内搜索 0 结果，"隔夜水"却能返回
    16 条对口文章。规则法只能做粗切，核心实体提取靠 rewrite_queries 的 LLM 通道。
    """
    t = "".join(" " if ch in _PUNCT_CHARS else ch for ch in (text or ""))
    for w in _STOPWORDS:
        t = t.replace(w, " ")
    parts = [p.strip() for p in t.split() if len(p.strip()) >= 2]
    if not parts:
        parts = [t.strip()] if t.strip() else []
    parts.sort(key=len, reverse=True)
    q = " ".join(parts[:2])[:max_len].strip()
    return q or (text or "")[:max_len]


def rewrite_queries(text: str, cfg: dict[str, Any], client: Any = None) -> list[str]:
    """把主张转成 1~3 个检索关键词（按优先级）。

    query_rewrite 配置：
      llm  —— 用 LLM 抽取核心名词短语（推荐；规则法提不出"隔夜水"这种核心实体）
      rule —— 纯规则切分，不花钱但召回差
      none —— 直接用原文
    任何一步失败都退回规则法，绝不因为改写失败而丢掉整条检索。
    """
    rcfg = cfg.get("cognitive", {}).get("retrieve", {}) or {}
    mode = (rcfg.get("query_rewrite") or "llm").lower()
    max_len = int(rcfg.get("query_max_len")
                  or (rcfg.get("search") or {}).get("query_max_len", 18))
    if mode == "none":
        return [text]
    if mode != "llm":
        return [rewrite_query_rule(text, max_len)]

    try:
        from ..prompts import load_prompt, render
        from ..llm import build_client
        from ..models import QueryRewrite

        system, user_tpl, _ver = load_prompt("query_rewrite", cfg)
        cli = client or build_client(cfg, stage="query_rewrite")
        data = cli.complete(render(user_tpl, claim=text), system, QueryRewrite,
                            stage="query_rewrite", prompt_version="v1")
        kws = [k for k in (data.get("keywords") or []) if k]
        if kws:
            return kws
    except Exception as exc:      # 改写失败不该让整条检索挂掉
        log.warning("LLM 查询改写失败，退回规则法：%s", exc)
    return [rewrite_query_rule(text, max_len)]


# ---------------------------------------------------------------------------
# HTTP 抓取
# ---------------------------------------------------------------------------
def _headers(cfg: dict[str, Any]) -> dict[str, str]:
    scfg = (cfg.get("cognitive", {}).get("retrieve", {}).get("search", {}) or {})
    ua = scfg.get("user_agent") or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    return {"User-Agent": ua, "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"}


def _fetch(url: str, cfg: dict[str, Any], timeout: float | None = None) -> str:
    scfg = (cfg.get("cognitive", {}).get("retrieve", {}).get("search", {}) or {})
    timeout = float(timeout or scfg.get("timeout_seconds", 20))
    try:
        import httpx
        r = httpx.get(url, headers=_headers(cfg), timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        return r.text
    except ImportError:
        import urllib.request
        req = urllib.request.Request(url, headers=_headers(cfg))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")


def _clean(s: str) -> str:
    return html_mod.unescape(_RE_TAG.sub("", s or "")).strip()


def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def _strip_page(raw: str) -> str:
    txt = _RE_SCRIPT_STYLE.sub(" ", raw)
    txt = re.sub(r"<br\s*/?>|</p>|</div>", "\n", txt, flags=re.I)
    return re.sub(r"\n{2,}", "\n", _clean(txt)).strip()


# ---------------------------------------------------------------------------
# 通用引擎解析
# ---------------------------------------------------------------------------
def parse_sogou(html: str) -> list[SearchHit]:
    """解析 Sogou 结果页。真实 URL 取自 `data-url` 属性（实测比解跳转可靠）。"""
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for m in _RE_DATA_BLOCK.finditer(html):
        attrs = m.group(0)
        url = html_mod.unescape(m.group(1)).strip()
        if not url.startswith("http") or url in seen:
            continue
        tm = _RE_DATA_TITLE.search(attrs)
        title = urllib.parse.unquote(tm.group(1)).strip() if tm else ""
        seen.add(url)
        hits.append(SearchHit(title=title, url=url, domain=_domain(url)))

    if hits and any(not h.title for h in hits):
        titles = [_clean(a) for a in
                  (_RE_A_TITLE.search(x).group(1) if _RE_A_TITLE.search(x) else "")
                  for x in _RE_H3.findall(html)]
        for h, t in zip(hits, [t for t in titles if t]):
            if not h.title:
                h.title = t
    return hits


def parse_bing(html: str) -> list[SearchHit]:
    out: list[SearchHit] = []
    for m in re.finditer(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        url, title = m.group(1), _clean(m.group(2))
        if url.startswith("http"):
            out.append(SearchHit(title=title, url=url, domain=_domain(url)))
    return out


def is_antispider(url_or_html: str) -> bool:
    """识别反爬拦截页——被拦了要退避，而不是当成"没搜到"。"""
    return "antispider" in url_or_html or "请输入验证码" in url_or_html


# ---------------------------------------------------------------------------
# 浏览器会话（站点适配器用）
# ---------------------------------------------------------------------------
class BrowserSession:
    """复用一个 Chrome 实例，避免每次检索都冷启动。

    headless=True + channel="chrome"：直接复用本机已安装的 Chrome，
    不需要 `playwright install` 下载内核（本机已验证 Chrome 存在）。
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._pw = None
        self._browser = None
        self._ctx = None

    def _ensure(self):
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright
        bcfg = (self.cfg.get("cognitive", {}).get("retrieve", {}).get("search", {}) or {})
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            channel=bcfg.get("browser_channel", "chrome"), headless=True)
        self._ctx = self._browser.new_context(
            user_agent=bcfg.get("user_agent") or None, locale="zh-CN")

    def page(self):
        self._ensure()
        return self._ctx.new_page()

    def close(self):
        for obj, closer in ((self._browser, "close"), (self._pw, "stop")):
            try:
                if obj:
                    getattr(obj, closer)()
            except Exception:
                pass
        self._browser = self._pw = self._ctx = None

    def __enter__(self) -> "BrowserSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# 站点适配器：域名 -> (构造搜索 URL, 从页面抽结果的 JS)
def _kepuchina_url(keyword: str) -> str:
    return f"{KEPUCHINA_SEARCH}?keyword={urllib.parse.quote(keyword)}"


# 抽 a 标签，过滤页脚/备案等噪声
_JS_LINKS = """els => els.map(e => [e.innerText.trim(), e.href])"""


def _filter_site_links(pairs: list[list[str]], domain: str) -> list[SearchHit]:
    """过滤掉页脚、备案、合作媒体等噪声链接，只留站内正文页。"""
    out: list[SearchHit] = []
    seen: set[str] = set()
    noise = ("beian", "miit", "xuexi.cn", "emuchong", "chinamtt", "icp", "gov.cn/portal")
    for title, url in pairs:
        if not title or len(title) < 8 or not url or url in seen:
            continue
        low = url.lower()
        if any(k in low for k in noise):
            continue
        if domain not in _domain(url) and "kepuchina" not in low:
            continue
        seen.add(url)
        out.append(SearchHit(title=title, url=url, domain=_domain(url)))
    return out


SITE_ADAPTERS: dict[str, Callable[..., list[SearchHit]]] = {}


def site_adapter(domain: str):
    def deco(fn):
        SITE_ADAPTERS[domain] = fn
        return fn
    return deco


@site_adapter("kepuchina.cn")
def search_kepuchina(keyword: str, cfg: dict[str, Any], n: int,
                     session: BrowserSession | None = None) -> list[SearchHit]:
    """科普中国站内搜索。JS 渲染，必须走浏览器。"""
    scfg = (cfg.get("cognitive", {}).get("retrieve", {}).get("search", {}) or {})
    wait_ms = int(scfg.get("browser_wait_ms", 4000))
    own = session is None
    session = session or BrowserSession(cfg)
    try:
        pg = session.page()
        pg.goto(_kepuchina_url(keyword), wait_until="load",
                timeout=int(scfg.get("browser_timeout_ms", 45000)))
        pg.wait_for_timeout(wait_ms)
        pairs = pg.eval_on_selector_all("a", _JS_LINKS)
        pg.close()
        return _filter_site_links(pairs, "kepuchina.cn")[:n]
    except Exception as exc:
        log.warning("科普中国站内搜索失败（%s）：%s", keyword, exc)
        return []
    finally:
        if own:
            session.close()


# ---------------------------------------------------------------------------
# 检索编排
# ---------------------------------------------------------------------------
def engine_search(query: str, cfg: dict[str, Any], site: str | None = None,
                  n: int | None = None, backend: str | None = None) -> list[SearchHit]:
    """通用搜索引擎检索（HTTP）。site 非空时加 site: 限定（仅 Sogou 有效）。"""
    scfg = (cfg.get("cognitive", {}).get("retrieve", {}).get("search", {}) or {})
    backend = backend or scfg.get("backend", "sogou")
    n = int(n or scfg.get("results_per_query", 10))
    if backend in ("mock", "browser"):
        return []

    q = f"{query} site:{site}" if (site and scfg.get("site_restricted", True)) else query
    if backend == "sogou":
        url = f"{SOGOU}?query={urllib.parse.quote(q)}"
    else:
        url = f"{BING}?q={urllib.parse.quote(q)}"
    try:
        raw = _fetch(url, cfg)
    except Exception as exc:
        log.warning("检索失败 %s：%s", q, exc)
        return []
    if is_antispider(raw):
        log.warning("被反爬拦截（%s）：%s", backend, q)
        return []
    hits = parse_sogou(raw) if backend == "sogou" else parse_bing(raw)
    if site:
        hits = [h for h in hits if site in h.domain]
    return hits[:n]


def authority_search(claim_text: str, cfg: dict[str, Any],
                     max_per_domain: int = 3,
                     session: BrowserSession | None = None,
                     client: Any = None) -> tuple[list[SearchHit], list[str]]:
    """按 tier 顺序逐域检索权威来源，凑够 top_k 即停。

    查询词会依次尝试（LLM 给出的优先级顺序）——第一个词没命中就试第二个，
    实测这是召回率的关键：整句 0 条、核心实体却能命中十几条。

    返回 (证据列表, 实际尝试过的查询词)。查询词一并返回，便于审计与复现。
    """
    rcfg = cfg.get("cognitive", {}).get("retrieve", {}) or {}
    scfg = rcfg.get("search", {}) or {}
    top_k = int(rcfg.get("top_k", 5))
    backend = scfg.get("backend", "sogou")
    queries = rewrite_queries(claim_text, cfg, client=client)

    domains = sorted(rcfg.get("authority_domains") or [], key=lambda d: int(d.get("tier", 9)))
    found: list[SearchHit] = []
    seen: set[str] = set()
    tried: list[str] = []

    own_session = False
    try:
        if backend == "browser" and session is None and any(
                d.get("domain") in SITE_ADAPTERS for d in domains):
            session, own_session = BrowserSession(cfg), True

        for q in queries:
            tried.append(q)
            for d in domains:
                dom, name, tier = d.get("domain", ""), d.get("name", ""), int(d.get("tier", 4))
                if backend == "browser":
                    if dom not in SITE_ADAPTERS:
                        continue
                    assert session is not None
                    hits = SITE_ADAPTERS[dom](q, cfg, max_per_domain, session)
                else:
                    hits = engine_search(q, cfg, site=dom, n=max_per_domain, backend=backend)
                    time.sleep(float(scfg.get("delay_seconds", 0)))
                for h in hits:
                    if h.url in seen:
                        continue
                    seen.add(h.url)
                    h.source_name, h.tier = name, tier
                    found.append(h)
            if len(found) >= top_k:
                break
    finally:
        if own_session and session is not None:
            session.close()
    return found[:top_k], tried


def fetch_snippet(url: str, cfg: dict[str, Any], max_chars: int | None = None) -> str:
    """抓证据正文片段。抓不到返回空串——宁可没有片段，也不编造。"""
    rcfg = cfg.get("cognitive", {}).get("retrieve", {}) or {}
    max_chars = int(max_chars or rcfg.get("evidence_max_chars", 1500))
    try:
        return _strip_page(_fetch(url, cfg))[:max_chars]
    except Exception as exc:
        log.warning("证据正文抓取失败 %s：%s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# L0：本地权威知识库（可复现）
# ---------------------------------------------------------------------------
def kb_search(claim_text: str, cfg: dict[str, Any], top_k: int = 5,
              client: Any = None) -> list[SearchHit]:
    """从本地知识库检索。库不存在就返回空——不自动降级到活检索，
    降级由 evidence_for 按 layers 配置显式控制，便于消融对照。"""
    path = (cfg.get("paths", {}) or {}).get("kb_path")
    if not path or not Path(path).exists():
        return []
    try:
        from ..kb.store import KnowledgeBase
    except ImportError:
        return []
    queries = rewrite_queries(claim_text, cfg, client=client)
    out: list[SearchHit] = []
    seen: set[str] = set()
    with KnowledgeBase(path) as kb:
        for q in queries:
            for row in kb.search(q, top_k=top_k):
                if row["url"] in seen:
                    continue
                seen.add(row["url"])
                out.append(SearchHit(
                    title=row["title"], url=row["url"], domain=_domain(row["url"]),
                    source_name=row.get("source_name", ""), tier=int(row.get("tier", 1)),
                    snippet=(row.get("text") or "")[:400],
                ))
                if len(out) >= top_k:
                    return out
    return out


# ---------------------------------------------------------------------------
# L2：DeepSeek 官方联网搜索（活检索，兜底）
# ---------------------------------------------------------------------------
def websearch_layer(claim_text: str, cfg: dict[str, Any], top_k: int = 5,
                    client: Any = None) -> list[SearchHit]:
    """兜底通道。返回结果按权威域过滤——不做限定的话会混进一堆自媒体。"""
    rcfg = cfg.get("cognitive", {}).get("retrieve", {}) or {}
    if not (rcfg.get("websearch", {}) or {}).get("enabled", True):
        return []
    try:
        from .websearch import web_search
    except ImportError:
        return []
    queries = rewrite_queries(claim_text, cfg, client=client)
    authorities = rcfg.get("authority_domains") or []
    out: list[SearchHit] = []
    seen: set[str] = set()
    from .websearch import WebSearchUnavailable
    for q in queries[:2]:                       # 兜底通道限流，最多试两个查询词
        try:
            hits = web_search(f"联网搜索「{q}」相关的权威健康科普或辟谣内容，给出标题和链接。", cfg)
        except WebSearchUnavailable as exc:     # 实时核查里兜底通道不可用就跳过，不中断核查
            log.warning("联网搜索不可用，跳过兜底通道：%s", exc)
            return out
        for h in hits:
            if h.url in seen:
                continue
            matched = next((d for d in authorities if d.get("domain") in h.domain), None)
            if matched is None:
                continue                        # 非登记权威域，丢弃
            seen.add(h.url)
            h.source_name = matched.get("name", "")
            h.tier = int(matched.get("tier", 1))
            out.append(h)
            if len(out) >= top_k:
                return out
    return out


# ---------------------------------------------------------------------------
# 三层编排
# ---------------------------------------------------------------------------
def _layer_search(layer: str, claim_text: str, cfg: dict[str, Any], need: int,
                  session: BrowserSession | None, client: Any) -> list[SearchHit]:
    if need <= 0:
        return []
    if layer == "kb":
        return kb_search(claim_text, cfg, top_k=need, client=client)
    if layer == "websearch":
        return websearch_layer(claim_text, cfg, top_k=need, client=client)
    if layer == "site":
        hits, _ = authority_search(claim_text, cfg, session=session, client=client)
        return hits[:need]
    log.warning("未知检索层：%s（可选 kb / site / websearch）", layer)
    return []


def evidence_for(claim: Claim, cfg: dict[str, Any], start_id: int = 0,
                 with_snippet: bool = True,
                 session: BrowserSession | None = None,
                 client: Any = None) -> list[Evidence]:
    """按 layers 配置逐层检索，返回带编号与 channel 标记的 Evidence 列表。

    每条证据都标 `channel`，评测时可以算出"多少比例靠活检索"——
    这是可复现性论证的关键数据，不能省。
    """
    rcfg = cfg.get("cognitive", {}).get("retrieve", {}) or {}
    layers = rcfg.get("layers") or ["kb", "site", "websearch"]
    top_k = int(rcfg.get("top_k", 5))

    out: list[Evidence] = []
    seen: set[str] = set()
    own_session = False
    try:
        for layer in layers:
            if len(out) >= top_k:
                break
            need = top_k - len(out)
            if layer == "site" and session is None and any(
                    d.get("domain") in SITE_ADAPTERS
                    for d in (rcfg.get("authority_domains") or [])):
                session, own_session = BrowserSession(cfg), True
            for h in _layer_search(layer, claim.text, cfg, need, session, client):
                if h.url in seen:
                    continue
                seen.add(h.url)
                snippet = h.snippet
                if with_snippet and not snippet:
                    snippet = fetch_snippet(h.url, cfg)
                out.append(Evidence(
                    id=start_id + len(out), claim_id=claim.id, title=h.title, url=h.url,
                    source_name=h.source_name, domain=h.domain, snippet=snippet,
                    tier=h.tier, channel=layer,  # type: ignore[arg-type]
                ))
                if len(out) >= top_k:
                    break
    finally:
        if own_session and session is not None:
            session.close()
    return out


def retrieve_evidence(claim: Claim, cfg: dict[str, Any], start_id: int = 0,
                      with_snippet: bool = True,
                      session: BrowserSession | None = None,
                      client: Any = None) -> list[Evidence]:
    """对一条主张检索证据（三层通道）。保留此名作为对外统一入口。"""
    return evidence_for(claim, cfg, start_id=start_id, with_snippet=with_snippet,
                        session=session, client=client)
