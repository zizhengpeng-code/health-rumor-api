"""内容接入：抓取文章链接 → 统一的 Document（CLAUDE.md 第 0 节）。

公众号文章（mp.weixin.qq.com）按微信页面结构解析：
    标题  —— og:title / var msg_title
    账号  —— var nickname / id="js_name"
    正文  —— id="js_content" 容器
其他网站用 trafilatura 通用正文抽取兜底。

去噪：公众号文末常见「推荐阅读」「往期回顾」「长按识别二维码」等区块。规则是
**尾部标记出现在正文后半部分（按字数算）时，从标记处截断**，再删掉含噪声关键词的短行——不能简单删掉
所有含「广告」的行，否则「警惕虚假广告」这种正文也会被误删。

抓不到时（被微信验证拦截、文章已删除）抛 IngestError，提示用户直接粘贴正文
（config: ingest.allow_paste），对应报告 2.5 的异常处理。
"""
from __future__ import annotations

import hashlib
import html as html_mod
import re
from typing import Any
from urllib.parse import urlparse

from ..config import doc_max_chars
from ..models import Document

BLOCKED_MARKERS = ("环境异常", "完成验证后即可继续访问", "该内容已被发布者删除",
                   "此内容因违规无法查看", "此内容被多人投诉", "参数错误")
TAIL_MARKERS = ("推荐阅读", "往期回顾", "往期推荐", "精彩回顾", "长按识别", "长按二维码",
                "点击关注", "扫码关注", "阅读原文", "END")
MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
             "(KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.47")


class IngestError(RuntimeError):
    """抓不到正文。消息里写明原因，并提示改用粘贴正文。"""


def _clean_text(raw_html: str) -> str:
    t = re.sub(r"<(script|style)\b.*?</\1>", " ", raw_html, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>|</p>|</section>|</div>|</h\d>|</li>", "\n", t, flags=re.I)
    t = html_mod.unescape(re.sub(r"<[^>]+>", "", t))
    lines = [re.sub(r"[ \t　\xa0]+", " ", ln).strip() for ln in t.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def strip_noise(text: str, noise_keywords: list[str] | tuple[str, ...] = ()) -> str:
    """去掉公众号文末的推广区块与含噪声关键词的短行。"""
    lines = text.split("\n")
    total = sum(len(ln) for ln in lines) or 1
    cut, seen = len(lines), 0
    for i, ln in enumerate(lines):
        # 按字数而不是行数判断位置：短文章行数少，按行数算会误判
        if seen >= total * 0.5 and any(ln.startswith(m) or ln == m for m in TAIL_MARKERS):
            cut = i
            break
        seen += len(ln)
    # 只删「几乎只有关键词本身」的行（如「广告」「长按识别二维码关注」）；
    # 「警惕虚假广告，……」这类正文句子虽然含关键词，但远长于关键词，必须保留
    kept = [ln for ln in lines[:cut]
            if not any(k in ln and len(ln) <= len(k) + 8 for k in noise_keywords)]
    return "\n".join(kept).strip()


def parse_wechat(page: str) -> dict[str, str]:
    """从公众号文章页 HTML 抽取标题、账号、正文（纯函数，便于离线测试）。"""
    for m in BLOCKED_MARKERS:
        if m in page and 'id="js_content"' not in page:
            raise IngestError(f"微信页面无法访问（{m}），请直接粘贴正文")

    def first(*patterns: str) -> str:
        for p in patterns:
            mm = re.search(p, page, re.S)
            if mm:
                return html_mod.unescape(mm.group(1)).strip()
        return ""

    title = first(r'<meta\s+property="og:title"\s+content="([^"]*)"',
                  r"var\s+msg_title\s*=\s*'([^']*)'", r'var\s+msg_title\s*=\s*"([^"]*)"',
                  r'<h1[^>]*class="rich_media_title[^"]*"[^>]*>(.*?)</h1>')
    account = first(r'var\s+nickname\s*=\s*"([^"]*)"', r"var\s+nickname\s*=\s*'([^']*)'",
                     r'id="js_name"[^>]*>\s*(.*?)\s*<')
    m = re.search(r'id="js_content"[^>]*>(.*?)(?=<script|<div[^>]+id="js_(?:tags|bottom|pc_qr)|$)',
                  page, re.S)
    body = _clean_text(m.group(1)) if m else ""
    return {"title": re.sub(r"<[^>]+>", "", title).strip(), "account": account, "text": body}


# 服务器抓不到的平台（D54）：登录墙 / 反爬 / 视频号没有公开网页。直接告诉用户改用截图或粘贴文字，
# 不去抓一个空壳页面再给出莫名其妙的结论。
UNSUPPORTED_PLATFORMS = {
    "xhslink.com": "小红书", "xiaohongshu.com": "小红书",
    "channels.weixin.qq.com": "视频号", "weixin.qq.com/sph": "视频号", "finder.video.qq.com": "视频号",
    "douyin.com": "抖音", "iesdouyin.com": "抖音",
    "kuaishou.com": "快手", "gifshow.com": "快手",
}


def extract_url(raw: str) -> str:
    """App 里「复制链接」常带一段分享文案（如「【标题】… http://xhslink.com/xxx 复制本条信息…」），取出第一个网址。"""
    m = re.search(r"https?://[^\s，。！？、）)】\]\"'<>]+", raw or "")
    return m.group(0) if m else (raw or "").strip()


def unsupported_platform(url: str) -> str:
    low = url.lower()
    return next((name for key, name in UNSUPPORTED_PLATFORMS.items() if key in low), "")


def fetch_document(url: str, cfg: dict[str, Any], fetcher: Any = None) -> Document:
    """抓取链接并返回 Document。fetcher 可注入（测试用），签名 fetcher(url) -> html。"""
    url = extract_url(url)
    platform = unsupported_platform(url)
    if platform:
        raise IngestError(f"{platform}的内容无法通过链接读取（平台有登录和防抓取限制）。"
                          f"请截图后用「上传截图」核查，或复制正文用「粘贴正文」核查。")
    icfg = cfg.get("ingest", {}) or {}
    if fetcher is None:
        import httpx

        def fetcher(u: str) -> str:
            r = httpx.get(u, headers={"User-Agent": MOBILE_UA, "Accept-Language": "zh-CN,zh;q=0.9"},
                          timeout=float(icfg.get("timeout_seconds", 20)), follow_redirects=True)
            r.raise_for_status()
            return r.text
    try:
        page = fetcher(url)
    except Exception as exc:                                    # noqa: BLE001
        raise IngestError(f"链接无法打开（{type(exc).__name__}），请直接粘贴正文") from exc

    host = urlparse(url).netloc.lower()
    if "mp.weixin.qq.com" in host:
        got = parse_wechat(page)
        source_type = "article"
    else:
        try:
            import trafilatura
            text = trafilatura.extract(page, include_comments=False, include_tables=False) or ""
            md = trafilatura.extract_metadata(page)
            title = (getattr(md, "title", "") or "") if md else ""
        except Exception:                                       # noqa: BLE001
            text, title = "", ""
        got = {"title": title, "account": host, "text": text or _clean_text(page)}
        source_type = "article"

    text = strip_noise(got["text"], icfg.get("noise_keywords") or [])
    if len(text) < 50:
        raise IngestError("没有抓到有效正文（可能需要在微信内打开），请直接粘贴正文")
    text = text[: doc_max_chars(cfg)]
    return Document(id="url-" + hashlib.md5(url.encode()).hexdigest()[:10],
                    source_type=source_type, title=got["title"], account=got["account"],
                    text=text, url=url)
