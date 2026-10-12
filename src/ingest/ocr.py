"""截图核查：把截图里的文字识别出来（D54）。

为什么要有：小红书、视频号、抖音的链接服务器抓不到（登录墙、反爬、视频号无公开网页），
而且小红书很多笔记的字写在图片上。截图是对所有平台都成立的统一入口，也不违反平台规则。

模型：阿里云百炼 qwen-vl-ocr（OpenAI 兼容接口，与向量检索共用百炼密钥）。
实测一张 900×420 的中文截图 2.3 秒、约 435 token，逐字准确。
同系列的 qwen-vl-max / qwen-vl-plus 在当前账号下未开通（403/404），专用 OCR 模型够用。

只做「识别文字」，不让模型判断真假或改写——识别结果原样进入后续两道滤网。
"""
from __future__ import annotations

import base64
import os
from typing import Any, Sequence

# 提示词必须短：实测加上「按从上到下的顺序」和一串限制后，qwen-vl-ocr 会改为输出文字框坐标
# （如 252,128,95,423,90），两次复现。下面两句都实测输出正确文字；第一句不行时用第二句重试。
OCR_PROMPTS = ("逐字识别图片中的全部文字，按原有顺序输出，不要解释。", "请输出图片中的全部文字。")


class OCRError(RuntimeError):
    pass


def _mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def looks_like_coordinates(text: str) -> bool:
    """识别结果几乎没有汉字或字母、主要是数字和逗号 → 模型输出的是坐标，不是文字。"""
    t = "".join(text.split())
    if not t:
        return False
    words = sum(1 for ch in t if ch.isalpha())                 # 汉字、字母都算 isalpha
    return words / len(t) < 0.2


def _settings(cfg: dict[str, Any]) -> dict[str, Any]:
    ocfg = ((cfg.get("ingest") or {}).get("ocr") or {})
    return {
        "model": os.getenv("OCR_MODEL") or ocfg.get("model", "qwen-vl-ocr"),
        "base_url": os.getenv("OCR_BASE_URL") or os.getenv("EMBED_BASE_URL")
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("OCR_API_KEY") or os.getenv("EMBED_API_KEY", ""),
        "max_images": int(ocfg.get("max_images", 3)),
        "max_bytes": int(ocfg.get("max_mb", 10)) * 1024 * 1024,
        "timeout": float(ocfg.get("timeout_seconds", 60)),
    }


def ocr_images(images: Sequence[bytes], cfg: dict[str, Any], client: Any = None) -> str:
    """识别多张截图，按顺序拼成一段正文。client 可注入，便于离线测试。"""
    s = _settings(cfg)
    if not images:
        raise OCRError("没有收到截图")
    if len(images) > s["max_images"]:
        raise OCRError(f"一次最多识别 {s['max_images']} 张截图")
    if client is None:
        if not s["api_key"]:
            raise OCRError("没有配置文字识别的密钥（OCR_API_KEY 或 EMBED_API_KEY）")
        from openai import OpenAI
        client = OpenAI(api_key=s["api_key"], base_url=s["base_url"], timeout=s["timeout"], max_retries=1)

    parts = []
    for i, data in enumerate(images, 1):
        if len(data) > s["max_bytes"]:
            raise OCRError(f"第 {i} 张截图太大，请压缩后再试")
        url = f"data:{_mime(data)};base64," + base64.b64encode(data).decode()
        text = ""
        for prompt in OCR_PROMPTS:
            try:
                r = client.chat.completions.create(model=s["model"], messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": prompt}]}])
            except Exception as exc:                            # noqa: BLE001
                raise OCRError(f"第 {i} 张截图识别失败：{exc}") from exc
            text = (r.choices[0].message.content or "").strip()
            if not looks_like_coordinates(text):
                break
        else:
            raise OCRError(f"第 {i} 张截图没能识别出文字，请换一张更清楚的截图")
        if text:
            parts.append(text)
    return "\n\n".join(parts)
