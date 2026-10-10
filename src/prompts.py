"""提示词加载。

约定：prompts/ 下的文件名为 `<name>_<version>.md`，正文用 Markdown 分区：

    ## System
    （系统提示词）

    ## User 模板
    （用户提示词模板，可用 {占位符}）

版本号写进文件名，改提示词就新建版本文件，**不覆盖旧版本**——这样评测结果
才能回溯到当时用的是哪一版（CLAUDE.md 工作约定第 3 条）。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.M)


class PromptNotFound(FileNotFoundError):
    pass


def prompt_dir(cfg: dict[str, Any]) -> Path:
    p = (cfg.get("paths", {}) or {}).get("prompts_dir")
    return Path(p) if p else Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=64)
def _read(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def _sections(text: str) -> dict[str, str]:
    """把 Markdown 按 `## 标题` 切段。代码块内的 ## 不会被误切（按行首匹配）。"""
    out: dict[str, str] = {}
    marks = list(_SECTION.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1).strip()] = text[m.end():end].strip()
    return out


def load_prompt(name: str, cfg: dict[str, Any]) -> tuple[str, str, str]:
    """返回 (system, user_template, 版本号)。版本号取自文件名。"""
    d = prompt_dir(cfg)
    matches = sorted(d.glob(f"{name}_*.md"))
    if not matches:
        raise PromptNotFound(f"找不到提示词 {name}_*.md（目录：{d}）")
    path = matches[-1]                      # 取最新版本
    version = path.stem[len(name) + 1:] if path.stem.startswith(name + "_") else path.stem
    secs = _sections(_read(str(path)))
    system = secs.get("System", "")
    user = secs.get("User 模板") or secs.get("User", "")
    if not system:
        raise PromptNotFound(f"{path.name} 缺少 `## System` 段落")
    return system, user, version


def render(template: str, **kwargs: Any) -> str:
    """安全填充模板：缺失的占位符原样保留，不抛异常。"""
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out
