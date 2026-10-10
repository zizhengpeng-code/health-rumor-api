"""配置加载：config.yaml + .env。

所有可调项集中在 config.yaml，代码里不硬编码。密钥只从环境变量读。
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


RETRIEVAL_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "local": {
        "embed": {"backend": "local", "model": "BAAI/bge-m3",
                  "local_path": "data/models/bge-m3", "device": "cuda"},
        "rerank": {"backend": "local", "model": "BAAI/bge-reranker-v2-m3",
                   "local_path": "data/models/bge-reranker-v2-m3", "device": "cuda"},
    },
    "api": {
        "embed": {"backend": "api", "model": "text-embedding-v4", "local_path": "", "device": "cpu"},
        "rerank": {"backend": "api", "model": "gte-rerank-v2", "local_path": "", "device": "cpu"},
    },
}


class Config(dict):
    """dict 的薄封装，支持 cfg.get_path("a.b.c", default) 点号取值。"""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """读 config.yaml，加载 .env，把相对路径解析成绝对路径。

    overrides 用于评测时的消融开关覆盖（浅合并到 cfg["_overrides"]，由各模块自行读取）。
    """
    load_dotenv(ROOT / ".env")
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(raw)

    # 相对路径 -> 绝对路径，避免因 cwd 不同而写到奇怪的地方
    paths = cfg.setdefault("paths", {})
    for key, val in list(paths.items()):
        if key.endswith("_dir") or key.endswith("_path"):
            v = Path(val)
            paths[key] = str(v if v.is_absolute() else ROOT / v)
    paths.setdefault("db_path", str(ROOT / "data" / "screen.db"))
    # 测试用：把结果目录重定向到临时目录，避免 pytest 覆盖 results/ 里的真实评测结果
    if os.environ.get("HRF_RESULTS_DIR"):
        paths["results_dir"] = os.environ["HRF_RESULTS_DIR"]
    # 扩库用：在知识库副本上建索引，不影响正在用旧库跑的评测（D41）
    if os.environ.get("HRF_KB_PATH"):
        paths["kb_path"] = os.environ["HRF_KB_PATH"]
    # 检索模型档案（D44）：演示用本地、正式成绩与上线用 API，同一份配置一键切换
    profile = os.environ.get("HRF_RETRIEVAL", "").lower()
    if profile in RETRIEVAL_PROFILES:
        kb = cfg.setdefault("kb", {})
        for part, values in RETRIEVAL_PROFILES[profile].items():
            kb.setdefault(part, {}).update(values)

    cfg["_root"] = str(ROOT)
    cfg["_overrides"] = dict(overrides or {})
    return cfg


def module_on(cfg: Config, name: str) -> bool:
    """判断某模块是否开启：评测配置的 override 优先于 config.yaml 的 enabled。"""
    ov = cfg.get("_overrides") or {}
    if name in ov:
        return bool(ov[name])
    return bool(cfg.get_path(f"{name}.enabled", True))


def doc_max_chars(cfg: dict[str, Any], module: str = "") -> int:
    """正文截断上限。**全链路统一规则**：默认继承 `ingest.max_chars`。

    `module` 传模块的配置路径（如 `"behavior"`、`"cognitive.baseline"`），
    该模块下的 `max_chars` 显式大于 0 时才覆盖，否则继承。

    为什么要有这条统一规则：如果某个模块偷偷用了更小的上限，它会因为"被喂得少"而失分，
    后面每加一个模块看起来都"有效"——但那个增益是假的（见 docs/decisions.md D18）。
    用 0 表示继承而不是把数字抄一遍，是为了避免两处数字日后漂移。
    """
    base = int((cfg.get("ingest", {}) or {}).get("max_chars", 6000))
    if not module:
        return base
    cur: Any = cfg
    for part in module.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return base
        cur = cur[part]
    own = (cur or {}).get("max_chars", 0) if isinstance(cur, dict) else 0
    try:
        own = int(own or 0)
    except (TypeError, ValueError):
        own = 0
    return own if own > 0 else base


def local_tz(cfg: Config) -> dt.timezone:
    return dt.timezone(dt.timedelta(hours=cfg.get("timezone_offset_hours", 8)))


def now(cfg: Config) -> dt.datetime:
    return dt.datetime.now(local_tz(cfg))


def run_stamp(cfg: Config) -> str:
    """本次运行的时间戳，用于结果目录与文件命名。"""
    return now(cfg).strftime("%Y%m%d-%H%M%S")
