"""向量化：把文本编码成稠密向量。

`backend` 在 config.yaml 切换（规格要求 API / 本地两路都留着）：
- `api`   —— OpenAI 兼容的 /embeddings 端点。当前用阿里云百炼 DashScope。
- `local` —— bge-m3 本地推理（FlagEmbedding / sentence-transformers）。
- `none`  —— 不向量化，向量路自动关闭（供 retrieve-bm25 消融行使用）。

**实测记录（重要，见 docs/decisions.md D29）**：规格原写 bge-m3，
但实测手上这个 DashScope key **没有 bge-m3**（HTTP 404 `Unsupported model`），
其自带的 `text-embedding-v4` 可用（1024 维）。所以：
- API 路走 `text-embedding-v4`
- 想要字面意义的 bge-m3，把 backend 切到 `local`（需装 FlagEmbedding）

两条路的向量**不通用**（维度和语义空间都不同），所以 `model` 名会写进
kb_vectors 与快照指纹——换模型即失效重算，不会拿旧向量冒充新模型的结果。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("kb.embed")

# 各后端一次最多提交多少条文本。DashScope 单次上限较小，给个保守值。
_BATCH = {"api": 10, "local": 32}


class EmbedError(RuntimeError):
    """编码失败（网络/限流/模型不可用）。"""


# 本地模型进程级缓存。检索时每次查询都会新建 Embedder/Reranker（search.py），
# 不缓存的话本地模式每查一次就要重新加载约 2GB 权重，演示时根本等不起。
_LOCAL_MODELS: dict[tuple[str, str, str], Any] = {}
ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_device(device: str) -> str:
    """配置写 cuda 但机器没显卡（或没装 CUDA 版 torch）时退回 cpu，不让流程崩。"""
    if not str(device).startswith("cuda"):
        return "cpu"
    try:
        import torch
        return device if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_local_model(cls: Any, path: str, device: str) -> Any:
    """加载并缓存 FlagEmbedding 模型；GPU 上用半精度（更快、显存减半，检索精度几乎不变）。"""
    p = Path(path)
    if not p.is_absolute() and (ROOT / p).exists():
        path = str(ROOT / p)                         # 配置里写相对项目根的路径
    dev = resolve_device(device)
    key = (cls.__name__, path, dev)
    if key not in _LOCAL_MODELS:
        fp16 = dev.startswith("cuda")
        try:
            _LOCAL_MODELS[key] = cls(path, use_fp16=fp16, devices=dev)
        except TypeError:                            # 旧版 FlagEmbedding 参数名是 device
            _LOCAL_MODELS[key] = cls(path, use_fp16=fp16, device=dev)
    return _LOCAL_MODELS[key]


def embed_config(cfg: dict[str, Any]) -> dict[str, Any]:
    e = (cfg.get("kb", {}) or {}).get("embed", {}) or {}
    return {
        "backend": str(e.get("backend", "api")).lower(),
        "model": e.get("model") or os.getenv("EMBED_MODEL", "text-embedding-v4"),
        "base_url": e.get("base_url") or os.getenv("EMBED_BASE_URL", ""),
        "api_key": os.getenv("EMBED_API_KEY", ""),
        "dim": int(e.get("dim", 0) or 0),          # 0 = 由模型决定
        "batch_size": int(e.get("batch_size", 0) or 0),
        "timeout": float(e.get("timeout_seconds", 60)),
        "max_attempts": int(e.get("max_attempts", 3)),
        "local_path": e.get("local_path", ""),     # 本地权重路径，空则用模型名
        "device": e.get("device", "cpu"),
    }


@dataclass
class Embedder:
    """统一编码接口。`model_tag` 会写进向量表与快照，用于判断缓存是否有效。"""

    backend: str
    model: str
    base_url: str = ""
    api_key: str = ""
    dim: int = 0
    batch_size: int = 0
    timeout: float = 60.0
    max_attempts: int = 3
    local_path: str = ""
    device: str = "cpu"
    calls: int = 0
    texts_encoded: int = 0
    last_error: str = ""          # 最近一次编码失败的原因，供上层做降级说明
    _local: Any = field(default=None, repr=False)

    @classmethod
    def build(cls, cfg: dict[str, Any]) -> "Embedder":
        return cls(**embed_config(cfg))

    @property
    def enabled(self) -> bool:
        return self.backend not in ("none", "", "off")

    @property
    def model_tag(self) -> str:
        """写进 kb_vectors.model。后端 + 模型名一起进 tag——同名模型不同后端也可能维度不同。"""
        return f"{self.backend}:{self.model}"

    # ---------- 编码 ----------
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.enabled:
            raise EmbedError("embed.backend = none，向量路未启用")
        fn = {"api": self._encode_api, "local": self._encode_local}.get(self.backend)
        if fn is None:
            raise EmbedError(f"未知的 embed.backend：{self.backend}")
        out: list[list[float]] = []
        bs = self.batch_size or _BATCH.get(self.backend, 16)
        for i in range(0, len(texts), bs):
            out.extend(fn(list(texts[i:i + bs])))
        return out

    def encode_one(self, text: str) -> list[float]:
        v = self.encode([text])
        return v[0] if v else []

    def _encode_api(self, batch: list[str]) -> list[list[float]]:
        import httpx

        if not self.base_url or not self.api_key:
            raise EmbedError("缺少 EMBED_BASE_URL / EMBED_API_KEY（见 .env）")
        url = self.base_url.rstrip("/") + "/embeddings"
        payload: dict[str, Any] = {"model": self.model, "input": batch}
        if self.dim:
            payload["dimensions"] = self.dim
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        last = ""
        for attempt in range(self.max_attempts):
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    rows = sorted(data["data"], key=lambda d: d.get("index", 0))
                    self.calls += 1
                    self.texts_encoded += len(batch)
                    return [list(x["embedding"]) for x in rows]
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                # 400/404 这类是请求本身有问题，重试没意义
                if r.status_code in (400, 401, 403, 404):
                    break
            except Exception as exc:                       # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_attempts - 1:
                time.sleep(1.5 * (attempt + 1))
        raise EmbedError(f"向量化失败（{self.model}）：{last}")

    def _encode_local(self, batch: list[str]) -> list[list[float]]:
        """bge-m3 本地推理。需要 `pip install FlagEmbedding`。"""
        if self._local is None:
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore
            except ImportError as exc:                     # pragma: no cover
                raise EmbedError(
                    "本地后端需要 FlagEmbedding：pip install FlagEmbedding。"
                    "或把 kb.embed.backend 切回 api。"
                ) from exc
            self._local = load_local_model(BGEM3FlagModel, self.local_path or self.model, self.device)
        out = self._local.encode(batch, batch_size=len(batch), max_length=1024)
        dense = out["dense_vecs"]
        self.calls += 1
        self.texts_encoded += len(batch)
        return [list(map(float, v)) for v in dense]

    def stats(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model, "model_tag": self.model_tag,
                "calls": self.calls, "texts_encoded": self.texts_encoded}
