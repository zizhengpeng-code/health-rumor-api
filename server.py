"""微信云托管后端（D53）：包装核查流程，供小程序 / 网页调用。

接口：
  GET  /api/health          健康检查；kb_ready 表示知识库是否已就绪
  POST /api/check           提交核查，立即返回 job_id（核查要 15–60 秒，不让请求一直挂着）
  GET  /api/result/{job_id} 查询结果；status = queued | running | done | error

与本机版（app/api/server.py）的区别：
  1. 知识库不打进镜像（代码包 ≤ 2MB、GitHub 单文件 ≤ 100MB），启动时从云托管对象存储下载
  2. 提交 / 查询分离，避免网关超时
  3. 访问限制：按微信用户（X-WX-OPENID，云托管自动注入）每日限次 + 全站每日上限，防止 API 余额被刷
  4. 整个进程共用一个 Checker，检索器只初始化一次

环境变量（在云托管「服务设置」里填，不要写进代码）：
  LLM_API_KEY / LLM_BASE_URL / LLM_MODEL            大模型（DeepSeek）
  EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL      向量（阿里云百炼）
  RERANK_API_KEY / RERANK_BASE_URL / RERANK_MODEL   重排（阿里云百炼）
  KB_FILEID      知识库在对象存储里的 fileid，形如 cloud://prod-xxx.xxxx-prod-xxx-123/kb/kb_cloud.db.gz
  KB_URL         （可选）直接给一个 HTTPS 下载地址，优先于 KB_FILEID
  DAILY_LIMIT_PER_USER  每个微信用户每天最多核查几次，默认 20
  DAILY_LIMIT_TOTAL     全站每天最多核查几次，默认 300
  ALLOW_ANONYMOUS       没有微信身份的请求（公网直连）是否放行，默认 0（不放行）
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
# 部署包里 src/ 与本文件同级；在项目里开发时 src/ 在上两级
for _p in (_HERE, _HERE.parent.parent):
    if (_p / "src").is_dir():
        sys.path.insert(0, str(_p))
        break

os.environ.setdefault("HRF_RETRIEVAL", "api")                       # 云上无显卡，检索走阿里云接口
os.environ.setdefault("HRF_KB_PATH", "/tmp/kb/kb_cloud.db")
os.environ.setdefault("HRF_RESULTS_DIR", "/tmp/results")

import httpx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.config import load_config  # noqa: E402
from src.ingest.fetch import IngestError, fetch_document  # noqa: E402
from src.models import CheckReport, Document  # noqa: E402
from src.report.render import PRIORITY_HINT, split_evidence, to_markdown  # noqa: E402

log = logging.getLogger("cloudrun")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KB_PATH = Path(os.environ["HRF_KB_PATH"])
PER_USER = int(os.environ.get("DAILY_LIMIT_PER_USER", "20"))
TOTAL = int(os.environ.get("DAILY_LIMIT_TOTAL", "300"))
ALLOW_ANON = os.environ.get("ALLOW_ANONYMOUS", "0") == "1"
JOB_TTL = 3600                                  # 结果保留 1 小时

app = FastAPI(title="健康谣言双滤网核查 API（云托管）", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


# ---------------------------------------------------------------------------
# 知识库：启动时下载
# ---------------------------------------------------------------------------
_kb = {"ready": KB_PATH.exists(), "error": ""}


def _kb_download_url() -> str:
    if os.environ.get("KB_URL"):
        return os.environ["KB_URL"]
    fileid = os.environ.get("KB_FILEID", "")
    if not fileid:
        raise RuntimeError("没有配置 KB_FILEID 或 KB_URL")
    env = fileid.split("://", 1)[1].split(".", 1)[0]
    # 云托管内网免鉴权调用开放接口（需在控制台开启「开放接口服务」并加白 /tcb/batchdownloadfile）
    r = httpx.post("http://api.weixin.qq.com/tcb/batchdownloadfile",
                   json={"env": env, "file_list": [{"fileid": fileid, "max_age": 3600}]}, timeout=20)
    data = r.json()
    if data.get("errcode") not in (0, None):
        raise RuntimeError(f"获取下载链接失败：{data}")
    item = (data.get("file_list") or [{}])[0]
    if not item.get("download_url"):
        raise RuntimeError(f"获取下载链接失败：{item}")
    return item["download_url"]


def _ensure_kb() -> None:
    if KB_PATH.exists():
        _kb["ready"] = True
        return
    try:
        t0 = time.time()
        url = _kb_download_url()
        KB_PATH.parent.mkdir(parents=True, exist_ok=True)
        part = KB_PATH.with_suffix(".part")
        with httpx.stream("GET", url, timeout=600, follow_redirects=True) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        with open(part, "rb") as f:
            gz = f.read(2) == b"\x1f\x8b"
        if gz:
            with gzip.open(part, "rb") as src, open(KB_PATH, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            part.unlink()
        else:
            part.replace(KB_PATH)
        _kb["ready"] = True
        log.info("知识库就绪：%s（%.0f MB，%.1f 秒）", KB_PATH, KB_PATH.stat().st_size / 2**20, time.time() - t0)
    except Exception as exc:                                    # noqa: BLE001
        _kb["error"] = str(exc)
        log.exception("知识库下载失败")


threading.Thread(target=_ensure_kb, daemon=True).start()

# ---------------------------------------------------------------------------
# 核查器：进程内只建一次
# ---------------------------------------------------------------------------
_checker = None
_checker_lock = threading.Lock()


def get_checker():
    global _checker
    with _checker_lock:
        if _checker is None:
            from src.pipeline import Checker
            _checker = Checker(load_config())
        return _checker


# ---------------------------------------------------------------------------
# 访问限制（进程内计数；多实例时各自计数，总量上限按实例数放大——演示场景足够）
# ---------------------------------------------------------------------------
_quota_lock = threading.Lock()
_quota: dict[str, Any] = {"day": "", "users": {}, "total": 0}


def _take_quota(who: str) -> str:
    """成功扣一次额度返回空串，否则返回拒绝原因。"""
    today = time.strftime("%Y-%m-%d")
    with _quota_lock:
        if _quota["day"] != today:
            _quota.update(day=today, users={}, total=0)
        if _quota["total"] >= TOTAL:
            return "今天的核查次数已用完，明天再来吧"
        n = _quota["users"].get(who, 0)
        if n >= PER_USER:
            return f"每人每天最多核查 {PER_USER} 次，明天再来吧"
        _quota["users"][who] = n + 1
        _quota["total"] += 1
        return ""


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------
_pool = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "2")))
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class CheckRequest(BaseModel):
    url: str = Field(default="", description="公众号文章链接")
    text: str = Field(default="", description="文章正文（粘贴模式）")
    title: str = Field(default="", description="文章标题（可选）")


def report_to_dict(r: CheckReport, source_url: str = "") -> dict[str, Any]:
    v = r.verdicts[0] if r.verdicts else None
    d, b = r.decision, r.behavior
    shown = split_evidence(v.evidence, v.used_evidence)[0] if v else []
    return {
        "ok": True, "error": "", "title": r.title, "source_url": source_url, "latency_s": r.latency_s,
        "priority": d.priority if d else "", "priority_hint": PRIORITY_HINT.get(d.priority, "") if d else "",
        "cognitive_label": v.label if v else "无法核查", "confidence": v.confidence if v else 0.0,
        "reasoning": v.reasoning if v else "",
        "behavior_level": b.risk_level if b else "低",
        "behaviors": [{"type": h.type, "evidence": h.evidence, "confidence": h.confidence}
                      for h in (b.behaviors if b else [])],
        "evidence": [{"id": i, "source_name": e.source_name or e.domain, "domain": e.domain,
                      "title": e.title, "url": e.url, "snippet": (e.snippet or "")[:200],
                      "tier": e.tier, "score": e.score} for i, e in shown],
        "rebuttal": r.rebuttal, "rebuttal_sources": r.rebuttal_sources, "notes": r.notes,
        "markdown_report": to_markdown(r),
    }


def _run(job_id: str, req: CheckRequest) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    try:
        for _ in range(120):                                    # 冷启动时等知识库下载，最多 2 分钟
            if _kb["ready"] or _kb["error"]:
                break
            time.sleep(1)
        if not _kb["ready"]:
            raise RuntimeError(f"知识库未就绪：{_kb['error'] or '下载超时'}")
        cfg = load_config()
        if req.url:
            doc = fetch_document(req.url, cfg)
        else:
            doc = Document(id=f"paste-{uuid.uuid4().hex[:10]}", source_type="paste",
                           title=req.title, text=req.text)
        job["result"] = report_to_dict(get_checker().check(doc), source_url=req.url)
        job["status"] = "done"
    except IngestError as exc:
        job.update(status="error", error=str(exc))
    except Exception as exc:                                    # noqa: BLE001
        traceback.print_exc()
        job.update(status="error", error=f"核查流程出错：{exc}")


def _gc_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        for k in [k for k, j in _jobs.items() if now - j["created"] > JOB_TTL]:
            _jobs.pop(k, None)


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "kb_ready": _kb["ready"], "kb_error": _kb["error"]}


@app.post("/api/check")
def submit(req: CheckRequest, request: Request):
    if not req.url and not req.text:
        return {"ok": False, "error": "请输入文章链接或粘贴正文"}
    if req.text and len(req.text.strip()) < 50:
        return {"ok": False, "error": "正文太短，请粘贴完整内容（至少 50 字）"}
    openid = request.headers.get("x-wx-openid", "")
    if not openid and not ALLOW_ANON:
        return {"ok": False, "error": "请在小程序里使用"}
    who = openid or (request.headers.get("x-forwarded-for", "") or (request.client.host if request.client else "anon"))
    reason = _take_quota(who)
    if reason:
        return {"ok": False, "error": reason}
    _gc_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "created": time.time(), "owner": who, "result": None, "error": ""}
    _pool.submit(_run, job_id, req)
    return {"ok": True, "job_id": job_id}


@app.get("/api/result/{job_id}")
def result(job_id: str, request: Request):
    job = _jobs.get(job_id)
    if job is None:
        return {"ok": False, "status": "error", "error": "任务不存在或已过期，请重新核查"}
    openid = request.headers.get("x-wx-openid", "")
    if openid and job["owner"] != openid:                       # 只能查自己的任务
        return {"ok": False, "status": "error", "error": "任务不存在或已过期，请重新核查"}
    if job["status"] == "done":
        return {"status": "done", **job["result"]}
    if job["status"] == "error":
        return {"ok": False, "status": "error", "error": job["error"]}
    return {"ok": True, "status": job["status"]}
