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
from src.ingest.ocr import OCRError, ocr_images  # noqa: E402
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


def _env_of(fileid: str) -> str:
    return fileid.split("://", 1)[1].split(".", 1)[0]


def _kb_download_url() -> str:
    if os.environ.get("KB_URL"):
        return os.environ["KB_URL"]
    fileid = os.environ.get("KB_FILEID", "")
    if not fileid:
        raise RuntimeError("没有配置 KB_FILEID 或 KB_URL")
    return _file_download_url(fileid)


def _file_download_url(fileid: str) -> str:
    # 云托管内网免鉴权调用开放接口（需在控制台开启「开放接口服务」并加白 /tcb/batchdownloadfile）
    r = httpx.post("http://api.weixin.qq.com/tcb/batchdownloadfile",
                   json={"env": _env_of(fileid), "file_list": [{"fileid": fileid, "max_age": 600}]}, timeout=20)
    data = r.json()
    if data.get("errcode") not in (0, None):
        raise RuntimeError(f"获取下载链接失败：{data}")
    item = (data.get("file_list") or [{}])[0]
    if not item.get("download_url"):
        raise RuntimeError(f"获取下载链接失败：{item}")
    return item["download_url"]


def _download_url_with_retry(total_s: int = 180) -> str:
    """开放接口服务是旁加载组件，与业务进程同时启动，刚启动时可能还没就绪（实测 Connection refused），
    所以失败后每 5 秒重试，最多 3 分钟。配置类错误（没填 KB_FILEID）不重试。"""
    deadline, last = time.time() + total_s, None
    while True:
        try:
            return _kb_download_url()
        except RuntimeError as exc:
            if "没有配置" in str(exc):
                raise
            last = exc
        except Exception as exc:                                # noqa: BLE001  连接被拒、返回非 JSON 等都重试
            last = exc
        if time.time() >= deadline:
            raise RuntimeError(f"多次重试仍失败：{last}")
        log.warning("获取知识库下载链接失败，5 秒后重试：%s", last)
        _kb["error"] = f"重试中：{last}"
        time.sleep(5)


_kb_lock = threading.Lock()


def _ensure_kb() -> None:
    if not _kb_lock.acquire(blocking=False):                   # 已有线程在下载
        return
    try:
        _ensure_kb_locked()
    finally:
        _kb_lock.release()


def _ensure_kb_locked() -> None:
    if KB_PATH.exists():
        _kb.update(ready=True, error="")
        return
    try:
        t0 = time.time()
        url = _download_url_with_retry()
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
        _kb.update(ready=True, error="")
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
# 截图核查（D54）
# ---------------------------------------------------------------------------
SHOT_PREFIX = "screenshots/"


def _check_image_ids(ids: list[str]) -> str:
    """只接受本环境对象存储 screenshots/ 目录下的文件，防止拿后端去下载任意文件。"""
    limit = int(((load_config().get("ingest") or {}).get("ocr") or {}).get("max_images", 3))
    if len(ids) > limit:
        return f"一次最多上传 {limit} 张截图"
    kb_env = _env_of(os.environ["KB_FILEID"]) if os.environ.get("KB_FILEID") else ""
    for f in ids:
        if not f.startswith("cloud://") or f"/{SHOT_PREFIX}" not in f or (kb_env and _env_of(f) != kb_env):
            return "截图地址不正确，请重新上传"
    return ""


def _delete_files(ids: list[str]) -> None:
    """识别完就删：家庭群截图里可能有昵称、头像等个人信息。需加白 /tcb/batchdeletefile；失败只记日志。"""
    try:
        r = httpx.post("http://api.weixin.qq.com/tcb/batchdeletefile",
                       json={"env": _env_of(ids[0]), "fileid_list": ids}, timeout=20)
        if r.json().get("errcode") not in (0, None):
            log.warning("删除截图失败：%s", r.text[:200])
    except Exception as exc:                                    # noqa: BLE001
        log.warning("删除截图失败：%s", exc)


def _ocr_screenshots(ids: list[str], cfg) -> str:
    try:
        images = []
        for f in ids:
            r = httpx.get(_file_download_url(f), timeout=60, follow_redirects=True)
            r.raise_for_status()
            images.append(r.content)
        text = ocr_images(images, cfg)
    finally:
        _delete_files(ids)
    min_chars = int(((cfg.get("ingest") or {}).get("ocr") or {}).get("min_chars", 20))
    if len(text.strip()) < min_chars:
        raise OCRError("截图里没识别出足够的文字，请换一张更清楚、文字更完整的截图")
    return text


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
    images: list[str] = Field(default_factory=list, description="截图在对象存储里的 fileID（截图核查，D54）")


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
        if not _kb["ready"] and not _kb_lock.locked():          # 之前彻底失败过：再试一次
            threading.Thread(target=_ensure_kb, daemon=True).start()
            time.sleep(1)
        for _ in range(240):                                    # 冷启动时等知识库下载（含重试），最多 4 分钟
            if _kb["ready"] or (_kb["error"] and not _kb["error"].startswith("重试中")
                                and not _kb_lock.locked()):
                break
            time.sleep(1)
        if not _kb["ready"]:
            raise RuntimeError(f"知识库未就绪：{_kb['error'] or '下载超时'}")
        cfg = load_config()
        recognized = ""
        if req.images:
            recognized = _ocr_screenshots(req.images, cfg)
            doc = Document(id=f"shot-{uuid.uuid4().hex[:10]}", source_type="screenshot",
                           title=req.title, text=recognized)
        elif req.url:
            doc = fetch_document(req.url, cfg)
        else:
            doc = Document(id=f"paste-{uuid.uuid4().hex[:10]}", source_type="paste",
                           title=req.title, text=req.text)
        job["result"] = report_to_dict(get_checker().check(doc), source_url=req.url)
        if recognized:
            job["result"]["recognized_text"] = recognized[:3000]
        job["status"] = "done"
    except (IngestError, OCRError) as exc:
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
    # 启动时那次彻底失败了（不是在重试中）：每次健康检查都在后台再试一次
    if not _kb["ready"] and _kb["error"] and not _kb["error"].startswith("重试中") and not _kb_lock.locked():
        threading.Thread(target=_ensure_kb, daemon=True).start()
    return {"status": "ok", "kb_ready": _kb["ready"], "kb_error": _kb["error"]}


@app.post("/api/check")
def submit(req: CheckRequest, request: Request):
    if not req.url and not req.text and not req.images:
        return {"ok": False, "error": "请输入文章链接、粘贴正文或上传截图"}
    if req.images:
        bad = _check_image_ids(req.images)
        if bad:
            return {"ok": False, "error": bad}
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
