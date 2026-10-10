"""SQLite 持久化：文档、预测、模型调用记录、评测运行。

移植自 SEA Credit Radar 的 `radar/store.py`，把 events/sources/updates 三张表
换成本项目的 documents / predictions / verdicts，保留 runs 表的成本记账能力。
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import CheckReport, Document, Prediction

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source_type  TEXT NOT NULL DEFAULT 'article',
    title        TEXT DEFAULT '',
    account      TEXT DEFAULT '',
    url          TEXT DEFAULT '',
    text         TEXT NOT NULL,
    char_count   INTEGER DEFAULT 0,
    published_at TEXT,
    fetched_at   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'new',  -- new | screened | failed
    note         TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id         TEXT NOT NULL,
    run_id         TEXT NOT NULL,
    config_name    TEXT DEFAULT '',
    cognitive_label TEXT,
    behavior_level TEXT,
    priority       TEXT,
    payload        TEXT,          -- 完整 CheckReport 的 JSON
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_doc ON predictions(doc_id);
CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_id);

CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    run_id            TEXT NOT NULL,
    stage             TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_version    TEXT DEFAULT '',
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd          REAL DEFAULT 0,
    latency_s         REAL DEFAULT 0,
    ok                INTEGER DEFAULT 1,
    note              TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_run ON runs(run_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    run_id        TEXT NOT NULL UNIQUE,
    config_name   TEXT NOT NULL,
    split         TEXT NOT NULL,
    n_samples     INTEGER DEFAULT 0,
    model         TEXT DEFAULT '',
    metrics       TEXT,           -- 全量指标的 JSON
    config_snapshot TEXT,         -- 配置快照，保证可复现
    prompt_versions TEXT,
    results_path  TEXT DEFAULT ''
);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- documents ----------
    def upsert_document(self, doc: Document) -> None:
        self.conn.execute(
            """INSERT INTO documents (id, source_type, title, account, url, text, char_count,
                                      published_at, fetched_at, status)
               VALUES (?,?,?,?,?,?,?,?,?, 'new')
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, account=excluded.account, text=excluded.text,
                 char_count=excluded.char_count, url=excluded.url""",
            (doc.id, doc.source_type, doc.title, doc.account, doc.url, doc.text,
             doc.char_count, _iso(doc.published_at), _iso(doc.fetched_at)),
        )
        self.conn.commit()

    def get_document(self, doc_id: str) -> Document | None:
        row = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return None
        return Document(
            id=row["id"], source_type=row["source_type"], title=row["title"],
            account=row["account"], url=row["url"], text=row["text"],
            published_at=_parse_dt(row["published_at"]), fetched_at=_parse_dt(row["fetched_at"]),
        )

    def set_doc_status(self, doc_id: str, status: str, note: str = "") -> None:
        self.conn.execute("UPDATE documents SET status=?, note=? WHERE id=?", (status, note, doc_id))
        self.conn.commit()

    # ---------- predictions ----------
    def save_report(self, report: CheckReport, run_id: str, config_name: str = "") -> None:
        self.conn.execute(
            """INSERT INTO predictions (doc_id, run_id, config_name, cognitive_label,
                                        behavior_level, priority, payload, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (report.doc_id, run_id, config_name,
             report.decision.cognitive_label if report.decision else None,
             report.behavior.risk_level if report.behavior else None,
             report.decision.priority if report.decision else None,
             report.model_dump_json(), _iso(dt.datetime.now())),
        )
        self.conn.commit()

    # ---------- runs（成本记账，被 llm.py 调用） ----------
    def log_run(self, run_id: str, stage: str, model: str, prompt_version: str = "",
                prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0,
                latency_s: float = 0.0, ok: bool = True, note: str = "") -> None:
        self.conn.execute(
            """INSERT INTO runs (ts, run_id, stage, model, prompt_version, prompt_tokens,
                                 completion_tokens, cost_usd, latency_s, ok, note)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (_iso(dt.datetime.now()), run_id, stage, model, prompt_version, prompt_tokens,
             completion_tokens, cost_usd, latency_s, 1 if ok else 0, note),
        )
        self.conn.commit()

    def cost_summary(self, run_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            """SELECT COUNT(*) n, COALESCE(SUM(prompt_tokens),0) pt,
                      COALESCE(SUM(completion_tokens),0) ct, COALESCE(SUM(cost_usd),0) cost,
                      COALESCE(AVG(latency_s),0) lat,
                      COALESCE(SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END),0) failed
               FROM runs WHERE run_id=?""", (run_id,)).fetchone()
        return {"calls": row["n"], "prompt_tokens": row["pt"], "completion_tokens": row["ct"],
                "cost_usd": round(row["cost"], 6), "avg_latency_s": round(row["lat"], 3),
                "failed_calls": row["failed"]}

    # ---------- eval_runs ----------
    def log_eval(self, run_id: str, config_name: str, split: str, n_samples: int, model: str,
                 metrics: dict[str, Any], config_snapshot: dict[str, Any],
                 prompt_versions: dict[str, str], results_path: str = "") -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO eval_runs
               (ts, run_id, config_name, split, n_samples, model, metrics,
                config_snapshot, prompt_versions, results_path)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (_iso(dt.datetime.now()), run_id, config_name, split, n_samples, model,
             json.dumps(metrics, ensure_ascii=False),
             json.dumps(config_snapshot, ensure_ascii=False, default=str),
             json.dumps(prompt_versions, ensure_ascii=False), results_path),
        )
        self.conn.commit()

    def list_evals(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT ts, run_id, config_name, split, n_samples, model FROM eval_runs "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ---------- 小工具 ----------
def _iso(v: dt.datetime | None) -> str | None:
    return v.isoformat() if isinstance(v, dt.datetime) else None


def _parse_dt(v: str | None) -> dt.datetime | None:
    if not v:
        return None
    try:
        return dt.datetime.fromisoformat(v)
    except ValueError:
        return None


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]
