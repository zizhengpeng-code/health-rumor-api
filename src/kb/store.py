"""本地权威知识库：可控信源、结果可复现。

为什么要有它（详见 docs/decisions.md D13）：
- 活检索（站内搜索 / 联网搜索）今天和下个月返回的东西不一样，
  消融实验的分数就没法横向比较。
- 把权威源抓下来冻结成带版本号的快照，检索才真正可复现。

中文检索用 SQLite FTS5 的 `trigram` 分词器——实测直接支持中文，
不需要引入 jieba。注意 trigram 要求查询串 **>= 3 个字符**，
更短的查询自动降级为 LIKE 匹配。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_docs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,          -- kepuchina / piyao / chinacdc / nhc
    source_name  TEXT DEFAULT '',        -- 机构中文名
    tier         INTEGER DEFAULT 1,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL UNIQUE,
    published    TEXT DEFAULT '',
    text         TEXT NOT NULL,
    seed_keyword TEXT DEFAULT '',        -- 由哪个种子关键词抓到的，便于审计覆盖度
    fetched_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kb_source ON kb_docs(source);

-- trigram 分词器：中文无需额外分词器
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
    title, text,
    content='kb_docs', content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS kb_ai AFTER INSERT ON kb_docs BEGIN
    INSERT INTO kb_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS kb_ad AFTER DELETE ON kb_docs BEGIN
    INSERT INTO kb_fts(kb_fts, rowid, title, text) VALUES ('delete', old.id, old.title, old.text);
END;
CREATE TRIGGER IF NOT EXISTS kb_au AFTER UPDATE ON kb_docs BEGIN
    INSERT INTO kb_fts(kb_fts, rowid, title, text) VALUES ('delete', old.id, old.title, old.text);
    INSERT INTO kb_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
END;

CREATE TABLE IF NOT EXISTS kb_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- 切块（M4）：整篇做检索单元太粗，一段命中会被当成整篇的证据
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id    TEXT PRIMARY KEY,        -- 形如 "12-c3"：文档 12 的第 3 块
    doc_id      INTEGER NOT NULL,
    idx         INTEGER NOT NULL,
    title       TEXT DEFAULT '',
    source      TEXT DEFAULT '',
    source_name TEXT DEFAULT '',
    tier        INTEGER DEFAULT 1,
    url         TEXT DEFAULT '',
    published   TEXT DEFAULT '',
    text        TEXT NOT NULL,
    char_len    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON kb_chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
    title, text,
    content='kb_chunks', content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS kc_ai AFTER INSERT ON kb_chunks BEGIN
    INSERT INTO kb_chunks_fts(rowid, title, text) VALUES (new.rowid, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS kc_ad AFTER DELETE ON kb_chunks BEGIN
    INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, title, text)
    VALUES ('delete', old.rowid, old.title, old.text);
END;

-- ---------------------------------------------------------------------------
-- 向量（M4）：冻结后不重复花钱；model 入库，换模型即失效重算
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb_vectors (
    chunk_id TEXT NOT NULL,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL,              -- float32 小端原始字节
    PRIMARY KEY (chunk_id, model)
);
"""

_TRIGRAM_MIN = 3          # FTS5 trigram 要求查询串至少 3 个字符
_RE_FTS_SPECIAL = re.compile(r'["\'()*:^\-]')


class KnowledgeBase:
    """权威来源本地库。快照带版本号与内容指纹，保证评测可复现。"""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "KnowledgeBase":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- 写入 ----------
    def add_document(self, *, source: str, title: str, url: str, text: str,
                     source_name: str = "", tier: int = 1, published: str = "",
                     seed_keyword: str = "") -> bool:
        """插入一篇。URL 已存在则跳过（返回 False）。"""
        if not url or not text or not title:
            return False
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO kb_docs
               (source, source_name, tier, title, url, published, text, seed_keyword, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (source, source_name, tier, title, url, published, text, seed_keyword,
             dt.datetime.now().isoformat(timespec="seconds")),
        )
        return cur.rowcount > 0

    def add_many(self, docs: Iterable[dict[str, Any]]) -> int:
        n = 0
        for d in docs:
            if self.add_document(**d):
                n += 1
        self.conn.commit()
        return n

    def commit(self) -> None:
        self.conn.commit()

    # ---------- 检索 ----------
    @staticmethod
    def _safe_query(q: str) -> str:
        """转义 FTS5 语法字符，避免用户输入里的引号/括号把查询搞崩。"""
        return _RE_FTS_SPECIAL.sub(" ", q or "").strip()

    def search(self, query: str, top_k: int = 5,
               sources: list[str] | None = None) -> list[dict[str, Any]]:
        """检索。trigram 需要 >=3 字符，更短的查询走 LIKE 降级。"""
        q = self._safe_query(query)
        if not q:
            return []
        src_clause, params = "", []
        if sources:
            src_clause = f" AND d.source IN ({','.join('?' * len(sources))})"
            params = list(sources)

        rows: list[sqlite3.Row] = []
        if len(q) >= _TRIGRAM_MIN:
            try:
                rows = self.conn.execute(
                    f"""SELECT d.*, bm25(kb_fts) AS rank
                        FROM kb_fts
                        JOIN kb_docs d ON d.id = kb_fts.rowid
                        WHERE kb_fts MATCH ?{src_clause}
                        ORDER BY rank
                        LIMIT ?""",
                    [q, *params, top_k],
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:                      # 查询串太短或 FTS 无命中 -> LIKE 兜底
            like = f"%{q}%"
            rows = self.conn.execute(
                f"""SELECT d.*, 0 AS rank FROM kb_docs d
                    WHERE (d.title LIKE ? OR d.text LIKE ?){src_clause}
                    ORDER BY LENGTH(d.text) ASC
                    LIMIT ?""",
                [like, like, *params, top_k],
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 切块 ----------
    def rebuild_chunks(self, cfg: dict[str, Any], *, drop: bool = True) -> dict[str, Any]:
        """从 kb_docs 派生切块。**不重爬**——现有文档已经抓下来了，切分是纯本地计算。

        入库前过质检（`is_usable_article`）：实测 963 篇里有 60 篇是 PDF 二进制
        或被抓成空壳，这些进 BM25 与向量库只会制造噪声。被剔除的记录写进 kb_meta，
        便于审计"到底丢了多少、为什么丢"。
        """
        from .chunk import chunk_document, is_usable_article, params_fingerprint

        if drop:
            self.conn.execute("DELETE FROM kb_chunks")
            # 切分参数变了 chunk_id 的含义就变了，旧向量必须一并作废
            self.conn.execute("DELETE FROM kb_vectors")

        docs = [dict(r) for r in self.conn.execute("SELECT * FROM kb_docs ORDER BY id")]
        skipped: list[dict[str, str]] = []
        n_chunks = 0
        batch: list[tuple] = []
        for d in docs:
            ok, why = is_usable_article(d["text"])
            if not ok:
                skipped.append({"doc_id": str(d["id"]), "url": d["url"], "reason": why})
                continue
            for ch in chunk_document(d, cfg):
                batch.append((ch.chunk_id, ch.doc_id, ch.idx, ch.title, ch.source,
                              ch.source_name, ch.tier, ch.url, ch.published,
                              ch.text, ch.char_len))
                n_chunks += 1
            if len(batch) >= 500:
                self._insert_chunks(batch)
                batch = []
        if batch:
            self._insert_chunks(batch)

        stats = {
            "docs_total": len(docs),
            "docs_used": len(docs) - len(skipped),
            "docs_skipped": len(skipped),
            "chunks": n_chunks,
            "chunk_params_fingerprint": params_fingerprint(cfg),
            "skipped": skipped[:200],
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO kb_meta(key, value) VALUES ('chunks', ?)",
            (json.dumps(stats, ensure_ascii=False),))
        self.conn.commit()
        return stats

    def _insert_chunks(self, rows: list[tuple]) -> None:
        self.conn.executemany(
            """INSERT OR REPLACE INTO kb_chunks
               (chunk_id, doc_id, idx, title, source, source_name, tier, url,
                published, text, char_len)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""", rows)

    def chunk_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) c FROM kb_chunks").fetchone()["c"])

    def chunks_for_embedding(self, model: str, limit: int | None = None) -> list[dict[str, Any]]:
        """还没算过该模型向量的块。已算过的直接跳过——向量是花钱的，别重复算。"""
        sql = """SELECT c.* FROM kb_chunks c
                 LEFT JOIN kb_vectors v ON v.chunk_id = c.chunk_id AND v.model = ?
                 WHERE v.chunk_id IS NULL ORDER BY c.chunk_id"""
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(sql, (model,))]

    def search_chunks(self, query: str, top_k: int = 20,
                      sources: list[str] | None = None) -> list[dict[str, Any]]:
        """块级 BM25 检索。trigram 要求 >=3 字符，更短走 LIKE 降级。"""
        q = self._safe_query(query)
        if not q:
            return []
        src_clause, params = "", []
        if sources:
            src_clause = f" AND c.source IN ({','.join('?' * len(sources))})"
            params = list(sources)

        rows: list[sqlite3.Row] = []
        if len(q) >= _TRIGRAM_MIN:
            try:
                rows = self.conn.execute(
                    f"""SELECT c.*, bm25(kb_chunks_fts) AS rank
                        FROM kb_chunks_fts
                        JOIN kb_chunks c ON c.rowid = kb_chunks_fts.rowid
                        WHERE kb_chunks_fts MATCH ?{src_clause}
                        ORDER BY rank LIMIT ?""",
                    [q, *params, top_k],
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            like = f"%{q}%"
            rows = self.conn.execute(
                f"""SELECT c.*, 0.0 AS rank FROM kb_chunks c
                    WHERE (c.title LIKE ? OR c.text LIKE ?){src_clause}
                    ORDER BY c.char_len ASC LIMIT ?""",
                [like, like, *params, top_k],
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 向量 ----------
    def put_vectors(self, model: str, items: list[tuple[str, Any]]) -> int:
        """写向量。items = [(chunk_id, np.ndarray)]。"""
        import numpy as np

        rows = []
        for cid, vec in items:
            arr = np.asarray(vec, dtype=np.float32)
            rows.append((cid, model, int(arr.shape[0]), arr.tobytes()))
        self.conn.executemany(
            "INSERT OR REPLACE INTO kb_vectors(chunk_id, model, dim, vec) VALUES (?,?,?,?)",
            rows)
        self.conn.commit()
        return len(rows)

    def vector_count(self, model: str) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM kb_vectors WHERE model = ?", (model,)).fetchone()["c"])

    def load_vectors(self, model: str) -> tuple[list[str], Any]:
        """载入全部向量 -> (chunk_ids, 矩阵 [n, dim] float32，已归一化)。

        规模量级（4088 块 × 1024 维 ≈ 17MB）用 numpy 暴力内积就够快，
        不值得为此引入 faiss 这类向量库依赖。
        """
        import numpy as np

        ids: list[str] = []
        vecs: list[Any] = []
        dim = 0
        for r in self.conn.execute(
                "SELECT chunk_id, dim, vec FROM kb_vectors WHERE model = ? ORDER BY chunk_id",
                (model,)):
            dim = int(r["dim"])
            ids.append(r["chunk_id"])
            vecs.append(np.frombuffer(r["vec"], dtype=np.float32))
        if not vecs:
            return [], np.zeros((0, 0), dtype=np.float32)
        mat = np.vstack(vecs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return ids, (mat / norms).astype(np.float32)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(chunk_ids), 400):
            part = chunk_ids[i:i + 400]
            q = f"SELECT * FROM kb_chunks WHERE chunk_id IN ({','.join('?' * len(part))})"
            for r in self.conn.execute(q, part):
                out[r["chunk_id"]] = dict(r)
        return out

    # ---------- 元信息 / 快照 ----------
    def doc_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) c FROM kb_docs").fetchone()["c"])

    def stats_by_source(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT source, COUNT(*) c FROM kb_docs GROUP BY source ORDER BY c DESC").fetchall()
        return {r["source"]: r["c"] for r in rows}

    def content_hash(self) -> str:
        """内容指纹：url 排序后取哈希。库变了指纹就变，用于锁定评测快照。"""
        h = hashlib.sha256()
        for r in self.conn.execute("SELECT url FROM kb_docs ORDER BY url"):
            h.update(r["url"].encode("utf-8"))
        return h.hexdigest()[:16]

    def freeze(self, version: str | None = None,
               embed_model: str = "", retrieval: dict[str, Any] | None = None) -> dict[str, Any]:
        """给当前库打快照标签，写进 kb_meta。评测时记录此版本号。

        快照必须把**影响检索结果的每一个变量**都记下来——切分参数、向量模型、
        检索模式。否则换一个参数重跑，"同一个快照"给出的证据就不一样了，
        复现论证会当场破功。
        """
        chunks_meta = self.conn.execute(
            "SELECT value FROM kb_meta WHERE key='chunks'").fetchone()
        chunks = json.loads(chunks_meta["value"]) if chunks_meta else {}
        meta = {
            "kb_version": version or dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
            "doc_count": self.doc_count(),
            "chunk_count": self.chunk_count(),
            "content_hash": self.content_hash(),
            "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
            "by_source": self.stats_by_source(),
            # --- 复现所需的全部参数 ---
            "chunk_params_fingerprint": chunks.get("chunk_params_fingerprint", ""),
            "docs_used": chunks.get("docs_used"),
            "docs_skipped": chunks.get("docs_skipped"),
            "embed_model": embed_model,
            "vector_count": self.vector_count(embed_model) if embed_model else 0,
            "retrieval": retrieval or {},
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO kb_meta(key, value) VALUES ('snapshot', ?)",
            (json.dumps(meta, ensure_ascii=False),))
        self.conn.commit()
        return meta

    def snapshot(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT value FROM kb_meta WHERE key='snapshot'").fetchone()
        return json.loads(row["value"]) if row else {}
