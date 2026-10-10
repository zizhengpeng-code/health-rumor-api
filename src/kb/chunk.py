"""把整篇文章切成块（chunk），供 BM25 与向量两路检索共用。

**为什么要切块**：整篇文章作为一个检索单元太粗——一篇 3000 字的辟谣文章里可能只有
一段跟当前主张相关，整篇命中会把无关内容当成证据喂给下游，稀释判断。
切块之后每条证据能精确到段落，核对时也容易定位。

切分策略（按 config 的 `kb.chunk` 配置）：
- 按段落（换行）切，尽量保持语义完整
- 块长落在 [min_chars, max_chars]，超长的单段再按句子切
- 相邻块重叠 overlap 字，避免答案正好被切在边界上
- 每块带上标题/来源/链接/发布日期/tier，供证据溯源

chunk_id 形如 `12-c3`（文档 12 的第 3 块）。切分参数变了 chunk_id 就会变——
所以切分参数必须进快照指纹，否则"同一个 chunk_id"指向的内容会变，复现就断了。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# 句末标点：单段过长时按它断开
_SENT_END = re.compile(r"(?<=[。！？；!?;])")
# 连续空白/换行归一
_BLANK = re.compile(r"\n{2,}")
_WS = re.compile(r"[ \t\u3000]+")

# 二进制/非正文内容特征。实测库里 53/963（5.5%）是 PDF 二进制被当成正文存了，
# 这些内容进 BM25 和向量库只会制造噪声——检索命中它们等于没命中。
_BINARY_MARKERS = ("%PDF-", "%PDF", "PK\x03\x04", "GIF89a", "\x89PNG", "<?xml")
_BINARY_SAMPLE = 400


def is_binary_text(text: str) -> bool:
    """判断正文是不是二进制垃圾（PDF/压缩包/图片被当正文存了）。

    两条判据，命中任一即判为二进制：
    1. 开头出现已知文件魔数（PDF/ZIP/GIF/PNG/XML 声明）
    2. 前 400 字里不可打印控制字符占比过高（正常中文正文不会这样）
    """
    if not text:
        return True
    head = text[:_BINARY_SAMPLE]
    if head.lstrip().startswith(_BINARY_MARKERS):
        return True
    ctrl = sum(1 for ch in head if ord(ch) < 32 and ch not in "\n\r\t")
    return ctrl / max(1, len(head)) > 0.05


def is_usable_article(text: str, min_chars: int = 200) -> tuple[bool, str]:
    """入库前的质检。返回 (是否可用, 原因)。原因用于记录被丢弃的文档，便于审计。"""
    if not text or not text.strip():
        return False, "空正文"
    if is_binary_text(text):
        return False, "二进制/PDF 内容"
    if len(text.strip()) < min_chars:
        return False, f"正文过短（{len(text.strip())} < {min_chars}）"
    return True, ""


@dataclass
class Chunk:
    """一个检索单元。"""

    chunk_id: str
    doc_id: int
    idx: int
    text: str
    title: str = ""
    source: str = ""            # 域名
    source_name: str = ""       # 机构名
    tier: int = 1
    url: str = ""
    published: str = ""

    @property
    def char_len(self) -> int:
        return len(self.text)

    def meta(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "idx": self.idx,
                "title": self.title, "source": self.source, "source_name": self.source_name,
                "tier": self.tier, "url": self.url, "published": self.published}


def split_paragraphs(text: str) -> list[str]:
    """按空行/换行切段，顺手归一空白。空段丢掉。"""
    text = _WS.sub(" ", (text or "").replace("\r\n", "\n").replace("\r", "\n"))
    parts = _BLANK.split(text)
    out: list[str] = []
    for p in parts:
        # 段内单换行也当分段——权威站正文常一行一句
        for line in p.split("\n"):
            line = line.strip()
            if line:
                out.append(line)
    return out


def _split_long(para: str, max_chars: int) -> list[str]:
    """单段超过 max_chars 时按句子切，再不行就硬切。"""
    out: list[str] = []
    buf = ""
    for sent in _SENT_END.split(para):
        if not sent:
            continue
        if len(buf) + len(sent) <= max_chars:
            buf += sent
            continue
        if buf:
            out.append(buf)
        # 单句本身就超长 -> 硬切
        while len(sent) > max_chars:
            out.append(sent[:max_chars])
            sent = sent[max_chars:]
        buf = sent
    if buf:
        out.append(buf)
    return out


def _cut_at_boundary(text: str, room: int) -> str:
    """在 room 字以内，尽量切在句末；找不到句末就硬切。

    用来把一段塞进"剩余空间"——直接整段塞会超上限，整段丢给下一块又会留下空洞。
    """
    if room <= 0:
        return ""
    if len(text) <= room:
        return text
    window = text[:room]
    best = 0
    for m in _SENT_END.finditer(window):
        best = m.end()
    return window[:best] if best else window


def _pack(atoms: list[str], body_max: int) -> list[str]:
    """贪心装箱：每个 body 不超过 body_max，装不下就把原子切开。

    这是块长上限的**唯一保证点**——早先的实现在"当前块还没到 min_chars"时
    会整段塞入，直接冲破上限（实测切出 796 字的块，配置写的是 500）。
    """
    bodies: list[str] = []
    cur: list[str] = []
    cur_len = 0
    pending = list(atoms)

    while pending:
        atom = pending.pop(0)
        sep = 1 if cur else 0
        if cur_len + sep + len(atom) <= body_max:
            cur.append(atom)
            cur_len += sep + len(atom)
            continue

        room = body_max - cur_len - sep
        cut = _cut_at_boundary(atom, room) if room > 0 else ""
        if cut:
            cur.append(cut)
            rest = atom[len(cut):].lstrip()
            if rest:
                pending.insert(0, rest)
            bodies.append("\n".join(cur))
            cur, cur_len = [], 0
            continue

        # 当前块已经满了：收尾，原子留到下一轮
        if cur:
            bodies.append("\n".join(cur))
            cur, cur_len = [], 0
            pending.insert(0, atom)
            continue

        # 空块仍装不下（原子比 body_max 还长）：硬切，保证不超上限
        bodies.append(atom[:body_max])
        rest = atom[body_max:].lstrip()
        if rest:
            pending.insert(0, rest)

    if cur:
        bodies.append("\n".join(cur))
    return bodies


def chunk_text(text: str, *, min_chars: int = 300, max_chars: int = 500,
               overlap: int = 50) -> list[str]:
    """把正文切成若干块（只返回文本，不涉及元数据）。

    **块长严格不超过 max_chars**：重叠部分**占用**本块预算，所以正文预算取
    `max_chars - overlap`。配置说 500 就是 500，不能因为"重叠不算"就切出 550。

    min_chars 是软下限：只在收尾时用来决定"这个短块要不要并进上一块"，
    不作为装箱时的硬约束——把 min_chars 当硬约束正是之前冲破上限的原因。
    """
    if max_chars <= 0:
        return []
    overlap = max(0, min(overlap, max_chars - 2))
    # -1 是重叠与正文之间那个换行符的位置，不能不算进预算（否则块长会超上限 1 个字）
    body_max = max(1, max_chars - overlap - 1)

    paras = split_paragraphs(text)
    if not paras:
        return []

    # 先把超长段落按句打散，保证原子单元不超过正文预算
    atoms: list[str] = []
    for p in paras:
        atoms.extend(_split_long(p, body_max) if len(p) > body_max else [p])

    bodies = _pack(atoms, body_max)

    # 收尾短块并进上一块——但要保证合并后仍不超上限
    if len(bodies) >= 2 and len(bodies[-1]) < min_chars // 2 \
            and len(bodies[-1]) + len(bodies[-2]) + 1 <= body_max:
        bodies[-2] = bodies[-2] + "\n" + bodies.pop()

    # 叠加重叠：第 i 块 = 第 i-1 块尾部 overlap 字 + 本块正文
    out: list[str] = []
    prev = ""
    for body in bodies:
        piece = (prev[-overlap:] + "\n" + body) if (prev and overlap) else body
        out.append(piece)
        prev = body
    return [c for c in out if c.strip()]


def chunk_params(cfg: dict[str, Any]) -> dict[str, int]:
    c = (cfg.get("kb", {}) or {}).get("chunk", {}) or {}
    return {
        "min_chars": int(c.get("min_chars", 300)),
        "max_chars": int(c.get("max_chars", 500)),
        "overlap": int(c.get("overlap", 50)),
    }


def params_fingerprint(cfg: dict[str, Any]) -> str:
    """切分参数指纹。参数变了，chunk_id 的含义就变了，快照必须能看出来。"""
    p = chunk_params(cfg)
    raw = f"min={p['min_chars']};max={p['max_chars']};overlap={p['overlap']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def chunk_document(doc: dict[str, Any], cfg: dict[str, Any]) -> list[Chunk]:
    """把一个 kb_docs 行切成带元数据的块。"""
    p = chunk_params(cfg)
    pieces = chunk_text(doc.get("text") or "", **p)
    doc_id = int(doc["id"])
    out: list[Chunk] = []
    for i, piece in enumerate(pieces):
        out.append(Chunk(
            chunk_id=f"{doc_id}-c{i}", doc_id=doc_id, idx=i, text=piece,
            title=doc.get("title") or "", source=doc.get("source") or "",
            source_name=doc.get("source_name") or "", tier=int(doc.get("tier") or 1),
            url=doc.get("url") or "", published=doc.get("published") or "",
        ))
    return out
