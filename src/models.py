"""数据模型（pydantic schema）。

所有模块之间的契约都在这里。任何 LLM 输出都必须先过这里的 schema 校验，
校验失败由 llm.py 带错误信息重试一次，再失败则记录错误并跳过。

命名注意：用 `import datetime as dt`，避免字段名与类型名冲突。
"""
from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 枚举（用 Literal 而非 Enum，便于直接序列化进 JSON 与 jsonl）
# ---------------------------------------------------------------------------
SourceType = Literal["article", "video", "paste"]
BehaviorType = Literal["转发诱导", "医疗行为建议", "消费导流", "金钱相关", "恐惧诉求"]
RiskLevel = Literal["高", "中", "低"]
CognitiveLabel = Literal["真", "假", "部分属实", "无法核查"]
Priority = Literal["尽快劝阻", "提醒留意", "告知误区", "无需处理"]
Topic = Literal["慢病", "用药", "饮食", "癌症", "养生器械", "其他"]

BEHAVIOR_TYPES: tuple[str, ...] = (
    "转发诱导", "医疗行为建议", "消费导流", "金钱相关", "恐惧诉求",
)
RISK_LEVELS: tuple[str, ...] = ("高", "中", "低")
COGNITIVE_LABELS: tuple[str, ...] = ("真", "假", "部分属实", "无法核查")
PRIORITIES: tuple[str, ...] = ("尽快劝阻", "提醒留意", "告知误区", "无需处理")
TOPICS: tuple[str, ...] = ("慢病", "用药", "饮食", "癌症", "养生器械", "其他")

# 认知面严重度：数字越大越严重，用于 risk_matrix 取"最严重的一条主张"
SEVERITY: dict[str, int] = {"假": 3, "部分属实": 2, "无法核查": 1, "真": 0}


# ---------------------------------------------------------------------------
# 0. ingest
# ---------------------------------------------------------------------------
class Document(BaseModel):
    """统一输入文档。文章、视频转写、手工粘贴正文都归一到这个结构。"""

    id: str
    source_type: SourceType = "article"
    title: str = ""
    account: str = ""            # 公众号名称 / 视频作者
    text: str                    # 正文（或转写文本）
    url: str = ""
    published_at: dt.datetime | None = None
    fetched_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now())
    topic_hint: str = ""         # 抓取阶段可给的主题提示，非必填

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("正文不能为空")
        return v

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# 1. behavior_screen（行为面滤网）
# ---------------------------------------------------------------------------
class BehaviorHit(BaseModel):
    """一条行为面命中。evidence 必须是原文片段，便于核对。"""

    type: BehaviorType
    evidence: str = Field(min_length=1, description="原文片段")
    confidence: float = Field(ge=0, le=1)


class BehaviorScreenOut(BaseModel):
    """behavior_screen 的**模型输出契约**：只有命中列表，没有风险等级。

    刻意不让模型输出 risk_level——等级由 `behavior.rules` 按 config 的规则表算。
    理由：一旦模型也能给等级，就有了两个真值来源，二者不一致时无从裁决；
    而且等级规则要能在**不重新调用模型**的前提下调整（换阈值、做消融）。
    模型负责"看到什么"，规则负责"算成多严重"，职责不重叠。
    """

    behaviors: list[BehaviorHit] = Field(default_factory=list)


class BehaviorResult(BaseModel):
    risk_level: RiskLevel
    behaviors: list[BehaviorHit] = Field(default_factory=list)

    @property
    def hit_types(self) -> list[str]:
        """按 BEHAVIOR_TYPES 的规范顺序去重排序。

        不用 sorted()——中文按 Unicode 码点排（恐 < 转）不符合直觉，
        固定成 config 里的声明顺序，评测与报告输出才稳定可 diff。
        """
        order = {t: i for i, t in enumerate(BEHAVIOR_TYPES)}
        return sorted({b.type for b in self.behaviors}, key=lambda t: order.get(t, 99))

    def to_eval(self) -> list[str]:
        """给评测用的行为类型列表。"""
        return self.hit_types


# ---------------------------------------------------------------------------
# 2~6. 认知面
# ---------------------------------------------------------------------------
class Claim(BaseModel):
    """可核查的事实性主张。"""

    id: int = Field(ge=0)
    text: str
    importance: int = Field(default=3, ge=1, le=5, description="1~5，越大越重要")
    checkable: bool = True


class FilteredClaim(BaseModel):
    """被过滤掉的非事实性内容，记录原因以便审计。"""

    text: str
    reason: str


class ClaimExtractResult(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    filtered: list[FilteredClaim] = Field(default_factory=list)


class QueryRewrite(BaseModel):
    """查询改写结果：1~3 个短关键词，按优先级从高到低。"""

    keywords: list[str] = Field(default_factory=list)

    @field_validator("keywords")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        out = [k.strip() for k in v if k and k.strip()]
        return out[:3]


class BaselineVerdict(BaseModel):
    """M2 baseline：单个 LLM 直接判断文章整体的事实性。

    刻意保持最小——baseline 是消融实验的参照基准，多一个字段就多一分"加料"嫌疑。
    """

    label: CognitiveLabel
    confidence: float = Field(ge=0, le=1, default=0.5)
    reasoning: str = ""


class Rebuttal(BaseModel):
    """辟谣稿。面向中老年读者，可直接转发到家庭群。"""

    text: str = ""
    sources: list[str] = Field(default_factory=list)


# 结尾的固定提示占 9 字，模型正文上限留出余量，拼上后总长仍 ≤ 300（CLAUDE.md 第 8 节）
REBUTTAL_ENDING = "具体用药请遵医嘱。"
REBUTTAL_BODY_MAX = 290


class RebuttalOut(BaseModel):
    """rebuttal 的模型输出契约。超长由 schema 拦下，llm.py 会带着错误信息让模型重写一次。"""

    text: str = Field(min_length=20)
    sources: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _length(cls, v: str) -> str:
        v = v.strip()
        if len(v) > REBUTTAL_BODY_MAX:
            raise ValueError(f"辟谣稿共 {len(v)} 字，超过 {REBUTTAL_BODY_MAX} 字上限，请压缩后重写")
        return v


class CheckReportResult(BaseModel):
    """核查报告生成的 LLM 输出（结构化部分）。"""

    summary: str = ""
    priority_reason: str = ""


class Evidence(BaseModel):
    """一条证据。必须带来源名称、链接、原文片段。"""

    id: int = Field(ge=0, description="证据编号，debate 中专家引用此编号")
    claim_id: int
    title: str
    url: str
    source_name: str = ""        # 来源机构名，如"中国互联网联合辟谣平台"
    domain: str = ""
    snippet: str = ""            # 原文片段
    published_at: dt.datetime | None = None
    tier: int = Field(default=4, ge=1, le=4, description="1 权威 ~ 4 自媒体")
    score: float = Field(default=0.0, ge=0, le=1, description="source_score 给的可信度分")
    # 证据来自哪一层检索通道。kb 可复现；site/websearch 是活检索，不可复现，
    # 评测时要单独统计占比——把不可复现的部分量化出来，而不是假装它不存在。
    channel: Literal["kb", "site", "websearch", "manual", ""] = ""
    # --- 块级检索（M4）---
    # 证据粒度从"整篇"下沉到"块"：整篇命中会把无关段落一起当证据喂给下游。
    chunk_id: str = ""           # 形如 "12-c3"，可回溯到具体段落
    relevance: float = 0.0       # 检索相关度分（BM25 / 余弦 / 重排分，量纲随模式变）
    matched_by: str = ""         # bm25 | vector | both —— 哪一路召回的，用于归因


class EvidenceVerdict(BaseModel):
    """evidence_judge 的模型输出契约：结合证据给出的事实性判定。

    `used_evidence` 是**刻意保留**的字段：它让"模型到底用了哪几条证据"可追溯。
    没有它，模型给了一个结论却指不出依据，核对时就只能重读全部证据。
    """

    label: CognitiveLabel
    confidence: float = Field(ge=0, le=1, default=0.5)
    reasoning: str = ""
    used_evidence: list[int] = Field(default_factory=list)


class ExpertOpinion(BaseModel):
    """一位专家在某一轮的发声。"""

    round: int = Field(ge=1)
    expert: str                  # 临床医学专家 / 药学专家 / ...
    claim_id: int
    stance: Literal["真", "假", "部分属实", "无法核查", "未定"]
    argument: str
    cited_evidence: list[int] = Field(default_factory=list)
    speculation: bool = False    # 无证据引用的论断标为推测


class DebateResult(BaseModel):
    claim_id: int
    opinions: list[ExpertOpinion] = Field(default_factory=list)
    rounds_used: int = 0
    terminated_by: Literal["consensus", "exhausted", "deadlock", "round_limit"] = "round_limit"
    verdict: CognitiveLabel = "无法核查"
    consensus: bool = False
    dissent: str = ""            # 未达成共识时如实呈现分歧


# --- M5 v2：按任务分工的辩论（D36）---
# 分工靠 schema 强制：每个角色的输出契约里**只有它该填的字段**。
# 模型多写了别的（比如信源核查专家顺手给了个结论），pydantic 会直接丢掉——想抢活也抢不了。
EvidenceStance = Literal["支持", "反驳", "中立"]


class EvidenceAssessment(BaseModel):
    """信源核查专家对一条证据的判断。编号与提示词里的 [n] 一致（1 基）。"""

    id: int = Field(ge=1)
    relevant: bool = Field(description="是否与文章核心说法直接相关")
    stance: EvidenceStance = Field(default="中立", description="相对文章核心说法的立场")
    out_of_context: bool = Field(default=False, description="是否被断章取义/答非所问")
    note: str = ""


class SourceCheckOut(BaseModel):
    """信源核查专家的输出契约：只评证据，**没有结论字段**。"""

    assessments: list[EvidenceAssessment] = Field(default_factory=list)


class AdvocatePoint(BaseModel):
    point: str
    cited_evidence: list[int] = Field(default_factory=list)


class AdvocateOut(BaseModel):
    """正方/反方的输出契约：只有论点与引用，**没有结论字段**。

    `concede=true` 表示在自己手上的证据里找不到站得住的论据，主动认输——
    这是辩论提前结束（达成共识）的信号。
    """

    points: list[AdvocatePoint] = Field(default_factory=list)
    rebuttal: str = ""
    concede: bool = False


class JudgeOut(BaseModel):
    """裁判（主持人）的输出契约：唯一有权下结论的角色。"""

    label: CognitiveLabel
    confidence: float = Field(ge=0, le=1, default=0.5)
    reasoning: str = ""
    decisive_evidence: list[int] = Field(default_factory=list)
    # D45：每轮结束后裁判决定继续辩论还是结案（Liang 等的判别模式）。最后一轮由代码强制结案。
    continue_debate: bool = False


class ReviewResult(BaseModel):
    """独立审核智能体的检查结果。"""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    revision_notes: str = ""


class CognitiveVerdict(BaseModel):
    """一条主张的最终认知面判定。"""

    claim_id: int
    claim_text: str
    label: CognitiveLabel
    confidence: float = Field(ge=0, le=1, default=0.5)
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    used_evidence: list[int] = Field(default_factory=list)   # 判定时实际用到的证据（1 基编号），报告只列这些
    debate: DebateResult | None = None
    review: ReviewResult | None = None
    needs_human: bool = False     # 打回超限 / 未达共识


# ---------------------------------------------------------------------------
# 7. risk_matrix
# ---------------------------------------------------------------------------
class RiskDecision(BaseModel):
    priority: Priority
    cognitive_label: CognitiveLabel      # 取最严重主张
    behavior_level: RiskLevel
    reason: str = ""


# ---------------------------------------------------------------------------
# 8. report
# ---------------------------------------------------------------------------
class CheckReport(BaseModel):
    """完整核查报告。"""

    doc_id: str
    title: str = ""
    generated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now())

    behavior: BehaviorResult | None = None
    verdicts: list[CognitiveVerdict] = Field(default_factory=list)
    decision: RiskDecision | None = None

    rebuttal: str = ""                    # 辟谣稿，≤300 字；为空表示无需辟谣（真 × 低风险）
    rebuttal_sources: list[str] = Field(default_factory=list)
    config_name: str = ""                 # 消融配置名，便于复现
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    debate_trace: dict = Field(default_factory=dict)   # 快判/辩论全过程，供界面展示讨论记录
    notes: list[str] = Field(default_factory=list)     # 降级与失败记录（见报告 2.5 异常处理）
    latency_s: float = 0.0

    @property
    def worst_label(self) -> CognitiveLabel:
        if not self.verdicts:
            return "无法核查"
        return max(self.verdicts, key=lambda v: SEVERITY[v.label]).label


# ---------------------------------------------------------------------------
# 评测用样本
# ---------------------------------------------------------------------------
class LabeledSample(BaseModel):
    """data/splits/*.jsonl 的一行。标签由队员人工标注。"""

    id: str
    title: str = ""
    text: str
    topic: Topic = "其他"
    cognitive_label: CognitiveLabel
    behaviors: list[BehaviorType] = Field(default_factory=list)
    behavior_level: RiskLevel | None = None   # 若为空则由 behaviors 规则推导
    source_url: str = ""
    notes: str = ""


class Prediction(BaseModel):
    """评测时对一条样本的预测。"""

    id: str
    cognitive_label: CognitiveLabel
    behaviors: list[str] = Field(default_factory=list)
    behavior_level: RiskLevel = "低"
    priority: Priority = "无需处理"
    confidence: float = 0.0     # 认知面置信度：定「何时信模型、何时检索」的门槛要用（D38 后续）
    reasoning: str = ""         # 判定理由：判错时不用花钱重放就能看到原因
    cost_usd: float = 0.0
    latency_s: float = 0.0
    tokens: int = 0
    note: str = ""              # 失败原因等，便于排查，不参与打分
