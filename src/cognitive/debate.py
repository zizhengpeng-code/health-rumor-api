"""M5 v2：按任务分工的多智能体辩论（D36）。

取代 v1「一次调用让模型扮演全部专家 + 审核员一票否决」。v1 的两个问题：
1. 专家之间没有真正交互，轮次和共识都是模型一口气编的；三位专家拿同一篇文章、同一组证据，互相重复。
2. 审核打回超限就把结论改成「无法核查」——按风险矩阵「无法核查 × 行为面低 = 放行」，审核卡住反而放走谣言。

v2 的分工（参考 ChatEval：角色相同会掉分；DebateCheck：按立场分证据；
两阶段健康辟谣辩论：没把握才辩论）：

    阶段一  快判：evidence_judge 判一次，把握够高且不是「部分属实/无法核查」就直接结束
    阶段二  辩论：
      ① 信源核查专家  只评证据（相关/立场/断章取义），输出里没有结论字段
      ② 正方 / 反方    各拿不同的证据（支持+中立 / 反驳+中立），只给论点与引用
      ③ 反驳轮         互相看对方论点，可引用任何相关证据指出误读；一方认输则提前结束
      ④ 裁判（主持人） 唯一能下结论的角色；每轮结束后判断结案还是继续（D45，Liang 等的
                       判别模式），最后一轮必须结案；多次裁决时多数票，平票取更严重的标签

分工靠两层强制：schema（角色的输出契约里只有它该填的字段）+ 代码（引用编号必须在该方可用范围内，
否则剔除并标为推测——原 review 的"证据是否支撑"改成程序核对）。全程自动，结论必为四类之一。
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..config import doc_max_chars
from ..models import (
    SEVERITY, AdvocateOut, Evidence, EvidenceVerdict, JudgeOut, SourceCheckOut,
)
from ..prompts import load_prompt, render
from .evidence_judge import EvidenceJudge
from .source_score import SourceScorer, format_scored_evidence

PROMPTS = ("evidence_judge", "debate_source_check", "debate_advocate", "debate_judge")

SIDES = {
    "pro": ("正方", "论证这篇文章的核心说法成立：与证据一致、表述基本准确。"),
    "con": ("反方", "论证这篇文章的核心说法有问题：错误、夸大、以偏概全，或缺少关键限定条件。"),
}


def pick_persona(cfg: dict[str, Any], title: str, text: str) -> str:
    """按主题给正反方一个专业身份（双方同一身份，差别只在立场与证据）。"""
    experts = (cfg.get("cognitive", {}).get("debate", {}) or {}).get("experts") or {}
    body = f"{title} {text[:800]}"
    if any(k in body for k in ("药", "剂量", "输液", "抗生素", "头孢", "停药", "疫苗")):
        key = "pharmacy"
    elif any(k in body for k in ("吃", "喝", "饮食", "食物", "营养", "保健品", "维生素")):
        key = "nutrition"
    else:
        key = "clinical"
    return (experts.get(key) or {}).get("name") or {
        "pharmacy": "药学专家", "nutrition": "营养学专家", "clinical": "临床医学专家"}[key]


def split_piles(evs: list[Evidence], assessments: dict[int, dict[str, Any]]
                ) -> tuple[list[int], list[int], list[int]]:
    """按信源核查结果分证据（1 基编号）。返回 (正方可用, 反方可用, 全部相关)。

    无关或断章取义的证据两边都不给。核查漏判的证据按「相关、中立」处理——
    宁可多给，也不要因为核查少写一条就让双方都看不到。
    """
    pro, con, relevant = [], [], []
    for i in range(1, len(evs) + 1):
        a = assessments.get(i, {"relevant": True, "stance": "中立", "out_of_context": False})
        if not a.get("relevant", True) or a.get("out_of_context", False):
            continue
        relevant.append(i)
        stance = a.get("stance", "中立")
        if stance in ("支持", "中立"):
            pro.append(i)
        if stance in ("反驳", "中立"):
            con.append(i)
    return pro, con, relevant


def check_citations(out: AdvocateOut, allowed: list[int]) -> tuple[AdvocateOut, list[int]]:
    """程序化引用核对：剔除不在可用范围里的编号；论点一条有效引用都没有 → 标为推测。"""
    dropped: list[int] = []
    for p in out.points:
        keep = [c for c in p.cited_evidence if c in allowed]
        dropped += [c for c in p.cited_evidence if c not in allowed]
        p.cited_evidence = keep
    return out, dropped


def vote(labels: list[str]) -> tuple[str, float]:
    """多数票；平票取更严重的标签（漏判谣言的代价大于误报）。返回 (标签, 得票率)。"""
    cnt = Counter(labels)
    top = max(cnt.values())
    tied = [lab for lab, n in cnt.items() if n == top]
    label = max(tied, key=lambda x: SEVERITY.get(x, 0))
    return label, round(top / len(labels), 4)


def _fmt_subset(evs: list[Evidence], ids: list[int]) -> str:
    """只渲染指定编号的证据，编号保持全局一致（不重新从 1 编）。"""
    if not ids:
        return "（无可用证据）"
    # 逐条渲染再改编号：不能把整段按 "\n[" 切——证据原文里本身就可能出现 "[3]" 这类引用
    return "\n".join("[" + str(i) + "]" + format_scored_evidence([evs[i - 1]])[3:]
                     for i in ids if 1 <= i <= len(evs))


def _fmt_points(name: str, out: AdvocateOut) -> str:
    if out.concede:
        return f"{name}：认输（在可用证据中找不到站得住的论据）"
    lines = [f"{name}："]
    for p in out.points:
        cites = ",".join(str(c) for c in p.cited_evidence)
        tag = f"证据[{cites}]" if cites else "（推测，无有效引用）"
        lines.append(f"  - {p.point} {tag}")
    if out.rebuttal:
        lines.append(f"  反驳：{out.rebuttal}")
    return "\n".join(lines)


@dataclass
class DebateJudge:
    """检索 + 信源评分 + 两阶段分工式辩论。client 均可注入，便于离线测试。"""

    cfg: dict[str, Any]
    retriever: EvidenceJudge
    debate_client: Any                       # 信源核查 + 正反方
    judge_client: Any | None = None          # 裁判；None 时用 debate_client
    scorer: SourceScorer | None = None
    max_chars: int = 6000
    last_trace: dict[str, Any] = field(default_factory=dict)
    last_debate_trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: dict[str, Any], store: Any = None, run_id: str = "adhoc",
              mode: str | None = None) -> "DebateJudge":
        from ..llm import build_client

        retriever = EvidenceJudge.build(cfg, store=store, run_id=run_id, mode=mode)
        debate_client = build_client(cfg, stage="debate", store=store, run_id=run_id)
        judge_client = build_client(cfg, stage="debate", store=store, run_id=run_id)
        t = (cfg.get("cognitive", {}).get("debate", {}) or {}).get("judge_temperature")
        if t is not None:
            judge_client.temperature = float(t)       # 多次裁决取多数票时需要 >0 才有意义
        return cls(cfg=cfg, retriever=retriever, debate_client=debate_client,
                   judge_client=judge_client, scorer=SourceScorer(cfg),
                   max_chars=doc_max_chars(cfg, "cognitive.baseline"))

    # ---------- 记账与配置 ----------
    @property
    def client(self) -> Any:
        return self.debate_client

    @property
    def clients(self) -> list[Any]:
        out = [getattr(self.retriever, "client", None), self.debate_client, self.judge_client]
        seen, uniq = set(), []
        for c in out:
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                uniq.append(c)
        return uniq

    @property
    def mode(self) -> str:
        return self.retriever.mode

    @property
    def dcfg(self) -> dict[str, Any]:
        return self.cfg.get("cognitive", {}).get("debate", {}) or {}

    def _complete(self, client: Any, name: str, schema: Any, stage: str, **kw: Any) -> dict:
        system, user_tpl, version = load_prompt(name, self.cfg)
        return client.complete(render(user_tpl, **kw), render(system, **kw), schema,
                               stage=stage, prompt_version=version)

    # ---------- 阶段一：快判 ----------
    def _quick(self, text: str, title: str, evs: list[Evidence]) -> EvidenceVerdict:
        data = self._complete(
            self.retriever.client, "evidence_judge", EvidenceVerdict, "evidence_judge",
            title=title or "（无标题）", text=text[: self.max_chars],
            evidence=format_scored_evidence(evs), n_evidence=len(evs), max_chars=self.max_chars)
        return EvidenceVerdict.model_validate(data)

    def _needs_debate(self, quick: EvidenceVerdict) -> bool:
        gate = self.dcfg.get("gate") or {}
        if not gate.get("enabled", True):
            return True
        if quick.label in (gate.get("always_debate_labels") or ["部分属实", "无法核查"]):
            return True
        return quick.confidence < float(gate.get("min_confidence", 0.85))

    # ---------- 阶段二：辩论 ----------
    def _source_check(self, text: str, title: str, evs: list[Evidence]) -> dict[int, dict]:
        data = self._complete(
            self.debate_client, "debate_source_check", SourceCheckOut, "debate_source_check",
            title=title or "（无标题）", text=text[: self.max_chars],
            evidence=format_scored_evidence(evs), n_evidence=len(evs))
        out = SourceCheckOut.model_validate(data)
        return {a.id: a.model_dump() for a in out.assessments if 1 <= a.id <= len(evs)}

    def _advocate(self, side: str, persona: str, text: str, title: str, evs: list[Evidence],
                  allowed: list[int], rnd: int, opponent: AdvocateOut | None
                  ) -> tuple[AdvocateOut, list[int]]:
        name, task = SIDES[side]
        other = SIDES["con" if side == "pro" else "pro"][0]
        opp = "（第一轮，尚无对方论点）" if opponent is None else \
            "对方上一轮的论点：\n" + _fmt_points(other, opponent)
        data = self._complete(
            self.debate_client, "debate_advocate", AdvocateOut, f"debate_{side}",
            side=name, task=task, persona=persona, title=title or "（无标题）",
            text=text[: self.max_chars], evidence=_fmt_subset(evs, allowed), opponent=opp,
            round=rnd, round_name="开场陈述" if rnd == 1 else "反驳")
        out = AdvocateOut.model_validate(data)
        if self.cfg.get("_overrides", {}).get("review", True):
            return check_citations(out, allowed)
        return out, []

    def _judge_once(self, text: str, title: str, evs: list[Evidence],
                    assessments: dict[int, dict], rounds: list[dict[str, AdvocateOut]],
                    final: bool) -> JudgeOut:
        """调用一次裁判。final=False 时裁判可选择继续辩论（D45 判别模式）。"""
        cited = sorted({c for r in rounds for o in r.values() for p in o.points
                        for c in p.cited_evidence})
        sc = "\n".join(
            f"[{i}] {'相关' if a.get('relevant') else '无关'}·{a.get('stance')}"
            f"{'·断章取义' if a.get('out_of_context') else ''} {a.get('note', '')}"
            for i, a in sorted(assessments.items())) or "（无）"
        transcript = "\n\n".join(
            f"第 {k} 轮\n{_fmt_points('正方', r['pro'])}\n{_fmt_points('反方', r['con'])}"
            for k, r in enumerate(rounds, 1))
        note = ("这是最后一轮，必须结案（continue_debate 必须为 false）。" if final else
                "如果双方论据已足以判断就结案；只有关键分歧未澄清时才要求再辩一轮。")
        data = self._complete(
            self.judge_client or self.debate_client, "debate_judge", JudgeOut, "debate_judge",
            title=title or "（无标题）", text=text[: self.max_chars], source_check=sc,
            evidence=_fmt_subset(evs, cited), transcript=transcript,
            round=len(rounds), max_rounds=self._n_rounds(), final_note=note)
        out = JudgeOut.model_validate(data)
        if final:
            out.continue_debate = False        # 最后一轮由代码兜底强制结案，不依赖模型听话
        return out

    def _judge(self, text: str, title: str, evs: list[Evidence], assessments: dict[int, dict],
               rounds: list[dict[str, AdvocateOut]], first: JudgeOut | None = None
               ) -> tuple[str, float, JudgeOut, list[str]]:
        """最终裁决。judge_runs>1 时多次独立裁决取多数票；`first` 是逐轮判断中已经结案的那次裁决。"""
        runs = max(1, int(self.dcfg.get("judge_runs", 1)))
        outs = [first] if first is not None else []
        while len(outs) < runs:
            outs.append(self._judge_once(text, title, evs, assessments, rounds, final=True))
        label, share = vote([o.label for o in outs])
        best = next(o for o in outs if o.label == label)
        conf = best.confidence if runs == 1 else share
        return label, conf, best, [o.label for o in outs]

    def _n_rounds(self) -> int:
        return max(1, min(int(self.dcfg.get("rounds", 2)), int(self.dcfg.get("max_rounds", 3))))

    # ---------- 入口 ----------
    def judge(self, text: str, title: str = "") -> tuple[EvidenceVerdict, list[Evidence]]:
        evs = self.retriever.retrieve(title or text[:60])
        if self.scorer is not None:
            self.scorer.apply(evs)
        self.last_trace = dict(self.retriever.last_trace or {})
        if evs:
            self.last_trace["avg_source_score"] = round(sum(e.score for e in evs) / len(evs), 4)

        quick = self._quick(text, title, evs)
        trace: dict[str, Any] = {
            "evidence": [e.model_dump(mode="json") for e in evs],
            "quick": quick.model_dump(mode="json"),
        }
        if not self._needs_debate(quick):
            self.last_trace["debated"] = False
            trace.update(debated=False, final_label=quick.label,
                         final_confidence=quick.confidence, final_reasoning=quick.reasoning)
            self.last_debate_trace = trace
            return quick, evs

        persona = pick_persona(self.cfg, title, text)
        assessments = self._source_check(text, title, evs) if evs else {}
        pro_ids, con_ids, relevant = split_piles(evs, assessments)

        n_rounds = self._n_rounds()
        adaptive = bool(self.dcfg.get("adaptive_break", True))
        rounds: list[dict[str, AdvocateOut]] = []
        dropped: dict[str, list[int]] = {"pro": [], "con": []}
        terminated = "round_limit"
        steps: list[dict[str, Any]] = []
        closing: JudgeOut | None = None
        last: dict[str, AdvocateOut | None] = {"pro": None, "con": None}
        for rnd in range(1, n_rounds + 1):
            this: dict[str, AdvocateOut] = {}
            for side, own in (("pro", pro_ids), ("con", con_ids)):
                allowed = own if rnd == 1 else relevant       # 反驳轮可引用任何相关证据
                opp = last["con" if side == "pro" else "pro"]
                out, drop = self._advocate(side, persona, text, title, evs, allowed, rnd, opp)
                this[side] = out
                dropped[side] += drop
            rounds.append(this)
            last = dict(this)
            if this["pro"].concede or this["con"].concede:
                terminated = "consensus"
                break
            # D45 判别模式：非最后一轮结束时由裁判决定结案还是继续（最后一轮直接进入最终裁决）
            if adaptive and rnd < n_rounds:
                step = self._judge_once(text, title, evs, assessments, rounds, final=False)
                steps.append({"round": rnd, "continue": step.continue_debate, "label": step.label})
                if not step.continue_debate:
                    terminated, closing = "judge_break", step
                    break

        label, conf, best, votes = self._judge(text, title, evs, assessments, rounds, first=closing)
        verdict = EvidenceVerdict(label=label, confidence=conf,
                                  reasoning=best.reasoning[:160],
                                  used_evidence=best.decisive_evidence)
        self.last_trace["debated"] = True
        trace.update(
            debated=True, persona=persona,
            source_check=assessments, piles={"pro": pro_ids, "con": con_ids, "relevant": relevant},
            rounds=[{s: o.model_dump(mode="json") for s, o in r.items()} for r in rounds],
            dropped_citations=dropped, terminated_by=terminated, judge_votes=votes,
            judge_steps=steps,
            final_label=label, final_confidence=conf, final_reasoning=verdict.reasoning,
        )
        self.last_debate_trace = trace
        return verdict, evs


def trace_to_markdown(trace: dict[str, Any]) -> list[str]:
    """把一条辩论轨迹渲染成讨论记录（eval.save_discussions 用）。"""
    lines = ["### 阶段一：快判", ""]
    q = trace.get("quick") or {}
    lines.append(f"{q.get('label')}（置信度 {q.get('confidence')}）— {q.get('reasoning', '')}")
    if not trace.get("debated"):
        lines += ["", "把握足够，未触发辩论。"]
        return lines
    lines += ["", f"### 阶段二：辩论（正反方身份：{trace.get('persona')}）", "",
              "**① 信源核查专家**", ""]
    for i, a in sorted((trace.get("source_check") or {}).items(), key=lambda x: int(x[0])):
        flag = "相关" if a.get("relevant") else "无关"
        ooc = "·断章取义" if a.get("out_of_context") else ""
        lines.append(f"- [{i}] {flag}·{a.get('stance')}{ooc}：{a.get('note', '')}")
    piles = trace.get("piles") or {}
    lines += ["", f"分给正方：{piles.get('pro')}；分给反方：{piles.get('con')}", ""]
    for k, r in enumerate(trace.get("rounds") or [], 1):
        lines += [f"**② 第 {k} 轮**", ""]
        for side, name in (("pro", "正方"), ("con", "反方")):
            o = r.get(side) or {}
            if o.get("concede"):
                lines.append(f"- {name}：认输")
                continue
            for p in o.get("points") or []:
                cites = ",".join(str(c) for c in p.get("cited_evidence") or []) or "推测"
                lines.append(f"- {name}：{p.get('point')}（证据 {cites}）")
            if o.get("rebuttal"):
                lines.append(f"- {name}反驳：{o['rebuttal']}")
        step = next((s for s in trace.get("judge_steps") or [] if s.get("round") == k), None)
        if step:
            lines.append(f"- 裁判：{'继续辩论' if step.get('continue') else '结案'}"
                         f"（当前倾向：{step.get('label')}）")
        lines.append("")
    drop = trace.get("dropped_citations") or {}
    if any(drop.values()):
        lines += [f"引用核对剔除的越界编号：{json.dumps(drop, ensure_ascii=False)}", ""]
    lines += ["**③ 裁判**", "",
              f"投票：{trace.get('judge_votes')} → **{trace.get('final_label')}**"
              f"（置信度 {trace.get('final_confidence')}，结束原因：{trace.get('terminated_by')}）",
              f"理由：{trace.get('final_reasoning', '')}"]
    return lines
