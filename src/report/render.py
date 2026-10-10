"""核查报告 → Markdown（两个滤网的结果、证据链、建议档位、发给爸妈的话）。读者是用小程序的子女（D51）。"""
from __future__ import annotations

from ..models import CheckReport, Evidence

PRIORITY_HINT = {
    "尽快劝阻": "内容不实，还在推人停药、花钱、买东西或转发——建议尽快把下面这段话发给爸妈",
    "提醒留意": "内容有争议或证据不足，或者内容属实但带有诱导话术——建议提醒爸妈先别急着照做",
    "告知误区": "内容不实，但没有推人行动的话术——有空跟爸妈说一声就行",
    "无需处理": "没发现明显问题",
}


def split_evidence(evs: list[Evidence], used: list[int], fallback: int = 3) -> tuple[list[tuple[int, Evidence]], int]:
    """报告只列判定时实际用到的证据（D52）；返回 [(原编号, 证据)] 和未列出的条数。

    检索结果里常混着无关片段（如标题党文章把检索带偏），全列出来子女会以为系统在凑数。
    判定没标编号时退回前几条——与「发给爸妈的话」取证据的规则一致（rebuttal.evidence_for_rebuttal）。
    """
    picked = [(i, evs[i - 1]) for i in dict.fromkeys(used) if 1 <= i <= len(evs)]
    if not picked:
        picked = list(enumerate(evs[:fallback], 1))
    return picked, len(evs) - len(picked)


def to_markdown(r: CheckReport) -> str:
    v = r.verdicts[0] if r.verdicts else None
    d = r.decision
    b = r.behavior
    lines = [f"# 核查报告：{r.title or '（无标题）'}", ""]

    if d:
        lines += [f"**建议：{d.priority}**　{PRIORITY_HINT.get(d.priority, '')}", ""]
    lines += ["| 项目 | 结果 |", "|---|---|"]
    if v:
        lines.append(f"| 真伪判定（认知面） | {v.label}（置信度 {v.confidence:.2f}） |")
    if b:
        lines.append(f"| 行为风险（行为面） | {b.risk_level} |")
    lines += ["", "## 判定理由", "", (v.reasoning if v else "") or "（无）", ""]

    lines += ["## 风险话术", ""]
    if b and b.behaviors:
        for h in b.behaviors:
            lines.append(f"- **{h.type}**：「{h.evidence}」")
    else:
        lines.append("未发现推人转发、停药、购买、付费或吓唬人的话术。")
    lines.append("")

    lines += ["## 证据链", ""]
    shown, rest = split_evidence(v.evidence, v.used_evidence) if v else ([], 0)
    if shown:
        if not (v and v.used_evidence):
            lines += ["（判断时没有标注具体用了哪几条，以下是检索到的前几条。）", ""]
        for i, e in shown:                       # 保留原编号，和判定理由里的「证据[4]」对得上
            src = e.source_name or e.domain
            link = f"[{e.title or src}]({e.url})" if e.url else (e.title or src)
            lines.append(f"{i}. **{src}** · {link}")
            if e.snippet:
                lines.append(f"   > {e.snippet[:200].replace(chr(10), ' ')}…")
        if rest:
            lines += ["", f"另检索到 {rest} 条参考资料，判断时未采用（完整记录见同名 JSON 文件）。"]
    else:
        lines.append("没有可用的权威证据。")
    lines.append("")

    lines += ["## 发给爸妈的话", ""]
    if r.rebuttal:
        lines += [r.rebuttal, ""]
        if r.rebuttal_sources:
            lines += [f"依据：{'、'.join(r.rebuttal_sources)}", ""]
    else:
        lines += ["（这次没能生成，见下方运行说明。）", ""]

    if r.notes:
        lines += ["## 运行说明", ""] + [f"- {n}" for n in r.notes] + [""]
    lines += ["---", f"耗时 {r.latency_s} 秒。本报告由系统自动生成，仅转述权威机构结论，不构成医疗建议。"]
    return "\n".join(lines) + "\n"
