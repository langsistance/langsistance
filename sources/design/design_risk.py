"""design_risk 纯函数域: term 推导 / 三档聚合 / 报告 / digest。

对应 spec: docs/superpowers/specs/2026-09-06-us-design-clearance-design.md §5.6/§6。
- term 规则: grant_date >= 2015-05-13 → +15 年; 早于 → +14 年; is_expired 用"今天"比较。
- 仅依赖标准库, 无 IO; 各函数可独立注入日期(断言与 TDD 的确定性)。
- design P1 T1, 后续 T2-T7 依赖本模块 dataclass 与签名。
"""
from dataclasses import dataclass, field
from datetime import date, datetime

_TERM_CUTOFF = date(2015, 5, 13)

# 报告/摘要长度口径 (spec §5.6/§6)
_DIGEST_LIMIT = 50


@dataclass(frozen=True)
class DesignCandidate:
    pub: str            # "USD504889S1"
    title: str
    grant_date: str     # "YYYY-MM-DD"
    assignee: str = ""
    status: str = "active"   # "active" | "expired"


@dataclass(frozen=True)
class JudgeVerdict:
    d_number: str
    score: float              # 0..1
    risk: str                 # "high" | "medium" | "low"
    dims: tuple = field(default_factory=tuple)   # ((name, score, note), ...)
    basis: str = ""
    difference: str = ""


def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _add_years(d: date, n: int) -> date:
    """加 n 年; 目标年 2/29 不存在时顺延为 2/28 (term 计算口径, 见测试)。"""
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


def effective_until(grant_date: str) -> str:
    """按 grant_date 推出届满日 (15y/14y 分界 = 2015-05-13)。"""
    g = _parse(grant_date)
    return _add_years(g, 15 if g >= _TERM_CUTOFF else 14).isoformat()


def is_expired(grant_date: str, today: str) -> bool:
    """届满当日(含)仍有效; 早于今日才过期。"""
    return effective_until(grant_date) < today


def risk_of_score(score: float) -> str:
    """按得分定档: >=0.7 high; >=0.45 medium; 其余 low。"""
    if score >= 0.7:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _dedup_highest(verdicts: list[JudgeVerdict]) -> list[JudgeVerdict]:
    """同 d_number 去重, 保留 score 最高者 (入参顺序即稳定序)。"""
    best: dict[str, JudgeVerdict] = {}
    for v in verdicts:
        prev = best.get(v.d_number)
        if prev is None or v.score > prev.score:
            best[v.d_number] = v
    return list(best.values())


def aggregate(
    active: list[DesignCandidate],
    verdicts: list[JudgeVerdict],
    expired: list[DesignCandidate],
    today: str,
) -> dict:
    """聚合: 按 score 分三档 (不信任 model risk 字段作分组依据, 但保留展示)。

    - 去重: 同 d_number 取最高 score。
    - expired_hits: 检索命中但已期满件, 单独提示, 不进判定。expired 由调用方在 L1 过滤
      (active 参数为判定过的活件)。
    - 返回 dict 承载分组 + 上下文 (候选/有效期口径), 供 build_report_md / build_digest 消费。
    """
    bins = {"high": [], "medium": [], "low": []}
    for v in _dedup_highest(verdicts):
        bins[risk_of_score(v.score)].append(v)
    return {
        "high": bins["high"],
        "medium": bins["medium"],
        "low": bins["low"],
        "expired_hits": expired,
        "active": active,
        "verdicts": verdicts,
        "today": today,
    }


def _candidate_by_dnumber(agg: dict) -> dict[str, DesignCandidate]:
    by_pub = {c.pub: c for c in agg["active"]}
    # 兼容: verdict.d_number 可能是 pub 本身 (USD...) 或 D 号前缀丢 S1。
    return by_pub


def _render_dim(note: str, score: float) -> str:
    return f"{note} (score={round(score, 2)})" if note else f"(score={round(score, 2)})"


def build_report_md(target_label: str, agg: dict) -> str:
    """固定报告模板: 总览 → 高危逐件 → 相关但已失效 → 判定说明与免责声明。"""
    caps = _candidate_by_dnumber(agg)
    high, medium, low = agg["high"], agg["medium"], agg["low"]
    expired: list = agg["expired_hits"]
    rows = []

    rows.append("# US 外观询检报告")
    rows.append("")
    rows.append(f"目标: {target_label}")
    rows.append(f"数据截至 {agg['today']}")
    rows.append("")
    rows.append("## 总览")
    rows.append("")
    rows.append(f"- 高危: {len(high)} 件")
    rows.append(f"- 中等: {len(medium)} 件")
    rows.append(f"- 低危: {len(low)} 件")
    if expired:
        rows.append(f"- 相关但已失效: {len(expired)} 件")
    rows.append("")

    rows.append("## 高危在先设计")
    rows.append("")
    if not high:
        rows.append("无。")
        rows.append("")
    for v in high:
        c = caps.get(v.d_number)
        d_label = v.d_number
        title = c.title if c else ""
        assignee = f"，权利人: {c.assignee}" if (c and c.assignee) else ""
        rows.append(f"### {d_label}{' — ' + title if title else ''}")
        rows.append("")
        if assignee:
            rows.append(f"{assignee.lstrip('，')}")
            rows.append("")
        rows.append(f"- 风险档位: {v.risk} (score={round(v.score, 2)})")
        rows.append(f"- 命中维度: {len(v.dims)} 项")
        for dim_name, dim_score, dim_note in v.dims:
            rows.append(
                f"  - {dim_name}: {_render_dim(dim_note, dim_score)}")
        if v.basis:
            rows.append(f"- 判定依据: {v.basis}")
        if v.difference:
            rows.append(f"- 关键差异: {v.difference}")
        rows.append("")

    rows.append("## 相关但已失效")
    rows.append("")
    if not expired:
        rows.append("无。")
        rows.append("")
    for c in expired:
        rows.append(f"- {c.pub} '{c.title}' (届满 {c.status})")
    rows.append("")

    rows.append("## 判定说明")
    rows.append("")
    rows.append(
        "本报告由多模态模型比对生成, 仅作选品风险参考, 不构成法律意见。"
        "高危仅代表在先权利外观高度相似, 是否构成侵权需持证代理人结合权利要求与"
        "具体产品评估。")
    rows.append("非法律意见。")
    return "\n".join(rows)


def build_digest(target_label: str, agg: dict) -> dict:
    """digest: 目标摘要 + 高危 D 号清单 (≤50), 供后端锚点写会话。"""
    high = [v.d_number for v in agg["high"]]
    return {
        "target": target_label,
        "type": "file",
        "result_ids": high[: _DIGEST_LIMIT],
        "totals": {
            "high": len(high),
            "medium": len(agg["medium"]),
            "low": len(agg["low"]),
        },
        "expired_count": len(agg["expired_hits"]),
    }
