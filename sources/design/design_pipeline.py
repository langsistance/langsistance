"""design_pipeline 编排层: L0→L1→L2→L3→risk→回执 (design P1 T7)。

对应 spec: docs/superpowers/specs/2026-09-06-us-design-clearance-design.md §7。
run(params, ctx) -> {"digest": ..., "report_md": str}; 顺序编排 + ctx.progress/warning
进度事件; 各环 try/except 按 spec fail-open 降级, 不中断整单。
- L1: country_param="" (US 结果侧过滤已内置); search 命中件按授权届满拆 active/expired,
  is_expired 以 design_risk.is_expired(grant_date, today=utc) 实判。
- L3: 缺在近候选 verdict 者补一轮重试; 仍缺 → 报告注明未比对件数。
- 澄清 → DesignNeedClarification; L1 限流 → DesignRuntimeError (executor 映射失败消息)。
- 各阶段经模块引用 sources.design.* 整模块可 mock; 测试禁真实网络。
L1 的 fetch (design_search.search_designs / design_image.fetch_design_pdf) 首参即本模块
_google_fetch 信号位——真实 Google 池化客户端由 executor (wire #4) 覆写注入,P1 未接线
绝不默认外呼。PDF 腿 (wire #6) 的 patentimages 直链下载经同款模块级 _pdf_fetch 信号位
注入 real-asis 客户端; 未由 executor 覆写时该占位抛错 (绝不落入 design_image 的同步
httpx 默认腿), 测试整模块 mock 永不到达。
"""
import base64
import datetime
import logging
import os

from sources.design import (design_image, design_judge, design_query,
                            design_risk, design_search, design_vision)

logger = logging.getLogger("design_pipeline")

# 通用零命中契约文案 (禁裸"未找到"): 含"未检出"+换词/上更清晰图引导。
ZERO_HITS_TEXT = (
    "未检出与目标外观高度相近的在先设计。"
    "可更换更贴切的关键词, 或上传更清晰、单一角度的产品图后重试。"
)


class DesignNeedClarification(Exception):
    """澄清信号: 目标不唯一或无法从输入判别单一产品 (executor 转 SSE 澄清)。"""


class DesignRuntimeError(Exception):
    """运行期硬错误 (现仅 L1 限流 → "服务暂不可用请稍后重试")。"""


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


async def _google_fetch(resource):
    """design_image.fetch_design_pdf / search 约定之 fetch 契约: (status, body)。

    P1 编排不直接外呼; 真实池化客户端由 executor 覆写(controller wiring)。本函数作为
    注入信号的纯占位, 永不被 mock 测试触达背景路径。
    """
    raise RuntimeError("google_fetch not wired")


async def _pdf_fetch(url):
    """design_image.fetch_design_pdf 之 pdf_fetch 契约: (status, bytes)。

    同步 httpx 默认腿 (design_image._fetch_pdf_http) 绝不作为编排默认 — 该腿将随真实
    外呼在离线测试跑满真实网络。executor 以同风格 async real-async patentimages 客户端
    覆写本占位; 未接线抛错 (P1 email/文本短腿不取附图也当安全跳过)。
    """
    raise RuntimeError("pdf_fetch not wired")


async def run(params: dict, ctx) -> dict:
    """编排全链: 返回 {"digest": {...}, "report_md": str}。fail-open 兜底保成功单。"""
    product_text = str(params.get("product_text") or "").strip()
    refs = list(params.get("product_image_refs") or [])
    today = _today()

    phase = "l0"
    try:
        prof = await _l0_resolve(product_text, refs, ctx)
        phase = "l1"
        cands = await _l1_search(prof, ctx)
        if not cands:
            phase = "zero"
            return _zero_result(_target(prof))
        active, expired = _partition(cands, today)
        phase = "l2"
        pdf_img = await _l2_pdf_images(active, ctx)
        phase = "l3"
        verdicts = await _l3_judge(refs, pdf_img, active, ctx)
        return _build_result(active, expired, verdicts, today, _target(prof))
    except DesignRuntimeError:
        raise
    except DesignNeedClarification:
        raise
    except Exception as exc:  # noqa: BLE001 —— 编排兜底防败单
        logger.warning("design_pipeline %s degrade: %s", phase, exc)
        ctx.warning("检索环节出现异常, 已降级处理, 请稍后重试。")
        label = (_target({}) or product_text or "")[:80]
        return _build_result([], [], [], _today(), label, degraded=True)


def _target(prof: dict) -> str:
    return str((prof or {}).get("en_name") or "").strip()[:120]


async def _l0_resolve(product_text: str, image_refs, ctx) -> dict:
    """L0: 目标产品画像。有产品图时以视觉 L0 为准; 无图走文本纯函数。

    有图 + 有文本: 用户文本常是请求语("帮我看…有没有外观专利风险")而非
    产品描述——整句当 en_name 检索只会零命中 (2026-09-07 线上 1.6s 空检)。
    因此文本仅作参考并入视觉提示(含英文品名时提示模型复用), 产品画像以
    图为准; 视觉拿不到 (clarify/失败) 且文本首段确像英文品名 (ASCII 字母
    过半) 才兜底文本, 否则澄清。无图: 维持文本解析路径不变。
    """
    product_images = _to_product_images(image_refs)
    if product_images:
        prof = await _vision_resolve(product_images, product_text)  # None=视觉失败
        if (prof is None
                or prof.get("needs_clarification")
                or not prof.get("en_name")):
            prof = _english_text_profile(product_text) or prof
    else:
        prof = design_query.l0_product_from_text(product_text)
        if prof.get("needs_clarification") or not prof.get("en_name"):
            prof = design_query.parse_l0_json(product_text or "")
    if (prof is None
            or prof.get("needs_clarification")
            or not prof.get("en_name")):
        ctx.warning("请补充说明单一目标产品(名称/SKU)后再检。")
        raise DesignNeedClarification(
            "需澄清目标产品: 请说明是单一产品并上传其产品图。")
    return prof


def _english_text_profile(product_text: str) -> dict | None:
    """文本兜底画像: 仅当文本首段像英文产品名(ASCII 字母过半)才可作 L1
    检索词; 中文请求语整句当产品名只会产出垃圾检索。返回 None 表示不可用。
    """
    if not (product_text or "").strip():
        return None
    prof = design_query.l0_product_from_text(product_text)
    name = str(prof.get("en_name") or "").strip()
    if not name:
        return None
    letters = sum(1 for ch in name if ch.isascii() and ch.isalpha())
    if letters < len(name) * 0.5:
        return None
    return prof


async def _vision_resolve(product_images, product_text: str = "") -> dict:
    """视觉 L0: call_vision 对产品图取 L0 JSON → parse 容错。

    用户文本作为参考并入提示(若含英文品名提示复用; 若是请求语则忽略)。
    call_vision 异常 → DesignVisionError → 澄清引导 (非硬失败)。
    parse_l0_json 同文本解析容错 (围栏/缺键/坏 JSON → needs_clarification)。
    生产零外呼在测试禁密 —— 测试整模块 mock design_vision.call_vision。
    """
    prompt = design_query.L0_PROMPT_ZH
    if (product_text or "").strip():
        prompt += (
            "\n用户附带文本(仅参考: 若其中给出该产品的英文名或品名请复用,"
            " 若是泛指请求语则忽略): " + product_text.strip()[:120])
    try:
        raw = await design_vision.call_vision(product_images, prompt)
    except design_vision.DesignVisionError as exc:
        logger.warning("design_pipeline L0 vision degrade: %s", exc)
        return None  # 交给 _l0_resolve: 英文品名文本兜底, 否则澄清
    return design_query.parse_l0_json(raw)


async def _l1_search(prof: dict, ctx) -> list:
    """L1: 在先在审/授权设计检索。rate_limited → DesignRuntimeError。"""
    ctx.progress("正在检索在先美国外观设计 …")
    out = await design_search.search_designs(
        _google_fetch,
        str(prof.get("en_name") or ""),
        list(prof.get("keywords") or []),
        country_param="")   # T3: US 结果侧过滤已内置
    if out.get("rate_limited"):
        ctx.progress("WARN: 检索服务暂不可用, 请稍后重试")
        raise DesignRuntimeError("rate_limited")
    return list(out.get("candidates") or [])


def _partition(cands, today) -> tuple[list, list]:
    """命中按届满拆 active/expired; expired 独立提示不进视觉判定。"""
    active, expired = [], []
    for c in cands:
        if (c.status == "expired" or (c.grant_date and design_risk.is_expired(
                c.grant_date, today))):
            expired.append(c)
        else:
            active.append(c)
    return active, expired


async def _l2_pdf_images(active, ctx):
    """L2: 每在近取附图 PDF → 页图 base64。单件失败跳过。全败 → 文本候选降级。"""
    from sources.long_task.patent_analyzer import _pdf_to_base64_images

    ctx.progress(f"正在读取 {len(active)} 件在先设计附图…")
    out = {}
    for c in active:
        try:
            pdf = await design_image.fetch_design_pdf(
                _google_fetch, c.pub, pdf_fetch=_pdf_fetch)
            out[c.pub] = _pdf_to_base64_images(pdf) if pdf else []
        except Exception as exc:  # noqa: BLE001
            logger.info("design_pipeline L2 skip %s: %s", c.pub, exc)
            out[c.pub] = []
    if active and not any(out.values()):
        ctx.warning("附图读取失败, 判定降级为文本维度候选。")
    return out


async def _l3_judge(refs, pdf_img, active, ctx):
    """L3: 视觉分批判定; 缺 verdict 补一轮重试; 仍缺记件数。"""
    ctx.progress("正在执行视觉比对, 每批模型复核一次 …")
    product_images = _to_product_images(refs)
    verdicts = await _judge_once(product_images, pdf_img)
    covered = {v.d_number for v in verdicts}
    pending = {c.pub for c in active if c.pub not in covered and c.pub in pdf_img}
    if pending:
        sub = {k: pdf_img[k] for k in pending}
        extra = await _judge_once(product_images, sub)
        verdicts = list(verdicts) + list(extra)
        missing = {c.pub for c in active} - {v.d_number for v in verdicts}
        if missing:
            ctx.warning(f"{len(missing)} 件未完成视觉比对, 结论文本参考。")
    return verdicts


async def _judge_once(product_images, pdf_by):
    try:
        return await design_judge.judge_product(product_images, pdf_by)
    except Exception as exc:  # noqa: BLE001 —— 单轮失败不拖垮
        logger.warning("design_pipeline judge skipped: %s", exc)
        return []


def _to_product_images(image_refs) -> list:
    """本地产品路径 → base64 data URI; 缺失跳过不抛。"""
    img = []
    for ref in image_refs or []:
        uri = _file_to_data_uri(str(ref))
        if uri:
            img.append(uri)
    return img


def _file_to_data_uri(path):
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext, "image/jpeg")
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    except OSError:
        return None


def _build_result(active, expired, verdicts, today, label,
                  degraded=False) -> dict:
    """聚合 (design_risk) → report_md + digest。零命中界内跑契约文案。"""
    agg = design_risk.aggregate(active, verdicts, expired, today)
    report = design_risk.build_report_md(label or "", agg)
    if not active and not expired and not verdicts:
        report = self_zero_report(label)
    covered = {v.d_number for v in verdicts}
    missing = [c.pub for c in active if c.pub not in covered]
    if missing:
        report += f"\n\n注: {len(missing)} 件未完成视觉比对, 相关结论仅作参考。"
    if degraded:
        report += "\n\n(本次部分检索环节受限, 建议稍后重试或补充信息后复核。)"
    return {"report_md": report, "digest": design_risk.build_digest(label or "", agg)}


def self_zero_report(label) -> str:
    """纯零命中报告 (通常 L1 未命中已由 run 提前分支, 兜底双保险)。"""
    return (f"# US 外观询检报告\n\n目标: {label or ''}\n\n{ZERO_HITS_TEXT}\n\n"
            "非法律意见。")


def _zero_result(label) -> dict:
    today = _today()
    return {
        "report_md": self_zero_report(label),
        "digest": {
            "target": label,
            "result_ids": [],
            "totals": {"high": 0, "medium": 0, "low": 0},
            "zero_hits": True,
        },
    }

