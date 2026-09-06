"""design_judge —— 分批判定与 schema 校验 (design P1 T6)。

规格: docs/superpowers/specs/2026-09-06-us-design-clearance-design.md §5.3/§5.5/§6;
task-6-brief.md。
产：分批 / JSON→JudgeVerdict[] (score 裁剪回 [0,1]、risk 以 risk_of_score(score) 重算覆盖,
不信任模型 risk 字段) / judge_product 逐批容错。批次 = 产品图全部视图 + 批内 2 件候选页图
→ 1 次 call_vision。消费 Task1 JudgeVerdict 与 Task5 call_vision; 页图格式即 _pdf_to_base64_images
输出 (data-uri 字符串), 本模块只拼接不产图。生产代码零产品词, 无 print。

判定口径 (写入 prompt 常量) —— 固定、通用、只指部位不指具体型号:
  ordinary observer 整体观感优先 + point of novelty 核对 + 色彩不作为独立判定维度 +
  0-1 score 三档口径 (>=0.7 高危) + 依据必须指向图中所见图部位 + 逐件给关键差异点。
"""
import json
import logging

from sources.design.design_risk import JudgeVerdict, risk_of_score
from sources.design.design_vision import call_vision

logger = logging.getLogger(__name__)

_MIN_SCORE = 0.0
_MAX_SCORE = 1.0

JUDGE_PROMPT_ZH = """你是美国外观设计专利的侵权比对审阅者。请按以下固定口径对每一在先设计专利逐件判定:
1. ordinary observer (本领域普通观察者) 的整体观感相似度优先, 而非孤立特征叠加;
2. 以产品的 point of novelty (新颖视觉点) 是否被复刻作为高危强信号;
3. 颜色仅作氛围背景, 不作为独立判定维度 (外观以形状/轮廓/布局为准);
4. 判分 0 到 1 (score), 0.7 及以上视为高危, 0.45 及以上为中等, 其余低危;
5. 判定依据 (basis) 必须具体指向附图中可辨认的部位/结构特征, 不写泛泛之词;
6. 逐件给出其与你对照产品最可能的关键差异点 (difference), 供庭审式抗辩参考。
把产品图与多件在先设计图一并查看。严格输出 JSON 数组, 每个元素:
{"d_number":"该件唯一专利号","score":0~1,"dims":[{"name":"维度名","score":0~1,"note":"该维度说明"}],
 "basis":"依据句","difference":"差异点"}
不得输出 JSON 之外的文字。无法判定某件时, 将该元素省略。"""


def split_batches(items: list, batch_size: int = 2) -> list[list]:
    """把列表按 batch_size 切分为连续子批 (默认 2 件/批)。"""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _clip_score(value) -> float:
    """把任意数值裁剪到 [0,1]; 非数值 → 0.0。"""
    try:
        s = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(_MIN_SCORE, min(_MAX_SCORE, s))


def _parse_dims_drop_bad(raw) -> tuple:
    """把 dims 数组规整为 ((name,score,note),...); 结构不合法项弃之。"""
    if not isinstance(raw, list):
        return ()
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        score = _clip_score(item.get("score"))
        note = str(item.get("note") or "").strip()
        if not name:
            continue
        out.append((name, score, note))
    return tuple(out)


def _verdict_from_item(item: dict) -> JudgeVerdict | None:
    """单件 dict → JudgeVerdict; 缺 d_number/score → None (弃件)。"""
    if not isinstance(item, dict):
        return None
    d_number = str(item.get("d_number") or "").strip()
    # score 缺省时以 0 处理 (仍产判定, 由聚合按档位归类), 但 d_number 必需。
    if not d_number or "score" not in item:
        return None  # 缺 d_number 或缺 score 的坏件 → 弃 (不产残判)
    score = _clip_score(item.get("score"))
    # 档位一律以 score 重算 (risk_of_score), 覆盖任何 model「risk」字段, 不信任模型档位。
    risk = risk_of_score(score)
    basis = str(item.get("basis") or "").strip()
    difference = str(item.get("difference") or "").strip()
    return JudgeVerdict(
        d_number=d_number,
        score=score,
        risk=risk,
        dims=_parse_dims_drop_bad(item.get("dims")),
        basis=basis,
        difference=difference,
    )


def parse_judge_json(raw: str) -> list[JudgeVerdict]:
    """把 vision 返回的 JSON 数组规整为 JudgeVerdict[]; 坏 JSON/非数组 → []。

    容错方向: 整体坏 JSON → [] (无部分结果可用); 数组里个别坏件 → 弃该件产其余。
    """
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        verdict = _verdict_from_item(item)
        if verdict is not None:
            out.append(verdict)
    return out


def _batch_images(product_images, items) -> list[str]:
    """批图 = 产品图全部视图 + 批内各候选的全部页图 (顺序随 pdf_images_by_d)。"""
    images = list(product_images or [])
    for _, pages in items:  # noqa: PERF102 —— 顺序由 items 承载, 逐候选拼接页面
        images.extend(str(page) for page in (pages or []))
    return images


def _d_numbers(items) -> list[str]:
    return [d for d, _ in items]


async def judge_product(
    product_images: list[str],
    pdf_images_by_d: dict[str, list[str]],
    call=call_vision,
) -> list[JudgeVerdict]:
    """对产品逐候选分批判定: 2 件/批 × 1 次 vision → 批内 JSON → verdicts。

    call: async (images_base64, prompt) -> 模型文本 (注入 mock 禁真实网络; 生产即
    design_vision.call_vision)。单批失败 → 记日志继续后续批; 返回已成功批的 verdicts。
    """
    candidates = list(pdf_images_by_d.items() if hasattr(pdf_images_by_d, "items") else [])
    verdicts: list[JudgeVerdict] = []
    for batch in split_batches(candidates):
        try:
            prompt = _build_batch_prompt(_d_numbers(batch))
            text = await call(_batch_images(product_images, batch), prompt)
        except Exception as exc:  # noqa: BLE001 —— 单批失败不拖垮整单 (fail-open)
            _log_batch_failure(_d_numbers(batch), exc)
            continue
        verdicts.extend(parse_judge_json(text))
    return verdicts


def _build_batch_prompt(d_numbers: list[str]) -> str:
    """批量 prompt: 固定口径常量 + 当批 D 号清单, 请模型逐件 JSON。"""
    joined = "，".join(d_numbers) if d_numbers else "(空)"
    return f"{JUDGE_PROMPT_ZH}\n\n本次请对以下在先设计件逐件判定: {joined}"


def _log_batch_failure(d_numbers, exc) -> None:
    logger.warning("design_judge batch skipped %s: %s", d_numbers, exc)
