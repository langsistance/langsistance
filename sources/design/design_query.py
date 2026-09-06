"""design_query L0 提示词与 JSON 解析 (design P1 T2)。

规格: docs/superpowers/specs/2026-09-06-us-design-clearance-design.md §L0; task-2-brief.md。
- L0_PROMPT_ZH / L0_PROMPT_EN: 通用视觉翻译提示, 零产品词固化。
- parse_l0_json: 剥 ```json 围栏→json.loads→缺键补默认; 坏 JSON 降级为
  needs_clarification=True 的空结构。
- l0_product_from_text: 文本降级入口, 仅产结构 (P1 不自动中译英)。
仅标准库, 纯函数无 IO。
"""
import json
import re

# 返回结构: {"en_name","keywords","visual_features","suggested_locarno","needs_clarification"}
# 结构基准: 键皆在, en_name 留空, needs_clarification 缺省为 False;
# 坏 JSON/空 en_name 在 parse 内各自置 True。
_EMPTY = {
    "en_name": "",
    "keywords": [],
    "visual_features": [],
    "suggested_locarno": [],
    "needs_clarification": False,
}


def parse_l0_json(raw: str) -> dict:
    """容错解析 L0 响应体 → dict。

    处理: markdown ```json 围栏 / 前后杂讯 → json.loads → 缺键补默认;
    无法解析的地板为 needs_clarification=True 的空结构 (不抛)。needs_clarification
    缺省时按 False 处理 (可信 en_name 存在即不算待澄清); 显式 true 或空 en_name
    才置 True。
    """
    body = _strip_fence(raw)
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return {**_EMPTY, "needs_clarification": True}
    if not isinstance(data, dict):
        return {**_EMPTY, "needs_clarification": True}
    explicit_clarify = bool(data.get("needs_clarification"))
    cleaned = dict(_EMPTY)
    cleaned.update(data)
    for key in ("keywords", "visual_features", "suggested_locarno"):
        if not isinstance(cleaned[key], list):
            cleaned[key] = []
    if explicit_clarify or not cleaned.get("en_name"):
        cleaned["needs_clarification"] = True
    return cleaned


def _strip_fence(raw: str) -> str:
    """剥去 ```json ... ``` 围栏, 保留围栏外可能的纯前导/尾随杂讯。"""
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


_EN_LIMIT = 60


def l0_product_from_text(text: str) -> dict:
    """文本降级入口: 纯文本第一段非空输入作 en_name (截 60), keywords=[原词]。

    P1 只产结构, 不做自动中译英; 中文引导由 L1/pipeline 层负责 (见 task-2 brief
    编排说明, 引导文案在 pipeline 常量层, 不在本模块)。
    """
    en_name = (text or "").strip()[:_EN_LIMIT]
    return {**_EMPTY, "en_name": en_name, "keywords": [en_name], "needs_clarification": False}


# 视觉翻译 schema 导引 (零产品词固化 —— 见 memory 规则: 禁止检索/提示词入提问词)。
_SCHEMA_HINT = (
    "对以下英文产品名 / 描述抽取出用于美国外观检索的 L0 结构化字段, "
    '返回严格 JSON, 不附加任何散文。键: "en_name"(字符串, 产品英文名)'
    '、"keywords"(字符串数组, 检索用英文关键词, 含同义/近义词)'
    '、"visual_features"(字符串数组, 可见外观特征)'
    '、"suggested_locarno"(字符串数组, 建议的洛迦诺大类小类号)'
    '、"needs_clarification"(布尔): 仅当输入是一张含多个不同产品的图片,'
    ' 或无法从输入判断出唯一一个产品时置 true, 并发空的 en_name。'
)

# 通用视觉翻译提示 (L0), 供 T7 pipeline 组装。
L0_PROMPT_ZH = (
    "你是美国外观专利初步检索的产品结构化助手。"
    "任务: 把用户给出的产品代表图翻译成可检索的英文结构化描述。"
    f"{_SCHEMA_HINT}"
    '翻译时: en_name 必须为英文; 若已给英文品名则直接复用。'
    '若图片含多个不同产品或无法判断为唯一产品, 必须置 needs_clarification=true'
    " 并清空 en_name, 交由澄清引导, 不要擅自挑选。"
    "只输出 JSON, 不要输出多余解释或 markdown 围栏之外的文字。"
)

# 英文等价提示 (与 ZH 仅语言差异), 供需要英文提示的编排保留同 schema。
L0_PROMPT_EN = (
    "You structure a product image into an L0 profile for US design searching. "
    'Respond with strict JSON only, no prose. Keys: "en_name"(string, English '
    'product name), "keywords"(string[]), "visual_features"(string[]), '
    '"suggested_locarno"(string[] of Locarno classes), "needs_clarification"'
    '(bool): set true and empty en_name only when the image shows multiple '
    "distinct products or no single product can be determined. Pass through an "
    "English name already given. Output the JSON object only."
)
