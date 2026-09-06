"""design_judge 分批判定与 schema 校验 (design P1 T6) — 契约测试。

规则: 2 件/批判定的分拆; parse_judge_json 容错坏 JSON/缺字段 → []; score 越界裁剪
[0,1]; risk 字段与 risk_of_score(score) 冲突时以 score 重算覆盖 (Task 1 口径)。
judge_product 逐批容错, 注入 mock call 禁真实网络; 批图 = 产品图 + 批内候选图。
通用句式零产品词 (不出现测试品名)。生产零产品词; 函数 <50 行, 无 print。
"""
import asyncio

from sources.design.design_judge import (
    JUDGE_PROMPT_ZH,
    judge_product,
    parse_judge_json,
    split_batches,
)
from sources.design.design_risk import risk_of_score


# ---------- 分批 ----------

def test_split_two_per_batch():
    assert split_batches(["a", "b", "c"]) == [["a", "b"], ["c"]]


def test_split_empty_and_single():
    assert split_batches([]) == []
    assert split_batches(["x"]) == [["x"]]


# ---------- parse_judge_json: schema 校验 ----------

_VALID_RAW = (
    '[{"d_number":"USD9A1","score":0.82,"risk":"high",'
    '"dims":[{"name":"整体轮廓","score":0.8,"note":"分节走向一致"}],'
    '"basis":"蛇头近似","difference":"尾部开关不同"}]'
)


def test_parse_ok_fields_map():
    vs = parse_judge_json(_VALID_RAW)
    assert len(vs) == 1
    v = vs[0]
    assert v.d_number == "USD9A1"
    assert v.score == 0.82
    assert v.risk == "high"
    assert v.basis == "蛇头近似"
    assert v.difference == "尾部开关不同"


def test_parse_dims_tuple_shape():
    v = parse_judge_json(_VALID_RAW)[0]
    assert v.dims == (("整体轮廓", 0.8, "分节走向一致"),)  # tuple of (name,score,note)


def test_parse_junk_empty():
    assert parse_judge_json("not json") == []


def test_parse_not_a_list_empty():
    assert parse_judge_json('{"score": 0.5}') == []


def test_parse_missing_required_field_empty():
    # 缺 d_number / score 的逐件 → 该件弃 (坏件不产出)。
    raw = '[{"d_number":"","score":0.6},{"d_number":"USD9B1"}]'
    assert parse_judge_json(raw) == []


def test_parse_score_high_capped_to_one():
    raw = '[{"d_number":"USDC1","score":1.7,"risk":"high"}]'
    v = parse_judge_json(raw)[0]
    assert v.score == 1.0


def test_parse_score_low_capped_to_zero():
    raw = '[{"d_number":"USDC2","score":-0.3,"risk":"low"}]'
    v = parse_judge_json(raw)[0]
    assert v.score == 0.0


def test_parse_risk_overridden_by_score():
    # 模型 risk:"high" 但 score=0.4 → 档位按 risk_of_score(0.4)=low 重算覆盖。
    raw = '[{"d_number":"USDD1","score":0.4,"risk":"high"}]'
    v = parse_judge_json(raw)[0]
    assert v.risk == "low"
    assert v.risk == risk_of_score(0.4)


def test_parse_risk_boundary_recompute():
    raw = '[{"d_number":"USDE1","score":0.70,"risk":"low"}]'
    assert parse_judge_json(raw)[0].risk == "high"


# ---------- JUDGE_PROMPT_ZH: 通用句式与固定判定口径 ----------

def test_prompt_generic_and_criteria():
    assert "toy snake" not in JUDGE_PROMPT_ZH.lower()
    assert "0.7" in JUDGE_PROMPT_ZH or "score" in JUDGE_PROMPT_ZH


def test_prompt_introduces_concepts():
    # 口径句: ordinary observer 整体观感 / point of novelty / 色彩非独立维度 / 依据句 / 差异点。
    assert "ordinary observer" in JUDGE_PROMPT_ZH.lower() or "整体观感" in JUDGE_PROMPT_ZH
    assert "novelty" in JUDGE_PROMPT_ZH.lower()
    assert "json" in JUDGE_PROMPT_ZH.lower()  # schema 说明


# ---------- judge_product: 逐批映射与容错 ----------

def test_judge_product_batches_two_images_and_verdicts():
    # 编排契约: 3 件 → 2 批 (2/1); 每批入图 = 产品图(s) + 批内候选各页; 返回批内全部 verdict。
    product_images = ["data:image/jpeg;base64,PROD", "data:image/jpeg;base64,PROD2"]
    pdf_by_d = {
        "USDF1": ["A1"],
        "USDF2": ["A2", "A2b"],
        "USFF3": ["B1"],
    }
    # 字典序 → kwargs 顺序: USDF1, USDF2 (批1), USFF3 (批2)。
    expected_batches = [
        ["USDF1", "USDF2"],
        ["USFF3"],
    ]
    received = []

    class RecordingCall:
        def __init__(self):
            self.batch_index = 0

        async def __call__(self, images_base64, prompt, override_shots=None):  # noqa: ARG001
            rec = {
                "images": list(images_base64),
                "idx": self.batch_index,
            }
            received.append(rec)
            # prompt 逐 D 号逐件 JSON (弱表单), 交由 judge_product 回填缺失维度。
            ds = expected_batches[self.batch_index]
            self.batch_index += 1
            return _raw_for_ds(ds)

    async def go():
        return await judge_product(product_images, pdf_by_d, call=RecordingCall())

    vs = asyncio.run(go())
    assert len(received) == 2                     # 3 件 → 2/1 两批判定
    assert len(vs) == 3                           # 每件产出 1 verdict
    assert {v.d_number for v in vs} == {"USDF1", "USDF2", "USFF3"}
    # 每批图 = 产品图在前, 批内两个候选的页图在后 (顺随 pdf_by_d 顺序)。
    assert received[0]["images"] == product_images + ["A1", "A2", "A2b"]
    assert received[1]["images"] == product_images + ["B1"]
    # 弱表单缺 basis/dims → 均以缺省保留 (dims 空, basis空), 不致命。
    for v in vs:
        assert 0.0 <= v.score <= 1.0


def _raw_for_ds(d_numbers):
    parts = []
    for dn in d_numbers:
        parts.append(
            '{"d_number":"%(d)s","score":0.6,"risk":"high"}' % {"d": dn}
        )  # risk 声称 high, score=0.6 → 以 score 档 (medium) 为准。
    return "[" + ",".join(parts) + "]"


def test_judge_product_batch_failure_skips_that_batch():
    # 单批异常/坏 JSON → 跳过该批, 其余批照常产出 (逐批容错)。
    product_images = ["PROD"]
    pdf_by_d = {"USG1": ["P1"], "USG2": ["P2"], "USG3": ["P3"], "USG4": ["P4"]}
    attempts = {"n": 0}

    class FailingFirstThenOk:
        async def __call__(self, images, prompt):  # noqa: ARG001
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("vision call errored for this batch")
            return _raw_for_ds(["USG2", "USG4"])  # 首批败 → 只回成功批 (批2/批4的 d)

    async def go():
        return await judge_product(product_images, pdf_by_d, call=FailingFirstThenOk())

    # 4 件 → 批1=[USG1,USG2], 批2=[USG3,USG4]。首批抛 → 仅批2产出。
    vs = asyncio.run(go())
    assert attempts["n"] == 2
    assert {v.d_number for v in vs} == {"USG2", "USG4"}
