# US 外观防侵权询检（P1 核心闭环）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 卖家上传产品图 → 检索 US 外观在先权利 → 视觉相似风险判定 → 文本报告 + 会话回执（P1：图像与文本双入口、无前端改版）。

**Architecture:** 新域 `sources/design/` 五个小模块（query/search/image/judge/risk）+ 薄 vision 适配器；编排为 celery 新 executor `execute_design_clearance`（镜像 families 成功/失败路径），入口在 core 上传分支加 design 场景路由与预检。判定全走多模态 LLM（复用 `[LONG_TASK] vision_*` 配置与 patent_analyzer 的 PDF→页图能力 `_pdf_to_base64_images`）。

**Tech Stack:** Python 3.12+（服务器 3.14 兼容）、asyncio、httpx、celery、pytest（PYTHONUTF8=1）、现有 vision 管道（minimax/MiniMax-M3 或 deepseek-vision，config 可配）。

**上游 spec:** `docs/superpowers/specs/2026-09-06-us-design-clearance-design.md`（V1–V6 实测定案）；可行性: `docs/superpowers/specs/2026-09-06-us-design-clearance-feasibility.md`。

## Global Constraints

- pytest 一律 `PYTHONUTF8=1 python -m pytest <file> -q`；已知 pre-existing 失败集（batch_prompt 12f、memory 5f、session_api 7e、searx 3f、patent_analyzer 3f、knowledge_candidates 1f、provider/browser_agent_parsing 收集失败）勿当回归
- 测试禁真实网络：httpx/vision/Google/USPTO 全 mock
- 零提问词固化红线：模块 prompt 常量只含通用句式，测试样例词不得进入产品 prompt/代码
- 不可变风格、early return、函数 <50 行、无 print 调试
- 每任务原子 commit（conventional，中文描述），每任务后 code review 门
- 服务器内存 <1GB：L2 全程流式，单件 PDF ≤ ~1MB、页图 base64 随批即弃不落库
- 时间：utcnow 计算均用 `datetime.datetime.now(datetime.timezone.utc)`

---

### Task 1: design_risk 纯函数域（term 推导/聚合/报告/digest）

**Files:**
- Create: `sources/design/__init__.py`
- Create: `sources/design/design_risk.py`
- Test: `tests/test_design_risk.py`

**Interfaces:**
- Consumes: 无（纯标准库）
- Produces（后续任务依赖，签名固定）:
```python
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
    risk: str                 # "high"|"medium"|"low"
    dims: tuple               # ((name, score, note), ...)
    basis: str
    difference: str

def effective_until(grant_date: str) -> str          # 15y/14y → "YYYY-MM-DD"
def is_expired(grant_date: str, today: str) -> bool
def risk_of_score(score: float) -> str               # >=0.7 high; >=0.45 medium; else low
def aggregate(active: list[DesignCandidate], verdicts: list[JudgeVerdict],
              expired: list[DesignCandidate],
              today: str) -> dict
def build_report_md(target_label: str, agg: dict) -> str
def build_digest(target_label: str, agg: dict) -> dict
```
term 规则（spec §5.3/5.6）：grant_date ≥ 2015-05-13 → +15 年；早于 → +14 年；`is_expired` 用今天比较；`expired` 列表 = 检索命中但已期满件（单独提示，不进判定）。

- [ ] **Step 1: 写失败测试**（tests/test_design_risk.py 全量先写）：
```python
from sources.design.design_risk import (
    effective_until, is_expired, risk_of_score, aggregate,
    build_report_md, build_digest, DesignCandidate, JudgeVerdict)


def test_term_2015_boundary():
    assert effective_until("2015-05-13") == "2030-05-13"   # 15y
    assert effective_until("2015-05-12") == "2029-05-12"   # 14y
    assert effective_until("2010-01-01") == "2024-01-01"


def test_term_leap_day_safe():
    assert effective_until("2016-02-29") == "2031-02-28"   # 加年后 2/29 不存在


def test_is_expired_boundary():
    # 语义: 届满当日(含)仍有效; 次日才过期。
    assert is_expired("2011-08-31", "2026-09-01") is True   # 15y → 2026-08-31 已过
    assert is_expired("2011-09-01", "2026-09-01") is False  # 届满当日仍有效
    assert is_expired("2011-09-02", "2026-09-01") is False


def test_risk_of_score():
    assert risk_of_score(0.70) == "high"
    assert risk_of_score(0.69) == "medium"
    assert risk_of_score(0.45) == "medium"
    assert risk_of_score(0.44) == "low"


def test_aggregate_splits_and_flags():
    active = [DesignCandidate("USD1A", "Toy snake", "2023-01-01", status="active")]
    expired = [DesignCandidate("USD9Z", "Toy snake old", "2008-01-01", status="expired")]
    verdicts = [JudgeVerdict("USD1A", 0.82, "high",
                             (("整体轮廓", 0.8, "分节走向一致"),), "蛇头近似", "尾部开关位置不同")]
    agg = aggregate(active, verdicts, expired, today="2026-09-01")
    assert agg["high"] == [verdicts[0]]
    assert agg["expired_hits"] == expired
    assert agg["medium"] == [] and agg["low"] == []


def test_report_and_digest_structures():
    active = [DesignCandidate("USD1A", "Toy snake", "2023-01-01", assignee="ACME")]
    verdicts = [JudgeVerdict("USD1A", 0.82, "high", (("整体轮廓", 0.8, "x"),),
                             "basis", "diff")]
    agg = aggregate(active, verdicts, [], today="2026-09-01")
    md = build_report_md("product.jpg", agg)
    assert "USD1A" in md and "0.82" in md and "ACME" in md
    assert "非法律意见" in md
    dig = build_digest("product.jpg", agg)
    assert dig["target"] == "product.jpg"
    assert dig["type"] == "file"
    assert "USD1A" in dig["result_ids"] and len(dig["result_ids"]) <= 50
```
（再补 2 例：score 0.5 medium 不入 high；expired 件不进 result_ids。）

- [ ] **Step 2: 跑确认失败**
Run: `PYTHONUTF8=1 python -m pytest tests/test_design_risk.py -q`
Expected: collection/import 失败（模块不存在）

- [ ] **Step 3: 实现**
`docs/superpowers/specs/2026-09-06-us-design-clearance-design.md` §5.6 语义。核心：
```python
from dataclasses import dataclass
from datetime import date, datetime

_TERM_CUTOFF = date(2015, 5, 13)

def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()

def _add_years(d: date, n: int) -> date:
    """加 n 年; 2/29 目标年不存在时顺延为 2/28(term 计算口径, 见测试)。"""
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)

def effective_until(grant_date: str) -> str:
    g = _parse(grant_date)
    return _add_years(g, 15 if g >= _TERM_CUTOFF else 14).isoformat()

def is_expired(grant_date: str, today: str) -> bool:
    return effective_until(grant_date) < today

def risk_of_score(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"
```
`aggregate`：high/medium/low 按 risk_of_score(verdict.score) 分组（不信任模型 risk 字段作分组依据，但保留展示）；`expired_hits` 透传；**同 d_number 去重取最高 score**。`build_report_md` 固定模板（总览 → 高危逐件含 D 号/标题/档位/score/维度/依据/差异 → 相关但已失效 → 免责声明）。`build_digest` 返回 `{"target": label, "type": "file", "result_ids": [high 的 d_number][:50]}`。
- [ ] **Step 4: 跑确认通过**（同上命令，Expected: PASS，~10 passed）
- [ ] **Step 5: Commit**
```bash
git add sources/design/__init__.py sources/design/design_risk.py tests/test_design_risk.py
git commit -m "feat: 外观询检 design_risk 纯函数域——15y/14y term 推导+三档聚合+报告/digest（design P1 T1）"
```

---

### Task 2: design_query —— L0 提示词与 JSON 解析

**Files:**
- Create: `sources/design/design_query.py`
- Test: `tests/test_design_query.py`

**Interfaces:**
- Consumes: 无（纯函数）；`L0_PROMPT_ZH/EN` 常量字符串（通用句式，零提问词固化）
- Produces:
```python
L0_PROMPT_ZH: str          # 见步骤 1 注释内容要求
def parse_l0_json(raw: str) -> dict   # 容错解析 → L0Product dict
def l0_product_from_text(text: str) -> dict   # 文本降级入口
```
返回 dict 形状：`{"en_name","keywords":[],"visual_features":[],"suggested_locarno":[],"needs_clarification":bool}`

- [ ] **Step 1: 失败测试**（含容错与降级契约）：
```python
from sources.design.design_query import parse_l0_json, l0_product_from_text, L0_PROMPT_ZH

def test_parse_ok():
    raw = '{"en_name":"Toy snake","keywords":["toy snake","robot"],"visual_features":["segmented"],"suggested_locarno":["21-01"],"needs_clarification":false}'
    p = parse_l0_json(raw)
    assert p["en_name"] == "Toy snake"
    assert p["needs_clarification"] is False

def test_parse_tolerates_markdown_fence_and_junk():
    p = parse_l0_json('```json\n{"en_name":"x","keywords":[],"visual_features":[],"suggested_locarno":[],"needs_clarification":false}\n```')
    assert p["en_name"] == "x"
    assert parse_l0_json("not json at all")["needs_clarification"] is True

def test_text_fallback_uses_first_meaningful_token():
    p = l0_product_from_text("帮我查遥控玩具蛇外观")
    assert p["en_name"]
    assert isinstance(p["keywords"], list)

def test_prompt_is_generic():
    assert "toy snake" not in L0_PROMPT_ZH.lower()
    assert "输出 JSON" in L0_PROMPT_ZH
```
- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现**：`parse_l0_json` 剥 ```json 围栏→json.loads→缺键补默认、坏 JSON 返回 `{"needs_clarification": True, ...空}`；`l0_product_from_text` 取输入非空文本作 `en_name`（截 60 字符），keywords=[原词]（中文给英文提示需求由 executor 在 L1 前做"中文→英文词"由 vision 或预置翻译调用——P1 文本入口要求用户给英文品名或由 vision 翻译：见 Task 7 编排说明，此处仅产结构）；`L0_PROMPT_ZH` 为通用视觉翻译提示（含 schema 描述与"多产品/无法判断→needs_clarification=true"指令）
- [ ] **Step 4: 跑通过**
- [ ] **Step 5: Commit** `feat: design_query L0 提示与容错解析（design P1 T2）`

---

### Task 3: design_search —— XHR 检索（阶梯/US 过滤/退避）

**Files:**
- Create: `sources/design/design_search.py`
- Test: `tests/test_design_search.py`

**Interfaces:**
- Consumes: `DesignCandidate`（Task 1）
- Produces:
```python
def build_ladder(en_name: str, keywords: list[str]) -> list[str]      # ≤6 条，紧→松
def parse_design_hits(xhr_json: dict) -> tuple[list[DesignCandidate], list[dict]]
    # → (active+expired 全量 US 件(带 status), em_cn_hits 日志项)
async def search_designs(fetch, en_name: str, keywords: list[str],
                         country_param: str = "US",
                         jitter: float = 1.0) -> dict
    # fetch: async (url_query:str) -> (status:int, text:str)
    # 返回 {"candidates":[...US 全量...], "em_cn":[...], "used_queries": n,
    #        "rate_limited": bool}
```
规则：查询 = `q="{term}"` + `type=DESIGN` +（country_param 存在时尝试 `&country=US`——由调用侧 flag 控制，P1 默认传空串 = 不用 country 参数，走结果侧过滤）；阶梯展开见 spec §5.3；503/429 → 退避 1s/3s ≤2 重试，全败置 `rate_limited=True`；US 件判定：`id` 以 `patent/USD` 开头；EM/CN 记录 `{"id":…,"country":…}` 不返回；解析字段来自 XHR item：`id / patent.publication_number / patent.title / patent.publication_date / patent.assignee`；`publication_date` 取 `[:10]`，缺省置 `""`（status 由调用方/risk 侧以 today 判定——此处仅算 status 需要日期，缺日期件 status 置 "unknown" 且设计上不参与 active 判定，Task 7 按 active-only 处理）。`DesignCandidate.pub` 用 publication_number。

- [ ] **Step 1: 失败测试**（mock fetch 注入；样例 JSON 用小写通用词构造，零提问词固化——样例词为通用品名词如 "robot toy" 可接受于测试数据）：
```python
import asyncio, json
from sources.design.design_search import build_ladder, parse_design_hits, search_designs

def _hit(pid, pub, date, title, assignee=""):
    return {"id": f"patent/{pid}/en",
            "patent": {"publication_number": pub, "title": title,
                       "publication_date": date, "assignee": assignee}}

def test_ladder_order_and_cap():
    lad = build_ladder("Robot toy", ["robot dog", "mechanical dog"])
    assert len(lad) <= 6
    assert lad[0] == "Robot toy robot dog"

def test_parse_us_only_and_status():
    xhr = {"results": {"cluster": [{"result": [
        _hit("USD1A", "USD1A", "2023-01-01", "Toy"),
        _hit("EM150000001S", "EM150000001S", "2023-02-01", "Toy"),
        _hit("CN309123456S", "CN309123456S", "2023-03-01", "Toy"),
    ]}]}}
    cands, em_cn = parse_design_hits(xhr)
    assert [c.pub for c in cands] == ["USD1A"]
    assert len(em_cn) == 2

async def _run(fetch, *a):
    return await search_designs(fetch, *a)

def test_search_rate_limit_sets_flag():
    async def boom(url_query):
        return 503, "sorry"
    res = asyncio.run(_run(boom, "Robot toy", []))
    assert res["rate_limited"] is True
    assert res["used_queries"] == 1
```
（再补：成功路径返回 candidates 且 used_queries==1；503 后第二次 200 → rate_limited False。）

- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现**（注意：`type=DESIGN` 与 `country` 放 XHR `url` 参数原文内拼为 `q={q}&type=DESIGN`；fetch 由调用方注入真实 httpx 包装（Task 7），本模块只拼 query 串并调用 fetch(`q={quote(term)}&type=DESIGN`)）
- [ ] **Step 4: 跑通过**
- [ ] **Step 5: Commit** `feat: design_search XHR 检索——阶梯/US 过滤/退避（design P1 T3）`

---

### Task 4: design_image —— 页面取 PDF 与降级

**Files:**
- Create: `sources/design/design_image.py`
- Test: `tests/test_design_image.py`

**Interfaces:**
- Consumes: 无（fixture HTML 常量）
- Produces:
```python
DESIGN_PAGE_URL = "https://patents.google.com/patent/{pid}/en"
def extract_pdf_url(html: str) -> str | None     # 首 patentimages …USD….pdf
async def fetch_design_pdf(fetch_page, pid: str) -> bytes | None
```
`fetch_page(pid) -> (status, html)`。失败/无 pdf → None（Task 7 降级"文本维度候选"）。

- [ ] **Step 1: 失败测试**：
```python
from sources.design.design_image import extract_pdf_url, fetch_design_pdf

HTML = ('<img src="https://patentimages.storage.googleapis.com/fa/21/81/'
        '670fbfd00d78ff/USD504889.pdf"><img src="https://example.com/other.png">')

def test_extract_first_pdf():
    assert extract_pdf_url(HTML).endswith("USD504889.pdf")

def test_extract_none_when_missing():
    assert extract_pdf_url("<html>no image</html>") is None

import asyncio
def test_fetch_degrade():
    async def bad(pid):
        return 404, "nope"
    assert asyncio.run(fetch_design_pdf(bad, "USD1A")) is None
```
- [ ] **Step 2: 失败 → Step 3 实现**（正则 `https://patentimages\.storage\.googleapis\.com/[^"'\s]+\.pdf` 首命中；`fetch_design_pdf` 先页再 `httpx.get(pdf_url)`，任一步非 200 → None；pdf 大小 >2MB 截断返回 None）→ **Step 4 通过 → Step 5 Commit** `feat: design_image 页→PDF 直链与降级（design P1 T4）`

---

### Task 5: design_vision —— 薄视觉适配器（镜像 patent_analyzer 参考实现）

**Files:**
- Create: `sources/design/design_vision.py`
- Test: `tests/test_design_vision.py`

**Interfaces:**
- Consumes: config `[LONG_TASK] vision_enabled/vision_provider/vision_model`（`configparser` 读 config.ini，可被 env 覆盖留待 Task 7）
- Produces:
```python
def load_vision_config(config_path: str = "config.ini") -> dict   # {"provider","model","enabled"}
async def call_vision(images_base64: list[str], prompt: str,
                      post=None, timeout: int = 90) -> str
    # post: async (url, headers, json, timeout) -> fake-response-like {status_code,text}
    # 返回模型文本; 非 200/异常 → 抛 DesignVisionError
class DesignVisionError(RuntimeError): ...
```
**镜像要求（写入代码注释与 review 关注点）**：先读 `sources/long_task/patent_analyzer.py:344-490`（`analyze_patent_with_vision`/`_call_vision_api`）与 `sources/long_task/config.py:19-60`——wire 格式（URL 构造、鉴权头、payload 字段、超时）与新适配器保持一致，不得另起炉灶。`call_vision` 的 `images_base64` 即 `_pdf_to_base64_images` 的输出格式（Task 6/7 复用该函数产页图）。

- [ ] **Step 1: 失败测试**（invariant 级，避免拷死 wire 细节）：
```python
import asyncio
from sources.design.design_vision import call_vision, DesignVisionError

def test_call_vision_sends_images_and_prompt():
    calls = {}
    async def fake_post(url, headers, json, timeout):
        calls["url"] = url; calls["json"] = json; calls["timeout"] = timeout
        class R: status_code, text = 200, '{"text":"ok"}'
        return R()
    out = asyncio.run(call_vision(["aGVsbG8="], "describe", post=fake_post))
    assert calls["url"].startswith("http")
    assert "aGVsbG8=" in str(calls["json"])
    assert calls["timeout"] == 90

def test_call_vision_error_raises():
    async def bad(url, headers, json, timeout):
        class R: status_code, text = 500, "boom"
        return R()
    try:
        asyncio.run(call_vision(["aGk="], "x", post=bad))
        assert False, "should raise"
    except DesignVisionError:
        pass
```
- [ ] **Step 2: 失败 → Step 3 实现**：`load_vision_config` 用 configparser；`call_vision` 组装参考实现同款请求（读 435-490 行对齐字段）；无 `post` 注入时默认 `httpx.AsyncClient.post`；`DesignVisionError` 含状态码与截断响应。`enabled=false` → 抛 `DesignVisionError("vision disabled")`。
- [ ] **Step 4: 通过 → Step 5 Commit** `feat: design_vision 适配器（镜像 patent_analyzer wire, design P1 T5）`

---

### Task 6: design_judge —— 分批判定与 schema 校验

**Files:**
- Create: `sources/design/design_judge.py`
- Test: `tests/test_design_judge.py`

**Interfaces:**
- Consumes: `JudgeVerdict`（Task 1）、`call_vision`（Task 5）、`_pdf_to_base64_images`（patent_analyzer:285，Task 7 注入页图）
- Produces:
```python
JUDGE_PROMPT_ZH: str   # 固定判定口径(ordinary observer/point of novelty/色彩说明/0-1 score/依据句/差异点)——通用句式
def split_batches(items: list, batch_size: int = 2) -> list[list]
def parse_judge_json(raw: str) -> list[JudgeVerdict]   # 容错: 坏 JSON/缺字段→[]
async def judge_product(product_images: list[str], pdf_images_by_d: dict[str, list[str]],
                        call=call_vision) -> list[JudgeVerdict]
```
- [ ] **Step 1: 失败测试**（批量拆分/容错/映射，样例用通用词）：
```python
from sources.design.design_judge import (split_batches, parse_judge_json,
                                          JUDGE_PROMPT_ZH)

def test_split_two_per_batch():
    assert split_batches(["a", "b", "c"]) == [["a", "b"], ["c"]]

def test_parse_ok_and_cap():
    raw = ('[{"d_number":"USD1A","score":0.82,"risk":"high",'
           '"dims":[{"name":"整体轮廓","score":0.8,"note":"一致"}],'
           '"basis":"蛇头近似","difference":"尾部开关不同"}]')
    vs = parse_judge_json(raw)
    assert vs[0].d_number == "USD1A" and vs[0].score == 0.82

def test_parse_junk_empty():
    assert parse_judge_json("not json") == []

def test_prompt_generic_and_criteria():
    assert "toy snake" not in JUDGE_PROMPT_ZH.lower()
    assert "0.7" in JUDGE_PROMPT_ZH or "score" in JUDGE_PROMPT_ZH
```
（再补 score>1/score<0 样本被裁剪到 [0,1]；risk 字段与 risk_of_score(score) 冲突时以 score 计算为准——parse 输出 risk=风险档按 Task 1 的 risk_of_score 重算并覆盖。）
- [ ] **Step 2: 失败 → Step 3 实现**（prompt 含固定判定口径句与 schema 说明；`judge_product` 按 2 件/批拼 prompt，`call(product_images + 该批图)`，逐批容错——单批异常记日志继续）→ **Step 4 通过 → Step 5 Commit** `feat: design_judge 分批判定与 schema 校验（design P1 T6）`

---

### Task 7: executor + 入口路由/预检集成

**Files:**
- Modify: `celery_worker.py`（新增 executor，镜像 families：装饰器 `@app.task(bind=True, max_retries=2, default_retry_delay=60, time_limit=1800, soft_time_limit=1770)`，参照 `execute_family_analysis` 1730 起结构与失败单点出口 2874/`_family_failed_terminal` 341-350 语义；成功端镜像 set_task_completed + anchor 写入模式——以 09-05 families 成功路径 ~2935 区域为准）
- Modify: `api_routes/core.py`（上传分支 design 场景路由 + 预检门小分支）
- Create: `sources/design/design_pipeline.py`（编排：入口→L0→L1→L2→L3→risk→回执；供 executor 薄调用、可整体 mock 测试）
- Test: `tests/test_design_pipeline.py`（全 mock）；集成冒烟并入 executor 所在既有 suites 范围自查

**Interfaces:**
- Consumes: Task 1–6 全部符号；`_pdf_to_base64_images`（patent_analyzer:285）、`failure_guidance`（status_manager）、anchor/回执 helper（沿用 09-05 命名：set_task_completed(anchor_payload=…)、append_task_message/build_result_digest 视既有签名——实现时先 grep 确认）
- Produces: `execute_design_clearance(task_id, params)`（params: `{"product_image_refs":[…],"product_text":str,"source":"us_design"}`）；`design_pipeline.run(params, logger, emit_progress) -> dict`（供 executor 与测试直调）

- [ ] **Step 1: 失败测试（design_pipeline，全 mock 链）**

打桩契约：pipeline 内以 `from sources.design import design_query, design_search, design_image, design_judge, design_risk` 模块引用调用（测试 patch 各模块函数即可生效）；`run(params, ctx)` 返回 `{"digest": {...}, "report_md": str}`，澄清信号以 `DesignNeedClarification(Exception)` 上抛。以下 6 用例逐一先跑 RED：

```python
import asyncio, json
from unittest import mock
from sources.design.design_pipeline import run, DesignNeedClarification
from sources.design.design_risk import DesignCandidate, JudgeVerdict

class _Ctx:
    def __init__(self):
        self.progress_msgs = []
    def progress(self, msg):
        self.progress_msgs.append(msg)
    def warning(self, msg):
        self.progress_msgs.append(f"WARN:{msg}")

PARAMS = {"product_image_refs": ["p.jpg"], "product_text": "",
          "source": "us_design"}

_USD_XHR = {"results": {"cluster": [{"result": [
    {"id": "patent/USD1A/en",
     "patent": {"publication_number": "USD1A", "title": "Toy",
                "publication_date": "2023-01-01", "assignee": "ACME"}}]}]}}
_PDF_HTML = ('<img src="https://patentimages.storage.googleapis.com/x/USD1A.pdf">')

async def _ok_fetch(url_query):
    return 200, json.dumps(_USD_XHR)

async def _ok_page(pid):
    return 200, _PDF_HTML

async def _ok_pdf(pid):
    return b"%PDF-1.4 fake small"

async def _ok_vision(images_base64, prompt, post=None, timeout=90):
    return ('[{"d_number":"USD1A","score":0.82,"risk":"high",'
            '"dims":[{"name":"整体轮廓","score":0.8,"note":"一致"}],'
            '"basis":"蛇头近似","difference":"尾部开关不同"}]')


async def _search_ok(en_name, keywords, country_param="", jitter=1.0):
    return {"candidates": [
        DesignCandidate("USD1A", "Toy", "2023-01-01", "ACME", "active")],
        "em_cn": [], "used_queries": 1, "rate_limited": False}


async def _judge_ok(product_images, pdf_images_by_d, call=None):
    return [JudgeVerdict("USD1A", 0.82, "high",
                         (("整体轮廓", 0.8, "一致"),),
                         "蛇头近似", "尾部开关不同")]

def _patch_all(**kw):
    base = dict(
        design_query=lambda: None, design_search=lambda: None,
        design_image=lambda: None, design_judge=lambda: None)
    base.update(kw)
    return base

def test_full_chain_success_digest_and_progress():
    ctx = _Ctx()
    with mock.patch("sources.design.design_pipeline.design_query.parse_l0_json",
                    return_value={"en_name": "Toy", "keywords": ["toy"],
                                  "visual_features": [], "suggested_locarno": [],
                                  "needs_clarification": False}), \
         mock.patch("sources.design.design_pipeline.design_query.l0_product_from_text",
                    return_value={"en_name": "Toy", "keywords": ["toy"],
                                  "visual_features": [], "suggested_locarno": [],
                                  "needs_clarification": False}), \
         mock.patch("sources.design.design_pipeline.design_search.search_designs",
                    new=_search_ok), \
         mock.patch("sources.design.design_pipeline.design_image.fetch_design_pdf",
                    new=_ok_pdf), \
         mock.patch("sources.design.design_pipeline.design_judge.judge_product",
                    new=_judge_ok):
        out = asyncio.run(run(PARAMS, ctx))
    assert "USD1A" in out["digest"]["result_ids"]
    assert "非法律意见" in out["report_md"]
    assert len(ctx.progress_msgs) >= 4
```
执行说明（Step 1 的 6 个用例，实现者须照此写全并各自 RED→GREEN）：
①**全链成功**（上例）——L0 patch 返回固定 dict、`_search_ok` 返回 `{"candidates":[DesignCandidate("USD1A","Toy","2023-01-01","ACME","active")], "em_cn":[], "used_queries":1, "rate_limited":False}`、`_judge_ok` 返回单条 JudgeVerdict → 断言 digest/report/进度 ≥4；
②**L1 rate_limited=True** → run 上抛 `DesignRuntimeError("rate_limited")`（自定义异常，executor 映射失败消息"服务暂不可用请稍后重试"），ctx 进度含该提示；
③**L1 0 命中**（candidates 空且非限流）→ run 正常返回，digest 含 0 命中契约文案（常量 `ZERO_HITS_TEXT`：含"未检出"与建议换词/上传更清晰图，禁裸"未找到"）；
④**L2 全败**（fetch_design_pdf 返 None）→ 报告含"文本维度候选"降级说明，digest 仍含著录件；
⑤**L3 全败**（judge_product 抛/返空）→ 报告注明"N 件未完成视觉比对"，digest 仍写；
⑥**L0 needs_clarification=True** → run 上抛 `DesignNeedClarification`。

辅助：`_search_ok`/`_judge_ok` 为测试模块级 async 函数（签名对齐被 patch 的模块函数）；pipeline 需自定义 `DesignRuntimeError`。
- [ ] **Step 2: 确认失败（上述 6 场景逐一 RED）→ Step 3 实现 pipeline（编排纯顺序 + 进度事件 + 每环 try/except 按 spec §7 降级；clarify→DesignNeedClarification 信号由 executor 转 SSE 澄清；0 命中契约文案常量）→ Step 4 通过 → Step 5 Commit** `feat: design_pipeline 编排 + execute_design_clearance + core 路由/预检（design P1 T7）`
- [ ] **Step 6: 入口冒烟**：`PYTHONUTF8=1 python -m pytest tests/test_design_*.py tests/test_design_pipeline.py -q` 全绿；并抽查 `import celery_worker` 无环（`python -c "import celery_worker"`）

---

### Task 8: 回归与收尾（controller-run）

- [ ] **Step 1**：新增 6 个测试文件全绿（Task 1–7 各自文件）
- [ ] **Step 2**：全量回归 `PYTHONUTF8=1 python -m pytest -q --ignore=tests/test_provider.py --ignore=tests/test_browser_agent_parsing.py`——失败画像须 = 既有 pre-existing 集 + 夹具互窜已知项（Phase A 记录），**新增失败 = 0**
- [ ] **Step 3**：终审（opus/code-reviewer，范围本 plan commits）——闭合回路核对 spec §2.3 自动化验收 1–6
- [ ] **Step 4**：账本 `.superpowers/sdd/progress.md` 追加 plan 段（沿用既有格式）
- 部署与轨迹 UAT（V4′/真实产品图端到端/限流表现）不在本 plan，列入部署清单（spec §10 收尾）

---

## Self-Review 备注（写完即查）
- Spec §2.3 验收 1–6 ↔ Task 1(term/聚合/报告/digest)、Task 3(阶梯/US/退避/0 命中)、Task 4(降级)、Task 6(schema/0.7 映射)、Task 1+7(回执/锚)、Task 7(预检/fail 单发)——已全覆盖
- §5.3 country=US 参数属 UAT（V4′），P1 默认结果侧 US 过滤（Task 3 契约）
- §11 时效实测未做：报告模板含"数据截至"字样（build_report_md 内嵌当日日期）作对冲（Task 1 实现含该行）
