# 检索后接地解读闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让技术解读达到智慧芽水平（本质句 + 技术维度骨架 + 每维度主线带代表申请人 + 数据驱动专精玩家），并把接地解读闭环反哺检索（补查询/补 CPC/rubric 真信号），同时增强打分并发（flash 与语义模型两侧）。

**Architecture:** 双阶段解读——预检索 terra 解读升级输出 `dimensions`（固定三层骨架，parse 硬控制 ≤3），查询按维度进 ladder；检索建池后新增 flash 接地合成模块（申请人/CPC 统计 + 按维度主线聚类，失败降级纯统计），单次触发反哺补查询/补 CPC/评分 rubric。打分并发：flash 批 10 + 信号量 6；语义 embedding 改 async + 分块 64 线程池并行。

**Tech Stack:** Python 3 / asyncio / FastAPI 后端、LangChain Provider（openrouter terra / deepseek flash）、bge-m3 embedding（SiliconFlow）、pytest（unittest 风格）。

## Global Constraints

- **通用性铁律（第三次重申）**：任何 prompt/代码零测试提问词汇（`RGB`、`控制放大器`、`独立控制` 不得出现在生产 prompt 中）；维度分层只用通用角色（核心器件/电路层、控制算法/电路层、场景应用层）；测试问题只出现在验证运行里。Task 1/3/7 有强制审计步骤。
- **测试命令**（Windows 基线）: `PYTHONUTF8=1 python -m pytest tests/ -q --ignore=tests/test_browser_agent_parsing.py --ignore=tests/test_provider.py`（当前基线 748 passed / 26 failed + 7 errors 为环境性预存，不得新增失败）。
- **Env 旋钮全部带垃圾值回退**（`_env_int` 模式，解析失败用默认值，绝不 raise）。
- **不可变模式**：`merge_interpretation_queries` 等改造保持返回新 dict，不 mutate 输入。
- **探针日志铁律**：任何静默路径必须有日志（family scoring probe 教训），接地合成每请求必打 probe。
- **绝不抛异常**：接地合成/评分/embedding 任何失败降级，流程不变。
- **提交规范**：conventional commits（feat:/refactor:/test:/docs:），无 Co-Authored-By（全局禁用）。
- 现有测试不得破坏（尤其 `tests/test_technical_interpretation.py` 33 个、`tests/test_react_tools.py` 134 个、`tests/test_chat_relevance.py`、`tests/test_semantic_rerank.py`）。

---

### Task 1: 预解读维度输出 + parse 硬控制（`technical_interpretation.py`）

**Files:**
- Modify: `sources/long_task/technical_interpretation.py`（INTERPRET_SYSTEM_PROMPT、parse_interpretation、新增 `_clean_str_list`/`_parse_dimensions`/`MAX_DIMENSIONS`、format_interpretation_rubric 加接地分支）
- Test: `tests/test_technical_interpretation.py`（新增 TestParseDimensions、TestGroundedRubric）

**Interfaces:**
- Produces: `parse_interpretation(raw) -> Optional[dict]` 返回 dict 新增 `"dimensions": [{"name","role","terms","queries"}]`（≤3、角色去重、空维度丢弃、查询经 sanitize_uspto_query）；`MAX_DIMENSIONS = 3`；`format_interpretation_rubric(interp) -> str` 对含 `line`/`representatives` 的维度输出接地分支（数据驱动玩家强信号措辞），预检索路径措辞不变。

- [ ] **Step 1: 写失败测试（维度解析）**

追加到 `tests/test_technical_interpretation.py` 末尾（`if __name__ == "__main__"` 之前）：

```python
class TestParseDimensions(unittest.TestCase):
    def test_dimensions_parsed_and_capped_at_three(self):
        raw = _valid_raw()
        raw["dimensions"] = [
            {"name": "d1", "role": "核心器件/电路层", "terms": ["a"],
             "queries": ['("a")']},
            {"name": "d2", "role": "控制算法/电路层", "terms": [], "queries": []},
            {"name": "d3", "role": "场景应用层", "terms": ["b"], "queries": []},
            {"name": "d4", "role": "多余层", "terms": ["c"], "queries": []},
        ]
        parsed = ti.parse_interpretation(raw)
        self.assertEqual(len(parsed["dimensions"]), ti.MAX_DIMENSIONS)
        self.assertEqual(parsed["dimensions"][0]["name"], "d1")

    def test_dimension_role_deduped_and_empties_dropped(self):
        raw = _valid_raw()
        raw["dimensions"] = [
            {"name": "d1", "role": "核心层", "terms": ["a"]},
            {"name": "d2", "role": "核心层", "terms": ["b"]},   # dup role
            {"name": "", "role": "", "terms": []},               # empty
            {"name": "d3", "role": "场景层", "terms": ["c"]},
        ]
        parsed = ti.parse_interpretation(raw)
        self.assertEqual([d["name"] for d in parsed["dimensions"]], ["d1", "d3"])

    def test_dimension_queries_sanitized(self):
        raw = _valid_raw()
        raw["dimensions"] = [
            {"name": "d1", "role": "核心层",
             "queries": ["中文垃圾 AND stuff", '("a")']},
        ]
        parsed = ti.parse_interpretation(raw)
        self.assertNotIn("中文", parsed["dimensions"][0]["queries"][0])
        self.assertEqual(parsed["dimensions"][0]["queries"], ['("a")'])

    def test_no_dimensions_key_returns_empty_list(self):
        parsed = ti.parse_interpretation(_valid_raw())
        self.assertEqual(parsed["dimensions"], [])


class TestGroundedRubric(unittest.TestCase):
    """format_interpretation_rubric 的接地分支：数据驱动玩家强信号。"""

    def _grounded(self):
        return {
            "scheme": "多通道恒流驱动",
            "structure_terms": ["error amplifier"],
            "dimensions": [
                {"name": "驱动核心", "role": "核心器件/电路层",
                 "line": "逐通道独立闭环恒流", "representatives": ["ERP Power"]},
            ],
            "players": ["ERP Power", "Samsung"],
        }

    def test_grounded_branch_renders_dimensions_and_strong_players(self):
        rubric = ti.format_interpretation_rubric(self._grounded())
        self.assertIn("驱动核心", rubric)
        self.assertIn("逐通道独立闭环恒流", rubric)
        self.assertIn("ERP Power", rubric)
        self.assertIn("真实玩家榜", rubric)
        self.assertIn("评分可上调 3-5 分", rubric)

    def test_grounded_branch_never_uses_weak_signal_wording(self):
        rubric = ti.format_interpretation_rubric(self._grounded())
        self.assertNotIn("本身不构成相关性依据", rubric)

    def test_pre_branch_keeps_weak_signal_wording(self):
        rubric = ti.format_interpretation_rubric(_valid_raw())
        self.assertIn("本身不构成相关性依据", rubric)

    def test_pre_dimensions_without_lines_stay_in_pre_branch(self):
        raw = _valid_raw()
        raw["dimensions"] = [{"name": "d1", "role": "核心层", "terms": ["a"]}]
        rubric = ti.format_interpretation_rubric(raw)
        self.assertIn("本身不构成相关性依据", rubric)

    def test_grounded_empty_returns_empty(self):
        self.assertEqual(ti.format_interpretation_rubric(
            {"players": [], "dimensions": []}), "")
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_technical_interpretation.py -q`
Expected: FAIL（`parse_interpretation` 返回无 `dimensions` 键；`ti.MAX_DIMENSIONS` 不存在）。

- [ ] **Step 3: 实现**

在 `sources/long_task/technical_interpretation.py` 中：

a) 顶部常量区（`MAX_INTERP_QUERIES` 旁）加：

```python
# The dimension skeleton is capped at three layers (core device /
# control circuit / application scenario); never let the model's
# output exceed it — parse enforces the cap, not the model.
MAX_DIMENSIONS = 3
```

b) `INTERPRET_SYSTEM_PROMPT` 在规则 5（queries）之后插入新规则 6，原 6/7/8 顺延为 7/8/9，JSON 说明加 dimensions：

```python
    "5. queries：用于 US 授权专利全文检索的布尔检索式 3-5 条，方案词"
    "优先、可直接执行；多词短语加双引号、同组同义词用 OR、组间用 "
    "AND；每条最多 12 个关键词、250 字符；禁止出现中文\n"
    "6. dimensions：把技术需求按纵深拆解为 2-3 个技术维度，默认三层"
    "骨架：核心器件/电路层（实现该功能的硬件核心）、控制算法/电路层"
    "（实现该功能的控制逻辑）、场景应用层（该功能落地的应用与接口）；"
    "某层在目标领域无实质内容时并入最接近的层并简要说明；禁止超过 3 "
    "个维度；每个维度输出 name（维度名）、role（分层角色）、terms"
    "（该维度方案级英文词 3-6 个）、queries（该维度布尔检索式 1-2 条，"
    "规则同第 5 条）\n"
    "7. 若提供 cpc_hints（该技术领域命中的专利分类号及分类标题），吸收"
    "其中与该需求相关的分类措辞——分类标题代表专利文献对这类技术的"
    "官方命名\n"
    "8. main_lines：该领域的主要技术路线划分（2-3 条，每条一句话，包含"
    "典型电路结构/实现方式——如某一领域可划分为模拟恒流环路与数字 PWM "
    "驱动两条路线，这类划分帮助检索覆盖不同实现流派；只写该领域真实"
    "存在的路线，禁止编造）\n"
    "9. key_players：该领域的主要申请人（3-5 个，英文公司名，必须是该"
    "领域真实活跃的专利申请人；不确定时宁可少给，禁止编造）\n"
    'Return JSON: {"scheme": "...", "structure_terms": [...], '
    '"independence_terms": [...], "scenarios": [...], "queries": [...], '
    '"dimensions": [{"name", "role", "terms", "queries"}], '
    '"main_lines": [...], "key_players": [...]}'
```

c) 新增模块级辅助 + `_parse_dimensions`，重构 `parse_interpretation`：

```python
def _clean_str_list(raw: dict, key: str) -> list:
    """Non-empty string list for a key; [] on missing/foreign shapes."""
    items = raw.get(key) or []
    if not isinstance(items, list):
        return []
    return [str(t).strip() for t in items
            if isinstance(t, str) and str(t).strip()]


def _parse_dimensions(raw: dict) -> list:
    """Sanitize the model's dimension output.

    Hard rules, enforced here not trusted to the model: at most
    MAX_DIMENSIONS entries, first occurrence of each role label wins,
    empty dimensions (no name and no terms) dropped, per-dimension
    queries run through the same sanitizer as top-level queries.
    """
    dims = raw.get("dimensions")
    if not isinstance(dims, list):
        return []
    out: list = []
    seen_roles: set = set()
    for d in dims:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "").strip()
        role = str(d.get("role") or "").strip()
        terms = _clean_str_list(d, "terms")[:10]
        if not name and not terms:
            continue
        if role:
            if role in seen_roles:
                continue
            seen_roles.add(role)
        queries: list = []
        qseen: set = set()
        for q in _clean_str_list(d, "queries")[:4]:
            q = sanitize_uspto_query(q)
            if q and q not in qseen:
                qseen.add(q)
                queries.append(q)
        out.append({"name": name, "role": role, "terms": terms,
                    "queries": queries[:2]})
        if len(out) >= MAX_DIMENSIONS:
            break
    return out
```

`parse_interpretation` 函数体替换为：

```python
def parse_interpretation(raw: Any) -> Optional[dict]:
    """Validate and sanitize the LLM interpretation into a canonical dict.

    Returns None when the output has neither a scheme nor structure
    terms — callers then skip the interpretation entirely.  Queries are
    sanitized the same way as rewrite queries (CJK stripped, length
    capped) so they are safe to hand to the USPTO search API.
    """
    if not isinstance(raw, dict):
        return None
    scheme = str(raw.get("scheme") or "").strip()
    structure_terms = _clean_str_list(raw, "structure_terms")
    if not scheme and not structure_terms:
        return None
    queries: list = []
    seen: set = set()
    for q in _clean_str_list(raw, "queries"):
        q = sanitize_uspto_query(q)
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return {
        "scheme": scheme,
        "structure_terms": structure_terms[:15],
        "independence_terms": _clean_str_list(raw, "independence_terms")[:10],
        "scenarios": _clean_str_list(raw, "scenarios")[:6],
        "main_lines": _clean_str_list(raw, "main_lines")[:3],
        "key_players": _clean_str_list(raw, "key_players")[:5],
        "dimensions": _parse_dimensions(raw),
        "queries": queries[:MAX_INTERP_QUERIES],
    }
```

（原闭包 `_str_list` 删除，被 `_clean_str_list` 替代。）

d) `format_interpretation_rubric` 顶部（`if not interp: return ""` 之后）插入接地分支：

```python
    dims = interp.get("dimensions") or []
    is_grounded = bool(interp.get("players")) or any(
            isinstance(d, dict) and (d.get("line") or d.get("representatives"))
            for d in dims)
    if is_grounded:
        parts = []
        scheme = interp.get("scheme")
        if scheme:
            parts.append(f"技术方案解读：{scheme}")
        terms = interp.get("structure_terms") or []
        if terms:
            parts.append(
                f"关键结构词：{' / '.join(str(t) for t in terms[:10])}")
        for d in dims[:MAX_DIMENSIONS]:
            seg = str(d.get("name") or "维度")
            role = d.get("role")
            if role:
                seg += f"（{role}）"
            line = d.get("line")
            if line:
                seg += f"：{line}"
            reps = d.get("representatives") or []
            if reps:
                seg += f"；代表申请人：{'、'.join(str(r) for r in reps[:3])}"
            parts.append("· " + seg)
        for line in (interp.get("cpc_hint_lines") or [])[:3]:
            parts.append("· CPC 主线线索：" + str(line))
        players = interp.get("players") or []
        if players:
            parts.append(
                "真实玩家榜（数据驱动，来自检索结果统计）："
                + " / ".join(str(p) for p in players[:5])
                + "。申请人命中该榜且其他相关性信号吻合时，视为同领域"
                "证据，评分可上调 3-5 分（满分 100）。"
            )
        if not parts:
            return ""
        return (
            "评分补充（来自检索后接地解读）：候选专利的标题/CPC/申请人"
            "若命中以下维度/玩家，即使不含查询字面词，也应视为同一技术"
            "方向，评分相应提高。\n" + "\n".join(parts)
        )
    parts = []
    # ... 原预检索分支代码保持原样（弱信号措辞不变）...
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_technical_interpretation.py -q`
Expected: PASS（33 旧 + 9 新）。

- [ ] **Step 5: 通用性审计（铁律）**

Run: `grep -n "RGB\|控制放大器\|独立控制" sources/long_task/technical_interpretation.py`
Expected: 无输出。若命中，改写 prompt 措辞。

- [ ] **Step 6: Commit**

```bash
git add sources/long_task/technical_interpretation.py tests/test_technical_interpretation.py
git commit -m "feat: 技术解读新增维度骨架输出（三层硬控制≤3）与接地 rubric 分支"
```

---

### Task 2: 按维度取查询（`merge_interpretation_queries`）

**Files:**
- Modify: `sources/long_task/technical_interpretation.py`（新增 `_dimension_queries`、改 `merge_interpretation_queries`）
- Test: `tests/test_technical_interpretation.py`（新增 TestMergeDimensionQueries）

**Interfaces:**
- Consumes: Task 1 的 `_parse_dimensions`（dimensions 各带 `queries`）、`expand_query_ladder`、`MAX_INTERP_LADDER_SLOTS`。
- Produces: `merge_interpretation_queries(rewrite, interp, cap=MAX_LADDER_QUERIES) -> dict` — 有维度时按维度取（各维度 chain 轮转交错，≤3 槽）；无维度时走原扁平链路径（行为不变，旧测试全绿）。

- [ ] **Step 1: 写失败测试**

```python
class TestMergeDimensionQueries(unittest.TestCase):
    def test_one_query_per_dimension_round_robin(self):
        interp = {"dimensions": [
            {"name": "d1", "queries": ['("a") AND ("b") AND ("c")']},
            {"name": "d2", "queries": ['("d") AND ("e")']},
            {"name": "d3", "queries": ['("f")']},
        ]}
        merged = ti.merge_interpretation_queries({"queries": []}, interp)
        self.assertEqual(merged["queries"][:3],
                         ['("a") AND ("b") AND ("c")',
                          '("d") AND ("e")', '("f")'])

    def test_two_dimensions_depth_fills_third_slot(self):
        interp = {"dimensions": [
            {"name": "d1", "queries": ['("a") AND ("b") AND ("c")']},
            {"name": "d2", "queries": ['("d") AND ("e")']},
        ]}
        merged = ti.merge_interpretation_queries({"queries": []}, interp)
        self.assertEqual(merged["queries"][:3],
                         ['("a") AND ("b") AND ("c")',
                          '("d") AND ("e")', '("a") AND ("b")'])

    def test_dimension_queries_dedupe_against_rewrite(self):
        interp = {"dimensions": [{"name": "d1", "queries": ['("a")']}]}
        merged = ti.merge_interpretation_queries(
            {"queries": ['("a")', "tail"]}, interp)
        self.assertEqual(merged["queries"], ['("a")', "tail"])

    def test_flat_path_unchanged_without_dimensions(self):
        merged = ti.merge_interpretation_queries(
            {"queries": []},
            {"queries": ['("a") AND ("b") AND ("c")']})
        self.assertEqual(
            merged["queries"],
            ['("a") AND ("b") AND ("c")', '("a") AND ("b")', '("a")'])
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_technical_interpretation.py -q`
Expected: FAIL（维度查询未进 ladder，首个断言空列表）。

- [ ] **Step 3: 实现**

新增（`expand_query_ladder` 之后）：

```python
def _dimension_queries(interp: Optional[dict]) -> list:
    """Per-dimension ladder head, round-robin interleaved.

    Each dimension contributes its tight-to-loose AND-drop chain; the
    chains are interleaved (d1[0], d2[0], d3[0], d1[1], ...) so the
    ladder head covers every facet before any single dimension's
    fallbacks, capped at MAX_INTERP_LADDER_SLOTS.  A dimension without
    queries contributes nothing.
    """
    dims = (interp or {}).get("dimensions") or []
    chains: list = []
    for d in dims:
        if not isinstance(d, dict):
            continue
        for q in (d.get("queries") or []):
            q = str(q).strip()
            if q:
                chains.append(expand_query_ladder(q))
                break
    interleaved: list = []
    i = 0
    while any(len(ch) > i for ch in chains) \
            and len(interleaved) < MAX_INTERP_LADDER_SLOTS:
        for ch in chains:
            if i < len(ch):
                interleaved.append(ch[i])
                if len(interleaved) >= MAX_INTERP_LADDER_SLOTS:
                    break
        i += 1
    return interleaved
```

`merge_interpretation_queries` 函数体替换为：

```python
def merge_interpretation_queries(rewrite: dict, interp: Optional[dict],
                                 cap: int = MAX_LADDER_QUERIES) -> dict:
    """Prepend the interpretation's queries to the rewrite ladder.

    With a dimension skeleton, each dimension contributes one ladder
    head entry (round-robin over its AND-drop chains) so retrieval
    covers every facet; without dimensions the flat interpretation
    queries expand into their full chains as before.  The auto-ladder
    and the blank-q injection both pick from the head.  Returns a new
    rewrite dict; the input is never mutated.  Dedupes against existing
    ladder entries.
    """
    out = dict(rewrite or {})
    existing = [q for q in (rewrite or {}).get("queries") or [] if q]
    dims = (interp or {}).get("dimensions") or []
    chain: list = []
    seen: set = set()
    if dims:
        for sub in _dimension_queries(interp):
            if sub and sub not in seen and sub not in existing:
                seen.add(sub)
                chain.append(sub)
    else:
        for q in (interp or {}).get("queries") or []:
            for sub in expand_query_ladder(str(q)):
                if sub and sub not in seen and sub not in existing:
                    seen.add(sub)
                    chain.append(sub)
    chain = chain[:MAX_INTERP_LADDER_SLOTS]
    # The flash rewrite's tail (its single-concept fallbacks) must stay
    # reachable, so the chain never starves it out of the ladder.
    merged = chain + existing
    out["queries"] = merged[:cap]
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_technical_interpretation.py -q`
Expected: PASS（全部，含旧扁平路径用例）。

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/technical_interpretation.py tests/test_technical_interpretation.py
git commit -m "feat: 解读查询按维度轮转进 ladder，覆盖所有技术层面"
```

---

### Task 3: 接地合成模块（新文件 `sources/long_task/grounded_interpretation.py`）

**Files:**
- Create: `sources/long_task/grounded_interpretation.py`
- Test: `tests/test_grounded_interpretation.py`（新建）

**Interfaces:**
- Consumes: `sanitize_uspto_query`（`sources.long_task.search_query_builder`）；Task 1 的预解读 dict 形状。
- Produces（Task 6 使用）:
  - `GROUNDED_ENABLED`、`GROUNDED_HEAD`、`GROUNDED_MIN`、`GROUNDED_MODEL`、`GROUNDED_PROVIDER`、`GROUNDED_TIMEOUT`
  - `candidate_stats(candidates) -> dict` — `{"applicants": [{"name","count"}], "cpc": [{"name","count"}]}`（频次降序）
  - `build_synthesis_input(question, pre_interp, candidates, stats, cpc_hints=None) -> dict`
  - `parse_grounded(raw) -> Optional[dict]` — `{"dimensions": [...], "players": [...], "supplementary_queries": [...], "supplementary_cpc": [...]}`
  - `merge_grounded(stats, llm_out, pre_interp=None) -> dict` — llm_out=None 时统计版保底
  - `synthesize_grounded(question, candidates, pre_interp=None, cpc_hints=None) -> Optional[dict]`（async；flash 失败 → 统计版；禁用/空问题 → None）
  - `GROUNDED_SYSTEM_PROMPT`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_grounded_interpretation.py`：

```python
"""Tests for the post-retrieval grounded interpretation module."""
import asyncio
import unittest
from unittest import mock

from sources.long_task import grounded_interpretation as gi
from sources.long_task import technical_interpretation as ti


def _cand(pid, title, applicant="ACME", cpc=("H05B45/20",), score=4,
          filing="2023-01-01"):
    return {"patent_id": pid, "title": title, "applicant": applicant,
            "cpc_codes": list(cpc), "relevance_score": score,
            "filing_date": filing}


class TestCandidateStats(unittest.TestCase):
    def test_applicant_and_cpc_frequency_desc(self):
        stats = gi.candidate_stats([
            _cand("1", "A", "ERP Power", ("H05B45/20", "H05B45/10")),
            _cand("2", "B", "ERP Power", ("H05B45/20",)),
            _cand("3", "C", "Samsung", ()),
        ])
        self.assertEqual(stats["applicants"][0],
                         {"name": "ERP Power", "count": 2})
        self.assertEqual(stats["cpc"][0],
                         {"name": "H05B45/20", "count": 2})
        self.assertEqual(len(stats["applicants"]), 2)

    def test_empty_input(self):
        self.assertEqual(gi.candidate_stats([]),
                         {"applicants": [], "cpc": []})


class TestParseGrounded(unittest.TestCase):
    def test_valid_output_parsed(self):
        parsed = gi.parse_grounded({
            "dimensions": [
                {"name": "D1", "role": "核心层", "line": "L1",
                 "representatives": ["ERP"], "players": ["ERP", "TI"],
                 "cpc": ["h05b45/20"]},
            ],
            "supplementary_queries": ["中文垃圾 AND stuff", '("a")'],
            "supplementary_cpc": ["h05b45/20", "H05B45/20"],
        })
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["dimensions"][0]["representatives"], ["ERP"])
        self.assertEqual(parsed["dimensions"][0]["cpc"], ["H05B45/20"])
        self.assertNotIn("中文", parsed["supplementary_queries"][0])
        self.assertEqual(parsed["supplementary_cpc"], ["H05B45/20"])  # dedup

    def test_unusable_returns_none(self):
        self.assertIsNone(gi.parse_grounded({}))
        self.assertIsNone(gi.parse_grounded("nope"))
        self.assertIsNone(gi.parse_grounded(
            {"dimensions": [], "players": [], "supplementary_queries": []}))


class TestMergeGrounded(unittest.TestCase):
    def test_llm_output_wins_over_stats(self):
        stats = {"applicants": [{"name": "X", "count": 5}], "cpc": []}
        llm = gi.parse_grounded({
            "dimensions": [{"name": "D", "line": "L"}],
            "players": ["ERP"], "supplementary_queries": ['("q")'],
            "supplementary_cpc": ["H05B45/20"],
        })
        out = gi.merge_grounded(stats, llm, {"scheme": "S",
                                             "structure_terms": ["t"]})
        self.assertEqual(out["players"], ["ERP"])
        self.assertEqual(out["scheme"], "S")
        self.assertEqual(out["structure_terms"], ["t"])

    def test_stats_only_fallback_when_llm_fails(self):
        stats = {"applicants": [{"name": "ERP Power", "count": 3},
                                {"name": "Samsung", "count": 2}],
                 "cpc": [{"name": "H05B45/20", "count": 3}]}
        out = gi.merge_grounded(stats, None, None)
        self.assertEqual(out["players"], ["ERP Power", "Samsung"])
        self.assertEqual(out["dimensions"], [])
        self.assertEqual(out["supplementary_queries"], [])


class TestSynthesizeGrounded(unittest.IsolatedAsyncioTestCase):
    class _FakeProvider:
        def __init__(self, result=None, fail=False):
            self.result = result
            self.fail = fail

        async def complete_json(self, system, user, max_retries=0):
            if self.fail:
                raise RuntimeError("down")
            return self.result

    async def test_success_returns_merged(self):
        llm = {"dimensions": [{"name": "D", "line": "L",
                               "representatives": ["ERP"]}],
               "players": ["ERP"],
               "supplementary_queries": ['("q")'],
               "supplementary_cpc": ["H05B45/20"]}
        provider = self._FakeProvider(result=llm)
        with mock.patch.object(gi, "_grounded_provider",
                               return_value=provider):
            out = await gi.synthesize_grounded(
                "问题", [_cand("1", "T")],
                pre_interp={"scheme": "S"})
        self.assertIsNotNone(out)
        self.assertEqual(out["dimensions"][0]["name"], "D")
        self.assertEqual(out["scheme"], "S")

    async def test_failure_falls_back_to_stats_only(self):
        provider = self._FakeProvider(fail=True)
        with mock.patch.object(gi, "_grounded_provider",
                               return_value=provider):
            out = await gi.synthesize_grounded(
                "问题", [_cand("1", "T", applicant="ERP")],
                pre_interp=None)
        self.assertIsNotNone(out)
        self.assertEqual(out["players"], ["ERP"])

    async def test_disabled_returns_none_without_calling(self):
        provider = self._FakeProvider(result={"players": ["X"]})
        with mock.patch.object(gi, "GROUNDED_ENABLED", False):
            with mock.patch.object(gi, "_grounded_provider",
                                   return_value=provider):
                self.assertIsNone(await gi.synthesize_grounded("问题", []))
        self.assertEqual(provider.fail, False)  # no call happened

    async def test_empty_question_returns_none(self):
        with mock.patch.object(gi, "_grounded_provider") as p:
            self.assertIsNone(await gi.synthesize_grounded("", []))
        p.assert_not_called()

    async def test_hanging_provider_times_out_to_stats_fallback(self):
        class _Slow:
            async def complete_json(self, system, user, max_retries=0):
                await asyncio.sleep(30)
                return {}

        with mock.patch.object(gi, "GROUNDED_TIMEOUT", 0.1):
            with mock.patch.object(gi, "_grounded_provider",
                                   return_value=_Slow()):
                out = await gi.synthesize_grounded(
                    "问题", [_cand("1", "T", applicant="ERP")])
        self.assertIsNotNone(out)
        self.assertEqual(out["players"], ["ERP"])


class TestPromptGenericity(unittest.TestCase):
    """通用性铁律（第三次重申）：生产 prompt 零测试提问词汇。"""

    FORBIDDEN = ("RGB", "控制放大器", "独立控制")

    def test_interpret_prompt_free_of_test_vocabulary(self):
        for word in self.FORBIDDEN:
            self.assertNotIn(word, ti.INTERPRET_SYSTEM_PROMPT)

    def test_grounded_prompt_free_of_test_vocabulary(self):
        for word in self.FORBIDDEN:
            self.assertNotIn(word, gi.GROUNDED_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_grounded_interpretation.py -q`
Expected: FAIL（模块不存在：ImportError / AttributeError）。

- [ ] **Step 3: 实现**

新建 `sources/long_task/grounded_interpretation.py`：

```python
"""Post-retrieval grounded interpretation of the user question.

The pre-retrieval interpretation (``technical_interpretation``) maps
the question to architecture vocabulary from model knowledge.  Its
players/main lines are knowledge-layer, though — production showed
lighting giants instead of the domain's specialised players.  This
module grounds the interpretation in the actual candidate pool: it
counts applicant/CPC frequencies over the scored candidates, asks the
Flash LLM to cluster the top candidates under the pre-interpretation's
dimension skeleton, and returns per-dimension main lines with
representatives plus supplementary queries/CPC codes for the loop.

Enhancement, not a dependency: any failure degrades to a stats-only
version (applicant frequency + CPC title groups), and a stats failure
returns None so callers keep their flow untouched.  The prompt is
generic — the question is passed at runtime, never baked in.
"""

import asyncio
import json
import os
from typing import Any, Optional

from sources.long_task.search_query_builder import sanitize_uspto_query

GROUNDED_ENABLED = os.getenv("REACT_GROUNDED_ENABLED", "1") == "1"
GROUNDED_MODEL = os.getenv("REACT_GROUNDED_MODEL", "deepseek-v4-flash")
GROUNDED_PROVIDER = os.getenv("REACT_GROUNDED_PROVIDER", "deepseek")
GROUNDED_HEAD = int(os.getenv("REACT_GROUNDED_HEAD", "30"))
GROUNDED_MIN = int(os.getenv("REACT_GROUNDED_MIN", "15"))
MAX_GROUNDED_DIMENSIONS = 3
MAX_GROUNDED_CANDIDATES = 30


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to *default* on garbage."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


GROUNDED_TIMEOUT = _env_int("REACT_GROUNDED_TIMEOUT", 30)

# Provider construction is expensive and must stay lazy; cache one.
_GROUNDED_PROVIDER_CACHE: dict = {}


def _grounded_provider():
    if "provider" not in _GROUNDED_PROVIDER_CACHE:
        from sources.llm_provider import Provider
        _GROUNDED_PROVIDER_CACHE["provider"] = Provider(
            provider_name=GROUNDED_PROVIDER, model=GROUNDED_MODEL,
            server_address="", is_local=False)
    return _GROUNDED_PROVIDER_CACHE["provider"]


def candidate_stats(candidates: list) -> dict:
    """Applicant and CPC frequencies over the candidates, desc.

    Pure code, zero LLM: this layer survives any model failure.
    """
    applicants: dict = {}
    cpc: dict = {}
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        name = str(c.get("applicant") or "").strip()
        if name:
            applicants[name] = applicants.get(name, 0) + 1
        for code in (c.get("cpc_codes") or []):
            code = str(code).strip().upper()
            if code:
                cpc[code] = cpc.get(code, 0) + 1

    def _top(counts: dict) -> list:
        return [{"name": k, "count": v} for k, v in
                sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    return {"applicants": _top(applicants), "cpc": _top(cpc)}


def build_synthesis_input(question: str, pre_interp: Optional[dict],
                          candidates: list, stats: dict,
                          cpc_hints: Optional[list] = None) -> dict:
    """Assemble the flash synthesis payload from deterministic facts."""
    pre: dict = {}
    if pre_interp:
        pre["scheme"] = str(pre_interp.get("scheme") or "")
        pre["dimensions"] = [
            {"name": str(d.get("name") or ""),
             "role": str(d.get("role") or ""),
             "terms": [str(t) for t in (d.get("terms") or [])[:6]]}
            for d in (pre_interp.get("dimensions") or [])[:3]
            if isinstance(d, dict)]
        pre["structure_terms"] = [
            str(t) for t in (pre_interp.get("structure_terms") or [])[:10]]
    cands = []
    for c in (candidates or [])[:MAX_GROUNDED_CANDIDATES]:
        if not isinstance(c, dict):
            continue
        cands.append({
            "id": str(c.get("patent_id") or ""),
            "title": str(c.get("title") or ""),
            "applicant": str(c.get("applicant") or ""),
            "cpc": [str(x).upper() for x in (c.get("cpc_codes") or [])[:5]],
            "score": c.get("relevance_score"),
            "filing": str(c.get("filing_date") or ""),
        })
    return {
        "question": str(question),
        "pre_interpretation": pre,
        "candidates": cands,
        "applicant_stats": (stats or {}).get("applicants") or [],
        "cpc_stats": (stats or {}).get("cpc") or [],
        "cpc_hints": [
            {"code": str(h.get("code", "")), "title": str(h.get("title", ""))}
            for h in (cpc_hints or [])[:8] if isinstance(h, dict) and h.get("code")],
    }


GROUNDED_SYSTEM_PROMPT = (
    "你是资深专利检索专家。系统给出一句技术需求、预检索技术解读"
    "（含技术维度骨架）、以及从检索结果抽取的候选专利事实（标题/"
    "申请人/CPC/相关度评分/申请年）与统计（申请人频次、CPC 频次）。\n"
    "你的任务：基于候选数据产出检索后接地解读。只输出 JSON，不要其他文字。\n"
    "规则：\n"
    "1. 以预解读的维度骨架为初始结构，依据候选数据验证/调整/合并/拆分"
    "维度；每个维度输出：name（维度名）、role（分层角色）、line（该"
    "维度在专利文献中的主线描述，一句话，含典型实现方式）、"
    "representatives（1-3 个代表申请人，必须来自候选数据中真实出现"
    "的申请人）、players（该维度活跃申请人 2-5 个，来自数据）、"
    "cpc（该维度对应 CPC 代码 1-3 个，来自数据统计）\n"
    "2. representatives 与 players 只能来自给定的申请人频次与候选"
    "数据，禁止编造；数据不足的维度宁可少给\n"
    "3. supplementary_queries：针对候选覆盖不足或缺失方向的布尔检索式"
    "2-4 条，方案词优先、可直接执行；多词短语加双引号、同组同义词 "
    "OR、组间 AND；每条最多 12 个关键词、250 字符；禁止中文\n"
    "4. supplementary_cpc：主线对应的 CPC 代码 1-4 个，来自 cpc 频次"
    "统计\n"
    'Return JSON: {"dimensions": [{"name", "role", "line", '
    '"representatives", "players", "cpc"}], "supplementary_queries": '
    '[...], "supplementary_cpc": [...]}'
)


def _clean_strs(raw: Any, key: str, cap: int) -> list:
    items = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    return [str(v).strip() for v in items
            if isinstance(v, str) and str(v).strip()][:cap]


def parse_grounded(raw: Any) -> Optional[dict]:
    """Validate/sanitize the synthesis LLM output.

    None when nothing usable — callers fall back to the stats-only
    version.  Queries go through the same sanitizer as pre-interpretation
    queries; CPC codes are uppercased and deduped.
    """
    if not isinstance(raw, dict):
        return None
    dims: list = []
    for d in (raw.get("dimensions") or []):
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "").strip()
        line = str(d.get("line") or "").strip()
        if not name and not line:
            continue
        dims.append({
            "name": name,
            "role": str(d.get("role") or "").strip(),
            "line": line,
            "representatives": _clean_strs(d, "representatives", 3),
            "players": _clean_strs(d, "players", 5),
            "cpc": list(dict.fromkeys(
                c for c in _clean_strs(d, "cpc", 3) if c and c.upper())),
        })
        if len(dims) >= MAX_GROUNDED_DIMENSIONS:
            break
    players = _clean_strs(raw, "players", 5)
    queries: list = []
    qseen: set = set()
    for q in (raw.get("supplementary_queries") or []):
        if not isinstance(q, str):
            continue
        q = sanitize_uspto_query(q)
        if q and q not in qseen:
            qseen.add(q)
            queries.append(q)
    cpc = list(dict.fromkeys(
        c.upper() for c in (raw.get("supplementary_cpc") or [])
        if isinstance(c, str) and c.strip()))[:4]
    if not dims and not players and not queries:
        return None
    return {"dimensions": dims, "players": players,
            "supplementary_queries": queries[:4],
            "supplementary_cpc": cpc}


_CPC_TITLES: Optional[dict] = None


def _cpc_title(code: str) -> str:
    """Lazy-load the CPC subgroup titles json once; "" on any failure."""
    global _CPC_TITLES
    if _CPC_TITLES is None:
        _CPC_TITLES = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "data/cpc/cpc_titles_subgroups.json"),
                      encoding="utf-8") as fh:
                for entry in json.load(fh) or []:
                    if isinstance(entry, dict) and entry.get("code"):
                        _CPC_TITLES[str(entry["code"])] = str(
                            entry.get("title") or "")
        except (OSError, ValueError):
            pass
    return _CPC_TITLES.get(code, "")


def merge_grounded(stats: dict, llm_out: Optional[dict],
                   pre_interp: Optional[dict] = None) -> dict:
    """Merge the synthesis result with the pre-interpretation.

    The pre-interpretation's scheme/structure terms ride along so the
    rubric keeps its architecture vocabulary.  *llm_out* None (flash
    failure) falls back to a stats-only version: applicant frequency
    players plus CPC-title group lines.  Never raises.
    """
    base: dict = {"dimensions": [], "players": [],
                  "supplementary_queries": [], "supplementary_cpc": [],
                  "cpc_hint_lines": []}
    for k in ("scheme", "structure_terms", "independence_terms"):
        v = (pre_interp or {}).get(k)
        if v:
            base[k] = v
    if llm_out:
        out = dict(base)
        out["dimensions"] = llm_out.get("dimensions") or []
        out["players"] = llm_out.get("players") or []
        out["supplementary_queries"] = llm_out.get("supplementary_queries") or []
        out["supplementary_cpc"] = llm_out.get("supplementary_cpc") or []
        return out
    out = dict(base)
    out["players"] = [e["name"] for e in (stats.get("applicants") or [])[:5]]
    for entry in (stats.get("cpc") or [])[:3]:
        title = _cpc_title(entry["name"])
        if title:
            out["cpc_hint_lines"].append(f"{entry['name']} {title}")
    return out


async def synthesize_grounded(question: str, candidates: list,
                              pre_interp: Optional[dict] = None,
                              cpc_hints: Optional[list] = None) -> Optional[dict]:
    """Grounded synthesis via the Flash LLM.  Never raises.

    Stats are computed first; the flash call either produces the full
    grounded interpretation or degrades to the stats-only version.
    Returns None only when disabled or the question is empty — callers
    then keep their pre-interpretation flow untouched.
    """
    if not GROUNDED_ENABLED:
        return None
    question = str(question or "").strip()
    if not question:
        return None
    stats = candidate_stats(candidates or [])
    try:
        provider = _grounded_provider()
        payload = build_synthesis_input(
            question, pre_interp, candidates or [], stats, cpc_hints)
        result = await asyncio.wait_for(
            provider.complete_json(
                GROUNDED_SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                max_retries=1),
            timeout=GROUNDED_TIMEOUT)
        return merge_grounded(stats, parse_grounded(result), pre_interp)
    except Exception:
        return merge_grounded(stats, None, pre_interp)
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_grounded_interpretation.py -q`
Expected: PASS（全部 11 个新用例）。

- [ ] **Step 5: 通用性审计**

Run: `grep -n "RGB\|控制放大器\|独立控制" sources/long_task/grounded_interpretation.py`
Expected: 无输出。

- [ ] **Step 6: Commit**

```bash
git add sources/long_task/grounded_interpretation.py tests/test_grounded_interpretation.py
git commit -m "feat: 检索后接地解读模块（统计+flash 合成+统计版保底）"
```

---

### Task 4: Flash 评分并发增强（`chat_relevance.py`）

**Files:**
- Modify: `sources/long_task/chat_relevance.py:26-27`（常量）、`score_candidates_concurrent`
- Test: `tests/test_chat_relevance.py`（更新 2 个既有用例 + 新增并发上限用例）

**Interfaces:**
- Consumes: 无。
- Produces: `SCORE_BATCH_SIZE`（默认 10，env `REACT_SCORE_BATCH_SIZE`）、`SCORE_MAX_CONCURRENCY`（默认 6，env `REACT_SCORE_MAX_CONCURRENCY`）；`score_candidates_concurrent(candidates, query, provider, rubric="") -> int` 签名不变（家族评分/补查询自动受益）。

- [ ] **Step 1: 更新既有用例并写失败测试**

`tests/test_chat_relevance.py`：

a) `TestSearchPoolConcurrentScoring.test_batches_scored_concurrently` 断言更新（25→10 批、并发被信号量压到 6）：

```python
    async def test_batches_scored_concurrently(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item(str(19500000 + i), f"T{i}")
                  for i in range(250)])
        provider = _ConcurrentProvider()
        scored = await pool.score_new(provider)
        # SCORE_BATCH_SIZE=10 → 250/10 = 25 batches; concurrency capped at 6
        self.assertEqual(provider.calls, 25)
        self.assertEqual(provider.max_inflight, 6)
        self.assertEqual(scored, 250)
```

b) `TestHeadScoringAndConcurrency.test_score_concurrent_uses_small_batches` 断言更新：

```python
    async def test_score_concurrent_uses_small_batches(self):
        from sources.long_task.chat_relevance import (
            SCORE_BATCH_SIZE, score_candidates_concurrent)
        from sources.long_task.candidate_metadata import build_candidates
        items = [_usp_raw_item(str(19500000 + i), f"T{i}")
                 for i in range(60)]
        cands = build_candidates(items)
        provider = _ConcurrentProvider()
        scored = await score_candidates_concurrent(cands, "q", provider)
        self.assertEqual(provider.calls, 6)          # 60/10 → 6 batches
        self.assertEqual(provider.max_inflight, 6)   # min(6 batches, cap 6)
        self.assertEqual(scored, 60)
```

c) 新增并发上限用例（`TestHeadScoringAndConcurrency` 内追加）：

```python
    async def test_concurrency_capped_by_semaphore(self):
        from sources.long_task import chat_relevance as cr
        from sources.long_task.candidate_metadata import build_candidates
        items = [_usp_raw_item(str(19500000 + i), f"T{i}")
                 for i in range(100)]
        cands = build_candidates(items)
        provider = _ConcurrentProvider()
        with mock.patch.object(cr, "SCORE_MAX_CONCURRENCY", 2):
            scored = await cr.score_candidates_concurrent(cands, "q", provider)
        self.assertEqual(provider.calls, 10)         # 100/10 → 10 batches
        self.assertEqual(provider.max_inflight, 2)   # capped at 2
        self.assertEqual(scored, 100)
```

文件头 import 加 `from unittest import mock`。

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_chat_relevance.py -q`
Expected: FAIL（calls=10/max_inflight=10 与新版断言不符；`SCORE_MAX_CONCURRENCY` 不存在）。

- [ ] **Step 3: 实现**

`chat_relevance.py` 常量区（`SCORE_BATCH_SIZE` 行）替换：

```python
SCORE_BATCH_SIZE = int(os.getenv("REACT_SCORE_BATCH_SIZE", "10"))
# Cap concurrent scoring calls so a large pool cannot hammer the
# gateway into 429s; the semaphore bounds the gather burst.
SCORE_MAX_CONCURRENCY = int(os.getenv("REACT_SCORE_MAX_CONCURRENCY", "6"))
```

`score_candidates_concurrent` 函数体替换（docstring 同步更新：25→10、并发上限）：

```python
async def score_candidates_concurrent(candidates: list, query: str,
                                      provider: Any,
                                      rubric: str = "") -> int:
    """Score unscored candidates in concurrent small batches.

    The Flash provider is slow on 100-entry batches (~50s); 10-entry
    batches gathered concurrently with a semaphore cap cut wall-clock
    substantially without hammering the gateway.  Dead candidates
    (expired/abandoned/PCT storage) are skipped — they rank below every
    live candidate regardless of score, so scoring them is pure waste.
    *rubric* (optional) is the architecture-level interpretation
    supplement passed through to the gate prompt.  Never raises.
    Returns how many candidates gained a score.
    """
    if provider is None:
        return 0
    pending = [c for c in candidates
               if "relevance_score" not in c
               and not is_dead_status(c.get("status"))]
    if not pending:
        return 0
    batches = [
        pending[i:i + SCORE_BATCH_SIZE]
        for i in range(0, len(pending), SCORE_BATCH_SIZE)
    ]
    sem = asyncio.Semaphore(max(1, SCORE_MAX_CONCURRENCY))

    async def _scored(batch: list) -> None:
        async with sem:
            await score_candidates(batch, query, provider, rubric)

    await asyncio.gather(*(_scored(batch) for batch in batches))
    return len(pending) - len([c for c in pending
                               if "relevance_score" not in c])
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_chat_relevance.py -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/chat_relevance.py tests/test_chat_relevance.py
git commit -m "perf: flash 评分批 10 + 信号量 6 并发上限"
```

---

### Task 5: 语义模型并发增强（`semantic_rerank.py` async + 分块）

**Files:**
- Modify: `sources/long_task/semantic_rerank.py`（`SEMANTIC_BATCH_SIZE`、`semantic_scores_batch`/`rerank_candidates` 改 async + to_thread 分块）
- Modify: `sources/agents/react_tools.py:629,699`（两处调用补 `await`）
- Test: `tests/test_semantic_rerank.py`（改 async 用例 + 分块用例）、`tests/test_react_tools.py`（TestSemanticRerankWiring 若签名报错则补 await）

**Interfaces:**
- Consumes: 无。
- Produces: `SEMANTIC_BATCH_SIZE = int(os.getenv("SEMANTIC_BATCH_SIZE", "64"))`；`async def semantic_scores_batch(query, candidates) -> dict`；`async def rerank_candidates(query, candidates, top_k=RERANK_TOP_K, alpha=RERANK_ALPHA) -> list`。签名参数不变、仅变 async；`cosine_similarity`/`fuse_ranking` 纯函数不动。

- [ ] **Step 1: 找全调用方**

Run: `grep -rn "semantic_scores_batch\|rerank_candidates" sources/ tests/ | grep -v "def \|semantic_rerank.py"`
Expected: 调用方仅 `sources/agents/react_tools.py:629`（semantic_scores_batch）与 `:699`（rerank_candidates）；测试仅 `tests/test_semantic_rerank.py`（直接用函数）与 `tests/test_react_tools.py` TestSemanticRerankWiring（经 _rank_pending_pool 间接）。

- [ ] **Step 2: 写失败测试（先改测试）**

`tests/test_semantic_rerank.py` 顶部 import 加 `import asyncio` 不需要（IsolatedAsyncioTestCase 自带）；`TestSemanticScoresBatch` 改为 `unittest.IsolatedAsyncioTestCase`，用例补 `await` 并新增分块用例：

```python
class TestSemanticScoresBatch(unittest.IsolatedAsyncioTestCase):
    async def test_returns_cosines_keyed_by_patent_id(self):
        with patch("sources.long_task.semantic_rerank.embed_texts",
                   return_value=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]):
            scores = await semantic_scores_batch("q", [
                {"patent_id": "a", "title": "A"},
                {"patent_id": "b", "title": "B"},
                {"patent_id": "c", "title": "   "},  # untitled — skipped
            ])
        self.assertAlmostEqual(scores["a"], 1.0)
        self.assertAlmostEqual(scores["b"], 0.0)
        self.assertNotIn("c", scores)

    async def test_embedding_failure_returns_empty(self):
        with patch("sources.long_task.semantic_rerank.embed_texts",
                   return_value=None):
            scores = await semantic_scores_batch(
                "q", [{"patent_id": "a", "title": "A"},
                      {"patent_id": "b", "title": "B"}])
        self.assertEqual(scores, {})

    async def test_fewer_than_two_titled_returns_empty(self):
        with patch("sources.long_task.semantic_rerank.embed_texts") as mock:
            scores = await semantic_scores_batch(
                "q", [{"patent_id": "a", "title": "A"}])
        self.assertEqual(scores, {})
        mock.assert_not_called()

    async def test_large_batch_chunked_concurrently(self):
        # query embed + 64-title chunk + 36-title chunk = 3 calls
        side = [[[1.0, 0.0]]] + [[[1.0, 0.0]] * 64] + [[[0.0, 1.0]] * 36]
        with patch("sources.long_task.semantic_rerank.SEMANTIC_BATCH_SIZE",
                   64), \
             patch("sources.long_task.semantic_rerank.embed_texts",
                   side_effect=side) as m:
            cands = [{"patent_id": str(i), "title": f"T{i}"}
                     for i in range(100)]
            scores = await semantic_scores_batch("q", cands)
        self.assertEqual(m.call_count, 3)
        self.assertEqual(len(scores), 100)
        self.assertAlmostEqual(scores["0"], 1.0)
        self.assertAlmostEqual(scores["99"], 0.0)
```

`tests/test_react_tools.py` 的 TestSemanticRerankWiring 若因 async 签名报错（`TypeError: object ... can't be used in 'await'` 或 mock 行为变化），在 `_rank_pending_pool` 内已由 `await` 调用——同步 patch `embed_texts` 在 to_thread 中仍生效，预期无需改动；若失败则在该用例调用链补对应调整（运行 Step 3 后见分晓）。

- [ ] **Step 3: 实现**

`semantic_rerank.py`：

a) 常量区（PRESCORE_ENABLED 之后）加：

```python
# Embedding calls run chunked in a thread pool: one giant synchronous
# batch blocks the event loop for seconds, while chunked to_thread
# calls parallelize and never stall the loop.
SEMANTIC_BATCH_SIZE = int(os.getenv("SEMANTIC_BATCH_SIZE", "64"))
```

b) 新增辅助（`embed_texts` 之后）：

```python
async def _embed_chunks(query: str, titles: list) -> Optional[list]:
    """Embed the query once, then the title chunks in parallel threads.

    Returns [query_vec, [chunk vectors...]] or None when the query
    embedding failed.  Chunk-level failures surface as None entries —
    callers decide how strict to be.
    """
    qv = await asyncio.to_thread(embed_texts, [str(query)])
    if qv is None or len(qv) != 1:
        return None
    chunks = [titles[i:i + SEMANTIC_BATCH_SIZE]
              for i in range(0, len(titles), SEMANTIC_BATCH_SIZE)]
    results = await asyncio.gather(
        *(asyncio.to_thread(embed_texts, chunk) for chunk in chunks))
    return qv[0], chunks, results
```

c) `semantic_scores_batch` 改 async：

```python
async def semantic_scores_batch(query: str, candidates: list) -> dict:
    """Semantic cosine scores for every titled candidate.

    Embeds the question once and the titles in concurrent chunks via
    thread pool (never blocks the event loop), returns
    {patent_id: cosine}.  Untitled candidates get no entry; any failure
    returns {} — prescoring is an enhancement, never a hard dependency.
    """
    if not query or not str(query).strip():
        return {}
    titled = [(c, str(c.get("title") or "").strip())
              for c in (candidates or []) if isinstance(c, dict)]
    titled = [(c, t) for c, t in titled if t]
    if len(titled) < 2:
        return {}
    start = time.monotonic()
    embedded = await _embed_chunks(query, [t for _, t in titled])
    if embedded is None:
        return {}
    query_vec, chunks, results = embedded
    scores: dict = {}
    pos = 0
    for ch, vecs in zip(chunks, results):
        if vecs is None or len(vecs) != len(ch):
            pos += len(ch)
            continue
        for (c, _), vec in zip(titled[pos:pos + len(ch)], vecs):
            pid = c.get("patent_id")
            if pid:
                scores[str(pid)] = cosine_similarity(query_vec, vec)
        pos += len(ch)
    logger.info(
        f"semantic prescore — candidates={len(titled)} "
        f"elapsed={round(time.monotonic() - start, 1)}s")
    return scores
```

d) `rerank_candidates` 改 async（签名参数不变）：

```python
async def rerank_candidates(query: str, candidates: list,
                            top_k: int = RERANK_TOP_K,
                            alpha: float = RERANK_ALPHA) -> list:
    """Semantically re-rank the top-*top_k* candidates of a ranked list.

    Candidates without a non-empty title keep their LLM-only order and
    are excluded from the fusion window (their positions never change
    relative to each other).  Embedding failures return the original
    list untouched.  Embedding runs chunked in threads — the event
    loop never blocks on the provider.
    """
    if len(candidates) < 2:
        return list(candidates)
    window = list(candidates[:top_k])
    titled = [c for c in window if str(c.get("title") or "").strip()]
    if len(titled) < 2:
        return list(candidates)
    start = time.monotonic()
    embedded = await _embed_chunks(
        query, [str(c.get("title") or "").strip() for c in titled])
    if embedded is None:
        return list(candidates)
    query_vec, chunks, results = embedded
    sem_scores: list = []
    for ch, vecs in zip(chunks, results):
        if vecs is None or len(vecs) != len(ch):
            return list(candidates)
        sem_scores.extend(cosine_similarity(query_vec, v) for v in vecs)
    logger.info(
        f"semantic rerank — candidates={len(titled)} "
        f"elapsed={round(time.monotonic() - start, 1)}s")
    fused_titled = fuse_ranking(titled, sem_scores, alpha)
    out = []
    fused_iter = iter(fused_titled)
    titled_ids = {id(c) for c in titled}
    for c in window:
        if id(c) in titled_ids:
            out.append(next(fused_iter))
        else:
            out.append(c)
    return out + list(candidates[top_k:])
```

e) `react_tools.py` 两处补 await：
- 行 629：`sem_map = semantic_scores_batch(pool.query, live)` → `sem_map = await semantic_scores_batch(pool.query, live)`
- 行 699：`ranked = rerank_candidates(pool.query, ranked, RERANK_TOP_K, RERANK_ALPHA)` → `ranked = await rerank_candidates(...)`（同参数）

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_semantic_rerank.py tests/test_react_tools.py tests/test_technical_interpretation.py -q`
Expected: PASS（若 TestSemanticRerankWiring 报 async 相关失败，按 Step 2 末尾说明处理——该用例走 `_rank_pending_pool`，await 已补，应直接通过）。

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/semantic_rerank.py sources/agents/react_tools.py tests/test_semantic_rerank.py tests/test_react_tools.py
git commit -m "perf: 语义模型 embedding 改 async 分块线程池并行"
```

---

### Task 6: 接地触发点与闭环反哺（`react_tools.py` + `general_agent.py`）

**Files:**
- Modify: `sources/agents/react_tools.py`（常量、`_rank_pending_pool` rubric 优先接地、新 `_grounded_synthesis_round`、execute_action 调用点、`_recall_expansion_round` codes 合并）
- Modify: `sources/agents/general_agent.py:1571` 附近（状态重置 3 行）
- Test: `tests/test_react_tools.py`（新 TestGroundedSynthesisRound + TestGroundedIntegration；`_agent_with_recall_pool` 提为模块级供两处复用）

**Interfaces:**
- Consumes: Task 3 的 `GROUNDED_ENABLED/GROUNDED_HEAD/GROUNDED_MIN/synthesize_grounded`；Task 5 的 await 版 scoring 调用链；既有 `_invoke_and_merge`/`_query_lines`/`AUTO_FEEDBACK_MAX`/`RECALL_MAX_CPC`。
- Produces: `_grounded_synthesis_round(agent, entry, lang) -> Optional[Tuple[list, str, str]]`；`agent._grounded_done/_grounded_interpretation/_grounded_cpc`；`GROUNDED_MIN`/`GROUNDED_HEAD` 模块常量（react_tools 内定义，env 读入，测试可 patch）。

- [ ] **Step 1: 写失败测试**

`tests/test_react_tools.py`：

a) 把 `TestRecallExpansionRound._agent_with_pool` 提为模块级函数 `_agent_with_recall_pool()`（原类内方法删除，类内调用改 `self._agent_with_pool()` → `_agent_with_recall_pool()`，4 处：test_merges_family_records_into_pool / test_fires_once_per_request / test_fetch_failure_degrades_to_none / test_scoring_head_spread_across_recall_batch）。新增模块级函数（内容与原方法一致）：

```python
def _agent_with_recall_pool():
    from sources.long_task.chat_relevance import SearchPool
    agent = _FakeAgent()
    agent._last_user_prompt = "干燥空气"
    agent._flash_llm = _ScoringProvider()
    pool = SearchPool("干燥空气")
    raw = {
        "applicationNumberText": "19511555",
        "applicationMetaData": {
            "inventionTitle": "AIR DRYER CONTROL USING HUMIDITY",
            "firstApplicantName": "ACME Corp",
            "filingDate": "2024-01-15",
            "applicationStatusDescriptionText": "Patented Case",
        },
        "childContinuityBag": [
            {"childPatentNumber": "7061668",
             "childApplicationNumberText": "10393563"},
        ],
    }
    pool.add([raw])
    agent._search_pool = pool
    agent._cpc_hints = [{"code": "H05B45/20", "title": "Colour control"}]
    return agent
```

b) 新测试类（追加在 TestRecallExpansionRound 之后）：

```python
class TestGroundedSynthesisRound(unittest.IsolatedAsyncioTestCase):
    def _agent_with_scored_pool(self, n=20):
        from sources.long_task.chat_relevance import SearchPool
        agent = _FakeAgent()
        agent._last_user_prompt = "q"
        agent.logger = _CaptureLogger()
        pool = SearchPool("q")
        cands = [{"patent_id": f"10000{i}", "title": f"T{i}",
                  "applicant": "ACME", "relevance_score": 4}
                 for i in range(n)]
        pool.add_from_candidates(cands)
        agent._search_pool = pool
        agent._search_interpretation = None
        return agent

    async def test_probe_logged_and_skipped_below_min(self):
        from sources.agents import react_tools as rt
        agent = self._agent_with_scored_pool(n=5)
        entry = _LadderEntry(agent)
        with patch("sources.agents.react_tools.GROUNDED_MIN", 15), \
             patch("sources.long_task.grounded_interpretation"
                   ".synthesize_grounded") as synth:
            result = await rt._grounded_synthesis_round(agent, entry, "zh")
        self.assertIsNone(result)
        synth.assert_not_awaited()
        self.assertIn("grounded_interpretation probe",
                      " ".join(agent.logger.lines))
        self.assertTrue(agent._grounded_done)

    async def test_synthesizes_once_and_executes_queries(self):
        from sources.agents import react_tools as rt
        agent = self._agent_with_scored_pool(n=20)
        entry = _LadderEntry(agent)
        grounded = {
            "dimensions": [{"name": "D1", "role": "核心层", "line": "L1",
                            "representatives": ["ERP"],
                            "players": ["ERP"]}],
            "players": ["ERP"],
            "supplementary_queries": ['("new") AND ("term")'],
            "supplementary_cpc": ["H05B45/20"],
        }
        with patch("sources.agents.react_tools.GROUNDED_MIN", 15), \
             patch("sources.long_task.grounded_interpretation"
                   ".synthesize_grounded",
                   new=AsyncMock(return_value=grounded)), \
             patch("sources.agents.react_tools._invoke_and_merge",
                   new=AsyncMock(return_value=([], "", 0))) as merge_mock:
            result = await rt._grounded_synthesis_round(agent, entry, "zh")
            self.assertIsNotNone(result)
            self.assertEqual(agent._grounded_interpretation, grounded)
            self.assertEqual(agent._grounded_cpc, ["H05B45/20"])
            self.assertIn("grounded_interpretation — lines=",
                          " ".join(agent.logger.lines))
            self.assertEqual(merge_mock.await_count, 1)
            q_used = merge_mock.await_args.args[2]
            self.assertEqual(q_used, '("new") AND ("term")')
            # fires once per request
            again = await rt._grounded_synthesis_round(agent, entry, "zh")
        self.assertIsNone(again)
        self.assertEqual(merge_mock.await_count, 1)

    async def test_no_pool_logs_probe_without_burning_queries(self):
        from sources.agents import react_tools as rt
        agent = _FakeAgent()
        agent.logger = _CaptureLogger()
        entry = _LadderEntry(agent)
        result = await rt._grounded_synthesis_round(agent, entry, "zh")
        self.assertIsNone(result)
        self.assertTrue(agent._grounded_done)

    async def test_rank_pending_pool_prefers_grounded_rubric(self):
        from sources.agents import react_tools as rt
        from sources.long_task.chat_relevance import SearchPool

        class _Recorder:
            def __init__(self):
                self.calls = 0
                self.prompts = []

            async def complete_json(self, system, user, max_retries=0):
                self.calls += 1
                self.prompts.append(system)
                return {"scores": []}

        agent = _FakeAgent()
        agent.llm = _Recorder()
        agent.logger = _CaptureLogger()
        agent._last_user_prompt = "q"
        agent._search_pool = SearchPool("q")
        agent._search_interpretation = {"scheme": "pre", "key_players": ["A"]}
        agent._grounded_interpretation = {
            "scheme": "pre",
            "dimensions": [{"name": "D1", "role": "核心层", "line": "L1",
                            "representatives": ["ERP"]}],
            "players": ["ERP"],
        }
        provider = _Recorder()
        with patch.object(rt, "_get_flash_provider", return_value=provider):
            await rt._rank_pending_pool(agent, [_cand("1001")], "zh")
        self.assertIn("真实玩家榜", provider.prompts[0])
        self.assertNotIn("本身不构成相关性依据", provider.prompts[0])

    async def test_rank_pending_pool_keeps_pre_rubric_without_grounded(self):
        from sources.agents import react_tools as rt
        from sources.long_task.chat_relevance import SearchPool

        class _Recorder:
            def __init__(self):
                self.prompts = []

            async def complete_json(self, system, user, max_retries=0):
                self.prompts.append(system)
                return {"scores": []}

        agent = _FakeAgent()
        agent.llm = _Recorder()
        agent._last_user_prompt = "q"
        agent._search_pool = SearchPool("q")
        agent._search_interpretation = {"scheme": "pre", "key_players": ["A"]}
        provider = _Recorder()
        with patch.object(rt, "_get_flash_provider", return_value=provider):
            await rt._rank_pending_pool(agent, [_cand("1001")], "zh")
        self.assertIn("本身不构成相关性依据", provider.prompts[0])

    async def test_recall_expansion_uses_grounded_cpc(self):
        from sources.agents.react_tools import _recall_expansion_round
        agent = _agent_with_recall_pool()
        agent._grounded_cpc = ["H05B45/30"]
        entry = _LadderEntry(agent)
        with patch("sources.agents.react_tools.fetch_by_numbers",
                   return_value=[]), \
             patch("sources.agents.react_tools.fetch_by_cpc") as fcp:
            fcp.return_value = []
            await _recall_expansion_round(agent, entry, "zh")
        codes = fcp.call_args[0][0]
        self.assertIn("H05B45/20", codes)
        self.assertIn("H05B45/30", codes)


class TestGroundedIntegration(unittest.IsolatedAsyncioTestCase):
    """execute_action 在主搜索路径调用接地轮（在 feedback 轮之前）。"""

    async def test_search_observation_merges_grounded_results(self):
        from sources.agents.react_tools import ToolEntry, make_action_executor
        from sources.long_task.chat_relevance import SearchPool
        agent = _FakeAgent()
        agent._last_user_prompt = "q"
        agent._flash_llm = _ScoringProvider()
        pool = SearchPool("q")
        pool.add([_usp_raw_item("19511555", "TITLE ONE")])
        agent._search_pool = pool

        class _SeqTool:
            def invoke(self, payload):
                agent._pending_raw_items = [
                    _usp_raw_item("19511555", "TITLE ONE")]
                agent._last_search_total = 1
                return "ok"

        tool_info = _ToolInfo(
            "uspto search",
            url="https://api.uspto.gov/api/v1/patent/applications/search")
        entry = ToolEntry(name="uspto_search", kind="knowledge",
                          knowledge=_Knowledge(3, ktype=1),
                          tool_info=tool_info, tool=_SeqTool())
        registry = {entry.name: entry}
        executor = await make_action_executor(agent, registry, None)
        merged_ranked = [{"patent_id": "18184836", "title": "NEW",
                          "applicant": "ACME", "status": "Patented Case",
                          "patent_number": "1",
                          "_raw": _usp_raw_item("18184836", "NEW TITLE")}]
        grounded_result = (merged_ranked, "重排", "已自动执行接地解读补检索式：\n- q1")
        with patch("sources.agents.react_tools._grounded_synthesis_round",
                   new=AsyncMock(return_value=grounded_result)), \
             patch("sources.long_task.search_query_builder"
                   ".build_feedback_queries", return_value=[]):
            result = await executor("uspto_search", {"q": "agent query"}, 1)
        self.assertEqual(result["kind"], "observation")
        self.assertIn("18184836", result["text"])
        self.assertIn("已自动执行接地解读补检索式", result["text"])
```

（`_LadderEntry` 已在测试文件定义；`AsyncMock` 已在 import 中。）

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -q`
Expected: FAIL（`_grounded_synthesis_round` 不存在；`rt.GROUNDED_MIN` 不存在；rubric 优先逻辑未实现）。

- [ ] **Step 3: 实现**

a) `general_agent.py` 状态重置区（`self._search_interpretation = None` 行之后）加：

```python
        self._grounded_done = False  # post-retrieval grounded synthesis, once per request
        self._grounded_interpretation = None  # data-grounded interpretation (players/lines)
        self._grounded_cpc = None  # supplementary CPC codes from the grounded synthesis
```

b) `react_tools.py` 模块常量区（FAMILY_SCORE_BUDGET 附近）加：

```python
# Post-retrieval grounded interpretation: fires once per request when
# the scored pool clears the minimum; clusters the scored head.
GROUNDED_MIN = int(os.getenv("REACT_GROUNDED_MIN", "15"))
GROUNDED_HEAD = int(os.getenv("REACT_GROUNDED_HEAD", "30"))
```

c) `_rank_pending_pool` rubric 段（现 640-644 行）替换：

```python
    _score_start = time.monotonic()
    try:
        from sources.long_task.technical_interpretation import (
            format_interpretation_rubric,
        )
        # Grounded interpretation (post-retrieval) wins over the
        # pre-retrieval one once it exists: its players/lines are
        # data-driven.
        _grounded = getattr(agent, "_grounded_interpretation", None)
        _rubric = format_interpretation_rubric(
            _grounded or getattr(agent, "_search_interpretation", None))
    except Exception:
        _rubric = ""
```

d) 新增 `_grounded_synthesis_round`（放在 `_auto_feedback_round` 之后、`_recall_expansion_round` 之前）：

```python
async def _grounded_synthesis_round(agent, entry, lang) -> Optional[Tuple[list, str, str]]:
    """Post-retrieval grounded synthesis with loop feedback.

    Once per request: when the scored pool clears GROUNDED_MIN, cluster
    the scored head into data-driven dimensions/players (Flash), store
    the grounded interpretation (rubric upgrades to real signals) and
    its supplementary CPC codes (recall expansion widens), then
    auto-execute its supplementary queries into the pool — mirroring
    the auto-feedback round.  The probe line logs every request (even
    skipped) so a silent path stays visible.  Returns
    (ranked, ranking_note, grounded_note) when queries executed; None
    otherwise.  Never raises.
    """
    if getattr(agent, "_grounded_done", False):
        return None
    agent._grounded_done = True
    from sources.long_task.grounded_interpretation import (
        GROUNDED_ENABLED, synthesize_grounded,
    )
    pool = getattr(agent, "_search_pool", None)
    _glog = getattr(agent, "logger", None)
    if pool is None:
        if _glog is not None:
            _glog.info(
                "grounded_interpretation probe — pool=0 scored=0 "
                f"trigger={GROUNDED_ENABLED}")
        return None
    scored = [
        c for c in pool._by_id.values()
        if isinstance(c.get("relevance_score"), (int, float))
    ]
    if _glog is not None:
        _glog.info(
            f"grounded_interpretation probe — pool={len(pool)} "
            f"scored={len(scored)} trigger={GROUNDED_ENABLED}")
    if not GROUNDED_ENABLED or len(scored) < GROUNDED_MIN:
        return None
    top = sorted(
        scored, key=lambda c: -(c.get("relevance_score") or 0))[:GROUNDED_HEAD]
    try:
        grounded = await synthesize_grounded(
            pool.query, top,
            pre_interp=getattr(agent, "_search_interpretation", None),
            cpc_hints=getattr(agent, "_cpc_hints", None))
    except Exception:
        grounded = None
    if not grounded:
        return None
    agent._grounded_interpretation = grounded
    agent._grounded_cpc = list(
        grounded.get("supplementary_cpc") or [])[:RECALL_MAX_CPC]
    lines = [str(d.get("name") or "") for d in
             (grounded.get("dimensions") or [])[:3]]
    players = ", ".join(str(p) for p in (grounded.get("players") or [])[:5])
    if _glog is not None:
        _glog.info(
            f"grounded_interpretation — lines={lines}"
            + (f" | players={players}" if players else ""))
    queries = [q for q in
               (grounded.get("supplementary_queries") or [])[:AUTO_FEEDBACK_MAX]
               if q]
    if not queries:
        return None
    executed: list = []
    gained = 0
    ranked: list = []
    ranking_note = ""
    for q in queries:
        merged = await _invoke_and_merge(agent, entry, q, lang)
        if merged is None:
            break
        executed.append(q)
        ranked, ranking_note, live = merged
        gained += live
    if not executed:
        return None
    if lang == "en":
        merged_note = (f"(merged {gained} new candidates)" if gained > 0
                       else "(no live hits)")
        grounded_note = (f"\n\nAuto-executed grounded queries {merged_note}:\n"
                         + _query_lines(executed))
    else:
        merged_note = (f"（并入 {gained} 条新候选）" if gained > 0
                       else "（均无有效命中）")
        grounded_note = (f"\n\n已自动执行接地解读补检索式{merged_note}：\n"
                         + _query_lines(executed))
    return ranked, ranking_note, grounded_note
```

e) `execute_action` 调用点：在 `text = await _maybe_append_feedback(agent, text, total, lang)` 与 `feedback = await _auto_feedback_round(agent, entry, lang)` 两行之间插入（镜像 feedback 合并块）：

```python
            grounded = await _grounded_synthesis_round(agent, entry, lang)
            if grounded is not None:
                ranked, ranking_note, grounded_note = grounded
                if ranked:
                    shown = [c["_raw"] for c in ranked]
                    agent._pending_raw_items = shown
                    agent._search_ranked = True
                    digest = _ranked_digest(ranked, lang=lang)
                    note = ranking_note + grounded_note
                    if lang == "en":
                        text = (f"Search results ({len(shown)} records, "
                                f"{note}):\n{digest}\n\n"
                                "The full list is displayed to the user.")
                    else:
                        text = (f"检索结果（{len(shown)} 条，{note}）：\n"
                                f"{digest}\n\n"
                                "完整列表已展示给用户。")
```

f) `_recall_expansion_round` codes 段（现 950-953 行）替换：

```python
    grounded_codes = [
        str(c).strip().upper()
        for c in (getattr(agent, "_grounded_cpc", None) or [])
        if str(c).strip()]
    codes = [str(h.get("code", "")).strip() for h in
             (getattr(agent, "_cpc_hints", None) or [])
             if isinstance(h, dict) and h.get("code")]
    codes = (codes + grounded_codes)[:RECALL_MAX_CPC]
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -q`
Expected: PASS（134 旧 - 0 破坏 + 8 新）。

- [ ] **Step 5: Commit**

```bash
git add sources/agents/react_tools.py sources/agents/general_agent.py tests/test_react_tools.py
git commit -m "feat: 接地解读单次触发闭环反哺（补查询/补CPC/rubric真信号）"
```

---

### Task 7: 全量回归 + 通用性审计 + 提交

**Files:** 无新改动（验证 + 收尾）。

- [ ] **Step 1: 全量测试**

Run: `PYTHONUTF8=1 python -m pytest tests/ -q --ignore=tests/test_browser_agent_parsing.py --ignore=tests/test_provider.py`
Expected: 748 passed / 26 failed + 7 errors 基线（新增测试全部通过，不得引入新失败）。若出现新失败：修复后重跑。

- [ ] **Step 2: 通用性审计（铁律）**

Run:
```bash
grep -rn "RGB\|控制放大器\|独立控制" sources/long_task/technical_interpretation.py sources/long_task/grounded_interpretation.py sources/agents/react_tools.py sources/agents/general_agent.py
```
Expected: 无输出。若命中：改写 prompt/代码措辞后重跑 Step 1。

- [ ] **Step 3: 确认工作区状态并收尾提交**

Run: `git status --short`
Expected: 干净（或仅剩未提交的 Task 6 改动——一并提交）。

```bash
git add -A
git commit -m "test: 接地解读闭环全量回归 + 通用性审计"
```
（若无未提交改动，跳过本步。）

---

## 部署与生产验证（计划外，交付后执行）

1. 同步 `/opt/langsistance` → 重启（uvicorn + celery）。
2. 跑测试问题生产轮，检查 `.logs/general_agent.log`：
   - `search_interpretation — scheme=...`（预解读，含维度）
   - `grounded_interpretation probe — pool=N scored=M trigger=True`
   - `grounded_interpretation — lines=[...] | players=...`（players 应为数据驱动专精玩家）
   - `relevance scoring — candidates=50 ... elapsed=...`（应显著低于 45-136s）
   - `semantic prescore — candidates=N elapsed=...`（正常且事件循环不卡）
   - `family scoring probe — seeds=... members=...`（玩家榜抬升后种子数可能增加）
3. 判据：核心器件维度 players 浮现专精玩家 + 主线 ≥2 带代表申请人 + 最终列表结构判据（本质句 + 维度骨架 + 玩家数据驱动）。
