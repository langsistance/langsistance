# 自适应专利检索（v4）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把检索的调参能力还给 agent：改写产物变为松紧阶梯并注入循环上下文、q 控制权还给 agent（显式传参优先，仅在缺失时兜底注入最紧级）、总命中数进观察，循环指导明确「同工具多调参 → 换工具需契合度判断 → 不契合则承认失败」的纪律。

**Architecture:** 四处修改全部落在既有模块：`search_query_builder`（阶梯 prompt + `format_ladder_guidance` 纯函数）、`general_agent`（create_agent 预改写 + 阶梯注入系统提示词 + `_last_search_total` 捕获与重置 + 循环指导两原则）、`react_tools`（`_maybe_rewrite_search_query` 从「强制覆盖」改为「仅缺失兜底」+ 观察文本带总命中数）。

**Tech Stack:** Python 3.14、unittest + `IsolatedAsyncioTestCase` + `AsyncMock`、既有 `Provider.complete_json`、既有 `build_search_queries` / `build_candidates` / `is_keyword_search_tool`（均已交付模块）。

## Global Constraints

- Python 3.14：禁用 `asyncio.get_event_loop()`；异步测试用 `IsolatedAsyncioTestCase`。
- 测试运行：`cd E:\online\workspace\copiioai\langsistance && PYTHONUTF8=1 python -m pytest tests/<file> -v`；全量：`PYTHONUTF8=1 python -m pytest tests/ -q --continue-on-collection-errors`。系统 python（C:/Python314），`PYTHONUTF8=1` 必带。
- 全量基线（本环境）：**402 passed / 26 failed / 9 errors**——26+9 为预存环境问题；验收标准 = 失败与错误数不变，passed = 402 + 新增测试数。
- **通用性（D5）**：任何模块不得为特定查询/技术领域硬编码关键词、阈值或分支。
- 所有 LLM 调用 guarded，失败降级，循环永不因新功能崩溃。
- 提交规范：`feat:` / `fix:`，无 attribution trailer。
- 分支：从当前 HEAD 拉出 `feature/adaptive-patent-search`。
- 前端零改动。

## File Structure

| 文件 | 职责 |
|---|---|
| `sources/long_task/search_query_builder.py` | **修改**。阶梯 prompt（最紧在前、允许有依据的领域限定）+ `format_ladder_guidance` 纯函数 |
| `sources/agents/general_agent.py` | **修改**。create_agent 预改写 + 阶梯注入 + `_last_search_total` 捕获/重置 + 循环指导两原则 |
| `sources/agents/react_tools.py` | **修改**。`_maybe_rewrite_search_query` 改为仅缺失兜底 + 观察带总命中数 |
| `tests/test_search_query_builder.py` | **修改**（追加阶梯测试） |
| `tests/test_react_tools.py` | **修改**（重写 C4 的 `TestMaybeRewriteSearchQuery` 为 v4 语义 + 追加观察总命中测试） |

---

### Task 1: 改写阶梯 + 阶梯指导格式化

**Files:**
- Modify: `sources/long_task/search_query_builder.py`
- Test: `tests/test_search_query_builder.py`（追加测试类）

**Interfaces:**
- Produces（Task 3/4 依赖）:
  - `build_search_queries(query, provider) -> {"concepts": [...], "queries": [最紧 ... 最松]}`（既有，顺序语义强化）
  - `format_ladder_guidance(rewrite: dict, lang: str = "zh") -> str`（纯函数；空 queries → `""`）

- [ ] **Step 1: 写失败测试**

在 `tests/test_search_query_builder.py` 末尾追加：

```python
# ── Ladder ordering + guidance formatting ────────────────────────────────────

from sources.long_task.search_query_builder import (
    REWRITE_SYSTEM_PROMPT,
    format_ladder_guidance,
)


class TestRewritePromptLadderRules(unittest.TestCase):
    def test_prompt_requires_tight_to_loose_ordering(self):
        self.assertIn("最紧", REWRITE_SYSTEM_PROMPT)
        self.assertIn("放宽", REWRITE_SYSTEM_PROMPT)

    def test_prompt_allows_justified_domain_constraints(self):
        self.assertIn("限定", REWRITE_SYSTEM_PROMPT)


class TestFormatLadderGuidance(unittest.TestCase):
    def test_renders_ordered_queries_with_lang_zh(self):
        rewrite = {"queries": [
            '("a" OR "b") AND ("c" OR "d") AND (x OR y)',
            '("a" OR "b") AND ("c" OR "d")',
            '("a" OR "b")',
        ]}
        text = format_ladder_guidance(rewrite, "zh")
        self.assertIn("由紧到松", text)
        pos_0 = text.find('("a" OR "b") AND ("c" OR "d") AND (x OR y)')
        pos_1 = text.find('("a" OR "b") AND ("c" OR "d")')
        pos_2 = text.find('("a" OR "b")')
        self.assertLess(pos_0, pos_1)
        self.assertLess(pos_1, pos_2)

    def test_empty_rewrite_returns_empty(self):
        self.assertEqual(format_ladder_guidance({}, "zh"), "")
        self.assertEqual(format_ladder_guidance({"queries": []}, "zh"), "")

    def test_english_variant(self):
        text = format_ladder_guidance(
            {"queries": ['"a" AND "b"']}, "en")
        self.assertIn("tightest", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_search_query_builder.py -v`
Expected: FAIL（ImportError: cannot import name 'format_ladder_guidance'）

- [ ] **Step 3: 写实现**

3a. 替换 `search_query_builder.py` 的 `REWRITE_SYSTEM_PROMPT` 为阶梯版：

```python
REWRITE_SYSTEM_PROMPT = (
    "你是一个专利检索式构造专家。把用户的自然语言技术问题改写为 "
    "USPTO Patent Application Search API 的检索式（q 参数）。"
    "本工具面向所有技术领域，不得为特定领域预设关键词。\n\n"
    "步骤：\n"
    "1. 从用户问题中抽取 2-4 个核心技术概念（忽略语气词和通用词）\n"
    "2. 每个概念翻译成该领域专利文献常用的英文术语，并给出 2-5 个"
    "同义/近义关键词（含行业缩写、上位/下位词、英美拼写变体）\n"
    "3. 可以为提高精度添加用户未提及的合理领域限定概念（如应用场景、"
    "设备载体），但每个限定必须有依据，且限定属于最弱概念层级\n"
    "4. 用检索式语法组装查询串：\n"
    "   - 多词短语必须用双引号包裹，如 \"3d printing\"\n"
    "   - 同一概念的同义词用 OR 连接并放在圆括号内："
    '("3d printing" OR "additive manufacturing" OR "rapid prototyping")\n'
    "   - 不同概念之间用 AND 连接\n"
    "   - 每个检索式最多 12 个关键词、250 字符，禁止出现中文\n"
    "5. 输出 2-4 个检索式，必须按松紧排序：\n"
    "   - 第一个为最紧的完整组合式（全部概念 + 领域限定）\n"
    "   - 后续逐级放宽（每级去掉最弱的限定/概念）\n"
    "   - 最后一个为最松的核心概念式\n\n"
    'Return JSON: {"concepts": [{"concept": "中文概念", '
    '"keywords": ["english term", ...]}], '
    '"queries": ["最紧", "较松", "最松"]}'
)
```

3b. 文件末尾追加：

```python
def format_ladder_guidance(rewrite: dict, lang: str = "zh") -> str:
    """Render the rewrite ladder for the loop system prompt.

    Queries are listed tightest-first so the LLM can pick a variant or
    adjust from it.  Returns "" when there is nothing to show.
    """
    queries = (rewrite or {}).get("queries") or []
    if not isinstance(queries, list) or not queries:
        return ""
    if lang == "en":
        header = (
            "Available search queries for the user's question, ordered "
            "tightest to loosest. You may call a search tool with one of "
            "these queries directly, or adjust them (drop a constraint to "
            "loosen, add a constraint to tighten) based on the result "
            "counts you observe:\n"
        )
    else:
        header = (
            "针对用户问题可用的检索式（由紧到松排列）。你可以直接用其中"
            "任一条调用搜索工具，也可以根据观察到的命中数自行调整"
            "（命中为 0 则去掉某组限定放宽，命中过多则添加限定收紧）：\n"
        )
    lines = [header]
    for i, q in enumerate(queries, start=1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_search_query_builder.py -v`
Expected: PASS（既有 10 + 新增 5 = 15 passed）

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/search_query_builder.py tests/test_search_query_builder.py
git commit -m "feat: add tight-to-loose query ladder and ladder guidance formatting"
```

---

### Task 2: 总命中数捕获 + 观察展示

**Files:**
- Modify: `sources/agents/general_agent.py`（`dynamic_backend_tool_function` 捕获 count；`create_agent` 重置一行）
- Modify: `sources/agents/react_tools.py`（观察文本带总命中数）
- Test: `tests/test_react_tools.py`（追加测试）

**Interfaces:**
- Produces: `agent._last_search_total`（int 或 None，每请求重置）——react_tools 观察消费

- [ ] **Step 1: 写失败测试**

在 `tests/test_react_tools.py` 末尾追加：

```python
# ── Total-hit count in search observations ───────────────────────────────────

class TestSearchObservationTotalCount(unittest.TestCase):
    def test_observation_includes_total_hits_when_present(self):
        agent = _FakeAgent()
        agent._last_search_total = 8750325
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(
            _registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "Air dryer humidity control"),
        ]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertIn("总命中 8750325", result["text"])

    def test_observation_omits_total_when_absent(self):
        agent = _FakeAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(
            _registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "Air dryer humidity control"),
        ]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertNotIn("总命中", result["text"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py::TestSearchObservationTotalCount -v`
Expected: FAIL（assertIn 失败——观察文本尚无「总命中」）

- [ ] **Step 3: 写实现**

3a. `general_agent.py` 的 `dynamic_backend_tool_function`，在 `tool_result = execute_backend_tool_request(tool_info, params)` 之后、`raw_items = ...` 之前插入：

```python
            # Capture the total hit count for the loop's adaptive search
            # (e.g. 0 hits → loosen the query, 8M hits → tighten it).
            _result_data = tool_result.get("data")
            if isinstance(_result_data, dict):
                _count = _result_data.get("count")
                if isinstance(_count, (int, float)):
                    self._last_search_total = int(_count)
```

3b. `general_agent.py` 的 `create_agent` 状态重置区（`self._search_rewrite = None` 行之后）加一行：

```python
        self._last_search_total = None   # total-hit count captured per request
```

3c. `react_tools.py` 的 `make_action_executor` knowledge 分支，观察文本组装处（zh/en 两个分支的 `retrieval` 文本）改为带总命中。当前代码（Task 3 版）：

```python
        pending = getattr(agent, "_pending_raw_items", None)
        if pending:
            capped, note = _cap_patent_list(entry.tool_info, pending, lang)
            agent._pending_raw_items = capped
            digest = _items_digest(capped, lang=lang)
            if lang == "en":
                text = (f"Search results ({len(capped)} records, {note}):\n"
                        f"{digest}\n\n"
                        "The full list is displayed to the user.")
            else:
                text = (f"检索结果（{len(capped)} 条，{note}）：\n"
                        f"{digest}\n\n"
                        "完整列表已展示给用户。")
            return {"kind": "observation", "text": text}
```

替换为：

```python
        pending = getattr(agent, "_pending_raw_items", None)
        if pending:
            capped, note = _cap_patent_list(entry.tool_info, pending, lang)
            agent._pending_raw_items = capped
            digest = _items_digest(capped, lang=lang)
            total = getattr(agent, "_last_search_total", None)
            total_note = ""
            if isinstance(total, int):
                total_note = (f", {total} total hits" if lang == "en"
                              else f"，总命中 {total}")
            if lang == "en":
                text = (f"Search results ({len(capped)} records{total_note}, {note}):\n"
                        f"{digest}\n\n"
                        "The full list is displayed to the user.")
            else:
                text = (f"检索结果（{len(capped)} 条{total_note}，{note}）：\n"
                        f"{digest}\n\n"
                        "完整列表已展示给用户。")
            return {"kind": "observation", "text": text}
```

注意：en 分支的 total_note 以 `, N total hits` 形式插入（与 `, {note}` 逗号衔接）；zh 分支用 `，总命中 N`。

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: PASS（既有 30 + 新增 2 = 32 passed；既有「已截断」断言不受影响——note 仍保留）

- [ ] **Step 5: Commit**

```bash
git add sources/agents/general_agent.py sources/agents/react_tools.py tests/test_react_tools.py
git commit -m "feat: capture total search hits and surface them in loop observations"
```

---

### Task 3: create_agent 预改写 + 阶梯注入系统提示词

**Files:**
- Modify: `sources/agents/general_agent.py`（`create_agent`）
- Test: `tests/test_search_query_builder.py`（追加 1 个集成型测试——`create_agent` 本身不在单测范围，阶梯块组装逻辑由 Task 1 的 `format_ladder_guidance` 覆盖；本任务测试：注入块内容断言放在 Task 4 的 fake-agent 流程测试中）

**Interfaces:**
- Consumes: Task 1 `build_search_queries` / `format_ladder_guidance`
- Produces: `agent._search_rewrite` 在 **create_agent 阶段**提前填充（Task 4 的执行器不再懒加载）；系统提示词含阶梯块

- [ ] **Step 1: 写失败测试**

在 `tests/test_react_tools.py` 的 `TestMaybeRewriteSearchQuery`（Task 4 重写后）中会覆盖「缓存提前存在时不再调用改写」——本任务的验证以 Task 4 测试为准。本任务先跑既有测试确认无回归：

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py tests/test_search_query_builder.py -q`
Expected: 既有全绿（32 + 15 = 47 passed，作为本任务改动前基线）

- [ ] **Step 2: 修改 create_agent**

2a. 在 `create_agent` 中，`system_prompt = (...)` 组装**之前**（`self.memory.reset(...)` 之前即可），插入预改写（替换现有的 `self._search_rewrite = None` 重置行为——保留重置 + 立即填充）：

现有代码：
```python
        self.knowledgeTool = (None, None)  # (knowledge_item, tool_info) — selected inside the loop
        self.tools = []
        lang = self._detect_lang(prompt)
        self._lang = lang
        if callback_handler:
            await _emit_status(callback_handler,
                "正在分析您的问题..." if lang == 'zh' else "Analyzing your question...")
```

在 `lang = self._detect_lang(prompt)` 之后插入：

```python
        # Pre-build the tight-to-loose query ladder for this request so the
        # loop system prompt can offer it to the LLM upfront.
        from sources.long_task.search_query_builder import build_search_queries
        try:
            self._search_rewrite = await build_search_queries(prompt, self.llm)
        except Exception:
            self._search_rewrite = {"concepts": [], "queries": []}
        self.logger.info(
            f"search_rewrite — queries={self._search_rewrite.get('queries')}"
        )
```

2b. `system_prompt` 组装处（现有）：
```python
        system_prompt = (
            self._get_fixed_system_prefix()
            + conversation_block
            + self._loop_system_guidance()
        )
```
替换为：
```python
        from sources.long_task.search_query_builder import format_ladder_guidance
        system_prompt = (
            self._get_fixed_system_prefix()
            + conversation_block
            + self._loop_system_guidance()
            + format_ladder_guidance(self._search_rewrite, lang)
        )
```

- [ ] **Step 3: 运行测试确认通过（Task 4 前先保持既有全绿）**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py tests/test_search_query_builder.py -q`
Expected: 47 passed（本任务未改测试文件；改动只影响 create_agent 运行路径，单测不触及）

- [ ] **Step 4: Commit**

```bash
git add sources/agents/general_agent.py
git commit -m "feat: pre-build query ladder in create_agent and inject it into the loop system prompt"
```

---

### Task 4: v4 执行器语义 + 循环指导两原则

**Files:**
- Modify: `sources/agents/react_tools.py`（`_maybe_rewrite_search_query` 改为仅缺失兜底）
- Modify: `sources/agents/general_agent.py`（`_loop_system_guidance` 加两原则）
- Test: `tests/test_react_tools.py`（**重写** `TestMaybeRewriteSearchQuery` 全部 8 个测试为 v4 语义 + 追加 2 个新测试）

**Interfaces:**
- Consumes: Task 1 阶梯、Task 3 的预缓存（`agent._search_rewrite` 已在 create_agent 填充；执行器保留懒加载兜底路径）
- Produces: `_maybe_rewrite_search_query(agent, tool_info, args) -> dict`——新语义：**LLM 显式非空 q 原样保留；q 槽缺失/为空时注入阶梯最紧级 `queries[0]`；无查询键的 args 原样返回**

- [ ] **Step 1: 重写测试（RED）**

用以下内容**整体替换** `tests/test_react_tools.py` 中的 `class TestMaybeRewriteSearchQuery`（从 `class TestMaybeRewriteSearchQuery` 到该类结束）：

```python
class TestMaybeRewriteSearchQuery(unittest.IsolatedAsyncioTestCase):
    """v4 semantics: the LLM's explicit q wins; the tightest ladder query
    is injected only when the q slot is absent or empty."""

    async def test_explicit_q_preserved(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b") AND ("c" OR "d")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"q": '("humidity control" OR "dew point") AND industrial'})
        self.assertEqual(
            out["q"], '("humidity control" OR "dew point") AND industrial')

    async def test_empty_q_injected_tightest(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b") AND ("c" OR "d")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        self.assertEqual(out["q"], '("a" OR "b") AND ("c" OR "d")')

    async def test_missing_q_injected_tightest(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b") AND ("c" OR "d")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"page": 1})
        self.assertEqual(out["q"], '("a" OR "b") AND ("c" OR "d")')
        self.assertEqual(out["page"], 1)

    async def test_explicit_query_key_preserved(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"query": "my own"})
        self.assertEqual(out["query"], "my own")

    async def test_params_json_with_explicit_q_preserved(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"params": '{"q": "mine", "pagination": {"offset": 0, "limit": 50}}'})
        import json
        parsed = json.loads(out["params"])
        self.assertEqual(parsed["q"], "mine")
        self.assertEqual(parsed["pagination"]["limit"], 50)

    async def test_params_json_without_q_injected(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"params": '{"pagination": {"offset": 0, "limit": 50}}'})
        import json
        parsed = json.loads(out["params"])
        self.assertEqual(parsed["q"], '("a" OR "b")')
        self.assertEqual(parsed["pagination"]["limit"], 50)

    async def test_skips_non_keyword_tools(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("get_patent_documents_application_number"), {"q": "raw"})
        self.assertEqual(out, {"q": "raw"})

    async def test_no_cache_no_queries_keeps_original(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": []}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        self.assertEqual(out, {"q": ""})

    async def test_args_without_query_key_unchanged(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"other": "x"})
        self.assertEqual(out, {"other": "x"})

    async def test_lazy_build_when_cache_missing(self):
        """Executor fallback: cache absent (non-create_agent path) → builds
        the rewrite once via the agent's provider."""
        agent = _RewriteAgent()
        agent._search_rewrite = None
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        self.assertEqual(agent.llm.calls, 1)
        self.assertIn('"compressed air dryer"', out["q"])
        # second call reuses the cache
        out2 = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        self.assertEqual(agent.llm.calls, 1)
        self.assertEqual(out2["q"], out["q"])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py::TestMaybeRewriteSearchQuery -v`
Expected: FAIL（v4 语义未实现——显式 q 仍被覆盖等）

- [ ] **Step 3: 写实现**

3a. 整体替换 `react_tools.py` 的 `_maybe_rewrite_search_query`：

```python
async def _maybe_rewrite_search_query(agent, tool_info, args) -> dict:
    """Inject the tightest ladder query ONLY when the q slot is absent.

    v4 semantics: the LLM owns q.  An explicit non-empty q the LLM passed
    (its own adaptation — loosened, tightened, or a ladder variant) is
    always respected.  The deterministic ladder (built in create_agent)
    fills in only when the q slot is missing or empty.  Applies only to
    backend (push=2) keyword search tools; every failure keeps the
    original args untouched.
    """
    if getattr(tool_info, "push", None) != 2 or not is_keyword_search_tool(tool_info):
        return args
    cached = getattr(agent, "_search_rewrite", None)
    if cached is None:
        from sources.long_task.search_query_builder import build_search_queries
        try:
            cached = await build_search_queries(
                getattr(agent, "_last_user_prompt", "") or "", agent.llm,
            )
        except Exception:
            cached = {"queries": []}
        agent._search_rewrite = cached
    queries = (cached or {}).get("queries") or []
    if not queries:
        return args
    tightest = queries[0]
    out = dict(args or {})

    def _blank(value) -> bool:
        return not str(value or "").strip()

    if "q" in out:
        if _blank(out.get("q")):
            out["q"] = tightest
        return out
    if "query" in out:
        if _blank(out.get("query")):
            out["query"] = tightest
        return out
    if "params" in out:
        try:
            import json
            if isinstance(out["params"], str):
                p = json.loads(out["params"])
            elif isinstance(out["params"], dict):
                p = dict(out["params"])
            else:
                return args
        except (ValueError, TypeError):
            return args
        if "q" in p:
            if _blank(p.get("q")):
                p["q"] = tightest
        elif "query" in p:
            if _blank(p.get("query")):
                p["query"] = tightest
        else:
            p["q"] = tightest
        out["params"] = json.dumps(p, ensure_ascii=False)
        return out
    return args
```

3b. 替换 `general_agent.py` 的 `_loop_system_guidance` 返回值：

```python
    def _loop_system_guidance(self) -> str:
        """Tool-usage guidance appended to the ReAct loop system prompt."""
        return """

## Tool Usage

- You have tools built from the user's knowledge base, plus `search_my_knowledge`
  to find more. Use them to complete the user's task.
- Work step by step: think about what is needed, call the right tool, observe
  the result, then decide the next step or write the final answer.
- You may call several tools in sequence and combine their results.
- Never fabricate tool results. If a tool fails, try another approach or
  explain the failure honestly.

## Adaptive Search Discipline

- The same search tool may be called multiple times with adjusted parameters:
  if a search returns 0 results, loosen the query (drop the weakest constraint
  or use a looser variant); if it returns far too many noisy results, tighten it
  (add a justified constraint). Keep adjusting until the result count and
  relevance are reasonable (suggested: at most 4 attempts per tool).
- Only after repeated adjustments on one tool still give poor results, consider
  another tool — and judge whether that tool actually fits the user's problem.
- If no available tool fits the problem, honestly report the search failure and
  its reason to the user. Do not keep chaining unsuitable tools.
"""
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v && PYTHONUTF8=1 python -m pytest tests/test_search_query_builder.py -q`
Expected: test_react_tools 32 + 2 = 34 passed；builder 16 passed

- [ ] **Step 5: Commit**

```bash
git add sources/agents/react_tools.py sources/agents/general_agent.py tests/test_react_tools.py
git commit -m "feat: hand q control back to the agent with ladder fallback and adaptive search discipline"
```

---

### Task 5: 全量回归 + 验收准备

**Files:** 无代码改动

- [ ] **Step 1: 全量回归**

Run: `PYTHONUTF8=1 python -m pytest tests/ -q --continue-on-collection-errors 2>&1 | tail -2`
Expected: **411 passed / 26 failed / 9 errors**（基线 402 + 新增 5 + 2 + 2 = 411；判据 = 失败与错误数 26/9 与基线一致，零新增；实施者以实际计数为准并写入报告）

- [ ] **Step 2: 手动验收清单（部署后执行）**

部署后端 + 前端后，复测干燥空气查询，逐项检查：
1. `general_agent.log` 出现 `search_rewrite — queries=[...]`（阶梯 2-4 级、由紧到松）
2. 搜索请求的 q 形态正常；若首级 0 命中，LLM 应同工具放宽重试（日志中出现同一工具的第二次调用且 q 更松）；观察含「总命中 N 条」
3. 若 LLM 换工具，其调用理由（观察流）应体现契合度判断；无契合工具时最终回答如实说明失败
4. 前端部署后步骤条显示
5. 对照 5 领域基准集其余查询各跑一遍

- [ ] **Step 3: 记录验收结果**（/save-session，控制器执行）

---

## Self-Review

**1. Spec coverage（v4 修订 §6 逐项）:**
- 改写阶梯（最紧在前、允许有依据的领域限定）→ Task 1 prompt + 测试 ✓
- q 控制权还给 agent（显式 q 直接用；缺失用 queries[0]）→ Task 4 执行器 + 重写测试 ✓
- 同工具多调参（0→放宽、过多→收紧、≤4 次建议）→ Task 4 `_loop_system_guidance`「Adaptive Search Discipline」✓
- 换工具契合度判断 + 承认失败 → 同上 ✓
- 总命中数进观察 → Task 2 ✓
- 阶梯注入系统提示词（LLM 可见阶梯才能选择/调整）→ Task 3 ✓
- 失败信号（阶梯全 0 → 如实报告）→ 观察携带总命中 0 + Task 4 指导语「honestly report」；未单列代码分支（0 命中时 digest 为空、观察如实呈现 0 条，LLM 按纪律处理）——与 spec 一致 ✓

**2. Placeholder scan:** 无 TBD/TODO；每步含完整代码。

**3. Type consistency:**
- `format_ladder_guidance(rewrite, lang="zh") -> str`：Task 1 定义，Task 3 在 create_agent 调用（`format_ladder_guidance(self._search_rewrite, lang)`）✓
- `agent._search_rewrite`：Task 3 预填充（含 `concepts/queries` 键），Task 4 执行器按 `(cached or {}).get("queries") or []` 消费 ✓；`agent._last_search_total`：Task 2 写入/重置，Task 2 观察消费（`getattr(agent, "_last_search_total", None)`）✓
- Task 4 测试引用的 `_RewriteAgent`（Task 4 旧版定义于测试文件，包含 `_last_user_prompt`、`llm`（带 `calls` 计数的 complete_json stub）、`_search_rewrite=None`）——重写测试时保持该类原样可用；`test_lazy_build_when_cache_missing` 断言 `agent.llm.calls == 1` 依赖该 stub 的计数 ✓
- 既有 C4 测试整体替换（不再有「强制覆盖」语义的断言），与 Task 4 实现一一对应 ✓

**4. 回归风险:** create_agent 每次请求 +1 次改写 LLM 调用（由懒加载改为前置）——成本可接受且换取阶梯提前可见；`_loop_system_guidance` 追加文本不影响既有测试（无测试断言该文本内容）；`dynamic_backend_tool_function` 的捕获代码在最外层 guarded 路径内（`execute_backend_tool_request` 已 try/except）✓
