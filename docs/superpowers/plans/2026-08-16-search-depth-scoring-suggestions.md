# 检索深度 + 评分增强 + 确定性收紧建议 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扩大聊天检索的候选深度（每调用翻页入池）、把 CPC 分类码加入 Flash 评分输入、命中过多时在观察里给确定性收紧建议——三项生产验证后的质量优化。

**Architecture:** `react_tools.execute_action` 对 USPTO 形状工具在 invoke 前构造「信封请求」（模板 body + 有效 q + 确保字段 + 翻页），翻页结果并入候选池；`relevance_gate` 的评分批次文本与 prompt 增加 CPC 分类码；观察文本按总命中数追加确定性建议句。

**Tech Stack:** Python 3.14（unittest + pytest）、LangChain StructuredTool、Flash LLM（`provider.complete_json`）

## Global Constraints

- 通用性 D5：不得硬编码领域关键词/阈值/分支；新增阈值一律环境变量可调
- 评分失败永不 raise；信封构造失败必须回退 raw args 直调（降级不崩溃）
- 前端零改动；长任务路径不动（长任务的 ensure_search_fields 已有，不受影响）
- 分支 feature/agent-react-loop 直接提交，不推送；测试命令 `PYTHONUTF8=1 python -m pytest <file> -q`
- 提交格式 `feat:/test:`，无 attribution trailer
- 全量回归基线：**463 passed / 26 failed / 9 errors**（26+9 为预存环境问题）

---

### Task A: USPTO 信封请求 + 翻页入池

**Files:**
- Modify: `sources/agents/react_tools.py`
- Test: `tests/test_react_tools.py`

**Interfaces:**
- Consumes: `sources.long_task.candidate_metadata`（`is_uspto_tool`、`build_candidates`、`ensure_search_fields`）；`sources.dynamic_tool_params._coerce_json_object`；`SearchPool`
- Produces:
  - `REACT_POOL_MAX_PAGES = int(os.getenv("REACT_POOL_MAX_PAGES", "2"))`（每调用最多取 3 页：第 1 页 + 2 页翻页）
  - `_effective_query(args) -> str`（提取真正送达 API 的 q：flat 首字符串 / body.q / params-json 的 q）
  - `_build_uspto_envelope(tool_info, q) -> dict`（模板 body + q + ensure_search_fields + 信封键）
  - `_collect_search_pages(agent, entry, args, first_raw) -> list`（翻页收集，去重哨兵 + 总命中提前停止）

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_react_tools.py`:

```python
# ── USPTO envelope + pagination ─────────────────────────────────────────────

from sources.agents.react_tools import (
    REACT_POOL_MAX_PAGES,
    _build_uspto_envelope,
    _collect_search_pages,
    _effective_query,
)


def _keyword_tool_template():
    """A realistic push=2 USPTO keyword tool template (tool_info.params)."""
    import json
    return json.dumps({
        "method": "POST",
        "query": {},
        "path": "/api/v1/patent/applications/search",
        "header": {},
        "body": {
            "q": "",
            "pagination": {"offset": 0, "limit": 50},
            "fields": ["applicationNumberText", "applicationMetaData.inventionTitle"],
            "sort": [{"field": "assignmentBag.assignmentRecordedDate", "order": "desc"}],
        },
    })


class _PagedToolInfo(_ToolInfo):
    def __init__(self):
        super().__init__("search_patent_by_key_word",
                         url="https://api.uspto.gov/api/v1/patent/applications/search")
        self.params = _keyword_tool_template()


class TestEffectiveQuery(unittest.TestCase):
    def test_flat_first_string_wins(self):
        self.assertEqual(
            _effective_query({"q": '("dry air" OR drying)', "page": 1, "pageSize": 10}),
            '("dry air" OR drying)')

    def test_body_q_extracted(self):
        self.assertEqual(
            _effective_query({"method": "POST", "body": {"q": "dryer AND humidit*"}}),
            "dryer AND humidit*")

    def test_params_json_q_extracted(self):
        self.assertEqual(
            _effective_query({"params": '{"q": "dry* AND humid*", "page": 1}'}),
            "dry* AND humid*")

    def test_missing_returns_empty(self):
        self.assertEqual(_effective_query({"page": 1}), "")
        self.assertEqual(_effective_query(None), "")


class TestBuildUsptoEnvelope(unittest.TestCase):
    def test_envelope_carries_template_and_ensures_fields(self):
        tool_info = _PagedToolInfo()
        envelope = _build_uspto_envelope(tool_info, 'dry* AND humid*')
        self.assertEqual(envelope["method"], "POST")
        self.assertEqual(envelope["body"]["q"], "dry* AND humid*")
        self.assertIn("applicationMetaData.cpcClassificationBag",
                      envelope["body"]["fields"])
        self.assertEqual(envelope["body"]["pagination"]["limit"], 50)
        self.assertEqual(envelope["query"], {})
        self.assertEqual(envelope["path"], "/api/v1/patent/applications/search")


class TestCollectSearchPages(unittest.IsolatedAsyncioTestCase):
    async def _agent_with_pages(self, pages_by_offset):
        agent = _PoolAgent()
        entry_k = _Knowledge(3, ktype=1)

        def get_dynamic_tool_for(knowledge_item, tool_info):
            from sources.agents.react_tools import StructuredTool
            class _Args(BaseModel):
                params: str = Field(description="params")
            def _noop(**kwargs):
                # production side effect: write page items + total hits
                import json as _json
                envelope = _json.loads(kwargs.get("params") or "{}")
                body = envelope.get("body") or {}
                offset = (body.get("pagination") or {}).get("offset", 0)
                page_items, total = pages_by_offset.get(offset, ([], 0))
                agent._pending_raw_items = page_items
                agent._last_search_total = total
                return f"The query returned {len(page_items)} items."
            return StructuredTool.from_function(_noop, name="search_patent_by_key_word",
                                                description="d", args_schema=_Args)
        agent.get_dynamic_tool_for = get_dynamic_tool_for
        tool_info = _PagedToolInfo()
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        return agent, entry

    async def test_pages_merged_until_total_or_cap(self):
        p0 = [_usp_raw_item("19500001", "P0-1"), _usp_raw_item("19500002", "P0-2")]
        p50 = [_usp_raw_item("19500003", "P50-1")]
        p100 = [_usp_raw_item("19500004", "P100-1")]
        agent, entry = await self._agent_with_pages(
            {0: (p0, 120), 50: (p50, 120), 100: (p100, 120)})
        collected = await _collect_search_pages(agent, entry, {"q": "dry*"}, p0)
        self.assertEqual([c["patent_id"] for c in collected],
                         ["19500001", "19500002", "19500003"])
        # page 3 (offset 100) NOT fetched: REACT_POOL_MAX_PAGES=2 pages after first

    async def test_total_exhausted_stops_early(self):
        p0 = [_usp_raw_item("19500001", "P0-1")]
        agent, entry = await self._agent_with_pages({0: (p0, 51), 50: ([], 51)})
        collected = await _collect_search_pages(agent, entry, {"q": "dry*"}, p0)
        self.assertEqual(len(collected), 1)

    async def test_duplicate_page_stops(self):
        p0 = [_usp_raw_item("19500001", "P0-1")]
        # template ignores offset → same items come back
        agent, entry = await self._agent_with_pages(
            {0: (p0, 200), 50: (p0, 200)})
        collected = await _collect_search_pages(agent, entry, {"q": "dry*"}, p0)
        self.assertEqual(len(collected), 1)

    async def test_no_q_returns_first_page_only(self):
        agent, entry = await self._agent_with_pages({})
        collected = await _collect_search_pages(agent, entry, {"page": 1}, [])
        self.assertEqual(collected, [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -q`
Expected: ImportError — `_effective_query` 等不存在

- [ ] **Step 3: Implement**

3a. 常量（`RELEVANCE_RANK_ENABLED` 之后）:

```python
REACT_POOL_MAX_PAGES = int(os.getenv("REACT_POOL_MAX_PAGES", "2"))
```

3b. 辅助函数（`_rank_pending_pool` 之前）:

```python
_ENVELOPE_KEYS = frozenset({"method", "body", "query", "path", "header"})


def _effective_query(args: dict) -> str:
    """Extract the query string that actually reaches the search API.

    Mirrors the flat-merge rule (first non-empty string value wins) and
    also understands body.q and params-JSON shapes.  Returns "" when no
    query can be recovered — callers then skip envelope building.
    """
    if not isinstance(args, dict):
        return ""
    body = args.get("body")
    if isinstance(body, dict):
        q = body.get("q")
        if isinstance(q, str) and q.strip():
            return q.strip()
    params = args.get("params")
    if isinstance(params, str) and params.strip():
        try:
            parsed = json.loads(params)
            if isinstance(parsed, dict):
                q = parsed.get("q")
                if isinstance(q, str) and q.strip():
                    return q.strip()
                return ""  # params dict carries no q — nothing to recover
        except (ValueError, TypeError):
            pass  # malformed params string — fall through to flat scan
    for value in args.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_uspto_envelope(tool_info, q: str) -> dict:
    """Build a template-faithful request envelope for a USPTO search tool.

    Carries the tool template's body (fields list included), injects *q*
    the way the flat-merge does, ensures the relevance fields
    (cpcClassificationBag etc.) are requested, and preserves method /
    query / path / header from the template.  Never raises — on any
    template problem returns a minimal envelope with just q.
    """
    try:
        from sources.dynamic_tool_params import _coerce_json_object
        template = _coerce_json_object(tool_info.params, "tool_info.params") or {}
        body = dict(template.get("body") or {})
    except Exception:
        template, body = {}, {}
    body["q"] = q
    try:
        body = ensure_search_fields({"body": body})["body"]
    except Exception:
        pass
    return {
        "method": template.get("method", "POST"),
        "body": body,
        "query": template.get("query", {}),
        "path": template.get("path"),
        "header": template.get("header", {}),
    }


async def _collect_search_pages(agent, entry, args, first_raw: list) -> list:
    """Fetch extra result pages for a USPTO search call and merge them.

    Stops early when total hits are exhausted, when a page returns items
    already seen (template ignored the offset), or after
    REACT_POOL_MAX_PAGES extra pages.  Never raises — failures return
    whatever was collected so far.
    """
    items = [c for c in build_candidates(first_raw or [])]
    q = _effective_query(args or {})
    if not q:
        return items
    seen_ids = {c["patent_id"] for c in items}
    total = getattr(agent, "_last_search_total", None)
    page_size = 50
    try:
        from sources.dynamic_tool_params import _coerce_json_object
        template = _coerce_json_object(entry.tool_info.params,
                                       "tool_info.params") or {}
        body = template.get("body") or {}
        page_size = int((body.get("pagination") or {}).get("limit", 50))
    except Exception:
        pass
    offset = page_size
    for _page in range(REACT_POOL_MAX_PAGES):
        if isinstance(total, int) and offset >= total:
            break
        envelope = _build_uspto_envelope(entry.tool_info, q)
        envelope["body"]["pagination"] = {"offset": offset, "limit": page_size}
        try:
            await asyncio.to_thread(entry.tool.invoke, envelope)
        except Exception:
            break
        raw = getattr(agent, "_pending_raw_items", None) or []
        fresh = [c for c in build_candidates(raw) if c["patent_id"] not in seen_ids]
        if not fresh:
            break  # offset ignored or universe exhausted
        for c in fresh:
            seen_ids.add(c["patent_id"])
        items.extend(fresh)
        offset += page_size
        if isinstance(total, int) and len(items) >= total:
            break
    return items
```

3c. `execute_action` knowledge 分支改造——把现有:

```python
        try:
            args = await _maybe_rewrite_search_query(agent, entry.tool_info, args)
            result = await asyncio.to_thread(entry.tool.invoke, args)
        except Exception as exc:
            return {"kind": "observation", "text": f"Error: {exc}"}
```

换成:

```python
        try:
            args = await _maybe_rewrite_search_query(agent, entry.tool_info, args)
            pool_eligible = _relevance_pool_applies_tool(agent, entry.tool_info)
            invoke_args = args
            if pool_eligible and isinstance(args, dict) \
                    and not (_ENVELOPE_KEYS.intersection(args)):
                q = _effective_query(args)
                if q:
                    invoke_args = _build_uspto_envelope(entry.tool_info, q)
            result = await asyncio.to_thread(entry.tool.invoke, invoke_args)
        except Exception as exc:
            return {"kind": "observation", "text": f"Error: {exc}"}
```

并把池分支的调用改为翻页收集:

```python
            if _relevance_pool_applies(agent, entry.tool_info, pending):
                collected = await _collect_search_pages(agent, entry, args, pending)
                ranked, note = await _rank_pending_pool(agent, collected, lang)
```

其中 `_rank_pending_pool` 的入参从 raw_items 改为 candidates 列表——把函数内第一行 `pool.add(raw_items)` 改为 `pool.add_from_candidates(collected)`（见下 3d）。

3d. 新增工具级判定函数（`_relevance_pool_applies` 之前）:

```python
def _relevance_pool_applies_tool(agent, tool_info) -> bool:
    """Tool-level (pre-invoke) half of the pool gate: switch on, backend,
    USPTO URL.  The parse check happens post-invoke on the results."""
    if not RELEVANCE_RANK_ENABLED:
        return False
    if getattr(tool_info, "push", None) != 2:
        return False
    return is_uspto_tool(tool_info)
```

3e. `SearchPool` 增加 `add_from_candidates`（`sources/long_task/chat_relevance.py`）:

```python
    def add_from_candidates(self, candidates: list) -> int:
        """Merge pre-built candidate dicts into the pool; return the
        number of NEW candidates added."""
        new = 0
        for c in candidates or []:
            pid = c.get("patent_id")
            if not pid or pid in self._by_id:
                continue
            self._by_id[pid] = c
            self._order.append(pid)
            new += 1
        return new
```

并把 `_rank_pending_pool` 的 `pool.add(raw_items)` 改为 `pool.add_from_candidates(raw_items)`（参数名改为 `candidates`）。

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py tests/test_chat_relevance.py -q`
Expected: all pass（含全部旧测试）

- [ ] **Step 5: Commit**

```bash
git add sources/agents/react_tools.py sources/long_task/chat_relevance.py tests/test_react_tools.py
git commit -m "feat: paginate USPTO search pages into the chat pool via template-faithful envelopes"
```

---

### Task B: CPC 分类码加入评分输入

**Files:**
- Modify: `sources/long_task/relevance_gate.py`
- Test: `tests/test_relevance_gate.py`

**Interfaces:**
- Consumes: 候选 dict 的 `cpc_codes` 字段（`build_candidates` 已产出；Task A 之后聊天路径响应也会带该字段）
- Produces: 修订后的 `GATE_SYSTEM_PROMPT`（允许依据 CPC）；`_batch_text` 输出行含 CPC

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_relevance_gate.py`:

```python
class TestBatchTextIncludesCpc(unittest.TestCase):
    def test_cpc_codes_rendered_in_batch_lines(self):
        from sources.long_task.relevance_gate import _batch_text
        candidates = [{
            "patent_id": "19511555",
            "title": "Air dryer humidity control",
            "applicant": "ACME",
            "filing_date": "2024-01-15",
            "cpc_codes": ["F26B 21/08", "B01D 53/26"],
        }]
        text = _batch_text(candidates, "干燥空气")
        self.assertIn("F26B 21/08", text)
        self.assertIn("B01D 53/26", text)

    def test_missing_cpc_renders_empty(self):
        from sources.long_task.relevance_gate import _batch_text
        candidates = [{
            "patent_id": "19511555", "title": "T", "applicant": "A",
            "filing_date": "2024-01-15", "cpc_codes": [],
        }]
        text = _batch_text(candidates, "q")
        self.assertIn("cpc=[]", text)


class TestGatePromptAllowsCpc(unittest.TestCase):
    def test_prompt_mentions_cpc(self):
        from sources.long_task.relevance_gate import GATE_SYSTEM_PROMPT
        self.assertIn("CPC", GATE_SYSTEM_PROMPT)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 python -m pytest tests/test_relevance_gate.py -q`
Expected: FAIL（batch 行无 CPC、prompt 无 CPC 字样）

- [ ] **Step 3: Implement**

3a. `_batch_text` 行格式改为:

```python
    lines = [
        f"- id={c['patent_id']} | title={c.get('title') or '(no title)'}"
        f" | applicant={c.get('applicant') or '?'}"
        f" | filing={c.get('filing_date') or '?'}"
        f" | cpc={c.get('cpc_codes') or []}"
        for c in candidates
    ]
```

3b. `GATE_SYSTEM_PROMPT` 中 `"≥3 分视为相关。只依据标题、申请人、日期判断，不要猜测未知内容。"` 换成:

```python
    "≥3 分视为相关。依据标题、申请人、CPC 分类码、日期判断，"
    "不要猜测未知内容。CPC 分类码是强信号：与用户问题的技术领域"
    "（如干燥/除湿、电气柜、半导体腔室）同族的分类码应显著加分。"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 python -m pytest tests/test_relevance_gate.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/relevance_gate.py tests/test_relevance_gate.py
git commit -m "feat: include CPC codes in relevance-gate scoring input"
```

---

### Task C: 命中过多确定性收紧建议

**Files:**
- Modify: `sources/agents/react_tools.py`
- Test: `tests/test_react_tools.py`

**Interfaces:**
- Consumes: `agent._last_search_total`
- Produces: `REACT_TIGHTEN_SUGGEST_THRESHOLD = int(os.getenv("REACT_TIGHTEN_SUGGEST_THRESHOLD", "5000"))`；`_tighten_hint(total, lang) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_react_tools.py`:

```python
from sources.agents.react_tools import (
    REACT_TIGHTEN_SUGGEST_THRESHOLD,
    _tighten_hint,
)


class TestTightenHint(unittest.TestCase):
    def test_hint_above_threshold_zh(self):
        self.assertIn("收紧", _tighten_hint(REACT_TIGHTEN_SUGGEST_THRESHOLD, "zh"))

    def test_hint_above_threshold_en(self):
        self.assertIn("tighten", _tighten_hint(REACT_TIGHTEN_SUGGEST_THRESHOLD, "en"))

    def test_no_hint_below_threshold(self):
        self.assertEqual(_tighten_hint(REACT_TIGHTEN_SUGGEST_THRESHOLD - 1, "zh"), "")

    def test_no_hint_for_non_int(self):
        self.assertEqual(_tighten_hint(None, "zh"), "")
        self.assertEqual(_tighten_hint("99999", "zh"), "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -q`
Expected: ImportError

- [ ] **Step 3: Implement**

常量 + 函数（`REACT_POOL_MAX_PAGES` 之后）:

```python
REACT_TIGHTEN_SUGGEST_THRESHOLD = int(os.getenv("REACT_TIGHTEN_SUGGEST_THRESHOLD", "5000"))


def _tighten_hint(total, lang: str = "zh") -> str:
    """Deterministic advice appended to observations when a search hit
    far too many results — the agent gets the suggestion from code
    instead of relying on its own judgment."""
    if not isinstance(total, int):
        return ""
    if total < REACT_TIGHTEN_SUGGEST_THRESHOLD:
        return ""
    if lang == "en":
        return ("\nToo many hits: retry with a tighter ladder query "
                "(e.g. ladder #1) or add a scene-constraint term.")
    return ("\n命中过多：建议改用更紧的阶梯检索式（如阶梯第 1 条）"
            "或添加场景限定词后重试。")
```

在 `execute_action` 共享段，把:

```python
            total_note = ""
            if isinstance(total, int):
                total_note = (f", {total} total hits" if lang == "en"
                              else f"，总命中 {total}")
```

换成:

```python
            total_note = ""
            if isinstance(total, int):
                total_note = (f", {total} total hits" if lang == "en"
                              else f"，总命中 {total}")
                total_note += _tighten_hint(total, lang)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add sources/agents/react_tools.py tests/test_react_tools.py
git commit -m "feat: append deterministic tighten suggestion when search hits exceed threshold"
```

---

### Task D: 全量回归 + 收尾（controller）

- [ ] **Step 1**: `PYTHONUTF8=1 python -m pytest tests/ -q --continue-on-collection-errors 2>&1 | tail -2`
  预期：26 failed / 9 errors 预存环境问题；passed = 463 + 新增（A:6、B:3、C:4 及相关）= 约 476；零新增失败
- [ ] **Step 2**: 记录账本 `.superpowers/sdd/progress.md`；向用户给部署清单（react_tools.py / chat_relevance.py / relevance_gate.py）+ 复测验收信号

---

## Self-Review Notes

- **规格覆盖**: ①②③ 对应 Task A/B/C；D5 通用性——新阈值全部 env 可调，无领域硬编码；CPC 提示为通用规则非特定领域
- **类型一致性**: `_collect_search_pages(agent, entry, args, first_raw) -> list[dict]`（candidates）；`_rank_pending_pool(agent, candidates, lang)` 第二参改为 candidates（`add_from_candidates`）；`_relevance_pool_applies_tool` 与 `_relevance_pool_applies` 分工：前者 invoke 前工具级判定，后者 invoke 后加解析判定
- **既有测试兼容**: `_PoolAgent` 无 `_search_pool` 之外的新依赖；`test_pool_path_ranks_observation_by_score` 等现有池测试仍走 `_rank_pending_pool` 但经 `_collect_search_pages` 收集——其 `_make_tool_with_pending` 返回字符串且不设 `_pending_raw_items`，`_collect_search_pages` 对 args `{"params": "{}"}` 提取 q 为空 → 直接返回第一页 candidates，行为不变
- **翻页哨兵**: `offset = len(items) if not total else 50 * (_page + 1)` 覆盖 total 缺失场景；重复页检测防模板忽略 offset 死循环
