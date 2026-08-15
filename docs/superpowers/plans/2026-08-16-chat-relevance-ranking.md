# 聊天路径专利相关性排序 + 通配符自适应 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让聊天 ReAct 路径检索出的专利列表按与用户问题的相关度排序呈现（合并同轮多次搜索、Flash 评分、Top-N 精选），并将改写通配符策略改为按概念明确程度自适应。

**Architecture:** 新增 `SearchPool` 候选池模块复用长任务 `relevance_gate` 的 Flash 评分与同族去重；`react_tools.execute_action` 在截断处注入池合并+排序分支（仅后端 push=2 关键词工具、USPTO 形状结果）；`general_agent` 注入 Top-N 精选纪律与排序感知摘要 prompt；`search_query_builder` 纯 prompt 改动实现通配符自适应。所有改动带环境开关 `REACT_RELEVANCE_RANK`（默认开）可一键回退。

**Tech Stack:** Python 3.14（unittest + pytest）、LangChain StructuredTool、Flash LLM（`provider.complete_json`）

## Global Constraints

- 通用性 D5：任何模块不得硬编码领域关键词/阈值/分支；prompt 示例跨领域多样（来自规格 ⑥ 与用户要求「修改要有通用性，不要处处按我们的测试问题去改」）
- 评分失败永不 raise：未评分候选沉底保留（与长任务闸门口径一致）
- 前端零改动；不加新依赖
- 长任务路径不动；每搜索调用仍只取 API 首页
- 分支：feature/agent-react-loop 上直接提交，不开新分支、不推送（用户明确指示）
- 测试命令：`PYTHONUTF8=1 python -m pytest <file> -q`；全量回归带 `--continue-on-collection-errors`
- 提交格式 `feat:/test:/docs:`，无 attribution trailer（已全局禁用）

---

### Task 1: SearchPool 候选池模块

**Files:**
- Create: `sources/long_task/chat_relevance.py`
- Test: `tests/test_chat_relevance.py`

**Interfaces:**
- Consumes: `sources.long_task.candidate_metadata.build_candidates`、`dedupe_candidates`；`sources.long_task.relevance_gate.score_candidates`（均已有）
- Produces: `SearchPool(query)` 类——`add(raw_items) -> int`（合并新候选数）、`score_new(provider) -> int`（本次新评分数）、`ranked(limit) -> list[dict]`（去重排序后候选）、`prune()`、`__len__()`；常量 `POOL_MAX_CANDIDATES = 300`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_relevance.py`:

```python
"""Tests for chat-path relevance ranking pool (chat_relevance)."""
import re
import unittest

from sources.long_task.chat_relevance import POOL_MAX_CANDIDATES, SearchPool


def _usp_raw_item(app_number, title, applicant="ACME Corp",
                  filing="2024-01-15", continuity_ids=None):
    item = {
        "applicationMetaData": {
            "applicationNumberText": app_number,
            "inventionTitle": title,
            "firstApplicantName": applicant,
            "filingDate": filing,
            "applicationStatusDescriptionText": "Patented Case",
        },
    }
    if continuity_ids:
        item["parentContinuityBag"] = [
            {"parentApplicationNumberText": i} for i in continuity_ids
        ]
    return item


class _FakeProvider:
    def __init__(self, scores=None, fail=False):
        self._scores = scores or {}
        self.fail = fail
        self.calls = 0

    async def complete_json(self, system, user):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        ids = re.findall(r"id=(\d+)", user)
        return {"scores": [
            {"id": i, "score": self._scores.get(i, 3)} for i in ids]}


class TestSearchPoolMerge(unittest.TestCase):
    def test_add_flattens_and_merges_new_ids(self):
        pool = SearchPool("测试问题")
        new = pool.add([_usp_raw_item("19511555", "A"),
                        _usp_raw_item("18184836", "B")])
        self.assertEqual(new, 2)
        self.assertEqual(len(pool), 2)

    def test_duplicate_ids_ignored(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A")])
        new = pool.add([_usp_raw_item("19511555", "A"),
                        _usp_raw_item("18184836", "B")])
        self.assertEqual(new, 1)
        self.assertEqual(len(pool), 2)

    def test_non_usp_shapes_add_nothing(self):
        pool = SearchPool("测试问题")
        new = pool.add([{"patentNumber": "US10150077B2"}])
        self.assertEqual(new, 0)
        self.assertEqual(len(pool), 0)

    def test_empty_adds_nothing(self):
        pool = SearchPool("测试问题")
        self.assertEqual(pool.add([]), 0)
        self.assertEqual(pool.add(None), 0)


class TestSearchPoolScoring(unittest.IsolatedAsyncioTestCase):
    async def test_scores_only_unscored_candidates(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A"),
                  _usp_raw_item("18184836", "B")])
        provider = _FakeProvider({"19511555": 5, "18184836": 2})
        scored = await pool.score_new(provider)
        self.assertEqual(scored, 2)
        self.assertEqual(provider.calls, 1)
        # second run: nothing left to score
        scored2 = await pool.score_new(provider)
        self.assertEqual(scored2, 0)
        self.assertEqual(provider.calls, 1)

    async def test_second_add_scores_only_new_ids(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A")])
        provider = _FakeProvider({"19511555": 5})
        await pool.score_new(provider)
        pool.add([_usp_raw_item("18184836", "B")])
        await pool.score_new(provider)
        # two batches: first with 1 id, second with 1 new id
        self.assertEqual(provider.calls, 2)

    async def test_provider_none_scores_nothing(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A")])
        self.assertEqual(await pool.score_new(None), 0)

    async def test_provider_failure_returns_zero_no_raise(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A")])
        scored = await pool.score_new(_FakeProvider(fail=True))
        self.assertEqual(scored, 0)
        self.assertEqual(pool.unscored().__len__(), 1)


class TestSearchPoolRanking(unittest.TestCase):
    def test_ranked_sorts_by_score_desc_unscored_sink(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A"),
                  _usp_raw_item("18184836", "B"),
                  _usp_raw_item("17222222", "C")])
        pool._by_id["19511555"]["relevance_score"] = 1
        pool._by_id["18184836"]["relevance_score"] = 5
        # C stays unscored
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked],
                         ["18184836", "19511555", "17222222"])

    def test_ranked_slices_to_limit(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item(str(19500000 + i), f"T{i}") for i in range(120)])
        self.assertEqual(len(pool.ranked(100)), 100)

    def test_ranked_dedupes_family_members(self):
        pool = SearchPool("测试问题")
        pool.add([
            _usp_raw_item("18184836", "Continuation child",
                          continuity_ids=["19511555"]),
            _usp_raw_item("19511555", "Original parent"),
        ])
        pool._by_id["18184836"]["relevance_score"] = 4
        pool._by_id["19511555"]["relevance_score"] = 5
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["19511555"])

    def test_ranked_dedupes_identical_titles(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "Same title"),
                  _usp_raw_item("18184836", "Same title")])
        pool._by_id["19511555"]["relevance_score"] = 5
        pool._by_id["18184836"]["relevance_score"] = 4
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["19511555"])


class TestSearchPoolPrune(unittest.TestCase):
    def test_prune_keeps_top_max(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item(str(19500000 + i), f"T{i}")
                  for i in range(POOL_MAX_CANDIDATES + 5)])
        pool.prune()
        self.assertEqual(len(pool), POOL_MAX_CANDIDATES)

    def test_prune_caps_only_never_gates(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "keep"),
                  _usp_raw_item("18184836", "drop")])
        pool._by_id["19511555"]["relevance_score"] = 5
        pool._by_id["18184836"]["relevance_score"] = 0
        pool.prune()
        self.assertIn("19511555", pool._by_id)
        self.assertIn("18184836", pool._by_id)  # prune only caps, never gates


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 python -m pytest tests/test_chat_relevance.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'sources.long_task.chat_relevance'`

- [ ] **Step 3: Write the implementation**

Create `sources/long_task/chat_relevance.py`:

```python
"""Relevance-ranked candidate pool for the chat ReAct path.

The chat loop may call the same backend keyword search tool several times
per turn (ladder discipline, tight to loose).  This module merges those
results into one candidate pool, scores newly arrived candidates against
the user's original question with the Flash LLM (same machinery as the
long-task relevance gate), and returns a ranked, family-deduped view for
display and observation digests.

Scoring failures never raise: unscored candidates sink to the bottom of
the ranking, keeping their pool order.
"""
from typing import Any, List

from sources.long_task.candidate_metadata import (
    build_candidates,
    dedupe_candidates,
)
from sources.long_task.relevance_gate import score_candidates

POOL_MAX_CANDIDATES = 300


class SearchPool:
    """Merged candidate pool for one chat turn."""

    def __init__(self, query: str):
        self.query = query
        self._by_id: dict = {}
        self._order: List[str] = []

    def __len__(self) -> int:
        return len(self._order)

    def add(self, raw_items: list) -> int:
        """Merge raw_items into the pool; return the number of NEW
        candidates added (already-known patent_ids are ignored)."""
        new = 0
        for c in build_candidates(raw_items or []):
            pid = c["patent_id"]
            if pid in self._by_id:
                continue
            self._by_id[pid] = c
            self._order.append(pid)
            new += 1
        return new

    def unscored(self) -> list:
        """Candidates without a relevance_score, in pool order."""
        return [self._by_id[pid] for pid in self._order
                if "relevance_score" not in self._by_id[pid]]

    async def score_new(self, provider: Any) -> int:
        """Score unscored candidates via the Flash LLM.

        Never raises.  Returns how many candidates gained a score.
        """
        if provider is None:
            return 0
        pending = self.unscored()
        if not pending:
            return 0
        await score_candidates(pending, self.query, provider)
        return len(pending) - len(self.unscored())

    def ranked(self, limit: int) -> list:
        """Family-deduped candidates ordered granted-first, then
        relevance_score desc, then filing date; unscored sink last.
        Sliced to *limit*."""
        kept, _ = dedupe_candidates(list(self._by_id.values()))
        return kept[:limit]

    def prune(self) -> None:
        """Keep only the top POOL_MAX_CANDIDATES ranked candidates."""
        kept = self.ranked(POOL_MAX_CANDIDATES)
        self._by_id = {c["patent_id"]: c for c in kept}
        self._order = [c["patent_id"] for c in kept]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 python -m pytest tests/test_chat_relevance.py -q`
Expected: all pass (14 tests)

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/chat_relevance.py tests/test_chat_relevance.py
git commit -m "feat: add chat-path relevance ranking pool with incremental Flash scoring"
```

---

### Task 2: react_tools 接线——execute_action 池路径 + 带分摘要 + 开关

**Files:**
- Modify: `sources/agents/react_tools.py`（imports、常量区、`_items_digest` 附近、`execute_action` 内 `pending` 分支）
- Test: `tests/test_react_tools.py`（追加测试类）

**Interfaces:**
- Consumes: `SearchPool`、`MAX_PATENT_LIST_ITEMS`（Task 1 / 既有）；`is_keyword_search_tool`、`build_candidates`（既有）
- Produces: 模块级 `RELEVANCE_RANK_ENABLED`（env `REACT_RELEVANCE_RANK`，默认开）；`_relevance_pool_applies(agent, tool_info, raw_items) -> bool`；`_ranked_digest(candidates, limit=SEARCH_DIGEST_LIMIT, lang) -> str`；`_rank_pending_pool(agent, raw_items, lang) -> (list, str)`；`execute_action` 池分支设置 `agent._search_ranked = True`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_react_tools.py`（文件末尾 `if __name__ == "__main__":` 之前）:

```python
# ── Chat-path relevance ranking pool ─────────────────────────────────────────

from sources.agents.react_tools import _ranked_digest


class _PoolAgent(_FakeAgent):
    def __init__(self):
        super().__init__()
        self._last_user_prompt = "工业干燥空气供应"
        self._search_pool = None
        self._search_ranked = False
        self._last_search_total = None

        class _LLM:
            def __init__(self):
                self.calls = 0
                self._scores = {}
                self.fail = False

            def set_scores(self, scores):
                self._scores = scores

            async def complete_json(self, system, user):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("down")
                ids = re.findall(r"id=(\d+)", user)
                return {"scores": [
                    {"id": i, "score": self._scores.get(i, 3)} for i in ids]}

        self.llm = _LLM()


async def _pool_executor(agent, registry):
    return await make_action_executor(agent, registry, None)


class TestRankedDigest(unittest.TestCase):
    def test_lines_carry_scores(self):
        from sources.long_task.candidate_metadata import build_candidates
        items = [_usp_raw_item("19511555", "Air dryer humidity control"),
                 _usp_raw_item("18184836", "Moisture control enclosure")]
        candidates = build_candidates(items)
        candidates[0]["relevance_score"] = 5
        text = _ranked_digest(candidates)
        self.assertIn("相关度5/5", text)
        # second candidate is unscored — its line carries no score suffix
        unscored_line = text.split("\n")[1]
        self.assertNotIn("相关度", unscored_line)

    def test_caps_at_20_with_total_note(self):
        from sources.long_task.candidate_metadata import build_candidates
        items = [_usp_raw_item(str(19500000 + i), f"Title {i}") for i in range(30)]
        text = _ranked_digest(build_candidates(items))
        self.assertNotIn("Title 20", text)
        self.assertIn("共 30 条", text)


class TestExecuteActionPoolPath(unittest.TestCase):
    def test_pool_path_ranks_observation_by_score(self):
        agent = _PoolAgent()
        agent.llm.set_scores({"19511555": 1, "18184836": 5})
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word")
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "Air dryer humidity control"),
            _usp_raw_item("18184836", "Moisture control enclosure"),
        ]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertEqual(result["kind"], "observation")
        self.assertIn("已按相关度排序", result["text"])
        self.assertIn("相关度5/5", result["text"])
        pos_high = result["text"].find("18184836")
        pos_low = result["text"].find("19511555")
        self.assertLess(pos_high, pos_low)
        self.assertTrue(agent._search_ranked)
        # display list reordered to ranked order
        ids = [str(i.get("applicationMetaData", {}).get("applicationNumberText"))
               for i in agent._pending_raw_items]
        self.assertEqual(ids, ["18184836", "19511555"])

    def test_pool_merges_across_calls(self):
        agent = _PoolAgent()
        agent.llm.set_scores({})
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word")
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [_usp_raw_item("19511555", "First call")]
        asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        agent._pending_raw_items = [_usp_raw_item("18184836", "Second call")]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 2))
        self.assertIn("19511555", result["text"])  # first call still present
        self.assertIn("18184836", result["text"])
        self.assertIn("池共 2 条", result["text"])

    def test_pool_skipped_for_non_keyword_title(self):
        agent = _PoolAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(_registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [_usp_raw_item("19511555", "A")]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertNotIn("已按相关度排序", result["text"])
        self.assertFalse(getattr(agent, "_search_ranked", False))

    def test_pool_skipped_for_non_usp_shape(self):
        agent = _PoolAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word")
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [{"patentNumber": "US10150077B2"}]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertNotIn("已按相关度排序", result["text"])

    def test_flag_off_falls_back_to_legacy(self):
        agent = _PoolAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word")
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        with patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", False):
            executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
            agent._pending_raw_items = [_usp_raw_item("19511555", "A")]
            result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertNotIn("已按相关度排序", result["text"])
        self.assertIn("完整列表已展示", result["text"])

    def test_scoring_failure_keeps_rank_stable(self):
        agent = _PoolAgent()
        agent.llm.fail = True
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word")
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "A"), _usp_raw_item("18184836", "B")]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertIn("已按相关度排序", result["text"])
        self.assertIn("本次新评分 0 条", result["text"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -q`
Expected: FAIL — `ImportError: cannot import name '_ranked_digest'`（`RELEVANCE_RANK_ENABLED` 同样不存在）

- [ ] **Step 3: Write the implementation**

3a. `sources/agents/react_tools.py` — 在 `from sources.long_task.candidate_metadata import (...)` 之后加 import:

```python
from sources.long_task.chat_relevance import SearchPool
```

3b. 在 `MAX_PATENT_LIST_ITEMS` 行后加开关常量:

```python
RELEVANCE_RANK_ENABLED = os.getenv("REACT_RELEVANCE_RANK", "1") != "0"
```

3c. 在 `_cap_patent_list` 函数之后插入两个辅助函数:

```python
def _relevance_pool_applies(agent, tool_info, raw_items) -> bool:
    """Pool + ranking applies only to backend keyword search tools whose
    results flatten via build_candidates (USPTO shape)."""
    if not RELEVANCE_RANK_ENABLED:
        return False
    if getattr(tool_info, "push", None) != 2:
        return False
    if not is_keyword_search_tool(tool_info):
        return False
    return bool(build_candidates(raw_items or []))


def _ranked_digest(candidates, limit: int = SEARCH_DIGEST_LIMIT,
                   lang: str = "zh") -> str:
    """Serialize ranked candidate dicts into a bounded digest with scores."""
    lines = []
    for c in candidates[:limit]:
        score = c.get("relevance_score")
        score_txt = ""
        if isinstance(score, (int, float)):
            score_txt = (f" 相关度{int(score)}/5" if lang == "zh"
                         else f" relevance {int(score)}/5")
        parts = [
            c.get("patent_id") or "?",
            c.get("title") or "(无标题)",
            c.get("applicant") or "?",
            c.get("filing_date") or "?",
            c.get("status") or "?",
        ]
        lines.append(" | ".join(str(p) for p in parts) + score_txt)
    text = "\n".join(lines)
    if len(candidates) > limit:
        note = (f"\n…共 {len(candidates)} 条" if lang == "zh"
                else f"\n...{len(candidates)} items total")
        text += note
    return text[:SEARCH_DIGEST_CHARS]


async def _rank_pending_pool(agent, raw_items, lang) -> Tuple[list, str]:
    """Merge raw_items into the turn's SearchPool, score new arrivals
    against the user's question, and return (ranked candidates, note).

    The pool lives on the agent for the whole request (created lazily;
    create_agent resets it per request).
    """
    pool = getattr(agent, "_search_pool", None)
    if pool is None:
        pool = SearchPool(getattr(agent, "_last_user_prompt", "") or "")
        agent._search_pool = pool
    pool.add(raw_items)
    scored = await pool.score_new(getattr(agent, "llm", None))
    pool.prune()
    ranked = pool.ranked(MAX_PATENT_LIST_ITEMS)
    if lang == "en":
        note = f"relevance-ranked — pool {len(pool)}, scored {scored} new"
    else:
        note = f"已按相关度排序（池共 {len(pool)} 条、本次新评分 {scored} 条）"
    return ranked, note
```

3d. 替换 `execute_action` 里的 pending 分支——把现有这段:

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

换成:

```python
        pending = getattr(agent, "_pending_raw_items", None)
        if pending:
            if _relevance_pool_applies(agent, entry.tool_info, pending):
                ranked, note = await _rank_pending_pool(agent, pending, lang)
                shown = [c["_raw"] for c in ranked]
                agent._pending_raw_items = shown
                agent._search_ranked = True
                digest = _ranked_digest(ranked, lang=lang)
            else:
                shown, note = _cap_patent_list(entry.tool_info, pending, lang)
                agent._pending_raw_items = shown
                digest = _items_digest(shown, lang=lang)
            total = getattr(agent, "_last_search_total", None)
            total_note = ""
            if isinstance(total, int):
                total_note = (f", {total} total hits" if lang == "en"
                              else f"，总命中 {total}")
            if lang == "en":
                text = (f"Search results ({len(shown)} records{total_note}, {note}):\n"
                        f"{digest}\n\n"
                        "The full list is displayed to the user.")
            else:
                text = (f"检索结果（{len(shown)} 条{total_note}，{note}）：\n"
                        f"{digest}\n\n"
                        "完整列表已展示给用户。")
            return {"kind": "observation", "text": text}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py tests/test_chat_relevance.py -q`
Expected: all pass（新 8 个 + 旧全部）

- [ ] **Step 5: Commit**

```bash
git add sources/agents/react_tools.py tests/test_react_tools.py
git commit -m "feat: wire relevance-ranked pool into chat search observations"
```

---

### Task 3: general_agent 接线——请求级重置 + Top-N 纪律 + 摘要 prompt 变体

**Files:**
- Modify: `sources/agents/general_agent.py`（常量区、`create_agent` 重置块、`_loop_system_guidance`、大列表摘要 prompt → 提取模块级函数 `_summary_system_prompt`）
- Test: `tests/test_react_integration_general_agent.py`（追加断言与测试类）

**Interfaces:**
- Consumes: `agent._search_pool`/`agent._search_ranked`（Task 2 设置）；`os.getenv`（既有）
- Produces: 模块常量 `RELEVANT_TOP_N`（env `REACT_RELEVANT_TOP_N`，默认 10）；模块级函数 `_summary_system_prompt(ranked: bool, lang: str) -> str`

- [ ] **Step 1: Write the failing tests**

Modify `tests/test_react_integration_general_agent.py`:

1a. 在 `TestCreateAgentWiring.test_answer_kind_returns_none_and_sets_flag` 的现有断言块:

```python
        # per-request state reset (pool reuse safety)
        self.assertIsNone(getattr(agent, "_pending_raw_items", "unset"))
```

之后追加两行:

```python
        self.assertIsNone(getattr(agent, "_search_pool", "unset"))
        self.assertFalse(getattr(agent, "_search_ranked", "unset"))
```

1b. 在文件末尾（`if __name__ == "__main__":` 之前）追加:

```python
from sources.agents.general_agent import (
    RELEVANT_TOP_N,
    _summary_system_prompt,
)


class TestLoopGuidanceTopN(unittest.TestCase):
    def test_guidance_requires_top_n_relevant_listing(self):
        agent = _make_agent()
        text = agent._loop_system_guidance()
        self.assertIn("relevance-ranked", text)
        self.assertIn(str(RELEVANT_TOP_N), text)

    def test_top_n_default_is_10(self):
        self.assertEqual(RELEVANT_TOP_N, 10)


class TestSummarySystemPrompt(unittest.TestCase):
    def test_ranked_variant_mentions_relevance_order_zh(self):
        text = _summary_system_prompt(True, "zh")
        self.assertIn("相关度排序", text)
        self.assertIn("摘要", text)

    def test_ranked_variant_mentions_relevance_order_en(self):
        text = _summary_system_prompt(True, "en")
        self.assertIn("relevance-ranked", text)

    def test_default_variant_has_no_ranking_note(self):
        self.assertNotIn("相关度排序", _summary_system_prompt(False, "zh"))
        self.assertNotIn("relevance-ranked", _summary_system_prompt(False, "en"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_integration_general_agent.py -q`
Expected: FAIL — `ImportError: cannot import name 'RELEVANT_TOP_N'`（`_summary_system_prompt` 同样不存在）

- [ ] **Step 3: Write the implementation**

3a. 在常量区 `USE_LARGE_LIST_SUMMARY = True` 行之后加:

```python
RELEVANT_TOP_N = int(os.getenv("REACT_RELEVANT_TOP_N", "10"))
```

3b. 在常量区之后、类定义之前加模块级函数:

```python
def _summary_system_prompt(ranked: bool, lang: str) -> str:
    """Build the large-list summary prompt; ranked lists get a
    relevance-order note so the summary leads with the most relevant."""
    ranked_note_zh = "列表已按相关度排序，开头优先呈现最相关项。"
    ranked_note_en = ("The list is already relevance-ranked — lead with "
                      "the most relevant items.")
    if lang == "en":
        base = (
            "You are a professional data analyst. Create a concise, well-structured summary of the data items below. "
            "Group similar items, highlight key patterns or trends, and present information clearly for non-technical readers. "
            "Use Markdown formatting — including tables where appropriate. Keep it under 600 words. "
            "Do NOT list every item individually; synthesize and summarize. "
            "Focus on: what the data shows overall, key differences between items, any notable outliers. "
            "IMPORTANT: The user asked in English — respond entirely in English."
        )
        return base + (" " + ranked_note_en if ranked else "")
    base = (
        "你是一名专业的数据分析师。请对以下数据项创建一个简洁、结构清晰的摘要。"
        "将相似的项目分组，突出关键模式或趋势，并以非技术读者易于理解的方式呈现。"
        "使用 Markdown 格式——包括适当的表格。保持在 600 字以内。"
        "不要逐项列出；综合和总结。"
        "重点关注：数据整体显示的内容、项目之间的关键差异、任何值得注意的异常值。"
        "重要：用户使用中文提问——请用中文回答。"
    )
    return base + (" " + ranked_note_zh if ranked else "")
```

3c. `create_agent` 重置块——在现有行:

```python
        self._last_search_total = None   # total-hit count captured per request
```

之后加:

```python
        self._search_pool = None   # relevance-ranked candidate pool, per request
        self._search_ranked = False  # True once a search list was relevance-ranked
```

3d. `_loop_system_guidance`——先把方法的 `return """` 改为 `return f"""`（否则 `{RELEVANT_TOP_N}` 不会被插值），然后在 `## Adaptive Search Discipline` 段末尾（`Do not keep chaining unsuitable tools.` 行之后）追加:

```python
- If a search observation shows relevance-ranked results (each line ends
  with a 相关度 score), your final answer must first list the most
  relevant top patents (at most {RELEVANT_TOP_N}) — application number,
  title, and one sentence on why each fits the user's question — before
  the overall summary.
```

3e. 大列表摘要路径——把 `_stream_raw_items` 里现有的整段:

```python
            if lang == 'en':
                summary_system_prompt = (
                    "You are a professional data analyst. Create a concise, well-structured summary of the data items below. "
                    "Group similar items, highlight key patterns or trends, and present information clearly for non-technical readers. "
                    "Use Markdown formatting — including tables where appropriate. Keep it under 600 words. "
                    "Do NOT list every item individually; synthesize and summarize. "
                    "Focus on: what the data shows overall, key differences between items, any notable outliers. "
                    "IMPORTANT: The user asked in English — respond entirely in English."
                )
            else:
                summary_system_prompt = (
                    "你是一名专业的数据分析师。请对以下数据项创建一个简洁、结构清晰的摘要。"
                    "将相似的项目分组，突出关键模式或趋势，并以非技术读者易于理解的方式呈现。"
                    "使用 Markdown 格式——包括适当的表格。保持在 600 字以内。"
                    "不要逐项列出；综合和总结。"
                    "重点关注：数据整体显示的内容、项目之间的关键差异、任何值得注意的异常值。"
                    "重要：用户使用中文提问——请用中文回答。"
                )
```

换成一行:

```python
            summary_system_prompt = _summary_system_prompt(
                bool(getattr(self, "_search_ranked", False)), lang)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_integration_general_agent.py -q`
Expected: all pass（新增 5 个 + 旧全部）

- [ ] **Step 5: Commit**

```bash
git add sources/agents/general_agent.py tests/test_react_integration_general_agent.py
git commit -m "feat: add Top-N relevant listing discipline and ranked-summary prompt"
```

---

### Task 4: 通配符自适应 prompt（search_query_builder）

**Files:**
- Modify: `sources/long_task/search_query_builder.py`（`REWRITE_SYSTEM_PROMPT` 步骤 2 与通配符示例行；`format_ladder_guidance` 追加句）
- Test: `tests/test_search_query_builder.py`（追加测试类）

**Interfaces:**
- Consumes: 无新依赖
- Produces: 改写后的 `REWRITE_SYSTEM_PROMPT`（含概念明确度判断规则、跨领域通配符示例）；`format_ladder_guidance` 输出含假性零命中重试句

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search_query_builder.py`（文件末尾）:

```python
class TestRewritePromptWildcardClarity(unittest.TestCase):
    def test_prompt_requires_clarity_judgment(self):
        self.assertIn("明确程度", REWRITE_SYSTEM_PROMPT)
        self.assertIn("一般性技术概念", REWRITE_SYSTEM_PROMPT)
        self.assertIn("判断不清", REWRITE_SYSTEM_PROMPT)

    def test_prompt_wildcard_examples_are_cross_domain(self):
        self.assertIn("filter*", REWRITE_SYSTEM_PROMPT)
        self.assertIn("cataly*", REWRITE_SYSTEM_PROMPT)

    def test_prompt_has_no_single_domain_anchor(self):
        self.assertNotIn("air dry", REWRITE_SYSTEM_PROMPT)
        self.assertNotIn("dehumidif", REWRITE_SYSTEM_PROMPT)


class TestLadderGuidanceWildcardRetry(unittest.TestCase):
    def test_guidance_suggests_wildcard_retry_on_false_zero_zh(self):
        text = format_ladder_guidance({"queries": ['("a" OR "b") AND ("c" OR "d")']}, "zh")
        self.assertIn("假性零命中", text)
        self.assertIn("通配符", text)

    def test_guidance_suggests_wildcard_retry_on_false_zero_en(self):
        text = format_ladder_guidance({"queries": ['("a" OR "b") AND ("c" OR "d")']}, "en")
        self.assertIn("false zero", text)
        self.assertIn("wildcard", text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 python -m pytest tests/test_search_query_builder.py -q`
Expected: FAIL — `AssertionError`（明确程度/一般性技术概念/filter*/假性零命中 等均不在当前 prompt）

- [ ] **Step 3: Write the implementation**

3a. `REWRITE_SYSTEM_PROMPT` 步骤 2——把现有:

```python
    "2. 每个概念翻译成该领域专利文献常用的英文术语，并给出 2-5 个"
    "同义/近义关键词（含行业缩写、上位/下位词、英美拼写变体）\n"
```

换成:

```python
    "2. 每个概念翻译成该领域专利文献常用的英文术语，并给出 2-5 个"
    "同义/近义关键词（含行业缩写、上位/下位词、英美拼写变体），"
    "并判断该概念的明确程度：\n"
    "   - 明确专有名词/品牌名/具体化合物名 → 精确词即可\n"
    "   - 一般性技术概念 → 关键词集中必须包含至少一个词尾通配符变体\n"
    "   - 判断不清 → 精确词与词尾通配符变体都放入关键词集\n"
```

3b. 通配符语法示例两行——把现有:

```python
    "   - 支持词尾通配符：air dry* 匹配 air dryer/drying/dried，"
    "dehumidif* 匹配 dehumidifier/dehumidification；通配符只在词尾生效，"
    "引号短语内不生效，词首通配符无效\n"
    "   - 为每个概念补充常见词形变体（dryer/drying、dehumidifier/"
    "dehumidification 等），或直接用通配符覆盖变体\n"
```

换成:

```python
    "   - 支持词尾通配符：filter* 匹配 filter/filtering/filtration，"
    "cataly* 匹配 catalyst/catalysis/catalytic；通配符只在词尾生效，"
    "引号短语内不生效，词首通配符无效\n"
    "   - 为每个概念补充常见词形变体，或直接用通配符覆盖变体\n"
```

3c. `format_ladder_guidance`——把函数末尾现有的 `    return "\n".join(lines)` 一行替换为:

```python
    if lang == "en":
        lines.append(
            "\nAlso: if a query returns 0 hits even though the technology "
            "clearly exists (a false zero), add word-ending wildcard "
            "variants to the concept terms and retry."
        )
    else:
        lines.append(
            "\n另外：当某级检索式在相关技术确实存在时仍返回 0 命中"
            "（假性零命中），可给概念词补充词尾通配符变体后重试。"
        )
    return "\n".join(lines)
```

（原 `return "\n".join(lines)` 行删除，由上面替换。）

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 python -m pytest tests/test_search_query_builder.py -q`
Expected: all pass（新增 5 个 + 旧全部，含旧断言 通配符/词尾/引号/最紧/放宽/限定/单组 全部保持）

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/search_query_builder.py tests/test_search_query_builder.py
git commit -m "feat: make wildcard rewrite adaptive to concept clarity with cross-domain examples"
```

---

### Task 5: 全量回归 + 收尾

**Files:**
- Test: `tests/`（全量）

- [ ] **Step 1: Run the full regression**

Run: `PYTHONUTF8=1 python -m pytest tests/ -q --continue-on-collection-errors`
Expected: 26 failed / 9 errors 为预存环境问题（系统 python 缺 ollama/markdownify），必须与基线逐条一致、零新增；passed 数应较基线 423 净增约 32（Task 1-4 新增 14+8+5+5 个测试）。若出现新增失败，先修再进下一步。

- [ ] **Step 2: Quick smoke of the kill switch**

Run: `REACT_RELEVANCE_RANK=0 PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -q`
Expected: all pass（flag-off 回退路径保持绿色）

- [ ] **Step 3: Commit any regression fixes, then final status**

```bash
git status --short
git log --oneline -6
```

预期：工作树干净；新提交 4 个 feat 落在 bbb3376 之上，分支 feature/agent-react-loop 未推送。

- [ ] **Step 4: Report for deployment acceptance**

向用户汇报：改动清单、回归基线对照、部署建议（后端 api-test + 前端 test.copiioai.com）与多领域基准验收清单（干燥空气/电池热管理/半导体腔温/AR 光学/医疗 AI，每查询切题率≥80%）。

---

## Self-Review Notes

- **Spec coverage**: 规格 ① → Task 1+2（池/评分/排序/开关）；② → Task 4（clarity 判断 + 跨领域示例 + 假性零命中重试句）；③ → Task 3（Top-N 纪律 + 摘要变体，下载 artifact 因输入自然排序无需改动）；④ → Task 1（score_new 永不 raise）+ Task 2（flag-off 测试）；⑤ → Task 1-4 单测 + Task 5 回归；⑥ → 各任务无越界改动
- **Type consistency**: `SearchPool.add/score_new/ranked/prune/__len__` 签名在 Task 1 定义、Task 2 使用一致；`agent._search_pool/_search_ranked` 在 Task 2 写入、Task 3 重置与读取一致；`_summary_system_prompt(ranked, lang)` 定义与调用一致
- **既有测试兼容**: `test_react_tools.py` 旧用例标题 "uspto search" 不含 "key" → `_relevance_pool_applies` 为 False → 走 legacy 路径不受影响；`test_search_query_builder.py` 旧断言词（通配符/词尾/引号/最紧/放宽/限定/只含一个核心概念/单组/由紧到松/tightest）在新 prompt 文本中全部保留
