# Fix Report — Whole-Branch Review Findings

## Status: All 11 findings resolved

## CRITICAL FIXES

### 1. celery_worker.py: `asyncio.run()` -> event-loop-safe pattern
- **File**: `celery_worker.py` (line 66)
- **Change**: Replaced `asyncio.run(_run_pipeline(...))` with `loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); loop.run_until_complete(...)` with `finally: loop.close()`
- **Commit**: 543cf19

### 2. api_routes/core.py: user_id BIGINT type mismatch
- **File**: `api_routes/core.py` (line 379)
- **Change**: Added `local_user_id = int(user_id)` cast before INSERT into `conversations` and `long_tasks` tables (which have `BIGINT UNSIGNED` columns). Redis caching may return bytes; explicit int() cast ensures safety.
- **Commit**: 543cf19

### 3. celery_worker.py: Sync export_pdf/export_docx block event loop
- **File**: `celery_worker.py` (lines 217-223, new async wrappers at bottom)
- **Change**: Wrapped `export_pdf` and `export_docx` as async functions (`export_pdf_async`, `export_docx_async`) using `loop.run_in_executor(None, _sync_export)`. Pipeline calls the async wrappers with `await`.
- **Commit**: 543cf19

### 4. general_agent.py: Remove dead `invoke_agent` handler for `intent == 'long_task'`
- **File**: `sources/agents/general_agent.py` (lines 1669-1679)
- **Change**: Removed the `isinstance(agent, dict) and agent.get('intent') == 'long_task'` branch in `invoke_agent`. This is dead code since `core.py` handles the long_task intent marker before `invoke_agent` is ever called.
- **Commit**: 543cf19

## HIGH FIXES

### 5. Firebase auth middleware for session.py and long_task.py
- **File**: `api_routes/session.py` (lines 38, 78), `api_routes/long_task.py` (lines 29, 38)
- **Change**: Added `verify_firebase_token(auth_header)` to `create_session`, `list_sessions`, `task_status`, and `download_report` endpoints. Added `http_request: Request` parameter to each.
- **Commit**: 72953c8

### 6. Session routes: Don't accept user_id from request body
- **File**: `api_routes/session.py` (line 19)
- **Change**: Removed `user_id: int` from `CreateSessionRequest` model. `user_id` is now extracted from Firebase auth token exclusively.
- **Commit**: 72953c8

### 7. api_routes/core.py: Update conversations.long_task_ids JSON column
- **File**: `api_routes/core.py` (lines 381-385)
- **Change**: Added `long_task_ids` column to INSERT into `conversations` table, setting it to `json.dumps([task_id])`.
- **Commit**: 543cf19

### 8. celery_worker.py: Update MySQL long_tasks status after each phase
- **File**: `celery_worker.py` (new helper + 3 call sites)
- **Change**: Added `_update_mysql_progress(task_id, current_phase, progress)` function that updates `long_tasks.status`, `current_phase`, and `progress` columns. Called after Phase 1 (generating_columns), Phase 2 (analyzing), Phase 3 (generating_report), and Phase 4 (exporting).
- **Commit**: 543cf19

### 9. celery_worker.py: Deterministic patent_ids dedup
- **File**: `celery_worker.py` (line 48)
- **Change**: Replaced `list(dict.fromkeys(...))` with `sorted(set(...))` for deterministic ordering.
- **Commit**: 543cf19

## MEDIUM FIXES

### 10. celery_worker.py: Remove unused `offset` parameter from `progress_pct`
- **File**: `celery_worker.py` (line 233 + all call sites)
- **Change**: Removed `offset: int` parameter from `progress_pct` and updated all call sites to pass 2 args instead of 3.
- **Commit**: 543cf19

### 11. sources/knowledge/selection.py: Verify conversation_history pass-through
- **File**: `sources/knowledge/knowledge.py` (lines 496-516)
- **Change**: Added `conversation_history: Optional[list] = None` parameter to `select_knowledge_tool_with_llm`. Forwarded it as `conversation_history=conversation_history` to `choose_knowledge_candidate`. The `choose_knowledge_candidate` function already had the parameter and passed it to `build_routing_user_content`.
- **Commit**: a94e512

## Commit Hashes
- CRITICAL: 543cf19
- HIGH: 72953c8
- MEDIUM: a94e512
