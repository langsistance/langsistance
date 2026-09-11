#!/usr/bin/env python3
"""Long task status polling and report download API routes.

Endpoints:
  GET /long_task/{task_id}/status
  GET /long_task/{task_id}/report?format=pdf|docx
"""

import re

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import Response
from sources.long_task.status_manager import (
    ERR_OTHER,
    ERR_UNRESOLVABLE_ID,
    failure_guidance,
    get_task_status,
    is_retryable_failure,
    lookup_query_task,
)
from sources.long_task.storage import create_storage, get_storage_config, LocalReportStorage
from sources.patent_id_translator import verdict_of
from sources.patent_id_utils import extract_us_patent_digits, kind_code_of
from sources.user.passport import verify_firebase_token


def _dispatch_from_mysql(user_id: str, task_id: str, logger) -> None:
    """Read task params from MySQL and dispatch via Celery.  Idempotent — safe
    to call even if the task was already dispatched."""
    import json as _json
    from sources.knowledge.knowledge import get_db_connection as _gdc
    conn = _gdc()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT input_params, session_id, scene_id, task_type
                   FROM long_tasks WHERE task_id = %s""",
                (task_id,),
            )
            row = cur.fetchone()
        if row:
            input_params = row.get('input_params')
            stored = _json.loads(input_params) if isinstance(input_params, str) else input_params
            next_params = {
                'query': stored.get('query', ''),
                # Single-patent tasks (family/prosecution/china/ep/jp) persist a
                # *singular* ``patent_id`` (no ``patent_ids`` key) in MySQL
                # input_params — forward it verbatim or the resumed executor
                # reads ``params.get('patent_id')`` == '' and flags the task
                # failed.  Batch rows carry ``patent_ids`` only.
                'patent_id': stored.get('patent_id'),
                'patent_ids': stored.get('patent_ids', []),
                'patent_source': stored.get('patent_source', 'auto'),
                'session_id': row.get('session_id') or '',
                'scene_id': row.get('scene_id'),
                'conversation_history': stored.get('conversation_history', []),
                'patent_file_refs': stored.get('patent_file_refs', []),
                'user_id': str(user_id),
            }
            if stored.get('patent_texts'):
                next_params['patent_texts'] = stored['patent_texts']
            # Dispatch by the stored task_type (A9): a paused family_analysis
            # must resume on the family executor, never the batch one.  Missing
            # type falls back to the batch executor (historical default).
            task_type = str(row.get('task_type') or 'patent_analysis')
            _dispatch_retry_task(task_type, task_id, next_params)
            logger.info(f"DISPATCHED task {task_id} via Celery")
    except Exception as e:
        logger.error(f"Failed to dispatch task {task_id}: {e}")
    finally:
        conn.close()


def _lookup_task_by_query_id_mysql(user_id: int, query_id: str) -> dict | None:
    """Fallback when Redis mapping expired but MySQL still has the task."""
    import json as _json
    from sources.knowledge.knowledge import get_db_connection
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT task_id, session_id, status, input_params
                   FROM long_tasks
                   WHERE user_id = %s
                     AND create_time >= NOW() - INTERVAL 2 HOUR
                   ORDER BY create_time DESC
                   LIMIT 30""",
                (user_id,),
            )
            rows = cur.fetchall() or []
        for row in rows:
            params = row.get('input_params')
            if isinstance(params, str):
                try:
                    params = _json.loads(params)
                except _json.JSONDecodeError:
                    continue
            if isinstance(params, dict) and params.get('query_id') == query_id:
                status = row.get('status') or 'pending'
                queue_status = 'queued' if status == 'queued' else 'running'
                return {
                    'task_id': row['task_id'],
                    'session_id': row.get('session_id') or '',
                    'status': queue_status,
                }
    except Exception:
        return None
    finally:
        conn.close()
    return None


def _task_owned_by(task_id: str, user_id: int) -> bool:
    """True when *task_id* belongs to *user_id*.

    安全闸口，失败时**保守拒绝**（返回 False → 调用方给 404）：DB 故障时
    宁可拒绝合法请求，也不能放行越权读取。
    """
    if not task_id:
        return False
    try:
        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM long_tasks WHERE task_id = %s AND user_id = %s",
                    (task_id, user_id),
                )
                return cur.fetchone() is not None
        finally:
            conn.close()
    except Exception as e:
        return False


def _owned_task_ids(task_ids: list[str], user_id: int) -> set[str]:
    """*task_ids* 中属于 *user_id* 的那些。

    比逐个调 _task_owned_by 少 N-1 次连接开关——批量轮询是热路径
    （客户端 1.5s 一轮、上限 20 个 id）。
    失败时保守返回空集：宁可不返回状态，也不放行越权读取。
    """
    ids = [t for t in task_ids if t]
    if not ids:
        return set()
    try:
        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                placeholders = ", ".join(["%s"] * len(ids))
                cur.execute(
                    "SELECT task_id FROM long_tasks "
                    "WHERE task_id IN (" + placeholders + ") AND user_id = %s",
                    (*ids, user_id),
                )
                rows = cur.fetchall() or []
            # 游标是 DictCursor（见 _dispatch_from_mysql 的 row.get 用法）
            return {str(r.get("task_id")) for r in rows if r.get("task_id")}
        finally:
            conn.close()
    except Exception:
        return set()


_TASK_TYPE_TO_SCENARIO = {
    "prosecution_analysis": "prosecution",
    "family_analysis": "families",
    "china_examination": "china_prosecution",
    "patent_analysis": "search",
}


def _task_type_to_scenario(task_type: str) -> str:
    """Map a stored long_tasks.task_type back to the celery scenario name."""
    return _TASK_TYPE_TO_SCENARIO.get(task_type, "search")


def _dispatch_retry_task(task_type: str, task_id: str, params: dict) -> None:
    """Dispatch a retried task to the matching Celery executor (需求 4)."""
    from celery_worker import (
        execute_china_examination_analysis,
        execute_family_analysis,
        execute_patent_analysis,
        execute_prosecution_analysis,
    )
    if task_type == "prosecution_analysis":
        execute_prosecution_analysis.delay(task_id=task_id, params=params)
    elif task_type == "family_analysis":
        execute_family_analysis.delay(task_id=task_id, params=params)
    elif task_type == "china_examination":
        execute_china_examination_analysis.delay(task_id=task_id, params=params)
    else:
        execute_patent_analysis.delay(task_id=task_id, params=params)


_UNRESOLVABLE_ERROR_ZH = "该专利号无法解析为可执行的分析任务。"
_UNRESOLVABLE_ERROR_EN = "This patent number cannot be resolved to a runnable analysis task."


def _unresolvable_response_detail(lang: str = "zh") -> dict:
    """Structured 422 body (detail) for a pre-check refusal (spec §5.3).

    Ruling ①: pre-check refusals produce an error-with-guidance, never a task
    row / Celery dispatch / analytics event.  ``guidance`` carries the shared
    ``failure_guidance`` template so submit/retry/detail present the same
    next-step as the chat main chain.
    """
    lang = lang if lang in ("zh", "en") else "zh"
    return {
        "error": _UNRESOLVABLE_ERROR_ZH if lang != "en" else _UNRESOLVABLE_ERROR_EN,
        "code": ERR_UNRESOLVABLE_ID,
        "guidance": failure_guidance("family", ERR_UNRESOLVABLE_ID, lang=lang),
    }


_DETERMINISTIC_ERROR_ZH = "该任务上次失败的根因无法通过重试解决。"
_DETERMINISTIC_ERROR_EN = ("The last failure of this task cannot be fixed "
                           "by retrying.")


def _deterministic_failure_detail(lang: str = "zh") -> dict:
    """Structured 422 body for refusing to re-queue a deterministically
    failed task (需求#17) — mirror shape of ``_unresolvable_response_detail``.
    """
    lang = lang if lang in ("zh", "en") else "zh"
    guidance = (
        "请核对输入的专利号或文件后重新发起分析；或换个方式描述需求。"
        if lang != "en" else
        "Please verify the patent number or file and resubmit the analysis, "
        "or restate your request."
    )
    return {
        "error": (_DETERMINISTIC_ERROR_ZH if lang != "en"
                  else _DETERMINISTIC_ERROR_EN),
        "code": ERR_OTHER,
        "guidance": guidance,
    }


def _verdict_of(*args, **kwargs):
    # Thin indirection so the pre-check gates read naturally and translator
    # faults decompose to "unknown" (fail-open per spec §7) at each call site.
    try:
        return verdict_of(*args, **kwargs)
    except Exception:
        return None


def _is_unresolvable_id(number: str) -> bool:
    """Whether *number* is *deterministically* unusable by any deep-analysis
    executor (PCT international number, unsupported/foreign prefix).  None /
    US grant-or-publication are fine; a translator fault is ``None`` (pass)."""
    return _verdict_of(number) == "unresolvable"


def _normalize_submit_patent_id(raw: str, scenario: str) -> str:
    """Validate and normalize a patent ID for the submit endpoint.

    - prosecution: must be an 8-digit US application number (prefix stripped)
    - family: any publication/application number with >= 6 alphanumerics;
      requires at least one digit in addition to the >= 6 alphanumerics
    """
    if scenario not in ("prosecution", "family"):
        raise ValueError(f"Unknown scenario: {scenario}")
    value = (raw or "").strip()
    if not value:
        raise ValueError("patent_id is required")
    if scenario == "prosecution":
        # Kind codes (B2/A1) carry digits — naive all-digits extraction
        # would pass "US9019058B2" through as the 8-digit "90190582".
        # A grant/publication number (any kind code) is never an
        # application number; reject it explicitly instead.
        if kind_code_of(value):
            raise ValueError(
                "Prosecution analysis requires an 8-digit US application number"
            )
        digits = extract_us_patent_digits(value)
        if len(digits) != 8:
            raise ValueError(
                "Prosecution analysis requires an 8-digit US application number"
            )
        return digits
    clean = re.sub(r"[^A-Za-z0-9]", "", value)
    if len(clean) < 6:
        raise ValueError("patent_id too short for family analysis")
    if not any(ch.isdigit() for ch in clean):
        raise ValueError("patent_id must contain digits")
    return value


def register_long_task_routes(logger, config):
    """Register long task polling and download routes with dependency injection."""
    router = APIRouter()

    @router.get("/long_task/recover")
    async def recover_long_task(
        query_id: str = Query(..., min_length=1),
        http_request: Request = None,
    ):
        """Recover a long task after SSE disconnect using the client query_id."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])
        hit = lookup_query_task(str(user_id), query_id)
        if not hit:
            hit = _lookup_task_by_query_id_mysql(user_id, query_id)
        if hit:
            return {"success": True, "found": True, **hit}
        return {"success": True, "found": False}

    @router.post("/long_task/submit")
    async def submit_long_task(http_request: Request):
        """Directly submit a prosecution/family long task (results-page button).

        Body: {scenario: "prosecution"|"family", patent_id, query?, lang?,
               session_id?}
        Reuses the existing queue + Celery dispatch; polling/download go
        through the existing status/report endpoints.
        """
        import json as _json
        import uuid as _uuid

        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user["uid"])

        try:
            body = await http_request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        scenario = body.get("scenario", "")
        try:
            patent_id = _normalize_submit_patent_id(
                body.get("patent_id", ""), scenario,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        patent_source = body.get("patent_source", "uspto")
        if not isinstance(patent_source, str) or not patent_source.strip():
            raise HTTPException(
                status_code=400, detail="patent_source must be a non-empty string",
            )
        patent_source = patent_source.strip()
        if len(patent_source) > 20:
            raise HTTPException(
                status_code=400, detail="patent_source must be at most 20 characters",
            )

        query = str(body.get("query") or "").strip() or (
            f"分析专利 {patent_id} 的审查历史" if scenario == "prosecution"
            else f"分析 {patent_id} 及其全球同族的审查差异"
        )
        lang = body.get("lang") if body.get("lang") in ("zh", "en") else "zh"

        # Resolvability pre-check (spec §5.3 入口2, ruling ①).  A family submit
        # whose id can never start an analysis (PCT international number,
        # unsupported/foreign prefix) refuses with guidance *before* any DB row /
        # Celery dispatch.  Prosecution is already constrained to an 8-digit US
        # application by ``_normalize_submit_patent_id`` above; a WO US-grant is
        # a valid family seed and passes.
        if scenario == "family" and _is_unresolvable_id(patent_id):
            raise HTTPException(
                status_code=422,
                detail=_unresolvable_response_detail(lang),
            )

        from sources.knowledge.knowledge import get_db_connection
        from sources.long_task.user_queue import try_start_user_task

        task_id = f"lt_{_uuid.uuid4().hex[:12]}"
        session_id = str(body.get("session_id") or "").strip()

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                if not session_id:
                    session_id = f"sess_{_uuid.uuid4().hex[:12]}"
                    cur.execute(
                        """INSERT INTO conversations
                           (session_id, user_id, title, messages, long_task_ids)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (session_id, user_id, query[:60],
                         _json.dumps([], ensure_ascii=False),
                         _json.dumps([task_id])),
                    )
                else:
                    cur.execute(
                        """SELECT long_task_ids FROM conversations
                           WHERE session_id = %s AND user_id = %s AND status != 2""",
                        (session_id, user_id),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise HTTPException(status_code=400, detail="Unknown session_id")
                    existing = _json.loads(row["long_task_ids"]) if isinstance(
                        row["long_task_ids"], str
                    ) else (row["long_task_ids"] or [])
                    existing.append(task_id)
                    cur.execute(
                        """UPDATE conversations SET long_task_ids = %s,
                           update_time = NOW() WHERE session_id = %s""",
                        (_json.dumps(existing), session_id),
                    )

                task_type = (
                    "prosecution_analysis" if scenario == "prosecution"
                    else "family_analysis"
                )
                cur.execute(
                    """INSERT INTO long_tasks
                       (task_id, session_id, user_id, task_type, input_params, status)
                       VALUES (%s, %s, %s, %s, %s, 'pending')""",
                    (task_id, session_id, user_id, task_type,
                     _json.dumps({
                         "query": query,
                         "patent_id": patent_id,
                         "patent_source": patent_source,
                         "lang": lang,
                     }, ensure_ascii=False)),
                )
                conn.commit()
        finally:
            conn.close()

        celery_params = {
            "query": query,
            "session_id": session_id,
            "user_id": str(user_id),
            "scenario": "prosecution" if scenario == "prosecution" else "families",
            "patent_id": patent_id,
            "patent_source": patent_source,
            "patent_id_type": (
                "application_number" if scenario == "prosecution" else "unknown"
            ),
            "lang": lang,
        }

        queue_result = try_start_user_task(str(user_id), task_id)
        status = "running"
        if queue_result == "running":
            if scenario == "prosecution":
                from celery_worker import execute_prosecution_analysis
                execute_prosecution_analysis.delay(task_id=task_id, params=celery_params)
            else:
                from celery_worker import execute_family_analysis
                execute_family_analysis.delay(task_id=task_id, params=celery_params)
        else:
            status = "queued"
        # M1: persist a "task created" message in the conversation so the
        # analysis task (and its later outcome) is visible in chat history.
        try:
            from sources.long_task.task_messages import append_task_message
            created_content = (
                f"已提交分析任务（{patent_id}），正在后台分析。\n\n"
                "任务完成后将在此展示结果摘要，可直接追问细节。"
                if queue_result == "running" else
                "分析任务已排队，将在当前任务完成后自动开始。"
            )
            append_task_message(
                task_id, event="created",
                content=created_content, patent_ids=[patent_id])
        except Exception as e:
            logger.warning(
                f"submit_long_task created message failed: {e}")
        logger.info(
            f"submit_long_task — task_id={task_id}, scenario={scenario}, "
            f"patent_id={patent_id}, queue={queue_result}"
        )
        return {
            "success": True,
            "task_id": task_id,
            "session_id": session_id,
            "status": status,
        }

    @router.post("/long_task/{task_id}/retry")
    async def retry_long_task(task_id: str, http_request: Request):
        """Re-submit a failed long task from its stored input params.

        需求 4 (失败可操作化): the frontend failure card offers one-click
        retry.  A NEW task_id is created (the old task record and its
        failure state stay intact), then queued/dispatched exactly like a
        fresh submission.
        """
        import json as _json
        import uuid as _uuid

        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user["uid"])

        from sources.knowledge.knowledge import get_db_connection
        from sources.long_task.user_queue import try_start_user_task

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT session_id, scene_id, task_type, input_params
                       FROM long_tasks
                       WHERE task_id = %s AND user_id = %s AND status != 2""",
                    (task_id, user_id))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Task not found")
                session_id = row.get("session_id") or ""
                scene_id = row.get("scene_id")
                task_type = row.get("task_type") or "patent_analysis"
                stored = row.get("input_params")
                input_params = _json.loads(stored) if isinstance(stored, str) else (stored or {})

                # Resolvability pre-check on retry (spec §5.3 入口3, ruling ①).
                # Retrying a family deep-analysis whose seed id can never resolve
                # (again a deterministically-unresolvable number) is refused here,
                # ABOVE the new-task INSERT — so no replacement pending row is
                # created and the stored original failed task is left untouched.
                if task_type == "family_analysis":
                    fam_target = (input_params or {}).get("patent_id")
                    if fam_target and _is_unresolvable_id(str(fam_target)):
                        _lang = (input_params or {}).get("lang") or "zh"
                        raise HTTPException(
                            status_code=422,
                            detail=_unresolvable_response_detail(_lang),
                        )

                # Deterministic-failure pre-check on retry (需求#17): a task
                # that failed for a *structural* reason (unresolvable id,
                # parameter/parse error) can never succeed by re-queuing —
                # refuse above the new-task INSERT just like the family gate.
                # The Redis status record carries the last error_message;
                # absent/transient errors keep the historical retry path.
                # Redis unavailability degrades to the historical path too —
                # our own infra must never block a user retry.
                old_error = ""
                try:
                    old_status = get_task_status(task_id)
                    old_error = str((old_status or {}).get("error_message") or "")
                except Exception:
                    old_error = ""
                if old_error and not is_retryable_failure(
                        old_error, task_type):
                    _lang = (input_params or {}).get("lang") or "zh"
                    raise HTTPException(
                        status_code=422,
                        detail=_deterministic_failure_detail(_lang),
                    )

                new_task_id = f"lt_{_uuid.uuid4().hex[:12]}"
                cur.execute(
                    """INSERT INTO long_tasks
                       (task_id, session_id, user_id, scene_id, task_type,
                        input_params, status)
                       VALUES (%s, %s, %s, %s, %s, %s, 'pending')""",
                    (new_task_id, session_id, user_id, scene_id, task_type,
                     _json.dumps(input_params, ensure_ascii=False)))
                conn.commit()
        finally:
            conn.close()

        celery_params = {
            "query": input_params.get("query", ""),
            "session_id": session_id,
            "scene_id": scene_id,
            "conversation_history": input_params.get("conversation_history", []),
            "user_id": str(user_id),
            "scenario": _task_type_to_scenario(task_type),
            "lang": input_params.get("lang", "zh"),
        }
        if input_params.get("patent_id"):
            celery_params["patent_id"] = input_params["patent_id"]
            celery_params["patent_id_type"] = input_params.get("patent_id_type", "unknown")
        if input_params.get("patent_ids"):
            celery_params["patent_ids"] = input_params["patent_ids"]
        celery_params["patent_source"] = input_params.get("patent_source", "auto")
        if input_params.get("patent_texts"):
            celery_params["patent_texts"] = input_params["patent_texts"]

        queue_result = try_start_user_task(str(user_id), new_task_id)
        status = "running" if queue_result == "running" else "queued"
        if queue_result == "running":
            _dispatch_retry_task(task_type, new_task_id, celery_params)
        logger.info(
            f"retry_long_task — old={task_id}, new={new_task_id}, "
            f"type={task_type}, queue={queue_result}"
        )
        return {
            "success": True,
            "task_id": new_task_id,
            "session_id": session_id,
            "status": status,
        }

    @router.get("/long_task/user_queue")
    async def user_queue_status(http_request: Request):
        """Return the current user's long-task queue (running + queued count)."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        from sources.long_task.user_queue import get_user_queue_status
        status = get_user_queue_status(str(user['uid']))
        return {"success": True, **status}

    @router.get("/long_task/{task_id}/status")
    async def task_status(task_id: str, http_request: Request):
        """Poll the current status of a long-running task."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])
        if not _task_owned_by(task_id, user_id):
            raise HTTPException(status_code=404, detail="Task not found")
        logger.info(f"Status poll for task: {task_id}")
        status = get_task_status(task_id)
        return {"success": True, **status}

    @router.post("/long_task/batch_status")
    async def batch_task_status(http_request: Request):
        """Poll status for multiple long-running tasks in one request."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])
        body = await http_request.json()
        task_ids = body.get("task_ids", []) or []
        if not isinstance(task_ids, list):
            task_ids = []
        # Cap at 20 to prevent abuse
        task_ids = task_ids[:20]
        # 一次查询取回全部归属，避免逐 id 开关连接（热路径，1.5s 一轮）
        owned = _owned_task_ids(task_ids, user_id)
        statuses = {}
        for tid in task_ids:
            # 非本人的任务直接略过（不报错、不返回 unknown）——
            # 报错或返回 unknown 都会变成 task_id 存在性的探针
            if tid not in owned:
                continue
            statuses[tid] = get_task_status(tid)
        return {"success": True, "statuses": statuses}

    @router.post("/long_task/{task_id}/pause")
    async def pause_task(task_id: str, http_request: Request):
        """Pause a running long task at its next checkpoint."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        if not _task_owned_by(task_id, int(user['uid'])):
            raise HTTPException(status_code=404, detail="Task not found")
        from sources.long_task.status_manager import (
            get_task_status, request_task_pause, is_task_paused,
        )
        status = get_task_status(task_id)
        if status.get('status') not in ('running', 'analyzing', 'searching', 'generating'):
            return {"success": False, "error": "Task is not running"}
        if is_task_paused(task_id):
            return {"success": False, "error": "Task is already paused"}
        request_task_pause(task_id)
        logger.info(f"Pause requested for task: {task_id}")
        return {"success": True, "message": "Pause requested"}

    @router.post("/long_task/{task_id}/resume")
    async def resume_task(task_id: str, http_request: Request):
        """Resume a paused task by re-queuing it. Dispatches immediately if
        no other task is running."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        if not _task_owned_by(task_id, int(user['uid'])):
            raise HTTPException(status_code=404, detail="Task not found")
        from sources.long_task.status_manager import (
            get_task_status, clear_task_pause,
        )
        from sources.long_task.user_queue import requeue_paused_task
        status = get_task_status(task_id)
        if status.get('status') != 'paused':
            return {"success": False, "error": "Task is not paused"}
        clear_task_pause(task_id)
        user_id = str(user.get('uid', ''))
        result = requeue_paused_task(user_id, task_id)
        logger.info(f"Resume requested for task: {task_id}, result={result}")
        if result == 'running':
            # No other task running — dispatch this one immediately
            _dispatch_from_mysql(user_id, task_id, logger)
        return {"success": True, "message": "Task re-queued"}

    @router.post("/long_task/{task_id}/stop")
    async def stop_task(task_id: str, http_request: Request):
        """Permanently stop and discard a long task."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        if not _task_owned_by(task_id, int(user['uid'])):
            raise HTTPException(status_code=404, detail="Task not found")
        from sources.long_task.status_manager import (
            get_task_status, request_task_stop, is_task_stopped,
        )
        status = get_task_status(task_id)
        if status.get('status') in ('completed', 'failed', 'cancelled'):
            return {"success": False, "error": "Task already in terminal state"}
        if is_task_stopped(task_id):
            return {"success": False, "error": "Stop already requested"}
        request_task_stop(task_id)
        logger.info(f"Stop requested for task: {task_id}")
        return {"success": True, "message": "Stop requested"}

    @router.get("/long_task/{task_id}/report")
    async def download_report(task_id: str, format: str = Query(..., pattern="^(pdf|docx)$"), http_request: Request = None):
        """Download a completed report file for a task."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])
        if not _task_owned_by(task_id, user_id):
            # 404 而非 403 —— 不暴露「存在但不属于你」
            raise HTTPException(status_code=404, detail="Report not found")
        logger.info(f"Report download for task: {task_id}, format: {format}")
        storage = create_storage(get_storage_config())
        filename = f"report.{format}"
        try:
            content = await storage.get(task_id, filename)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")
        except Exception as e:
            # Primary storage failed — try local fallback
            logger.warning(
                f"Primary storage get failed for {task_id}/{filename}: {e}, "
                f"trying local fallback"
            )
            try:
                local = LocalReportStorage()
                content = await local.get(task_id, filename)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Report not found")
            except Exception as e2:
                logger.error(f"Local fallback also failed: {e2}")
                raise HTTPException(status_code=500, detail="Failed to retrieve report")

        media_type = "application/pdf" if format == "pdf" else \
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="patent_analysis_{task_id}.{format}"'
            },
        )

    return router
