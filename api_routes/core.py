#!/usr/bin/env python3

from fastapi import APIRouter, Request, Form, UploadFile, File as FastAPIFile
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
import uuid
import json
import asyncio
from functools import lru_cache
from datetime import datetime, timedelta

from sources.schemas import QueryResponse
from sources.logger import Logger
from api_routes.models import QueryRequest, QuestionRequest
from sources.knowledge.knowledge import get_knowledge_tool
from sources.user.passport import verify_firebase_token, check_and_increase_usage
from sources.callback.sse_callback import SSECallbackHandler

router = APIRouter()

# Agent pool for reusing agent instances
_agent_pool = {}
_agent_pool_lock = asyncio.Lock()
AGENT_POOL_MAX_SIZE = 3  # reduced from 10 for 2C2G memory budget
AGENT_POOL_MAX_IDLE_TIME = 300  # 5 minutes

# Token cache with TTL
_token_cache = {}
_token_cache_lock = asyncio.Lock()
TOKEN_CACHE_TTL = 86400  # 24 hours

async def get_cached_token_validation(auth_header: str):
    """Cache token validation results to reduce Firebase API calls"""
    async with _token_cache_lock:
        if auth_header in _token_cache:
            cached_data, expiry = _token_cache[auth_header]
            if datetime.now() < expiry:
                return cached_data
            else:
                del _token_cache[auth_header]

    # Validate token if not cached
    user = await asyncio.to_thread(verify_firebase_token, auth_header)

    async with _token_cache_lock:
        _token_cache[auth_header] = (user, datetime.now() + timedelta(seconds=TOKEN_CACHE_TTL))
        # Cleanup old entries
        current_time = datetime.now()
        expired_keys = [k for k, (_, exp) in _token_cache.items() if current_time >= exp]
        for k in expired_keys:
            del _token_cache[k]

    return user

async def get_or_create_agent(create_agent_func, app_logger):
    """Get an agent from pool or create new one"""
    async with _agent_pool_lock:
        current_time = datetime.now()

        # Clean up expired agents
        expired_keys = [k for k, (_, last_used) in _agent_pool.items()
                       if (current_time - last_used).total_seconds() > AGENT_POOL_MAX_IDLE_TIME]
        for k in expired_keys:
            del _agent_pool[k]
            app_logger.info(f"Removed expired agent from pool: {k}")

        # Try to reuse an existing agent
        for agent_id, (agent, _) in _agent_pool.items():
            _agent_pool[agent_id] = (agent, current_time)
            app_logger.info(f"Reusing agent from pool: {agent_id}")
            return agent

        # Create new agent if pool is not full
        if len(_agent_pool) < AGENT_POOL_MAX_SIZE:
            agent = await create_agent_func()
            agent_id = str(uuid.uuid4())
            _agent_pool[agent_id] = (agent, current_time)
            app_logger.info(f"Created new agent and added to pool: {agent_id}")
            return agent

        # Pool is full, create temporary agent (not cached)
        app_logger.warning("Agent pool is full, creating temporary agent")
        return await create_agent_func()

async def check_usage_async(user_id: str) -> bool:
    """Async wrapper for check_and_increase_usage"""
    return await asyncio.to_thread(check_and_increase_usage, user_id)

def _detect_patent_source(
    scene_id=None,
    conv_history: list = None,
    query: str = "",
    app_logger=None,
) -> str:
    """Detect patent source (uspto/cnipa) from conversation context and scene tools.

    Detection order (text keywords take priority — scene tools only provide fallback):
      1. Conversation / query text keywords (most reliable — what user is searching)
      2. Scene tools URL heuristics (fallback)
      3. Default: let the pipeline auto-detect via patent ID format
    """
    conv_history = conv_history or []

    # ── Build combined text ──
    combined = query + " " + " ".join(
        m.get("content", "") for m in (conv_history or [])
        if isinstance(m, dict)
    )
    combined_lower = combined.lower()

    # ── 1. Text keywords (primary — user intent) ──
    uspto_keywords = ["uspto", "美国专利", "美国专利商标局", "united states patent",
                      "us patent", "us application"]
    cnipa_keywords = ["cnipa", "中国专利", "中国国家知识产权", "国家知识产权局",
                      "chinese patent", "china patent",
                      "zldsj"]
    if any(kw in combined_lower for kw in uspto_keywords):
        source = "uspto"
        if app_logger:
            app_logger.info(f"patent_source: text_keywords → uspto")
        return source
    if any(kw in combined_lower for kw in cnipa_keywords):
        source = "cnipa"
        if app_logger:
            app_logger.info(f"patent_source: text_keywords → cnipa")
        return source

    # ── 2. Scene tools (fallback) ──
    if scene_id:
        try:
            from sources.long_task.scene_tools import get_scene_knowledge_tools
            candidates = get_scene_knowledge_tools(scene_id)
            urls = [
                c.get("tool_url", "") for c in candidates
                if c.get("tool_url")
            ]
            url_text = " ".join(urls).lower()
            if "uspto" in url_text:
                source = "uspto"
            elif "zldsj" in url_text or "cnipa" in url_text:
                source = "cnipa"
            else:
                source = "auto"
            if app_logger:
                app_logger.info(
                    f"patent_source: scene_id={scene_id}, "
                    f"tool_urls={urls[:3]}, → {source}"
                )
            return source
        except Exception:
            pass

    # ── 3. Default ──
    if app_logger:
        app_logger.info(f"patent_source: no signal found → auto")
    return "auto"


async def _classify_long_task_async(
    query: str,
    conv_history: list,
    scene_id=None,
    app_logger=None,
) -> dict:
    """Use LLM to classify scenario, extract patent IDs, and detect source.

    Returns dict with: scenario, patent_ids, patent_source, reasoning
    """
    from sources.llm_provider import Provider
    from sources.long_task.config import get_long_task_config

    ltc = get_long_task_config()
    flash = Provider(
        provider_name=ltc['provider_family'],
        model='deepseek-chat' if ltc['provider_family'] == 'deepseek' else 'MiniMax-M2.7-highspeed',
        server_address='', is_local=False,
    )

    # Build conversation summary for the LLM prompt.
    # Patent search results can be extremely long (multiple patents with full
    # metadata, each easily 5k+ chars). Use a high limit to avoid truncating IDs.
    MAX_MSG_CHARS = 200000
    conv_summary = []
    for msg in (conv_history or []):
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = str(msg.get('content', ''))
            conv_summary.append(
                f"[{role}]: {content[:MAX_MSG_CHARS]}"
                f"{'...(truncated)' if len(content) > MAX_MSG_CHARS else ''}"
            )

    system_prompt = (
        "你是一个专利分析任务分类器。分析用户查询 + 对话历史，判断场景、提取专利ID、识别来源。\n\n"
        "## 三种场景\n"
        "1. direct_ids — 用户当前提问本身包含专利申请号，是独立问题"
        "（不依赖历史对话即可理解）。\n"
        "   例：「分析专利申请 17429113, 18012525, 18331482」\n"
        "   例：「帮我查一下202310123456.7这个专利」\n"
        "2. conversation_refs — 用户的问题是针对历史对话中出现的专利结果的追问。"
        "当前提问如果不看历史对话就无法确定要分析哪些专利。\n"
        "   例：（历史对话返回了3个专利）用户：「从这3条中筛选出橡胶垫圈相关专利」\n"
        "   例：（历史对话返回了专利列表）用户：「上述专利里哪些已授权」\n"
        "   例：（历史对话返回了专利结果）用户：「帮我分析第一个专利」\n\n"
        "## 专利ID格式\n"
        "- USPTO（美国）: 纯8位数字，如 17429113, 18331482\n"
        "- CNIPA（中国）: 20XX + 8~9位数字，可选 .校验位，如 202310123456.7\n\n"
        "## 专利来源判断\n"
        "- 如果对话提到 USPTO、美国专利、美国专利商标局 → uspto\n"
        "- 如果对话提到 CNIPA、中国专利、国家知识产权局 → cnipa\n"
        "- 纯8位数字ID → uspto\n"
        "- 20XX开头的长数字ID → cnipa\n"
        "- 不确定 → unknown\n\n"
        "## 输出JSON\n"
        '{"scenario": "direct_ids"|"conversation_refs", '
        '"patent_ids": ["id1","id2"], '
        '"patent_source": "uspto"|"cnipa"|"unknown", '
        '"reasoning": "简要说明"}'
    )

    user_text = (
        f"用户当前提问：{query}\n\n"
        f"对话历史：\n" + "\n".join(conv_summary) if conv_summary else "(无历史对话)"
    )

    try:
        result = await flash.complete_json(system_prompt, user_text)
    except Exception as e:
        if app_logger:
            app_logger.warning(f"LLM scenario classification failed: {e}, falling back to regex")
        return {
            "scenario": "unknown",
            "patent_ids": [],
            "patent_source": "auto",
            "reasoning": f"LLM error: {e}",
        }

    if not result or not isinstance(result, dict):
        return {
            "scenario": "unknown",
            "patent_ids": [],
            "patent_source": "auto",
            "reasoning": "LLM returned empty/invalid response",
        }

    return {
        "scenario": result.get("scenario", "unknown"),
        "patent_ids": result.get("patent_ids", []) or [],
        "patent_source": result.get("patent_source", "auto"),
        "reasoning": result.get("reasoning", ""),
    }


def _prepare_long_task_inputs(
    query: str,
    conv_history: list,
    scene_id=None,
    patent_file_refs: list = None,
    app_logger=None,
    llm_result: dict = None,
) -> dict:
    """Prepare patent analysis inputs, optionally enriched by LLM classification.

    Three scenarios:
      1. "conversation_refs" — query is a FOLLOW-UP about previous results.
         IDs come from conversation history (LLM extracts them).
      2. "direct_ids" — query ITSELF contains patent application IDs.
         Self-contained question. IDs come from query.
      3. "file_upload" — user uploaded patent specification files.

    When ``llm_result`` is provided (from ``_classify_long_task_async``),
    scenario and patent_source from the LLM are used directly.
    Otherwise falls back to regex extraction + keyword detection.

    Returns dict with keys:
      scenario, patent_ids, patent_source, patent_texts
    """
    import re as _patent_id_re

    conv_history = conv_history or []
    patent_file_refs = patent_file_refs or []

    # ── Scenario 3: file upload (unambiguous, no LLM needed) ──
    if patent_file_refs:
        patent_ids = [
            ref.get("filename", "").rsplit(".", 1)[0]
            for ref in patent_file_refs
        ]
        patent_source = _detect_patent_source(
            scene_id=scene_id,
            conv_history=conv_history,
            query=query,
            app_logger=app_logger,
        )
        if app_logger:
            app_logger.info(
                f"Long task scenario=file_upload — "
                f"files={len(patent_file_refs)}, ids={patent_ids[:5]}, "
                f"patent_source={patent_source}"
            )
        return {
            "scenario": "file_upload",
            "patent_ids": patent_ids,
            "patent_source": patent_source,
            "patent_texts": None,
        }

    # ── Regex sweep on full text (always run — catches IDs LLM might miss) ──
    text_sources = [query or ""]
    for msg in conv_history:
        if isinstance(msg, dict):
            text_sources.append(msg.get("content", ""))
    combined_text = "\n".join(text_sources)
    uspto_matches = _patent_id_re.findall(r'\b(\d{8})\b', combined_text)
    slash_matches = _patent_id_re.findall(r'\b(\d{2})/(\d{6})\b', combined_text)
    uspto_matches += [a + b for a, b in slash_matches]
    cnipa_matches = _patent_id_re.findall(r'\b(20[12]\d{8,9}(?:\.\d)?)\b', combined_text)
    regex_ids = list(dict.fromkeys(uspto_matches + cnipa_matches))

    # ── Use LLM result if available ──
    if llm_result and llm_result.get("scenario") != "unknown":
        scenario = llm_result["scenario"]
        patent_source = llm_result.get("patent_source", "auto")
        llm_ids = llm_result.get("patent_ids", []) or []

        # Merge LLM IDs + regex IDs (LLM may miss IDs due to truncation)
        patent_ids = list(dict.fromkeys(llm_ids + regex_ids))

        if scenario == "conversation_refs":
            patent_texts = {}
            for msg in conv_history:
                if msg.get('role') == 'assistant' and msg.get('patent_data'):
                    for p in msg['patent_data']:
                        if isinstance(p, dict) and 'patent_id' in p:
                            pid = str(p['patent_id'])
                            if pid not in patent_ids:
                                patent_ids.append(pid)
                            st = p.get('spec_text', '')
                            if st and len(st) > 100:
                                patent_texts[pid] = st
            patent_texts = patent_texts if patent_texts else None
        else:
            patent_texts = None
    else:
        # ── Fallback: regex + keyword detection (no LLM available) ──
        patent_ids = list(dict.fromkeys(regex_ids))
        patent_texts = {}
        for msg in conv_history:
            if msg.get('role') == 'assistant' and msg.get('patent_data'):
                for p in msg['patent_data']:
                    if isinstance(p, dict) and 'patent_id' in p:
                        pid = p['patent_id']
                        if pid not in patent_ids:
                            patent_ids.append(pid)
                        st = p.get('spec_text', '')
                        if st and len(st) > 100:
                            patent_texts[pid] = st

        query_uspto = _patent_id_re.findall(r'\b(\d{8})\b', query or "")
        query_slash = _patent_id_re.findall(r'\b(\d{2})/(\d{6})\b', query or "")
        query_uspto += [a + b for a, b in query_slash]
        query_cnipa = _patent_id_re.findall(r'\b(20[12]\d{8,9}(?:\.\d)?)\b', query or "")
        query_has_ids = bool(query_uspto or query_cnipa)

        followup_keywords = ["这", "上述", "前面", "以上", "从中", "其中", "筛选", "过滤",
                            "挑出", "选出", "哪些", "哪个", "这几个", "那几个"]
        query_is_followup = any(kw in (query or "") for kw in followup_keywords)

        scenario = "direct_ids" if (query_has_ids and not query_is_followup) else "conversation_refs"
        patent_source = _detect_patent_source(
            scene_id=scene_id, conv_history=conv_history, query=query, app_logger=app_logger,
        )
        patent_texts = patent_texts if patent_texts else None

    if app_logger:
        app_logger.info(
            f"Long task scenario={scenario} — "
            f"patent_ids_count={len(patent_ids)}, "
            f"patent_source={patent_source}, "
            f"patent_ids={patent_ids[:10]}{'...' if len(patent_ids) > 10 else ''}"
        )

    return {
        "scenario": scenario,
        "patent_ids": patent_ids,
        "patent_source": patent_source,
        "patent_texts": patent_texts,
    }


def register_core_routes(app_logger, interaction_ref, query_resp_history_ref, config_ref, is_generating_flag, think_wrapper_func, create_agent_func):
    """注册核心路由并传递所需的依赖"""

    @router.get("/latest_answer")
    async def get_latest_answer():
        app_logger.info("Latest answer endpoint called")
        if interaction_ref.current_agent is None:
            return JSONResponse(status_code=404, content={"error": "No agent available"})

        uid = str(uuid.uuid4())
        if not any(q["answer"] == interaction_ref.current_agent.last_answer for q in query_resp_history_ref):
            query_resp = {
                "done": "false",
                "answer": interaction_ref.current_agent.last_answer,
                "reasoning": interaction_ref.current_agent.last_reasoning,
                "agent_name": interaction_ref.current_agent.agent_name if interaction_ref.current_agent else "None",
                "success": interaction_ref.current_agent.success,
                "blocks": {f'{i}': block.jsonify() for i, block in enumerate(interaction_ref.get_last_blocks_result())} if interaction_ref.current_agent else {},
                "status": interaction_ref.current_agent.get_status_message if interaction_ref.current_agent else "No status available",
                "uid": uid
            }
            interaction_ref.current_agent.last_answer = ""
            interaction_ref.current_agent.last_reasoning = ""
            query_resp_history_ref.append(query_resp)
            return JSONResponse(status_code=200, content=query_resp)

        if query_resp_history_ref:
            return JSONResponse(status_code=200, content=query_resp_history_ref[-1])
        return JSONResponse(status_code=404, content={"error": "No answer available"})

    @router.post("/query")
    async def process_query(request: QueryRequest, http_request: Request):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)

        user_id = user['uid']

        app_logger.info(f"[user={user_id}] Processing query: {request.query}")

        allowed = check_and_increase_usage(user_id)
        if not allowed:
            return JSONResponse(status_code=429, content="Daily API usage limit exceeded (100/day)")

        # 如果没有提供 query_id，自动生成一个
        if not request.query_id:
            request.query_id = str(uuid.uuid4())

        query_resp = QueryResponse(
            done="false",
            answer="",
            reasoning="",
            agent_name="Unknown",
            success=False,
            blocks={},
            status="Ready",
            uid=str(uuid.uuid4())
        )

        if is_generating_flag:
            app_logger.warning("Another query is being processed, please wait.")
            return JSONResponse(status_code=429, content=query_resp.jsonify())

        try:
            # is_generating = True  # Uncomment if needed
            # 调用 think_wrapper_func 来处理查询
            success = await think_wrapper_func(user_id, interaction_ref, request.query, request.query_id)

            if not success:
                query_resp.answer = interaction_ref.last_answer
                query_resp.reasoning = interaction_ref.last_reasoning
                return JSONResponse(status_code=400, content=query_resp.jsonify())

            if interaction_ref.current_agent:
                blocks_json = {f'{i}': block.jsonify() for i, block in enumerate(interaction_ref.current_agent.get_blocks_result())}
            else:
                app_logger.error("No current agent found")
                blocks_json = {}
                query_resp.answer = "Error: No current agent"
                return JSONResponse(status_code=400, content=query_resp.jsonify())

            app_logger.info(f"Answer: {interaction_ref.last_answer}")
            app_logger.info(f"Blocks: {blocks_json}")
            query_resp.done = "true"
            query_resp.answer = interaction_ref.last_answer
            query_resp.reasoning = interaction_ref.last_reasoning
            query_resp.agent_name = interaction_ref.current_agent.agent_name
            query_resp.success = interaction_ref.last_success
            query_resp.blocks = blocks_json

            query_resp_dict = {
                "done": query_resp.done,
                "answer": query_resp.answer,
                "agent_name": query_resp.agent_name,
                "success": query_resp.success,
                "blocks": query_resp.blocks,
                "status": query_resp.status,
                "uid": query_resp.uid
            }
            query_resp_history_ref.append(query_resp_dict)

            app_logger.info("Query processed successfully")
            return JSONResponse(status_code=200, content=query_resp.jsonify())

        except Exception as e:
            app_logger.error(f"An error occurred: {str(e)}")
            # sys.exit(1)  # 不应该在路由中退出应用
            return JSONResponse(status_code=500, content={"error": "Internal server error"})
        finally:
            app_logger.info("Processing finished")
            if config_ref.getboolean('MAIN', 'save_session'):
                interaction_ref.save_session()

    @router.get("/screenshot")
    async def get_screenshot():
        app_logger.info("Screenshot endpoint called")
        screenshot_path = ".screenshots/updated_screen.png"
        if os.path.exists(screenshot_path):
            return FileResponse(screenshot_path)
        app_logger.error("No screenshot available")
        return JSONResponse(
            status_code=404,
            content={"error": "No screenshot available"}
        )

    @router.post("/find_knowledge_tool")
    async def find_knowledge_tool(request: QuestionRequest, http_request: Request):
        """根据用户问题查找最相关的知识及其对应的工具"""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)

        user_id = user['uid']

        app_logger.info(f"Finding knowledge tool for user: {user_id} with question: {request.question}")

        try:
            # 参数校验

            if not request.question:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "question is required"
                    }
                )

            # 调用knowledge.py中的方法获取知识项和工具信息
            knowledge_item, tool_info = get_knowledge_tool(
                user_id,
                request.question,
                request.top_k,
                0
            )

            if not knowledge_item:
                app_logger.info("No matching knowledge found above similarity threshold")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "message": "No matching knowledge found above similarity threshold"
                    }
                )

            # 构建返回的知识记录对象
            knowledge_response = {
                "userId": knowledge_item.user_id,
                "question": knowledge_item.question,
                "description": knowledge_item.description,
                "answer": knowledge_item.answer,
                "public": knowledge_item.public,
                "modelName": knowledge_item.model_name,
                "toolId": knowledge_item.tool_id,
                "params": knowledge_item.params
            }

            response_data = {
                "success": True,
                "message": "Knowledge and tool found successfully",
                "knowledge": knowledge_response
            }

            if tool_info:
                # 检查tool_info的push字段
                if tool_info.push == 2:
                    # 如果push为2，返回空对象
                    tool_response = {}
                else:
                    # 如果push不为2，保持原有逻辑
                    tool_response = {
                        "id": tool_info.id,
                        "title": tool_info.title,
                        "description": tool_info.description,
                        "url": tool_info.url,
                        "params": tool_info.params
                    }

                tool_response["push"] = tool_info.push
                response_data["tool"] = tool_response

            app_logger.info(f"Successfully found knowledge and tool for user: {user_id}")
            return JSONResponse(
                status_code=200,
                content=response_data
            )

        except Exception as e:
            app_logger.error(f"Error in find_knowledge_tool: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Internal server error: {str(e)}"
                }
            )

    async def _handle_file_upload_query(http_request: Request):
        """Handle multipart file-upload query: extract text, dispatch long task."""
        import traceback

        try:
            from sources.long_task.text_extractor import extract_text_from_binary
        except Exception as e:
            app_logger.error(f"File upload: failed to import text_extractor: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Server configuration error: text_extractor not available"},
            )

        try:
            form = await http_request.form()

            query = (form.get("query") or "").strip()
            query_id = form.get("query_id") or f"q_{uuid.uuid4().hex[:8]}"
            conv_json = form.get("conversation_history") or "[]"
            try:
                conversation_history = json.loads(conv_json) if isinstance(conv_json, str) else []
            except json.JSONDecodeError:
                conversation_history = []

            if not query:
                return JSONResponse(status_code=400, content={"error": "query is required"})

            # Auth + usage
            auth_header = http_request.headers.get("Authorization")
            try:
                user = await get_cached_token_validation(auth_header)
            except Exception as e:
                app_logger.error(f"Token validation failed: {str(e)}")
                return JSONResponse(status_code=401, content={"error": "Invalid token"})
            user_id = user["uid"]
            allowed = await check_usage_async(user_id)
            if not allowed:
                return JSONResponse(status_code=429, content={"error": "Daily API usage limit exceeded (100/day)"})

            if is_generating_flag:
                return JSONResponse(status_code=429, content={"error": "Another query is being processed"})

            # Collect uploaded patent files (handle multiple files with same field name)
            MAX_FILES = 100
            MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
            patent_files: list[tuple[str, bytes, str]] = []  # (filename, content, content_type)
            for key in form:
                # Use getlist to get ALL files for multi-value fields
                fields = form.getlist(key)
                for field in fields:
                    if hasattr(field, "filename") and field.filename:
                        if len(patent_files) >= MAX_FILES:
                            break
                        content = await field.read()
                        if len(content) > MAX_FILE_BYTES:
                            continue
                        if len(content) < 50:
                            continue
                        patent_files.append((field.filename, content, field.content_type or ""))
                if len(patent_files) >= MAX_FILES:
                    break

            if not patent_files:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No valid patent files uploaded (PDF/DOCX only, min 50 bytes, max 10 MB each)"},
                )

            app_logger.info(
                f"[user={user_id}] File upload: received {len(patent_files)} files, query={query[:80]}"
            )

            # Save files to disk for async processing by Celery worker.
            # Text extraction (esp. OCR) can take minutes — we must not block the HTTP response.
            task_id = f"lt_{uuid.uuid4().hex[:12]}"
            local_user_id = int(user_id)

            upload_dir = os.path.join("/app", ".uploads", task_id)
            os.makedirs(upload_dir, exist_ok=True)
            patent_file_refs: list[dict] = []
            for filename, content, ct in patent_files:
                safe_name = os.path.basename(filename) or f"upload_{len(patent_file_refs)}"
                file_path = os.path.join(upload_dir, safe_name)
                with open(file_path, "wb") as f:
                    f.write(content)
                patent_file_refs.append({
                    "filename": filename,
                    "path": file_path,
                    "content_type": ct,
                })

            # Use filenames (without extension) as patent identifiers
            patent_ids = [f.rsplit(".", 1)[0] for f, _, _ in patent_files]
            app_logger.info(
                f"[user={user_id}] File upload: saved {len(patent_file_refs)} files to {upload_dir}"
            )

            # ── Session reuse for file upload path ──
            from sources.knowledge.knowledge import get_db_connection

            # Detect patent source from query + conversation context
            # (pipeline will further auto-detect from document content)
            inputs = _prepare_long_task_inputs(
                query=query,
                conv_history=conversation_history,
                patent_file_refs=patent_file_refs,
                app_logger=app_logger,
            )
            patent_source = inputs["patent_source"]

            reused_session = False
            existing_session_id = (form.get("session_id") or "").strip()
            conn = get_db_connection()
            try:
                with conn.cursor() as cur:
                    if existing_session_id:
                        cur.execute(
                            """SELECT id, long_task_ids FROM conversations
                               WHERE session_id = %s AND user_id = %s AND status != 2""",
                            (existing_session_id, local_user_id))
                        existing = cur.fetchone()
                        if existing:
                            existing_task_ids = json.loads(existing['long_task_ids']) if isinstance(existing['long_task_ids'], str) else (existing['long_task_ids'] or [])
                            existing_task_ids.append(task_id)
                            cur.execute(
                                """UPDATE conversations
                                   SET messages = %s, long_task_ids = %s, update_time = NOW()
                                   WHERE session_id = %s""",
                                (json.dumps(conversation_history, ensure_ascii=False),
                                 json.dumps(existing_task_ids),
                                 existing_session_id))
                            session_id = existing_session_id
                            reused_session = True
                            app_logger.info(f"File upload: reusing session {session_id}")

                    if not reused_session:
                        session_id = f"sess_{uuid.uuid4().hex[:12]}"
                        cur.execute(
                            """INSERT INTO conversations (session_id, user_id, scene_id, title, messages, long_task_ids)
                               VALUES (%s, %s, %s, %s, %s, %s)""",
                            (session_id, local_user_id, None,
                             f"{query[:60]}",
                             json.dumps(conversation_history, ensure_ascii=False),
                             json.dumps([task_id])))
                        app_logger.info(f"File upload: created new session {session_id}")

                    cur.execute(
                        """INSERT INTO long_tasks
                           (task_id, session_id, user_id, scene_id, task_type, input_params, status)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (task_id, session_id, local_user_id, None,
                         "patent_analysis",
                         json.dumps({
                             "query": query,
                             "patent_ids": patent_ids,
                             "patent_source": patent_source,
                             "patent_file_refs": patent_file_refs,
                         }, ensure_ascii=False),
                         "pending"))
                    conn.commit()
            finally:
                conn.close()

            from celery_worker import execute_patent_analysis
            celery_params = {
                "query": query,
                "patent_ids": patent_ids,
                "patent_source": patent_source,
                "session_id": session_id,
                "scene_id": None,
                "conversation_history": conversation_history,
                "patent_file_refs": patent_file_refs,
            }
            execute_patent_analysis.delay(task_id=task_id, params=celery_params)

            app_logger.info(f"[user={user_id}] File upload: dispatched task={task_id}")

            # Return SSE with long_task_created
            async def generate_sse():
                event_data = json.dumps({
                    "type": "long_task_created",
                    "task_id": task_id,
                    "session_id": session_id,
                    "patent_ids": patent_ids,
                    "patent_count": len(patent_ids),
                    "source": "file_upload",
                }, ensure_ascii=False)
                yield f"data: {event_data}\n\n"
                yield "data: {\"type\": \"end\"}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as e:
            app_logger.error(
                f"File upload: unhandled error: {e}\n{traceback.format_exc()}"
            )
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal server error: {str(e)}"},
            )

    @router.post("/query_stream")
    async def process_query_stream(http_request: Request):
        content_type = http_request.headers.get("content-type", "")

        # ── Multipart file-upload path ──
        if "multipart/form-data" in content_type:
            return await _handle_file_upload_query(http_request)

        # ── JSON path (existing) ──
        body = await http_request.json()
        request = QueryRequest(**body)

        # Optimize: Use cached token validation
        auth_header = http_request.headers.get("Authorization")
        try:
            user = await get_cached_token_validation(auth_header)
        except Exception as e:
            app_logger.error(f"Token validation failed: {str(e)}")
            return JSONResponse(status_code=401, content={"error": "Invalid token"})

        user_id = user['uid']

        app_logger.info(f"[user={user_id}] Processing query_stream: {request.query}")

        # Optimize: Make usage check async
        allowed = await check_usage_async(user_id)
        if not allowed:
            return JSONResponse(status_code=429, content={"error": "Daily API usage limit exceeded (100/day)"})

        # 如果没有提供 query_id，自动生成一个
        if not request.query_id:
            app_logger.warning("query id is none.")
            return JSONResponse(status_code=400, content={"error": "query_id is required"})

        if is_generating_flag:
            app_logger.warning("Another query is being processed, please wait.")
            return JSONResponse(status_code=429, content={"error": "Another query is being processed"})

        async def generate():
            # Optimize: Reuse agent from pool instead of creating new one
            general_agent = await get_or_create_agent(create_agent_func, app_logger)
            queue = asyncio.Queue()
            handler = SSECallbackHandler(queue)

            # Run the entire agent pipeline (create + invoke) in a background
            # task so the queue reading loop can yield status events to the
            # client as soon as they arrive, rather than buffering them until
            # create_agent completes.
            async def run_pipeline():
                openai_agent = None
                try:
                    openai_agent = await general_agent.create_agent(
                        user_id, request.query, request.query_id,
                        request.tool_data, handler,
                        push_filter=request.push_filter
                    )
                except Exception as e:
                    app_logger.error(f"Failed to create agent: {str(e)}")
                    await queue.put({'type': 'error', 'message': f'Failed to create agent: {str(e)}'})
                    await queue.put({'type': 'end'})
                    return
                finally:
                    if openai_agent is None:
                        handler.queue.put_nowait({'type': 'done'})

                # Long task handling: create_agent returned a long_task intent marker
                if isinstance(openai_agent, dict) and openai_agent.get('intent') == 'long_task':
                    app_logger.info(f"Long task intent received, setting up pipeline...")
                    try:
                        import json
                        from sources.knowledge.knowledge import get_db_connection

                        task_id = f"lt_{uuid.uuid4().hex[:12]}"
                        conv_history = request.conversation_history or []

                        local_user_id = int(user_id)
                        matched_knowledge = openai_agent.get('knowledge', {})
                        scene_id = getattr(matched_knowledge, 'scene_id', None) if matched_knowledge else None

                        # ── LLM: classify scenario → extract IDs → detect source ──
                        llm_result = await _classify_long_task_async(
                            query=request.query,
                            conv_history=conv_history,
                            scene_id=scene_id,
                            app_logger=app_logger,
                        )
                        inputs = _prepare_long_task_inputs(
                            query=request.query,
                            conv_history=conv_history,
                            scene_id=scene_id,
                            app_logger=app_logger,
                            llm_result=llm_result,
                        )
                        patent_ids = inputs["patent_ids"]
                        patent_source = inputs["patent_source"]
                        patent_texts = inputs["patent_texts"] or {}

                        # ── Session reuse: if the client already has a session, append to it ──
                        reused_session = False
                        session_id = request.session_id.strip() if request.session_id else ""
                        app_logger.info(f"Long task: connecting to DB...")
                        conn = get_db_connection()
                        app_logger.info(f"Long task: DB connected, inserting records...")
                        try:
                            with conn.cursor() as cur:
                                if session_id:
                                    # Validate session exists and belongs to this user
                                    cur.execute(
                                        """SELECT id, long_task_ids FROM conversations
                                           WHERE session_id = %s AND user_id = %s AND status != 2""",
                                        (session_id, local_user_id))
                                    existing = cur.fetchone()
                                    if existing:
                                        # Reuse: update messages and append new task_id
                                        existing_task_ids = json.loads(existing['long_task_ids']) if isinstance(existing['long_task_ids'], str) else (existing['long_task_ids'] or [])
                                        existing_task_ids.append(task_id)
                                        cur.execute(
                                            """UPDATE conversations
                                               SET messages = %s, long_task_ids = %s, update_time = NOW()
                                               WHERE session_id = %s""",
                                            (json.dumps(conv_history, ensure_ascii=False),
                                             json.dumps(existing_task_ids),
                                             session_id))
                                        reused_session = True
                                        app_logger.info(f"Long task: reusing session {session_id}, long_task_ids={existing_task_ids}")

                                if not reused_session:
                                    # Create new session (existing behavior)
                                    session_id = f"sess_{uuid.uuid4().hex[:12]}"
                                    cur.execute(
                                        """INSERT INTO conversations (session_id, user_id, scene_id, title, messages, long_task_ids)
                                           VALUES (%s, %s, %s, %s, %s, %s)""",
                                        (session_id, local_user_id, scene_id,
                                         f"专利分析 - {request.query[:50]}",
                                         json.dumps(conv_history, ensure_ascii=False),
                                         json.dumps([task_id])))
                                    app_logger.info(f"Long task: created new session {session_id}")

                                cur.execute(
                                    """INSERT INTO long_tasks
                                       (task_id, session_id, user_id, scene_id, task_type, input_params, status)
                                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                                    (task_id, session_id, local_user_id, scene_id,
                                     'patent_analysis',
                                     json.dumps({
                                         'query': request.query,
                                         'patent_ids': patent_ids,
                                         'patent_source': patent_source,
                                         **({'patent_texts': patent_texts} if patent_texts else {}),
                                     }, ensure_ascii=False),
                                     'pending'))
                                conn.commit()
                            app_logger.info(f"Long task: DB records inserted")
                        finally:
                            conn.close()

                        app_logger.info(f"Long task: submitting to Celery...")
                        from celery_worker import execute_patent_analysis
                        celery_params = {
                            'query': request.query,
                            'patent_ids': patent_ids,
                            'patent_source': patent_source,
                            'session_id': session_id,
                            'scene_id': scene_id,
                            'conversation_history': conv_history,
                        }
                        if patent_texts:
                            celery_params['patent_texts'] = patent_texts
                        execute_patent_analysis.delay(task_id=task_id, params=celery_params)
                        app_logger.info(f"Long task: Celery task submitted")

                        await queue.put({
                            'type': 'status',
                            'message': '批量专利分析任务已提交',
                            'transient': False,
                        })
                        await queue.put({
                            'type': 'long_task_created',
                            'task_id': task_id,
                            'session_id': session_id,
                        })
                        await queue.put({'type': 'end'})
                        app_logger.info(f"Long task: SSE events pushed, pipeline done")
                        return
                    except Exception as e:
                        app_logger.error(f"Long task setup failed: {str(e)}")
                        import traceback
                        app_logger.error(traceback.format_exc())
                        await queue.put({'type': 'error', 'message': f'批量分析任务启动失败: {str(e)}'})
                        await queue.put({'type': 'end'})
                        return

                try:
                    await general_agent.invoke_agent(openai_agent, handler)
                    await queue.put({'type': 'end', 'content': '[DONE]'})
                except Exception as e:
                    app_logger.error(f"invoke agent fail. An error occurred: {str(e)}")
                    await queue.put({'type': 'error', 'message': str(e)})
                    await queue.put({'type': 'end'})
                finally:
                    handler.queue.put_nowait({'type': 'done'})

            task = asyncio.create_task(run_pipeline())

            # Optimize: Batch tokens to reduce serialization overhead
            token_buffer = []
            buffer_size = 5  # Send 5 tokens at once
            last_flush_time = asyncio.get_event_loop().time()
            last_stream_time = last_flush_time
            flush_interval = 0.05  # Flush every 50ms
            heartbeat_interval = 15.0

            try:
                while True:
                    try:
                        # Try to get event with timeout to enable periodic flushing
                        event = await asyncio.wait_for(queue.get(), timeout=flush_interval)

                        if event['type'] == 'token':
                            token_buffer.append(event['content'])

                            current_time = asyncio.get_event_loop().time()
                            # Flush if buffer is full or enough time has passed
                            if len(token_buffer) >= buffer_size or (current_time - last_flush_time) >= flush_interval:
                                # Send buffered tokens
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                                last_flush_time = current_time
                                last_stream_time = current_time

                        elif event['type'] == 'status':
                            if token_buffer:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                            status_json = json.dumps(event)
                            yield f"data:{status_json}\n\n"
                            current_time = asyncio.get_event_loop().time()
                            last_flush_time = current_time
                            last_stream_time = current_time

                        elif event['type'] in {'artifact_start', 'artifact_chunk', 'artifact_end'}:
                            if token_buffer:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                            artifact_json = json.dumps(event)
                            yield f"data:{artifact_json}\n\n"
                            current_time = asyncio.get_event_loop().time()
                            last_flush_time = current_time
                            last_stream_time = current_time

                        elif event['type'] == 'long_task_created':
                            if token_buffer:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                            yield f"data:{json.dumps({'type': 'long_task_created', 'task_id': event.get('task_id'), 'session_id': event.get('session_id')})}\n\n"
                            current_time = asyncio.get_event_loop().time()
                            last_flush_time = current_time
                            last_stream_time = current_time

                        elif event['type'] == 'long_task_intent':
                            # Intermediate event from invoke_agent — process if needed
                            continue

                        elif event['type'] == 'end':
                            # Flush remaining tokens before ending
                            if token_buffer:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                                last_stream_time = asyncio.get_event_loop().time()
                            break

                        elif event['type'] == 'error':
                            # Flush tokens and send error
                            if token_buffer:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                                last_stream_time = asyncio.get_event_loop().time()
                            error_json = json.dumps({'error': event.get('message', 'Unknown error')})
                            yield f"data:{error_json}\n\n"
                            last_stream_time = asyncio.get_event_loop().time()
                            break

                    except asyncio.TimeoutError:
                        # Periodic flush even without new tokens
                        current_time = asyncio.get_event_loop().time()
                        if token_buffer:
                            if (current_time - last_flush_time) >= flush_interval:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                                last_flush_time = current_time
                                last_stream_time = current_time
                        elif (current_time - last_stream_time) >= heartbeat_interval:
                            yield ": ping\n\n"
                            last_stream_time = current_time

            except Exception as e:
                app_logger.error(f"Error in generate loop: {str(e)}")
                error_json = json.dumps({'error': str(e)})
                yield f"data:{error_json}\n\n"
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    @router.get("/copiioai_statistics")
    async def get_statistics():
        """获取今日知识创建和用户提问统计"""
        from datetime import datetime

        # Mock 数据
        today = datetime.now().strftime("%Y-%m-%d")

        stats = {
            "date": today,
            "knowledge_created": 42,  # 今日创建的知识数量
            "knowledge_shared": 41,  # 今日创建的知识数量
            "user_questions": 156,  # 今日用户提问次数
            "active_users": 23,  # 今日活跃用户数
            "knowledge_tools_used": 18,  # 今日知识工具使用次数
            "success_rate": "92.3%"  # 成功率
        }

        return JSONResponse(status_code=200, content=stats)

    return router
