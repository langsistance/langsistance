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
from sources.analytics import track_event
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


def _register_long_task_for_recovery(
    user_id: int | str,
    query_id: str | None,
    task_id: str,
    session_id: str,
    queue_status: str,
) -> None:
    """Record query_id 鈫?task mapping so the client can recover after SSE timeout."""
    if not query_id:
        return
    from sources.long_task.status_manager import register_query_task
    register_query_task(str(user_id), query_id, task_id, session_id, queue_status)

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

    Detection order (text keywords take priority 鈥?scene tools only provide fallback):
      1. Conversation / query text keywords (most reliable 鈥?what user is searching)
      2. Scene tools URL heuristics (fallback)
      3. Default: let the pipeline auto-detect via patent ID format
    """
    conv_history = conv_history or []

    # 鈹€鈹€ Build combined text 鈹€鈹€
    combined = query + " " + " ".join(
        m.get("content", "") for m in (conv_history or [])
        if isinstance(m, dict)
    )
    combined_lower = combined.lower()

    # 鈹€鈹€ 1. Text keywords (primary 鈥?user intent) 鈹€鈹€
    uspto_keywords = ["uspto", "缇庡浗涓撳埄", "缇庡浗涓撳埄鍟嗘爣灞€", "united states patent",
                      "us patent", "us application"]
    cnipa_keywords = ["cnipa", "涓浗涓撳埄", "涓浗鍥藉鐭ヨ瘑浜ф潈", "鍥藉鐭ヨ瘑浜ф潈灞€",
                      "chinese patent", "china patent",
                      "zldsj"]
    # Chinese company names 鈫?infer cnipa
    cn_company_keywords = [
        "鍗庝负", "灏忕背", "oppo", "vivo", "鑵捐", "闃块噷宸村反", "鐧惧害",
        "姣斾簹杩?, "瀹佸痉鏃朵唬", "涓叴", "澶х枂", "瀛楄妭璺冲姩", "涓姱鍥介檯",
        "浜笢鏂?, "鏍煎姏", "缇庣殑", "娴峰皵", "鑱旀兂", "钄氭潵", "灏忛箯", "鐞嗘兂",
        "瀵掓绾?, "鍦板钩绾?, "绱厜", "闀挎睙瀛樺偍", "闀块懌",
    ]
    # US company names 鈫?infer uspto
    us_company_keywords = [
        "apple", "google", "microsoft", "tesla", "intel", "amd",
        "nvidia", "qualcomm", "ibm", "meta", "amazon", "broadcom",
        "micron", "cisco", "oracle", "hp", "dell",
    ]
    if any(kw in combined_lower for kw in uspto_keywords):
        source = "uspto"
        if app_logger:
            app_logger.info(f"patent_source: text_keywords 鈫?uspto")
        return source
    if any(kw in combined_lower for kw in cnipa_keywords):
        source = "cnipa"
        if app_logger:
            app_logger.info(f"patent_source: text_keywords 鈫?cnipa")
        return source

    # Company name inference (lower priority than explicit patent office keywords)
    if any(kw in combined_lower for kw in us_company_keywords):
        source = "uspto"
        if app_logger:
            app_logger.info(f"patent_source: us_company 鈫?uspto")
        return source
    if any(kw in combined_lower for kw in cn_company_keywords):
        source = "cnipa"
        if app_logger:
            app_logger.info(f"patent_source: cn_company 鈫?cnipa")
        return source

    # 鈹€鈹€ 2. Scene tools (fallback) 鈹€鈹€
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
                    f"tool_urls={urls[:3]}, 鈫?{source}"
                )
            return source
        except Exception:
            pass

    # 鈹€鈹€ 3. Default 鈹€鈹€
    if app_logger:
        app_logger.info(f"patent_source: no signal found 鈫?auto")
    return "auto"


def _detect_query_language(query: str) -> str:
    """Detect the language of a user query for report generation.

    Uses CJK character ratio: if >15% of characters are CJK Unified
    Ideographs (U+4E00鈥揢+9FFF), returns 'zh'; otherwise 'en'.

    This is a GLOBAL PRINCIPLE: report language matches query language.
    """
    if not query:
        return 'zh'
    cjk_count = sum(1 for c in query if '涓€' <= c <= '榭?)
    alpha_count = sum(1 for c in query if c.isalpha())
    total = cjk_count + alpha_count
    if total == 0:
        return 'zh'
    return 'zh' if cjk_count / max(total, 1) > 0.15 else 'en'


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
        "浣犳槸涓€涓笓鍒╁垎鏋愪换鍔″垎绫诲櫒銆傚垎鏋愮敤鎴锋煡璇?+ 瀵硅瘽鍘嗗彶锛屽垽鏂満鏅€佹彁鍙栦笓鍒㊣D銆佽瘑鍒潵婧愩€俓n\n"
        "## 鏍稿績鍘熷垯\n"
        "涓撳埄ID鐨勬潵婧愬彇鍐充簬鐢ㄦ埛鐨勬剰鍥撅細\n"
        "- 濡傛灉鐢ㄦ埛鎻愰棶涓寘鍚簡鏄庣‘鐨勪笓鍒╁彿 鈫?浠庢彁闂腑鎻愬彇ID\n"
        "- 濡傛灉鐢ㄦ埛鎻愰棶鐨勬剰鎬濇槸銆屽熀浜庝箣鍓嶇殑缁撴灉缁х画鎿嶄綔銆嶁啋 浠庡巻鍙蹭腑鎻愬彇ID\n"
        "- 濡傛灉鐢ㄦ埛鎻愰棶鏄竴涓叏鏂扮殑銆佺嫭绔嬬殑璇濋锛堜笌鍘嗗彶鏃犲叧锛夆啋 涓嶆彁鍙栦换浣旾D锛?
        "patent_ids 杩斿洖绌烘暟缁勶紝绯荤粺浼氳嚜鍔ㄨЕ鍙戞悳绱㈡ā寮廫n\n"
        "## 鍥涚鍦烘櫙\n"
        "1. prosecution 鈥?鐢ㄦ埛瑕佸垎鏋?*鍗曚釜**涓撳埄鐨勫鏌ュ巻鍙诧紙瀹℃煡杩囩▼銆丱ffice Action銆?
        "绛旇京銆丆laim淇敼銆佹巿鏉冨師鍥犵瓑锛夈€俓n"
        "   鍏抽敭璇嶏細瀹℃煡鍘嗗彶銆佸鏌ヨ繃绋嬨€佸鏌ユ剰瑙併€丱ffice Action銆乸rosecution銆?
        "绛旇京銆佺瓟澶嶃€乧laim淇敼銆佹巿鏉冨師鍥犮€丱A鍥炲銆乺ejection銆乤llowance\n"
        "   - patent_ids 涓彧鎻愬彇 **1涓?* 涓撳埄鍙穃n"
        "   - patent_source 蹇呴』鏄?uspto锛堝鏌ュ巻鍙插垎鏋愪粎鏀寔USPTO锛塡n"
        "   渚嬶細銆屽府鎴戝垎鏋愪竴涓嬩笓鍒?17429113 鐨勫鏌ュ巻鍙层€嶁啋 scenario: prosecution\n"
        "   渚嬶細銆屽垎鏋愪笓鍒╃敵璇?18331482 鐨?Office Action 鍜岀瓟杈╄繃绋嬨€嶁啋 scenario: prosecution\n"
        "   渚嬶細銆孉nalyze the prosecution history of patent 17429113銆嶁啋 scenario: prosecution\n"
        "2. direct_ids 鈥?鐢ㄦ埛鎻愰棶鏈韩鍖呭惈涓撳埄鐢宠鍙凤紝鎴栬€呯敤鎴锋彁鍑轰簡涓€涓叏鏂扮嫭绔嬬殑闂銆俓n"
        "   - 濡傛灉鎻愰棶涓湁涓撳埄鍙?鈫?patent_ids 浠庢彁闂腑鎻愬彇\n"
        "   - 濡傛灉鎻愰棶涓病鏈変笓鍒╁彿锛堝銆屽府鎴戠湅鐪嬬壒鏂媺鐨勮嚜鍔ㄩ┚椹朵笓鍒┿€嶏級鈫?patent_ids 杩斿洖 []\n"
        "   渚嬶細銆屽垎鏋愪笓鍒╃敵璇?17429113, 18012525, 18331482銆嶁啋 patent_ids: [\"17429113\", ...]\n"
        "   渚嬶細銆屽府鎴戠湅鐪嬬壒鏂媺杩戞湡鐨勪笓鍒╁湪鑷姩椹鹃┒棰嗗煙鏈変粈涔堟柊杩涘睍銆嶁啋 patent_ids: []\n"
        "3. conversation_refs 鈥?鐢ㄦ埛鐨勯棶棰樻槸閽堝鍘嗗彶瀵硅瘽涓嚭鐜扮殑涓撳埄缁撴灉鐨勮拷闂紝"
        "涓嶇湅鍘嗗彶瀵硅瘽灏辨棤娉曠‘瀹氳鍒嗘瀽鍝簺涓撳埄銆俓n"
        "   - patent_ids 浠庡巻鍙插璇濅腑鎻愬彇\n"
        "   渚嬶細锛堝巻鍙插璇濊繑鍥炰簡3涓笓鍒╋級鐢ㄦ埛锛氥€屼粠杩?鏉′腑绛涢€夊嚭姗¤兌鍨湀鐩稿叧涓撳埄銆峔n"
        "   渚嬶細锛堝巻鍙插璇濊繑鍥炰簡涓撳埄鍒楄〃锛夌敤鎴凤細銆屼笂杩颁笓鍒╅噷鍝簺宸叉巿鏉冦€峔n"
        "   渚嬶細锛堝巻鍙插璇濊繑鍥炰簡涓撳埄缁撴灉锛夌敤鎴凤細銆屽府鎴戝垎鏋愮涓€涓笓鍒┿€峔n\n"
        "## 涓撳埄ID鏍煎紡\n"
        "- USPTO锛堢編鍥斤級: 绾?浣嶆暟瀛楋紝濡?17429113, 18331482\n"
        "- CNIPA锛堜腑鍥斤級: 20XX + 8~9浣嶆暟瀛楋紝鍙€?.鏍￠獙浣嶏紝濡?202310123456.7\n\n"
        "## 涓撳埄鏉ユ簮鍒ゆ柇\n"
        "- 濡傛灉瀵硅瘽鎻愬埌 USPTO銆佺編鍥戒笓鍒┿€佺編鍥戒笓鍒╁晢鏍囧眬 鈫?uspto\n"
        "- 濡傛灉瀵硅瘽鎻愬埌 CNIPA銆佷腑鍥戒笓鍒┿€佸浗瀹剁煡璇嗕骇鏉冨眬 鈫?cnipa\n"
        "- 绾?浣嶆暟瀛桰D 鈫?uspto\n"
        "- 20XX寮€澶寸殑闀挎暟瀛桰D 鈫?cnipa\n"
        "- 濡傛灉鐢ㄦ埛鎻愬埌浜嗕腑鍥藉叕鍙革紙濡傚崕涓恒€佸皬绫炽€丱PPO銆乿ivo銆佽吘璁€侀樋閲屽反宸淬€佺櫨搴︺€乗n"
        "  姣斾簹杩€佸畞寰锋椂浠ｃ€佷腑鍏淬€佸ぇ鐤嗐€佸瓧鑺傝烦鍔ㄣ€佷腑鑺浗闄呫€佷含涓滄柟绛夛級锛屼笖娌℃湁鏄庣‘\n"
        "  鎻愬埌缇庡浗涓撳埄 鈫?cnipa\n"
        "- 濡傛灉鐢ㄦ埛鎻愬埌浜嗙編鍥藉叕鍙革紙濡侫pple銆丟oogle銆丮icrosoft銆乀esla銆両ntel銆乗n"
        "  AMD銆丯VIDIA銆丵ualcomm銆両BM绛夛級锛屼笖娌℃湁鏄庣‘鎻愬埌涓浗涓撳埄 鈫?uspto\n"
        "- 涓嶇‘瀹?鈫?unknown\n\n"
        "## 杈撳嚭JSON\n"
        '{"scenario": "prosecution"|"direct_ids"|"conversation_refs", '
        '"patent_ids": ["id1","id2"], '
        '"patent_source": "uspto"|"cnipa"|"unknown", '
        '"reasoning": "绠€瑕佽鏄?}'
    )

    user_text = (
        f"鐢ㄦ埛褰撳墠鎻愰棶锛歿query}\n\n"
        f"瀵硅瘽鍘嗗彶锛歕n" + "\n".join(conv_summary) if conv_summary else "(鏃犲巻鍙插璇?"
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
      1. "conversation_refs" 鈥?query is a FOLLOW-UP about previous results.
         IDs come from conversation history (LLM extracts them).
      2. "direct_ids" 鈥?query ITSELF contains patent application IDs.
         Self-contained question. IDs come from query.
      3. "file_upload" 鈥?user uploaded patent specification files.

    When ``llm_result`` is provided (from ``_classify_long_task_async``),
    scenario and patent_source from the LLM are used directly.
    Otherwise falls back to regex extraction + keyword detection.

    Returns dict with keys:
      scenario, patent_ids, patent_source, patent_texts
    """
    import re as _patent_id_re

    conv_history = conv_history or []
    patent_file_refs = patent_file_refs or []

    # 鈹€鈹€ Scenario 3: file upload (unambiguous, no LLM needed) 鈹€鈹€
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
                f"Long task scenario=file_upload 鈥?"
                f"files={len(patent_file_refs)}, ids={patent_ids[:5]}, "
                f"patent_source={patent_source}"
            )
        return {
            "scenario": "file_upload",
            "patent_ids": patent_ids,
            "patent_source": patent_source,
            "patent_texts": None,
        }

    # 鈹€鈹€ Regex sweep on full text (always run 鈥?catches IDs LLM might miss) 鈹€鈹€
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

    # 鈹€鈹€ Use LLM result if available 鈹€鈹€
    if llm_result and llm_result.get("scenario") != "unknown":
        scenario = llm_result["scenario"]
        patent_source = llm_result.get("patent_source", "auto")
        llm_ids = llm_result.get("patent_ids", []) or []

        # For direct_ids: trust the LLM's patent_ids exactly as returned.
        # If it returned an empty list, the query has no IDs and should
        # fall through to search mode 鈥?do NOT backfill from regex on
        # history (which would pick up unrelated 8-digit numbers).
        # For conversation_refs: fall back to regex if LLM returned no IDs.
        if scenario == "direct_ids":
            patent_ids = list(dict.fromkeys(llm_ids))
        else:
            patent_ids = list(dict.fromkeys(llm_ids)) if llm_ids else list(dict.fromkeys(regex_ids))

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
        # 鈹€鈹€ Fallback: regex + keyword detection (no LLM available) 鈹€鈹€
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

        followup_keywords = ["杩?, "涓婅堪", "鍓嶉潰", "浠ヤ笂", "浠庝腑", "鍏朵腑", "绛涢€?, "杩囨护",
                            "鎸戝嚭", "閫夊嚭", "鍝簺", "鍝釜", "杩欏嚑涓?, "閭ｅ嚑涓?]
        query_is_followup = any(kw in (query or "") for kw in followup_keywords)

        scenario = "direct_ids" if (query_has_ids and not query_is_followup) else "conversation_refs"
        patent_source = _detect_patent_source(
            scene_id=scene_id, conv_history=conv_history, query=query, app_logger=app_logger,
        )
        patent_texts = patent_texts if patent_texts else None

    if app_logger:
        app_logger.info(
            f"Long task scenario={scenario} 鈥?"
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
    """娉ㄥ唽鏍稿績璺敱骞朵紶閫掓墍闇€鐨勪緷璧?""

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
        track_event("query", user_id=str(user_id), query_text=request.query,
                    query_id=request.query_id)

        allowed = check_and_increase_usage(user_id)
        if not allowed:
            return JSONResponse(status_code=429, content="Daily API usage limit exceeded (100/day)")

        # 濡傛灉娌℃湁鎻愪緵 query_id锛岃嚜鍔ㄧ敓鎴愪竴涓?
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
            # 璋冪敤 think_wrapper_func 鏉ュ鐞嗘煡璇?
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
            # sys.exit(1)  # 涓嶅簲璇ュ湪璺敱涓€€鍑哄簲鐢?
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
        """鏍规嵁鐢ㄦ埛闂鏌ユ壘鏈€鐩稿叧鐨勭煡璇嗗強鍏跺搴旂殑宸ュ叿"""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)

        user_id = user['uid']

        app_logger.info(f"Finding knowledge tool for user: {user_id} with question: {request.question}")
        track_event("knowledge:find", user_id=str(user_id), query_text=request.question)

        try:
            # 鍙傛暟鏍￠獙

            if not request.question:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "question is required"
                    }
                )

            # 璋冪敤knowledge.py涓殑鏂规硶鑾峰彇鐭ヨ瘑椤瑰拰宸ュ叿淇℃伅
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

            # 鏋勫缓杩斿洖鐨勭煡璇嗚褰曞璞?
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
                # 妫€鏌ool_info鐨刾ush瀛楁
                if tool_info.push == 2:
                    # 濡傛灉push涓?锛岃繑鍥炵┖瀵硅薄
                    tool_response = {}
                else:
                    # 濡傛灉push涓嶄负2锛屼繚鎸佸師鏈夐€昏緫
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
                    content={"error": "No valid patent files uploaded (PDF/DOCX/XML only, min 50 bytes, max 10 MB each)"},
                )

            app_logger.info(
                f"[user={user_id}] File upload: received {len(patent_files)} files, query={query[:80]}"
            )

            # Save files to disk for async processing by Celery worker.
            # Text extraction (esp. OCR) can take minutes 鈥?we must not block the HTTP response.
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

            # 鈹€鈹€ Session reuse for file upload path 鈹€鈹€
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
            scenario = inputs["scenario"]

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
                        track_event("session:new", user_id=str(local_user_id),
                                    session_id=session_id, query_text=query)

                    cur.execute(
                        """INSERT INTO long_tasks
                           (task_id, session_id, user_id, scene_id, task_type, input_params, status)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (task_id, session_id, local_user_id, None,
                         "patent_analysis",
                         json.dumps({
                             "query": query,
                             "query_id": query_id,
                             "patent_ids": patent_ids,
                             "patent_source": patent_source,
                             "patent_file_refs": patent_file_refs,
                         }, ensure_ascii=False),
                         "pending"))
                    conn.commit()
            finally:
                pass  # conn is closed below after both branches

            from sources.long_task.user_queue import try_start_user_task
            queue_result = try_start_user_task(str(local_user_id), task_id)

            celery_params = {
                "query": query,
                "patent_ids": patent_ids,
                "patent_source": patent_source,
                "session_id": session_id,
                "scene_id": None,
                "conversation_history": conversation_history,
                "patent_file_refs": patent_file_refs,
                "user_id": str(local_user_id),
                "scenario": scenario,
            }

            if queue_result == "running":
                from celery_worker import execute_patent_analysis
                execute_patent_analysis.delay(task_id=task_id, params=celery_params)
                app_logger.info(f"[user={user_id}] File upload: dispatched task={task_id}")
                track_event("long_task:submit", user_id=str(local_user_id),
                            task_id=task_id, patent_count=len(patent_ids),
                            patent_source=patent_source,
                            session_id=session_id or None,
                            query_text="[file upload]")
                event_type = "long_task_created"
                event_status = "running"
                conn.close()
            else:
                # Queued 鈥?update MySQL status, worker will pick it up later
                conn.ping(reconnect=True)
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE long_tasks SET status = 'queued' WHERE task_id = %s",
                        (task_id,),
                    )
                    conn.commit()
                app_logger.info(f"[user={user_id}] File upload: queued task={task_id}")
                track_event("long_task:queued", user_id=str(local_user_id),
                            task_id=task_id, patent_count=len(patent_ids),
                            patent_source=patent_source,
                            session_id=session_id or None,
                            query_text="[file upload]")
                event_type = "long_task_created"
                event_status = "queued"
                conn.close()

            _register_long_task_for_recovery(
                local_user_id, query_id, task_id, session_id, event_status,
            )

            # Return SSE with long_task_created
            async def generate_sse():
                event_data = json.dumps({
                    "type": event_type,
                    "task_id": task_id,
                    "session_id": session_id,
                    "patent_ids": patent_ids,
                    "patent_count": len(patent_ids),
                    "source": "file_upload",
                    "status": event_status,
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

        # 鈹€鈹€ Multipart file-upload path 鈹€鈹€
        if "multipart/form-data" in content_type:
            return await _handle_file_upload_query(http_request)

        # 鈹€鈹€ JSON path (existing) 鈹€鈹€
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
        track_event("query_stream", user_id=str(user_id), query_text=request.query,
                    query_id=request.query_id,
                    session_id=request.session_id or None)

        # Optimize: Make usage check async
        allowed = await check_usage_async(user_id)
        if not allowed:
            return JSONResponse(status_code=429, content={"error": "Daily API usage limit exceeded (100/day)"})

        # 濡傛灉娌℃湁鎻愪緵 query_id锛岃嚜鍔ㄧ敓鎴愪竴涓?
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

                        # 鈹€鈹€ LLM: classify scenario 鈫?extract IDs 鈫?detect source 鈹€鈹€
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
                        scenario = inputs["scenario"]

                        # 鈹€鈹€ Session reuse: if the client already has a session, append to it 鈹€鈹€
                        reused_session = False
                        session_id = request.session_id.strip() if request.session_id else ""
                        app_logger.info(
                            f"Long task: session_id from request="
                            f"'{request.session_id}', resolved='{session_id}'"
                        )
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
                                         f"{request.query[:60]}",
                                         json.dumps(conv_history, ensure_ascii=False),
                                         json.dumps([task_id])))
                                    app_logger.info(f"Long task: created new session {session_id}")
                                    track_event("session:new", user_id=str(local_user_id),
                                                session_id=session_id,
                                                query_text=request.query)

                                cur.execute(
                                    """INSERT INTO long_tasks
                                       (task_id, session_id, user_id, scene_id, task_type, input_params, status)
                                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                                    (task_id, session_id, local_user_id, scene_id,
                                     'prosecution_analysis' if scenario == 'prosecution' else 'patent_analysis',
                                     json.dumps({
                                         'query': request.query,
                                         'query_id': request.query_id,
                                         **({'patent_id': patent_ids[0]} if scenario == 'prosecution' and patent_ids else {'patent_id': ''}),
                                         **({'patent_ids': patent_ids} if scenario != 'prosecution' else {}),
                                         'patent_source': patent_source,
                                         **({'patent_texts': patent_texts} if patent_texts else {}),
                                         **({'lang': _detect_query_language(request.query)} if scenario == 'prosecution' else {}),
                                     }, ensure_ascii=False),
                                     'pending'))
                                conn.commit()
                            app_logger.info(f"Long task: DB records inserted")
                        finally:
                            pass  # conn is closed below after both branches

                        from sources.long_task.user_queue import try_start_user_task

                        is_prosecution = (scenario == 'prosecution')
                        query_lang = _detect_query_language(request.query)

                        celery_params = {
                            'query': request.query,
                            'session_id': session_id,
                            'scene_id': scene_id,
                            'conversation_history': conv_history,
                            'user_id': str(local_user_id),
                            'scenario': scenario,
                        }
                        celery_params['lang'] = query_lang
                        if is_prosecution:
                            celery_params['patent_id'] = patent_ids[0] if patent_ids else ''
                        else:
                            celery_params['patent_ids'] = patent_ids
                            celery_params['patent_source'] = patent_source
                        if patent_texts:
                            celery_params['patent_texts'] = patent_texts

                        queue_result = try_start_user_task(str(local_user_id), task_id)

                        if queue_result == 'running':
                            app_logger.info(f"Long task: submitting to Celery...")
                            if is_prosecution:
                                from celery_worker import execute_prosecution_analysis
                                execute_prosecution_analysis.delay(task_id=task_id, params=celery_params)
                            else:
                                from celery_worker import execute_patent_analysis
                                execute_patent_analysis.delay(task_id=task_id, params=celery_params)
                            app_logger.info(f"Long task: Celery task submitted")
                            track_event("long_task:submit", user_id=str(local_user_id),
                                        task_id=task_id,
                                        patent_count=1 if is_prosecution else len(patent_ids),
                                        patent_source=patent_source,
                                        session_id=session_id or None,
                                        query_text=request.query)
                            conn.close()
                        else:
                            # Queued 鈥?update MySQL, Celery will pick it up when dequeued
                            app_logger.info(f"Long task: queued (user already has running task)")
                            track_event("long_task:queued", user_id=str(local_user_id),
                                        task_id=task_id,
                                        patent_count=1 if is_prosecution else len(patent_ids),
                                        patent_source=patent_source,
                                        session_id=session_id or None,
                                        query_text=request.query)
                            conn.ping(reconnect=True)
                            with conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE long_tasks SET status = 'queued' WHERE task_id = %s",
                                    (task_id,),
                                )
                                conn.commit()
                            conn.close()

                        _register_long_task_for_recovery(
                            local_user_id, request.query_id, task_id, session_id, queue_result,
                        )

                        if is_prosecution:
                            status_msg = (
                                '涓撳埄瀹℃煡鍘嗗彶鍒嗘瀽浠诲姟宸叉彁浜? if queue_result == 'running'
                                else '涓撳埄瀹℃煡鍘嗗彶鍒嗘瀽浠诲姟宸叉帓闃燂紝灏嗗湪褰撳墠浠诲姟瀹屾垚鍚庤嚜鍔ㄥ紑濮?
                            )
                        else:
                            status_msg = (
                                '鎵归噺涓撳埄鍒嗘瀽浠诲姟宸叉彁浜? if queue_result == 'running'
                                else '鎵归噺涓撳埄鍒嗘瀽浠诲姟宸叉帓闃燂紝灏嗗湪褰撳墠浠诲姟瀹屾垚鍚庤嚜鍔ㄥ紑濮?
                            )
                        await queue.put({
                            'type': 'status',
                            'message': status_msg,
                            'transient': False,
                        })
                        await queue.put({
                            'type': 'long_task_created',
                            'task_id': task_id,
                            'session_id': session_id,
                            'status': queue_result,
                        })
                        await queue.put({'type': 'end'})
                        app_logger.info(f"Long task: SSE events pushed, pipeline done")
                        return
                    except Exception as e:
                        app_logger.error(f"Long task setup failed: {str(e)}")
                        import traceback
                        app_logger.error(traceback.format_exc())
                        await queue.put({'type': 'error', 'message': f'鎵归噺鍒嗘瀽浠诲姟鍚姩澶辫触: {str(e)}'})
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
            task.add_done_callback(
                lambda t: (
                    app_logger.error(f"SSE background task crashed: {t.exception()}")
                    if t.exception() and not t.cancelled() else None
                )
            )

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
                            yield f"data:{json.dumps({'type': 'long_task_created', 'task_id': event.get('task_id'), 'session_id': event.get('session_id'), 'status': event.get('status')})}\n\n"
                            current_time = asyncio.get_event_loop().time()
                            last_flush_time = current_time
                            last_stream_time = current_time

                        elif event['type'] == 'long_task_intent':
                            # Intermediate event from invoke_agent 鈥?process if needed
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
        """鑾峰彇浠婃棩鐭ヨ瘑鍒涘缓鍜岀敤鎴锋彁闂粺璁?""
        from datetime import datetime

        # Mock 鏁版嵁
        today = datetime.now().strftime("%Y-%m-%d")

        stats = {
            "date": today,
            "knowledge_created": 42,  # 浠婃棩鍒涘缓鐨勭煡璇嗘暟閲?
            "knowledge_shared": 41,  # 浠婃棩鍒涘缓鐨勭煡璇嗘暟閲?
            "user_questions": 156,  # 浠婃棩鐢ㄦ埛鎻愰棶娆℃暟
            "active_users": 23,  # 浠婃棩娲昏穬鐢ㄦ埛鏁?
            "knowledge_tools_used": 18,  # 浠婃棩鐭ヨ瘑宸ュ叿浣跨敤娆℃暟
            "success_rate": "92.3%"  # 鎴愬姛鐜?
        }

        return JSONResponse(status_code=200, content=stats)

    return router
