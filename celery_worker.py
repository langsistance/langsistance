"""Celery worker entry point for long-running patent analysis tasks."""

import asyncio
import json
import os
import sys

# Ensure the project root is on sys.path so `sources.*` imports work
# both under docker-compose (volume mount at /app) and bare Docker runs.
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from celery import Celery
from sources.logger import Logger

_pipeline_logger = Logger("long_task_pipeline.log")

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

app = Celery('patent_tasks', broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def execute_patent_analysis(self, task_id: str, params: dict):
    """Batch patent analysis -- 4-phase serial pipeline with checkpointing."""
    _pipeline_logger.info(
        f"[task={task_id}] START — "
        f"query={params.get('query', '')[:120]}, "
        f"patent_source={params.get('patent_source', 'cnipa')}, "
        f"session_id={params.get('session_id', '')}, "
        f"scene_id={params.get('scene_id', '')}"
    )
    from sources.long_task.status_manager import (
        update_task_status, set_task_completed, set_task_failed,
        save_checkpoint, load_checkpoint,
    )
    from sources.long_task.config import get_long_task_config
    from sources.long_task.patent_analyzer import (
        generate_table_columns, download_patent_document,
        analyze_single_patent, generate_patent_summary, build_failed_row,
    )
    from sources.long_task.report_generator import (
        generate_report_outline, generate_report_section,
    )
    from sources.long_task.storage import create_storage
    from sources.llm_provider import Provider

    ltc = get_long_task_config()
    model_family = ltc['provider_family']
    max_patents = ltc['max_patents']

    # ---- Input dedup + truncation (deterministic ordering) ----
    patent_ids = sorted(set(params.get('patent_ids', [])))
    patent_ids = patent_ids[:max_patents]
    total = len(patent_ids)

    _pipeline_logger.info(
        f"[task={task_id}] CONFIG — "
        f"model_family={model_family}, max_patents={max_patents}, "
        f"patent_ids_count={len(patent_ids)}, "
        f"patent_ids={patent_ids[:10]}{'...' if len(patent_ids) > 10 else ''}"
    )

    # ---- Provider setup ----
    if model_family == 'minimax':
        flash_provider = Provider(provider_name='minimax', model='MiniMax-M2.7-highspeed',
                                  server_address='', is_local=False)
        pro_provider = Provider(provider_name='minimax', model='MiniMax-M2.7-highspeed',
                                server_address='', is_local=False)
    else:
        flash_provider = Provider(provider_name='deepseek', model='deepseek-chat',
                                  server_address='', is_local=False)
        pro_provider = Provider(provider_name='deepseek', model='deepseek-reasoner',
                                server_address='', is_local=False)

    try:
        # ---- Run the 4-phase pipeline using event-loop-safe pattern ----
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_run_pipeline(
                task_id=task_id,
                params=params,
                patent_ids=patent_ids,
                total=total,
                max_patents=max_patents,
                flash_provider=flash_provider,
                pro_provider=pro_provider,
                update_task_status=update_task_status,
                set_task_completed=set_task_completed,
                set_task_failed=set_task_failed,
                save_checkpoint=save_checkpoint,
                load_checkpoint=load_checkpoint,
                generate_table_columns=generate_table_columns,
                download_patent_document=download_patent_document,
                analyze_single_patent=analyze_single_patent,
                generate_patent_summary=generate_patent_summary,
                build_failed_row=build_failed_row,
                generate_report_outline=generate_report_outline,
                generate_report_section=generate_report_section,
                create_storage=create_storage,
            ))
            _pipeline_logger.info(
                f"[task={task_id}] PIPELINE_DONE — status={result.get('status')}"
            )
            return result
        finally:
            loop.close()
    except Exception as e:
        _pipeline_logger.error(
            f"[task={task_id}] FAILED — error={e}"
        )
        set_task_failed(task_id, str(e))
        raise self.retry(exc=e)


def _update_mysql_progress(task_id: str, current_phase: str, progress: int) -> None:
    """Update the long_tasks MySQL row with current progress."""
    try:
        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE long_tasks
                       SET status = 'running', current_phase = %s, progress = %s
                       WHERE task_id = %s""",
                    (current_phase, progress, task_id))
                conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # Non-fatal: MySQL update failure should not break the pipeline


async def _run_pipeline(
    task_id: str,
    params: dict,
    patent_ids: list,
    total: int,
    max_patents: int,
    flash_provider,
    pro_provider,
    update_task_status,
    set_task_completed,
    set_task_failed,
    save_checkpoint,
    load_checkpoint,
    generate_table_columns,
    download_patent_document,
    analyze_single_patent,
    generate_patent_summary,
    build_failed_row,
    generate_report_outline,
    generate_report_section,
    create_storage,
) -> dict:
    """Internal async pipeline orchestrator."""

    # ---- Crash recovery: load checkpoint ----
    checkpoint = load_checkpoint(task_id)
    if checkpoint and checkpoint.get('pending'):
        table_rows = checkpoint.get('completed_rows', [])
        pending = checkpoint['pending']
    else:
        table_rows = []
        pending = patent_ids

    scene_id = params.get('scene_id')
    # Load scene knowledge+tools once for use across all phases
    scene_candidates = None
    if scene_id:
        from sources.long_task.scene_tools import get_scene_knowledge_tools
        scene_candidates = get_scene_knowledge_tools(scene_id)
        _pipeline_logger.info(
            f"[task={task_id}] PHASE0 — scene_id={scene_id}, "
            f"scene_candidates_count={len(scene_candidates) if scene_candidates else 0}"
        )
        if scene_candidates:
            for c in scene_candidates:
                _pipeline_logger.info(
                    f"[task={task_id}] PHASE0 candidate — "
                    f"knowledge_id={c.get('knowledge_id')}, "
                    f"knowledge_type={c.get('knowledge_type')}, "
                    f"question={c.get('knowledge_question', '')[:80]}, "
                    f"tool_title={c.get('tool_title', '')}, "
                    f"tool_url={c.get('tool_url', '')}"
                )
    id_url_map = {}          # patent_id → document_url from search results

    # ==== Phase 0-prep: Extract patent IDs from conversation history via LLM ====
    # When the user says "从这3条中筛选"/"filter these 3", the patent IDs are
    # in the previous assistant response text but not in structured patent_data.
    if not patent_ids:
        conv_history = params.get('conversation_history', [])
        if conv_history:
            history_text = "\n".join(
                msg.get('content', '') for msg in conv_history
                if msg.get('role') == 'assistant'
            )
            if history_text and len(history_text) > 100:
                _pipeline_logger.info(
                    f"[task={task_id}] PHASE0 extract_patent_ids_from_history — "
                    f"history_text_length={len(history_text)}"
                )
                try:
                    extract_result = await flash_provider.complete_json(
                        "You are a patent ID extractor. "
                        "Extract all patent APPLICATION NUMBERS from the provided text. "
                        "For USPTO: application numbers are PURE 8-DIGIT NUMBERS "
                        "(e.g. 18331482) or slash format (e.g. 18/234567). "
                        "Publication numbers like US20230310100A1 are NOT application "
                        "numbers — find the corresponding 8-digit application number "
                        "in the text instead. "
                        "For CNIPA: format CNxxxxxxxxxA (e.g. CN202111325396A). "
                        "Return ONLY a JSON object: "
                        '{"patent_ids": ["id1", "id2", ...], '
                        '"source": "uspto" or "cnipa" or "unknown"}. '
                        "Only include application numbers, not publication numbers or grant numbers.",
                        f"Extract all patent/application IDs from:\n\n{history_text[:4000]}",
                    )
                    if extract_result and isinstance(extract_result, dict):
                        extracted = extract_result.get('patent_ids', [])
                        if isinstance(extracted, list) and extracted:
                            import re as _re2
                            # Normalize: strip commas, US prefix, kind codes, whitespace
                            normalized = []
                            for pid in extracted:
                                pid = str(pid).strip().replace(',', '').replace('/', '')
                                # Strip "US" prefix
                                if pid.upper().startswith('US') and len(pid) > 2:
                                    pid = pid[2:]
                                # Strip kind code suffix (A1, A2, B1, B2, etc.)
                                pid = _re2.sub(r'[AB]\d$', '', pid)
                                # US application numbers are EXACTLY 8 digits
                                if pid and pid.isdigit() and len(pid) == 8:
                                    normalized.append(pid)
                            patent_ids = sorted(set(normalized))[:max_patents]
                            params['patent_source'] = extract_result.get('source', 'cnipa')
                            _pipeline_logger.info(
                                f"[task={task_id}] PHASE0 llm_extracted_patent_ids — "
                                f"count={len(patent_ids)}, "
                                f"source={params['patent_source']}, "
                                f"patent_ids={patent_ids}"
                            )
                except Exception as e:
                    _pipeline_logger.warning(
                        f"[task={task_id}] PHASE0 extract_patent_ids LLM failed: {e}"
                    )
            if patent_ids:
                total = len(patent_ids)
                pending = patent_ids

    # ==== Phase 0: Search patents via scene tools (if no patent_ids provided) ====
    if not patent_ids and scene_candidates:
        from sources.long_task.scene_tools import (
            select_tool,
            execute_tool,
            extract_patent_ids,
            extract_patent_id_url_map,
        )
        update_task_status(task_id, 'searching_patents', 0,
                           f'已发现 {len(scene_candidates)} 个场景工具，正在选择检索方案...')
        selected = await select_tool(
            'search patents', params['query'],
            scene_candidates, flash_provider,
        )
        _pipeline_logger.info(
            f"[task={task_id}] PHASE0 select_tool — "
            f"selected={selected is not None}, "
            f"tool_title={selected.get('tool', {}).title if selected else 'N/A'}, "
            f"tool_url={selected.get('tool', {}).url if selected else 'N/A'}, "
            f"reason={selected.get('reason', '') if selected else ''}"
        )
        if selected:
            update_task_status(task_id, 'searching_patents', 2,
                               f'正在检索专利：{selected.get("reason", "")}')
            result = await execute_tool(selected['tool'], selected['params'])
            raw_items = result.get('raw_items', []) or []
            patent_ids = extract_patent_ids(raw_items)
            id_url_map = extract_patent_id_url_map(raw_items)
            _pipeline_logger.info(
                f"[task={task_id}] PHASE0 search_result — "
                f"raw_items_count={len(raw_items)}, "
                f"patent_ids_found={len(patent_ids)}, "
                f"patent_ids={patent_ids[:10]}{'...' if len(patent_ids) > 10 else ''}, "
                f"id_url_map_size={len(id_url_map)}"
            )
            params['patent_source'] = _infer_source_from_tool(
                selected['tool'], params.get('patent_source', 'cnipa'),
            )
        if patent_ids:
            total = len(patent_ids)
            pending = patent_ids
            update_task_status(task_id, 'searching_patents', 5,
                               f'检索到 {len(patent_ids)} 个专利，开始分析',
                               patent_ids=patent_ids)
        else:
            set_task_failed(task_id, '未找到匹配的专利')
            return {'status': 'failed', 'task_id': task_id,
                    'error': 'No patents found matching the search criteria'}

    # ==== Phase 1: Generate columns (Flash) ====
    update_task_status(task_id, 'generating_columns', 5,
                       f'正在生成分析框架（{total} 个专利）...')
    _pipeline_logger.info(
        f"[task={task_id}] PHASE1 generate_table_columns — "
        f"query={params['query'][:100]}, patent_count={total}, "
        f"provider={flash_provider.model if hasattr(flash_provider, 'model') else 'flash'}"
    )
    columns = await generate_table_columns(
        query=params['query'],
        patent_count=total,
        provider=flash_provider,
    )
    _pipeline_logger.info(
        f"[task={task_id}] PHASE1 columns_generated — "
        f"column_count={len(columns)}, columns={columns}"
    )
    update_task_status(task_id, 'generating_columns', 5,
                       f'分析维度：{" | ".join(columns[1:4])}...',
                       table_columns=columns)
    # Update MySQL long_tasks table after phase 1
    _update_mysql_progress(task_id, 'generating_columns', 5)

    # ==== Phase 2: Per-patent download -> analyze -> summarize ====
    _pipeline_logger.info(
        f"[task={task_id}] PHASE2 START — pending_count={len(pending)}, "
        f"total={total}"
    )
    for i, patent_id in enumerate(pending):
        patent_index = len(table_rows) + 1
        _pipeline_logger.info(
            f"[task={task_id}] PHASE2 patent[{patent_index}/{total}] — "
            f"patent_id={patent_id}, "
            f"has_scene_candidates={scene_candidates is not None}, "
            f"doc_url_from_search={(id_url_map or {}).get(patent_id, '') or '(none)'}"
        )
        try:
            update_task_status(task_id, 'analyzing',
                               progress_pct(i, total),
                               f'正在下载专利文件（{patent_index}/{total}）...',
                               table_rows=table_rows)

            # Try scene tool download first, fall back to hardcoded download
            patent_text = await _download_patent_via_scene_or_fallback(
                patent_id=patent_id,
                params=params,
                scene_candidates=scene_candidates,
                flash_provider=flash_provider,
                download_patent_document=download_patent_document,
                id_url_map=id_url_map,
            )

            _pipeline_logger.info(
                f"[task={task_id}] PHASE2 patent[{patent_index}/{total}] download_done — "
                f"patent_id={patent_id}, "
                f"text_length={len(patent_text) if patent_text else 0}"
            )

            update_task_status(task_id, 'analyzing',
                               progress_pct(i, total),
                               f'正在分析（{patent_index}/{total}）：{patent_id}',
                               table_rows=table_rows)

            row = await analyze_single_patent(
                patent_id=patent_id, patent_text=patent_text,
                columns=columns, query=params['query'],
                provider=pro_provider, timeout=60,
            )
            _pipeline_logger.info(
                f"[task={task_id}] PHASE2 patent[{patent_index}/{total}] analyze_done — "
                f"patent_id={patent_id}, row_keys={list(row.keys()) if row else 'None'}"
            )

            row['_summary'] = await generate_patent_summary(
                patent_id=patent_id, row=row, query=params['query'],
                provider=pro_provider,
            )
            _pipeline_logger.info(
                f"[task={task_id}] PHASE2 patent[{patent_index}/{total}] summary_done — "
                f"patent_id={patent_id}, "
                f"summary_length={len(row.get('_summary', '')) if row.get('_summary') else 0}"
            )

        except Exception as e:
            _pipeline_logger.error(
                f"[task={task_id}] PHASE2 patent[{patent_index}/{total}] FAILED — "
                f"patent_id={patent_id}, error={e}"
            )
            row = build_failed_row(patent_id, str(e))

        table_rows.append(row)
        _pipeline_logger.info(
            f"[task={task_id}] PHASE2 table_updated — "
            f"completed={len(table_rows)}/{total}, "
            f"successful={len([r for r in table_rows if not r.get('_failed')])}, "
            f"failed={len([r for r in table_rows if r.get('_failed')])}"
        )
        save_checkpoint(task_id, {
            'completed': [r['patent_id'] for r in table_rows if not r.get('_failed')],
            'current': patent_id,
            'pending': pending[i+1:],
            'completed_rows': table_rows,
            'failed': [r['patent_id'] for r in table_rows if r.get('_failed')],
        })

        update_task_status(task_id, 'analyzing',
                           progress_pct(i + 1, total),
                           f'已完成 {len(table_rows)}/{total} 个专利分析',
                           table_rows=table_rows)
    # Update MySQL long_tasks table after phase 2
    _pipeline_logger.info(
        f"[task={task_id}] PHASE2 COMPLETE — "
        f"total_rows={len(table_rows)}, table_columns={columns}"
    )
    _update_mysql_progress(task_id, 'analyzing', 75)

    # ==== Phase 3: Generate report (Pro, dynamic) ====
    _pipeline_logger.info(
        f"[task={task_id}] PHASE3 generate_report — "
        f"columns={columns}, table_rows_count={len(table_rows)}, "
        f"provider={pro_provider.model if hasattr(pro_provider, 'model') else 'pro'}"
    )
    update_task_status(task_id, 'generating_report', 80,
                       '正在规划报告结构...')
    try:
        outline = await generate_report_outline(
            query=params['query'], columns=columns,
            table_rows=table_rows, provider=pro_provider,
        )
    except Exception as e:
        _pipeline_logger.error(
            f"[task={task_id}] PHASE3 outline FAILED — {e}, falling back to default"
        )
        outline = {
            'title': '专利分析报告',
            'sections': [{'heading': '分析结果', 'description': ''}],
        }
    _pipeline_logger.info(
        f"[task={task_id}] PHASE3 outline — "
        f"title={outline.get('title', '')}, "
        f"sections_count={len(outline.get('sections', []))}, "
        f"sections={[s.get('heading', '') for s in outline.get('sections', [])]}"
    )

    report_parts = []
    sections = outline.get('sections', [{'heading': '分析结果', 'description': ''}])
    for idx, section in enumerate(sections):
        sec_pct = 80 + int((idx + 1) / len(sections) * 10)
        update_task_status(task_id, 'generating_report', sec_pct,
                           f'正在撰写：{section["heading"]}')
        try:
            text = await generate_report_section(
                section=section, query=params['query'],
                columns=columns, table_rows=table_rows,
                provider=pro_provider,
            )
        except Exception as e:
            _pipeline_logger.error(
                f"[task={task_id}] PHASE3 section[{idx+1}/{len(sections)}] FAILED — {e}"
            )
            text = f"（{section['heading']} 生成失败）"
        report_parts.append(f"## {section['heading']}\n\n{text}")
        _pipeline_logger.info(
            f"[task={task_id}] PHASE3 section[{idx+1}/{len(sections)}] — "
            f"heading={section['heading']}, text_length={len(text)}"
        )

    report_text = f"# {outline.get('title', '专利分析报告')}\n\n" + "\n\n".join(report_parts)
    _pipeline_logger.info(
        f"[task={task_id}] PHASE3 report_text — "
        f"total_length={len(report_text)}, sections_written={len(report_parts)}"
    )
    update_task_status(task_id, 'generating_report', 90,
                       '报告撰写完成', result_summary=report_text)
    _update_mysql_progress(task_id, 'generating_report', 90)

    # ==== Phase 4: Export files ====
    from sources.long_task.storage import get_storage_config
    storage_cfg = get_storage_config()
    try:
        storage = create_storage(storage_cfg)
    except Exception as e:
        _pipeline_logger.error(
            f"[task={task_id}] PHASE4 storage_init FAILED — {e}, falling back to local"
        )
        storage = create_storage({'report_storage_backend': 'local'})
    _pipeline_logger.info(
        f"[task={task_id}] PHASE4 export — "
        f"storage_backend={storage_cfg.get('report_storage_backend', 'local')}"
    )

    report_files = []
    try:
        update_task_status(task_id, 'exporting', 92, '正在生成 PDF 文件...')
        pdf_bytes = await export_pdf_async(report_text, table_rows, columns)
        await storage.put(task_id, 'report.pdf', pdf_bytes)
        _pipeline_logger.info(
            f"[task={task_id}] PHASE4 pdf — size_bytes={len(pdf_bytes)}"
        )
        report_files.append({'format': 'pdf', 'filename': 'report.pdf', 'size': len(pdf_bytes)})
    except Exception as e:
        _pipeline_logger.error(f"[task={task_id}] PHASE4 pdf FAILED — {e}")

    try:
        update_task_status(task_id, 'exporting', 96, '正在生成 Word 文件...')
        docx_bytes = await export_docx_async(report_text, table_rows, columns)
        await storage.put(task_id, 'report.docx', docx_bytes)
        _pipeline_logger.info(
            f"[task={task_id}] PHASE4 docx — size_bytes={len(docx_bytes)}"
        )
        report_files.append({'format': 'docx', 'filename': 'report.docx', 'size': len(docx_bytes)})
    except Exception as e:
        _pipeline_logger.error(f"[task={task_id}] PHASE4 docx FAILED — {e}")

    _pipeline_logger.info(
        f"[task={task_id}] COMPLETED — "
        f"patents_analyzed={len(table_rows)}, "
        f"report_files={[f['format'] for f in report_files]}"
    )
    _update_mysql_progress(task_id, 'exporting', 100)
    set_task_completed(task_id, report_files)
    return {'status': 'completed', 'task_id': task_id}


# ── Phase 0 helpers ─────────────────────────────────────────────────────────

def _infer_source_from_tool(tool_info, default: str = 'cnipa') -> str:
    """Infer patent source (cnipa/uspto) from a tool's URL."""
    url = getattr(tool_info, 'url', '') or ''
    if 'uspto' in url.lower():
        return 'uspto'
    if 'zldsj' in url.lower():
        return 'cnipa'
    return default


async def _download_patent_via_scene_or_fallback(
    patent_id: str,
    params: dict,
    scene_candidates: list | None,
    flash_provider,
    download_patent_document,
    id_url_map: dict | None = None,
) -> str:
    """Download patent text via scene tool, direct USPTO API, or fallback."""
    doc_url = (id_url_map or {}).get(patent_id, '')
    patent_source = params.get('patent_source', 'cnipa')

    # ── Step 1: Try scene tool download ──
    if scene_candidates:
        from sources.long_task.scene_tools import select_tool, execute_tool

        context = f'patent_id={patent_id}'
        if doc_url:
            context += f', document_url={doc_url}'

        selected = await select_tool(
            'download patent document',
            context,
            scene_candidates,
            flash_provider,
        )
        if selected:
            _pipeline_logger.info(
                f"[download] scene_tool_selected — patent_id={patent_id}, "
                f"tool_title={selected.get('tool', {}).title if selected.get('tool') else 'N/A'}, "
                f"tool_url={selected.get('tool', {}).url if selected.get('tool') else 'N/A'}"
            )
            result = await execute_tool(selected['tool'], selected['params'])
            from sources.long_task.scene_tools import extract_document_text
            text = extract_document_text(result)
            _pipeline_logger.info(
                f"[download] scene_tool_result — patent_id={patent_id}, "
                f"text_found={text is not None}, "
                f"text_length={len(text) if text else 0}"
            )
            # If the result looks like a document list (JSON metadata, not patent
            # text), go to direct USPTO download to fetch the actual specification.
            is_doc_list = (
                text and len(text) > 50
                and ('"documentBag"' in text or '"documentTypeCode"' in text
                     or '"downloadOptionBag"' in text)
            )
            if text and len(text) > 50 and not is_doc_list:
                return text
            if is_doc_list:
                _pipeline_logger.info(
                    f"[download] scene_tool_returned_doc_list "
                    f"({len(text)} chars), "
                    f"trying direct USPTO API for specification"
                )
            else:
                _pipeline_logger.info(
                    f"[download] scene_tool_short_text "
                    f"({len(text) if text else 0} chars), "
                    f"trying direct USPTO API"
                )

    # ── Step 2: Direct USPTO API for US patents ──
    if patent_source == 'uspto':
        uspto_text = await _download_uspto_patent_direct(patent_id, flash_provider)
        if uspto_text and len(uspto_text) > 100:
            return uspto_text
        _pipeline_logger.info(
            f"[download] direct_uspto_failed — patent_id={patent_id}, "
            f"falling back to hardcoded download"
        )

    # ── Step 3: Hardcoded download ──
    _pipeline_logger.info(
        f"[download] fallback — patent_id={patent_id}, "
        f"patent_source={patent_source}"
    )
    return await download_patent_document(patent_id, patent_source)


async def _download_uspto_patent_direct(patent_id: str, flash_provider=None) -> str | None:
    """Download USPTO patent document text directly (two-step).

    Step 1: GET /api/v1/patent/applications/{appNumber}/documents → document list
    Step 2: LLM picks the specification → GET its download URL → return text
    """
    import asyncio
    import json as _json

    try:
        from sources.http_outbound import outbound_http

        # Normalize: strip commas, slashes, non-digits (US app numbers are pure digits)
        app_number = patent_id.strip().replace(',', '').replace('/', '')
        app_number = ''.join(c for c in app_number if c.isdigit())
        if not app_number or len(app_number) < 8:
            _pipeline_logger.warning(
                f"[download] uspto_invalid_app_number — patent_id={patent_id}"
            )
            return None
        headers = {'Accept': 'application/json'}
        uspto_key = os.getenv('USPTO_API_KEY', '')
        if uspto_key:
            headers['X-API-Key'] = uspto_key

        # Step 1: Get document list
        doc_list_url = (
            f"https://api.uspto.gov/api/v1/patent/applications/"
            f"{app_number}/documents"
        )
        _pipeline_logger.info(
            f"[download] uspto_step1 — url={doc_list_url}"
        )
        resp = await asyncio.to_thread(
            outbound_http.get, doc_list_url, purpose='patent_download',
            headers=headers, timeout=20,
        )
        if resp.status_code != 200:
            _pipeline_logger.warning(
                f"[download] uspto_step1_failed — status={resp.status_code}"
            )
            return None

        doc_list = resp.json() if resp.text else {}
        documents = (
            doc_list.get('documentBag', [])
            if isinstance(doc_list, dict)
            else []
        )
        # Log raw document list for debugging
        if documents:
            _pipeline_logger.info(
                f"[download] uspto_raw_doc[0] — {_json.dumps(documents[0], ensure_ascii=False)[:3000]}"
            )
            # Compact summary: index | typeCode | typeName | format | fileName
            doc_summary = []
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    continue
                doc_summary.append(
                    f"[{i}] code={doc.get('documentTypeCode','?')} "
                    f"type={doc.get('documentTypeName','?')} "
                    f"fmt={doc.get('documentFormat','?')} "
                    f"file={doc.get('fileName','?')}"
                )
            _pipeline_logger.info(
                f"[download] uspto_doc_summary —\n" + "\n".join(doc_summary)
            )
        _pipeline_logger.info(
            f"[download] uspto_step1_done — doc_count={len(documents)}"
        )

        if not documents:
            _pipeline_logger.warning(
                f"[download] uspto_no_documents — patent_id={app_number}"
            )
            return None

        # Helper: get download URL from a document (may be nested in downloadOptionBag)
        def _get_download_url(doc: dict) -> str:
            # Direct field
            for key in ('downloadUrl', 'documentUrl', 'url'):
                val = doc.get(key, '')
                if val:
                    return val
            # Nested in downloadOptionBag
            options = doc.get('downloadOptionBag', [])
            if isinstance(options, list) and options:
                for opt in options:
                    if isinstance(opt, dict):
                        for key in ('downloadUrl', 'url'):
                            val = opt.get(key, '')
                            if val:
                                return val
            return ''

        # Step 2: LLM picks the specification document
        spec_doc = None
        if flash_provider and len(documents) > 1:
            # Build a compact summary for the LLM
            doc_lines = []
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    continue
                doc_lines.append(_json.dumps({
                    'index': i,
                    'documentTypeCode': doc.get('documentTypeCode', ''),
                    'documentTypeName': doc.get('documentTypeName', ''),
                    'documentFormat': doc.get('documentFormat', ''),
                    'fileName': doc.get('fileName', ''),
                    'pageCount': doc.get('pageCount', ''),
                    'hasDownloadOptionBag': bool(doc.get('downloadOptionBag')),
                }, ensure_ascii=False))
            try:
                selection = await flash_provider.complete_json(
                    "You are a patent document classifier. From a list of USPTO "
                    "patent application documents, identify the specification "
                    "(说明书). The specification is typically:\n"
                    "- documentTypeCode = 'SPEC'\n"
                    "- The main detailed description of the invention\n"
                    "- Usually the longest text document, NOT drawings/figures\n"
                    "- NOT the abstract, claims-only, or search report\n\n"
                    "Return JSON: {\"selected_index\": <index of specification>, "
                    "\"reason\": \"<brief explanation>\"}",
                    f"Patent application: {app_number}\n"
                    f"Available documents:\n" + "\n".join(doc_lines),
                )
                if selection and isinstance(selection, dict):
                    idx = selection.get('selected_index')
                    if isinstance(idx, int) and 0 <= idx < len(documents):
                        spec_doc = documents[idx]
                        _pipeline_logger.info(
                            f"[download] llm_selected_spec — index={idx}, "
                            f"type={spec_doc.get('documentTypeCode')}, "
                            f"reason={selection.get('reason', '')}"
                        )
            except Exception as e:
                _pipeline_logger.warning(
                    f"[download] llm_spec_selection_failed: {e}"
                )

        # Fallback: heuristic search for SPEC document
        if not spec_doc:
            for doc in documents:
                if not isinstance(doc, dict):
                    continue
                doc_type = str(doc.get('documentTypeCode', '') or doc.get('documentType', ''))
                if 'SPEC' in doc_type.upper():
                    spec_doc = doc
                    _pipeline_logger.info(f"[download] heuristic_spec — type={doc_type}")
                    break

        # Last fallback: first TXT/XML document
        if not spec_doc:
            for doc in documents:
                if not isinstance(doc, dict):
                    continue
                fmt = str(doc.get('documentFormat', '') or doc.get('format', '')).upper()
                if 'TXT' in fmt or 'XML' in fmt:
                    spec_doc = doc
                    _pipeline_logger.info(f"[download] fallback_txt — format={fmt}")
                    break

        if not spec_doc:
            _pipeline_logger.warning(
                f"[download] uspto_no_spec_found — patent_id={app_number}"
            )
            return None

        # Step 3: Download the specification (may need redirect following)
        _pipeline_logger.info(
            f"[download] uspto_spec_selected — "
            f"code={spec_doc.get('documentTypeCode','?')}, "
            f"type={spec_doc.get('documentTypeName','?')}, "
            f"fmt={spec_doc.get('documentFormat','?')}, "
            f"file={spec_doc.get('fileName','?')}"
        )
        return await _download_uspto_spec_with_redirect(
            spec_doc, app_number, headers
        )
    except Exception as e:
        _pipeline_logger.warning(
            f"[download] uspto_direct_error — patent_id={patent_id}, error={e}"
        )
        return None


def progress_pct(completed: int, total: int) -> int:
    """Map completed/total to progress percentage in [5, 75]."""
    if total == 0:
        return 5
    return 5 + min(70, int(completed / total * 70))


async def export_pdf_async(report_text: str, table_rows: list, columns: list) -> bytes:
    """Export report as PDF using weasyprint, run in executor."""
    import asyncio
    import traceback

    def _sync_export():
        html = _build_report_html(report_text, table_rows, columns)
        try:
            import weasyprint
            return weasyprint.HTML(string=html).write_pdf()
        except Exception:
            _pipeline_logger.error(f"weasyprint PDF export failed:\n{traceback.format_exc()}")
            raise

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_export)


async def export_docx_async(report_text: str, table_rows: list, columns: list) -> bytes:
    """Export report as DOCX using python-docx, run in executor."""
    import asyncio
    import io
    from docx import Document

    def _sync_export():
        doc = Document()
        for line in report_text.split('\n'):
            if line.startswith('# '):
                doc.add_heading(line[2:], level=1)
            elif line.startswith('## '):
                doc.add_heading(line[3:], level=2)
            elif line.strip():
                doc.add_paragraph(line.strip())

        if table_rows and columns:
            table = doc.add_table(rows=1, cols=len(columns))
            table.style = 'Table Grid'
            for i, col in enumerate(columns):
                table.rows[0].cells[i].text = col
            for row_data in table_rows:
                row = table.add_row()
                for i, col in enumerate(columns):
                    row.cells[i].text = str(row_data.get(col, ''))

        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_export)


def _build_report_html(report_text: str, table_rows: list, columns: list) -> str:
    """Build HTML for PDF export."""
    import html as html_mod
    rows_html = ""
    if table_rows and columns:
        header = "<tr>" + "".join(f"<th>{html_mod.escape(str(c))}</th>" for c in columns) + "</tr>"
        body = ""
        for r in table_rows:
            body += "<tr>" + "".join(f"<td>{html_mod.escape(str(r.get(c, '')))}</td>" for c in columns) + "</tr>"
        rows_html = f"<table>{header}{body}</table>"

    text_html = report_text.replace('\n', '<br>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #ddd;padding:8px;font-size:12px;text-align:left;}}
th{{background:#f5f5f5;}}</style></head>
<body>{text_html}{rows_html}</body></html>"""


# ── USPTO download helpers ────────────────────────────────────────────────────

import re as _re

def _get_download_url_from_doc(doc: dict) -> str:
    """Extract the download URL from a USPTO documentBag entry.

    The URL may be at the top level or nested in downloadOptionBag[].downloadUrl.
    """
    for key in ('downloadUrl', 'documentUrl', 'url'):
        val = doc.get(key, '')
        if val:
            return val
    options = doc.get('downloadOptionBag', [])
    if isinstance(options, list) and options:
        for opt in options:
            if isinstance(opt, dict):
                for key in ('downloadUrl', 'url'):
                    val = opt.get(key, '')
                    if val:
                        return val
    return ''


async def _download_uspto_spec_with_redirect(
    spec_doc: dict,
    app_number: str,
    headers: dict,
) -> str | None:
    """Download USPTO specification, following redirect URLs if needed.

    USPTO download URLs may return a text/JSON body containing another URL
    (e.g. "Please use redirect URL: https://...").  We follow at most one
    redirect.  Pattern taken from uspto_download.py.
    """
    import asyncio

    from sources.http_outbound import outbound_http

    spec_url = _get_download_url_from_doc(spec_doc)
    if not spec_url:
        _pipeline_logger.warning(
            f"[download] uspto_spec_no_url — app={app_number}, "
            f"spec_doc_keys={list(spec_doc.keys()) if spec_doc else 'N/A'}"
        )
        return None
    _pipeline_logger.info(
        f"[download] uspto_spec_url_extracted — url={spec_url}"
    )

    for hop in range(2):  # max 1 redirect
        _pipeline_logger.info(
            f"[download] uspto_spec_hop{hop} — url={spec_url[:120]}"
        )
        resp = await asyncio.to_thread(
            outbound_http.get, spec_url, purpose='patent_download',
            headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            _pipeline_logger.warning(
                f"[download] uspto_spec_hop{hop}_failed — status={resp.status_code}"
            )
            return None

        content_type = resp.headers.get('Content-Type', '').lower()
        content = resp.text or ''

        # If response looks like a file (not text/JSON), return it directly
        if content_type and not any(t in content_type for t in ('text/', 'json', 'xml', 'html')):
            _pipeline_logger.info(
                f"[download] uspto_spec_binary — type={content_type}, len={len(resp.content)}"
            )
            # For binary, return the text if decodeable
            try:
                return resp.content.decode('utf-8', errors='replace')
            except Exception:
                return content

        # Check if the text response contains a redirect URL
        stripped = content.strip()
        if not stripped:
            _pipeline_logger.warning(
                f"[download] uspto_spec_empty — app={app_number}"
            )
            return None

        # Try to extract a redirect URL from the response
        from sources.dynamic_tool_params import _extract_first_url
        redirect_url = _extract_first_url(stripped)
        if redirect_url and redirect_url != spec_url:
            _pipeline_logger.info(
                f"[download] uspto_spec_redirect — to={redirect_url[:120]}"
            )
            spec_url = redirect_url
            continue

        # No redirect: this IS the content
        _pipeline_logger.info(
            f"[download] uspto_spec_done — len={len(stripped)}"
        )
        return stripped

    return None
