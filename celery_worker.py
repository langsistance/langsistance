"""Celery worker entry point for long-running patent analysis tasks."""

import asyncio
import os
from celery import Celery

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

    # ---- Provider setup ----
    if model_family == 'minimax':
        flash_provider = Provider(provider_name='minimax', model='minimax-2.7-highspeed',
                                  server_address='', is_local=False)
        pro_provider = Provider(provider_name='minimax', model='minimax-2.7-highspeed',
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
            return result
        finally:
            loop.close()
    except Exception as e:
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
    id_url_map = {}          # patent_id → document_url from search results

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
        if selected:
            update_task_status(task_id, 'searching_patents', 2,
                               f'正在检索专利：{selected.get("reason", "")}')
            result = await execute_tool(selected['tool'], selected['params'])
            raw_items = result.get('raw_items', []) or []
            patent_ids = extract_patent_ids(raw_items)
            id_url_map = extract_patent_id_url_map(raw_items)
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
    columns = await generate_table_columns(
        query=params['query'],
        patent_count=total,
        provider=flash_provider,
    )
    update_task_status(task_id, 'generating_columns', 5,
                       f'分析维度：{" | ".join(columns[1:4])}...',
                       table_columns=columns)
    # Update MySQL long_tasks table after phase 1
    _update_mysql_progress(task_id, 'generating_columns', 5)

    # ==== Phase 2: Per-patent download -> analyze -> summarize ====
    for i, patent_id in enumerate(pending):
        try:
            update_task_status(task_id, 'analyzing',
                               progress_pct(i, total),
                               f'正在下载专利文件（{len(table_rows)+1}/{total}）...',
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

            update_task_status(task_id, 'analyzing',
                               progress_pct(i, total),
                               f'正在分析（{len(table_rows)+1}/{total}）：{patent_id}',
                               table_rows=table_rows)

            row = await analyze_single_patent(
                patent_id=patent_id, patent_text=patent_text,
                columns=columns, query=params['query'],
                provider=pro_provider, timeout=60,
            )
            row['_summary'] = await generate_patent_summary(
                patent_id=patent_id, row=row, query=params['query'],
                provider=pro_provider,
            )

        except Exception as e:
            row = build_failed_row(patent_id, str(e))

        table_rows.append(row)
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
    _update_mysql_progress(task_id, 'analyzing', 75)

    # ==== Phase 3: Generate report (Pro, dynamic) ====
    update_task_status(task_id, 'generating_report', 80,
                       '正在规划报告结构...')
    outline = await generate_report_outline(
        query=params['query'], columns=columns,
        table_rows=table_rows, provider=pro_provider,
    )

    report_parts = []
    sections = outline.get('sections', [{'heading': '分析结果', 'description': ''}])
    for idx, section in enumerate(sections):
        sec_pct = 80 + int((idx + 1) / len(sections) * 10)
        update_task_status(task_id, 'generating_report', sec_pct,
                           f'正在撰写：{section["heading"]}')
        text = await generate_report_section(
            section=section, query=params['query'],
            columns=columns, table_rows=table_rows,
            provider=pro_provider,
        )
        report_parts.append(f"## {section['heading']}\n\n{text}")

    report_text = f"# {outline.get('title', '专利分析报告')}\n\n" + "\n\n".join(report_parts)
    update_task_status(task_id, 'generating_report', 90,
                       '报告撰写完成', result_summary=report_text)
    _update_mysql_progress(task_id, 'generating_report', 90)

    # ==== Phase 4: Export files ====
    from sources.long_task.storage import get_storage_config
    storage_cfg = get_storage_config()
    storage = create_storage(storage_cfg)

    update_task_status(task_id, 'exporting', 92, '正在生成 PDF 文件...')
    pdf_bytes = await export_pdf_async(report_text, table_rows, columns)
    await storage.put(task_id, 'report.pdf', pdf_bytes)

    update_task_status(task_id, 'exporting', 96, '正在生成 Word 文件...')
    docx_bytes = await export_docx_async(report_text, table_rows, columns)
    await storage.put(task_id, 'report.docx', docx_bytes)

    report_files = [
        {'format': 'pdf', 'filename': 'report.pdf', 'size': len(pdf_bytes)},
        {'format': 'docx', 'filename': 'report.docx', 'size': len(docx_bytes)},
    ]
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
    """Download patent text via scene tool or fall back to hardcoded download.

    Phase 0 search results include a document URL — if available, the
    download tool fetches it directly.  Otherwise the Flash LLM selects
    a scene download tool by patent_id.  Falls back to the hardcoded
    download_patent_document() as last resort.
    """
    doc_url = (id_url_map or {}).get(patent_id, '')

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
            result = await execute_tool(selected['tool'], selected['params'])
            from sources.long_task.scene_tools import extract_document_text
            text = extract_document_text(result)
            if text:
                return text

    # Fallback: existing hardcoded download
    patent_source = params.get('patent_source', 'cnipa')
    return await download_patent_document(patent_id, patent_source)


def progress_pct(completed: int, total: int) -> int:
    """Map completed/total to progress percentage in [5, 75]."""
    if total == 0:
        return 5
    return 5 + min(70, int(completed / total * 70))


async def export_pdf_async(report_text: str, table_rows: list, columns: list) -> bytes:
    """Export report as PDF using weasyprint, run in executor."""
    import asyncio
    import weasyprint

    def _sync_export():
        html = _build_report_html(report_text, table_rows, columns)
        return weasyprint.HTML(string=html).write_pdf()

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
