#!/usr/bin/env python3
"""Replace family analysis Phase 2 with pipelined concurrency (same as US pattern)."""
import sys

with open('celery_worker.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Find old Phase 2 start
old_start = source.find('# Phase 2: Per-document download')
if old_start < 0:
    old_start = source.find('# Phase 2: Sequential download + pipelined')
if old_start < 0:
    print("ERROR: Phase 2 start not found")
    sys.exit(1)

# Walk back to line start
old_start = source.rfind('═══', 0, old_start)
if old_start < 0:
    print("ERROR: Phase 2 header not found")
    sys.exit(1)
old_start = source.rfind('\n', 0, old_start) + 1

# Find Phase 2 end: look for COMPLETE log
complete_marker = 'FAMILY PHASE2 COMPLETE'
complete_idx = source.find(complete_marker, old_start + 500)
if complete_idx < 0:
    print("ERROR: COMPLETE marker not found")
    sys.exit(1)

# Find end of COMPLETE block (skip all continuation lines)
old_end = source.find('\n', complete_idx) + 1
# The COMPLETE line spans multiple lines with continuation, find where the log line ends
for _ in range(5):
    next_line = source.find('\n', old_end)
    if next_line > 0 and (source[old_end:next_line].startswith('            f"') or source[old_end:next_line].startswith('            ')):
        old_end = next_line + 1
    else:
        break

# Also find and remove the following "docs_with_text" line if it's separate
next_block = source.find('docs_with_text', old_end, old_end + 200)
if 0 < next_block < old_end + 200:
    old_end = source.find('\n', next_block) + 1

print(f"Old Phase 2: bytes {old_start} to {old_end} ({old_end - old_start} chars)")

# New Phase 2 code
new_phase2 = """        # ═════════════════════════════════════════════════════════════════
        # Phase 2: Sequential download + pipelined parallel analysis
        # ═════════════════════════════════════════════════════════════════
        # Each document is downloaded sequentially.  As soon as download
        # completes, analysis (OCR -> LLM analyze -> LLM summarize) is launched
        # immediately via asyncio.create_task().  All analysis steps run under
        # a Semaphore(100), so multiple OCR + LLM calls execute concurrently.
        # (Same pattern as execute_prosecution_analysis Phase 2.)
        total_dl = len(docs_to_download)
        _pipeline_logger.info(
            f"[task={task_id}] FAMILY PHASE2 START — "
            f"pending_count={total_dl}, total={total_dl}"
        )
        update_task_status(task_id, 'downloading', 15,
                           _t('prosecution_downloading', lang, current=0, total=total_dl))

        async def _fetch_prosecution(url: str, hdrs: dict, timeout: int):
            return await _uspto_get_with_retry(url, hdrs, timeout)

        _table_rows: list[dict] = [None] * total_dl  # preserve order by index
        _analyze_sem = asyncio.Semaphore(100)  # caps concurrent OCR + LLM calls
        _pending_tasks = []  # list of asyncio.Task
        _downloaded = 0  # documents downloaded so far
        _analyzed = 0    # documents with completed (or skipped) analysis

        async def _analyze_one(_doc: Any, _i: int) -> None:
            nonlocal _analyzed, _downloaded
            async with _analyze_sem:
                doc_index = _i + 1

                # ── Text extraction (local-first -> Vision fallback) ──
                if not _doc.text and _doc.binary:
                    from sources.long_task.text_extractor import extract_text_from_binary
                    _local_text = extract_text_from_binary(
                        _doc.binary, skip_pdf_extraction=False,
                    )
                    if _local_text and len(_local_text.strip()) > 50:
                        _doc.text = _local_text.strip()
                        _pipeline_logger.info(
                            f"[task={task_id}] FAMILY PHASE2 local_extract_ok — "
                            f"code={_doc.document_code}, idx={doc_index}/{total_dl}, "
                            f"chars={len(_doc.text)}"
                        )
                    elif vision_enabled:
                        try:
                            _text = await _extract_text_via_vision(
                                _doc.binary, _doc.description, vision_provider,
                            )
                            if _text and len(_text.strip()) > 50:
                                _doc.text = _text.strip()
                                _pipeline_logger.info(
                                    f"[task={task_id}] FAMILY PHASE2 vision_ok — "
                                    f"code={_doc.document_code}, idx={doc_index}/{total_dl}, "
                                    f"chars={len(_doc.text)}"
                                )
                        except Exception as _e:
                            _pipeline_logger.warning(
                                f"[task={task_id}] FAMILY PHASE2 vision_error — "
                                f"code={_doc.document_code}: {type(_e).__name__}: {_e}"
                            )

                if not _doc.text or len(_doc.text.strip()) < 50:
                    row = build_failed_row(_doc.document_code, "text extraction failed", columns, lang)
                    row["_failed"] = True
                    row["_summary"] = ""
                    _table_rows[_i] = row
                    _analyzed += 1
                    return

                # ── Analyze (LLM) ──
                try:
                    row = await analyze_single_document(
                        doc_text=_doc.text,
                        doc_code=_doc.document_code,
                        doc_desc=_doc.description,
                        doc_category=_doc.category,
                        columns=columns,
                        query=query,
                        provider=pro_provider,
                        lang=lang,
                    )
                except Exception as e:
                    _pipeline_logger.warning(
                        f"[task={task_id}] FAMILY PHASE2 analyze_error — "
                        f"code={_doc.document_code}, idx={doc_index}/{total_dl}: "
                        f"{type(e).__name__}: {e}"
                    )
                    row = build_failed_row(_doc.document_code, str(e), columns, lang)
                    row["_failed"] = True

                # ── Summarize (LLM) ──
                try:
                    summary = await generate_document_summary(
                        doc_text=_doc.text, row=row, query=query,
                        provider=pro_provider, lang=lang,
                    )
                except Exception as e:
                    _pipeline_logger.warning(
                        f"[task={task_id}] FAMILY PHASE2 summary_error — "
                        f"code={_doc.document_code}: {type(e).__name__}: {e}"
                    )
                    summary = ""
                row["_summary"] = summary
                _table_rows[_i] = row
                _analyzed += 1

                _p = max(_downloaded, _analyzed)
                update_task_status(
                    task_id, 'analyzing',
                    progress_pct(_p, total_dl),
                    _t('prosecution_analyzing', lang,
                       current=_analyzed, total=total_dl,
                       desc=_doc.description[:40]),
                    table_rows=[r for r in _table_rows if r is not None],
                )
                _pipeline_logger.info(
                    f"[task={task_id}] FAMILY PHASE2 analysis_done — "
                    f"code={_doc.document_code}, idx={doc_index}/{total_dl}, "
                    f"analyzed={_analyzed}/{total_dl}"
                )

        # ── Main pipeline: download sequentially, launch analysis immediately ──
        for _i, _doc in enumerate(docs_to_download):
            doc_index = _i + 1

            _result = _handle_task_stop(task_id, user_id, _downloaded, total_dl)
            if _result:
                return _result
            _result = _handle_task_pause(task_id, user_id, _downloaded, total_dl, {
                'completed': [r.get(columns[0], '') for r in _table_rows if r is not None and not r.get('_failed')],
                'pending': [d.document_code for d in docs_to_download[_i:]],
                'completed_rows': [r for r in _table_rows if r is not None],
                'failed': [r.get(columns[0], '') for r in _table_rows if r is not None and r.get('_failed')],
                'columns': columns,
                'downloaded': _downloaded,
                'analyzed': _analyzed,
            })
            if _result:
                return _result

            _visible_progress = max(_downloaded, _analyzed)
            update_task_status(
                task_id, 'downloading',
                progress_pct(_visible_progress, total_dl),
                _t('prosecution_downloading', lang, current=doc_index, total=total_dl),
            )
            await download_single_document(_doc, _fetch_prosecution, us_app_number, headers)
            _downloaded += 1

            _pipeline_logger.info(
                f"[task={task_id}] FAMILY PHASE2 download_done — "
                f"code={_doc.document_code}, idx={doc_index}/{total_dl}, "
                f"text_len={len(_doc.text) if _doc.text else 0}"
            )

            _pending_tasks.append(asyncio.create_task(_analyze_one(_doc, _i)))

        # ── Wait for all in-flight analyses to finish ──
        docs_with_text = [d for d in docs_to_download if d.text and len(d.text.strip()) >= 50]
        _pipeline_logger.info(
            f"[task={task_id}] FAMILY PHASE2 all_downloads_done — "
            f"with_text={len(docs_with_text)}/{total_dl}, "
            f"waiting_for_{len(_pending_tasks)}_analyses"
        )
        if _pending_tasks:
            await asyncio.gather(*_pending_tasks)

        # Reconstruct table_rows in original order
        table_rows = [r for r in _table_rows if r is not None]

"""

source = source[:old_start] + new_phase2 + source[old_end:]

with open('celery_worker.py', 'w', encoding='utf-8') as f:
    f.write(source)
print("Phase 2 replaced successfully")
