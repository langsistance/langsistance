# Non-Final Rejection Patent Finder

**Date:** 2026-07-29
**Status:** Approved
**Type:** Script

## Goal

Find 10 US granted patents from major tech companies that went through a full Non-Final Rejection cycle (CTNF → Amendment → NOA), outputting application numbers, grant numbers, company names, titles, and grant dates.

## Companies

Apple, Tesla, NVIDIA, SpaceX, Samsung, Qualcomm

## Date Range

Granted within the last 5 years (2021-07-29 to 2026-07-29)

## Pipeline

1. **Search** — POST `api.uspto.gov/api/v1/patent/applications/search` per company, filtering by assignee + grant status + date range
2. **Get documents** — GET `api.uspto.gov/api/v1/patent/applications/{appNumber}/documents` per patent
3. **Classify** — Reuse `prosecution_downloader._classify_single_document()` to check for CTNF + Amendment (CLM/WCLM or description match) + NOA
4. **Accumulate** — Stop when 10 matches found
5. **Output** — Table to stdout with company, title, grant date, app number, grant number

## Reused Modules

- `sources/http_outbound.py` — USPTO HTTP client with retry + rate limiting
- `sources/long_task/prosecution_downloader.py` — `_classify_single_document()` for document type matching
- `sources/dynamic_tool_params.py` — `USPTO_DOWNLOAD_API_PREFIX`
- Env: `USPTO_API_KEY`

## Output

Tabular format to stdout with all 5 fields per match.

## Location

`scripts/find_non_final_rejection_patents.py`
