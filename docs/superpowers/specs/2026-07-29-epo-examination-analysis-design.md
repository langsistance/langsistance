# EPO Examination History Analysis

**Date:** 2026-07-29
**Status:** Approved
**Type:** Feature — Multi-jurisdiction examination analysis

## Summary

Add European Patent Office (EPO) examination history analysis to the existing multi-jurisdiction patent prosecution system, achieving depth comparable to USPTO analysis where data permits. Integrates into the cross-jurisdiction family analysis pipeline alongside US, CN, and JP.

## Data Sources

Two EPO OPS API paths, both using existing OAuth2 Client Credentials:

| Path | Endpoint | Purpose |
|------|----------|---------|
| Register — biblio | `/register/application/epodoc/{app}/biblio` | Bibliographic data |
| Register — events | `/register/application/epodoc/{app}/events` | Legal event timeline |
| Register — procedural | `/register/application/epodoc/{app}/procedural-steps` | Examination procedure steps |
| Published Data | `/published-data/publication/epodoc/{pub}/biblio` | Find search report publication (A1/A3) |
| Published Data | `/published-data/publication/epodoc/{pub}/fulltext` | Search Opinion / Written Opinion full text |

**Key constraint:** Communication of Examining Division and Applicant Responses have NO full-text API (only procedural metadata). Search Opinion/Written Opinion (published A1/A3) DO have full-text.

## Analysis Pipeline (4 Phases)

### Phase 0 — Resolution & Data Fetching
- Input patent ID → EPO Family API → extract EP member → app_number + pub_number
- Parallel fetch: biblio, events, procedural-steps, search report full-text

### Phase 1 — Document Analysis (full-text items)
- Search Opinion / Written Opinion text → Flash LLM column generation → Pro LLM per-section analysis
- Covers: novelty assessment, inventive step, cited prior art, examiner's preliminary stance

### Phase 2 — Procedural Timeline Analysis (metadata items)
- From procedural-steps + events → build chronological timeline
- AI inference: search→examination→response→grant/refusal, duration per step, PACE acceleration, opposition

### Phase 3 — Conclusion & Integration
- Final status (granted/refused/pending), allowed claims count, scope changes
- Data fed into `generate_family_prosecution_report()` as `ep_exam_data`

## Files

| File | Action |
|------|--------|
| `sources/epo_ops_client.py` | **NEW** — EPO OPS general client (OAuth2, Register, Published Data) — refactored from `patent_family.py` |
| `sources/long_task/epo_examination.py` | **NEW** — EPO examination data fetching + AI analysis pipeline |
| `celery_worker.py` | **MODIFY** — Add `execute_epo_examination_analysis` task; add EPO Phase 0.x in `execute_family_analysis` |
| `prosecution_analyzer.py` | **MODIFY** — Extend `generate_family_prosecution_report()` with `ep_exam_data` parameter |
| `api_routes/core.py` | **MODIFY** — Add `epo_prosecution` scenario routing |
| `config.ini` | **MODIFY** — Add `[EPO]` section if needed (reuses `[FAMILY]` OAuth credentials) |

## Integration

EPO analysis slots into the existing `execute_family_analysis` flow between JP (Phase 0.4) and USPTO (Phase 0.5), and outputs feed into `generate_family_prosecution_report` alongside US/CN/JP data.
