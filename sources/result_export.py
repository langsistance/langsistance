import csv
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone
from html import escape
from typing import Any

from sources.export_labels import uspto_field_label

DEFAULT_EXPORT_MIN_ROWS = 6
MAX_XLSX_CELL_CHARS = 32767

CSV_MIME_TYPE = "text/csv;charset=utf-8"
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Default column width for the Results sheet (accommodates CJK + Latin labels)
_DEFAULT_COL_WIDTH = 28

# ── Metadata label localizations ──────────────────────────────────────
_METADATA_LABELS: dict[str, dict[str, str]] = {
    "query_id":        {"zh": "查询ID",           "en": "Query ID"},
    "original_count":  {"zh": "原始结果数",        "en": "Original Count"},
    "exported_count":  {"zh": "导出结果数",        "en": "Exported Count"},
    "filter_applied":  {"zh": "已应用筛选",        "en": "Filter Applied"},
    "generated_at":    {"zh": "生成时间",          "en": "Generated At"},
}


def _export_min_rows() -> int:
    try:
        return max(1, int(os.getenv("RESULT_EXPORT_MIN_ROWS", DEFAULT_EXPORT_MIN_ROWS)))
    except ValueError:
        return DEFAULT_EXPORT_MIN_ROWS


# ── Column roles for the structured JSON artifact ─────────────────────────
# Closed set consumed by the frontend results view.  Unknown keys are "text".
_ROLE_SUFFIXES: list[tuple[str, str]] = [
    # (role, lowercase suffix) — first match wins, checked in order
    ("document_title", "documenttitle"),
    ("document_date", "documentdate"),
    ("application_number", "applicationnumbertext"),
    ("application_number", "applicationnumber"),
    ("application_number", "application_number"),
    ("patent_id", "patentnumber"),
    ("patent_id", "publicationnumber"),
    ("assignee", "assigneeentityname"),
    ("assignee", "assignee"),
    ("assignee", "applicant"),
    ("inventors", "inventorname"),
    ("inventors", "inventors"),
    ("filing_date", "filingdate"),
    ("filing_date", "applicationdate"),
    ("publication_date", "publicationdate"),
    ("publication_date", "grantdate"),
    ("ipc", "ipcclass"),
    ("ipc", "cpcclass"),
    ("ipc", "ipc"),
    ("abstract", "abstracttext"),
    ("abstract", "abstract"),
    ("title", "patenttitle"),
    ("title", "inventiontitle"),
    ("title", "title"),
    ("publication_number", "earliestpublicationnumber"),
    ("publication_number", "pctpublicationnumber"),
    ("url", "pdfurl"),
    ("url", "downloadurl"),
    ("url", "download_url"),
    ("url", "document_url"),
    ("url", "url"),
]


def infer_column_role(key: str) -> str:
    """Map a flattened result column key to a frontend rendering role.

    Keys may carry prefixes (e.g. ``applicationMetaData.patentTitle``) —
    only the last path segment is compared.  Unknown keys map to ``text``.
    """
    segment = str(key or "").lower().rsplit(".", 1)[-1].strip()
    for role, suffix in _ROLE_SUFFIXES:
        if segment == suffix:
            return role
    return "text"


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _flatten_value(prefix: str, value: Any, row: dict[str, str]) -> None:
    if isinstance(value, dict):
        if not value and prefix:
            row[prefix] = "{}"
            return
        for key, nested_value in value.items():
            nested_key = f"{prefix}.{key}" if prefix else str(key)
            _flatten_value(nested_key, nested_value, row)
        return

    if isinstance(value, list):
        if prefix:
            row[prefix] = _stringify_cell(value)
        else:
            row["value"] = _stringify_cell(value)
        return

    row[prefix or "value"] = _stringify_cell(value)


def normalize_result_rows(items: list[Any]) -> tuple[list[str], list[dict[str, str]]]:
    columns: list[str] = []
    seen_columns = set()
    rows: list[dict[str, str]] = []

    for item in items:
        row: dict[str, str] = {}
        if isinstance(item, dict):
            _flatten_value("", item, row)
        else:
            row["value"] = _stringify_cell(item)

        for column in row:
            if column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)
        rows.append(row)

    return columns, rows


def build_csv_bytes(columns: list[str], rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue().encode("utf-8-sig")


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell_text(value: Any) -> str:
    text = _stringify_cell(value)
    if len(text) <= MAX_XLSX_CELL_CHARS:
        return text
    suffix = "... [truncated]"
    return text[:MAX_XLSX_CELL_CHARS - len(suffix)] + suffix


def _cell_xml(row_index: int, column_index: int, value: Any, *, style_id: int | None = None) -> str:
    ref = f"{_column_name(column_index)}{row_index}"
    text = escape(_xlsx_cell_text(value), quote=False)
    style_attr = f' s="{style_id}"' if style_id is not None else ""
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


# ── OOXML styles.xml ──────────────────────────────────────────────────
# Style 0 = normal, Style 1 = bold + light-blue fill (header row)
def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts>'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills>'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill><fgColor rgb="FFDDEBF7"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '</cellXfs>'
        '</styleSheet>'
    )


def _sheet_view_xml(freeze_row: bool = False) -> str:
    """Return a ``<sheetViews>`` block, optionally freezing the first row."""
    if not freeze_row:
        return ""
    return (
        '<sheetViews>'
        '<sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView>'
        '</sheetViews>'
    )


def _cols_xml(column_count: int, width: int = _DEFAULT_COL_WIDTH) -> str:
    """Return a ``<cols>`` block setting uniform column widths."""
    if column_count < 1:
        return ""
    return (
        '<cols>'
        f'<col min="1" max="{column_count}" width="{width}" customWidth="1"/>'
        '</cols>'
    )


def _worksheet_xml(
    columns: list[str],
    rows: list[dict[str, str]],
    *,
    lang: str = "zh",
    header_style_id: int | None = None,
    freeze_header: bool = False,
    column_width: int = _DEFAULT_COL_WIDTH,
    localize_headers: bool = True,
) -> str:
    """Build a worksheet XML string with optional formatting.

    Args:
        columns: Header column names (raw API field paths or pre-localized).
        rows: Data rows keyed by *columns* values.
        lang: Language for header localization (``"zh"`` / ``"en"``).
        header_style_id: XF style index applied to the header row, or ``None``.
        freeze_header: If ``True``, freeze the first row via ``<sheetViews>``.
        column_width: Uniform column width, or 0 to omit the ``<cols>`` block.
        localize_headers: If ``True``, translate *columns* via
            :func:`uspto_field_label` before writing them as headers.
    """
    # Localize header display text (row lookups still use raw column keys)
    display_columns = [
        uspto_field_label(col, lang) if localize_headers else col
        for col in columns
    ]

    xml_parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
    ]
    if freeze_header:
        xml_parts.append(_sheet_view_xml(freeze_row=True))
    if column_width > 0 and columns:
        xml_parts.append(_cols_xml(len(columns), column_width))

    xml_parts.append('<sheetData>')

    # Header row (r=1)
    header_cells = "".join(
        _cell_xml(1, column_index, display_name, style_id=header_style_id)
        for column_index, display_name in enumerate(display_columns, start=1)
    )
    xml_parts.append(f'<row r="1">{header_cells}</row>')

    # Data rows (r=2 .. n+1)
    for row_offset, row in enumerate(rows, start=2):
        cells = "".join(
            _cell_xml(row_offset, column_index, row.get(column, ""))
            for column_index, column in enumerate(columns, start=1)
        )
        xml_parts.append(f'<row r="{row_offset}">{cells}</row>')

    xml_parts.append('</sheetData>')
    xml_parts.append('</worksheet>')
    return "".join(xml_parts)


# ── Instructions sheet ────────────────────────────────────────────────
def _instructions_rows(lang: str) -> list[dict[str, str]]:
    """Return row data for the localized instructions sheet."""
    if lang == "en":
        return [
            {"A": "How to Use This File",
             "B": "",
             "C": "CopiioAI Patent Search Results"},
            {"A": "", "B": "", "C": ""},
            {"A": "This workbook contains 3 sheets:",
             "B": "", "C": ""},
            {"A": "1. Instructions (this sheet)",
             "B": "Usage guide and field descriptions.",
             "C": ""},
            {"A": '2. "Results" sheet',
             "B": "Patent search results — one patent per row, key fields as columns.",
             "C": ""},
            {"A": '3. "Metadata" sheet',
             "B": "Query statistics (search scope, result count, timestamp).",
             "C": ""},
            {"A": "", "B": "", "C": ""},
            {"A": "Tips:",
             "B": "", "C": ""},
            {"A": "• Some cells contain JSON data (e.g. event history, assignments).",
             "B": 'Enable "Wrap Text" and adjust row height to view full content.',
             "C": ""},
            {"A": "• Use Excel AutoFilter (Data → Filter) to sort and filter by any column.",
             "B": "", "C": ""},
            {"A": "• Patent numbers and application numbers can be searched on Google Patents or USPTO.",
             "B": "", "C": ""},
        ]
    return [
        {"A": "使用说明",
         "B": "",
         "C": "CopiioAI 专利检索结果"},
        {"A": "", "B": "", "C": ""},
        {"A": "本工作簿包含 3 个工作表：",
         "B": "", "C": ""},
        {"A": "1. 使用说明（本工作表）",
         "B": "引导说明和字段描述。",
         "C": ""},
        {"A": '2. "Results" 结果工作表',
         "B": "专利检索结果——每行一条专利，关键字段按列排列。",
         "C": ""},
        {"A": '3. "Metadata" 元数据工作表',
         "B": "查询统计信息（搜索范围、结果数量、生成时间）。",
         "C": ""},
        {"A": "", "B": "", "C": ""},
        {"A": "使用提示：",
         "B": "", "C": ""},
        {"A": "• 部分单元格包含 JSON 数据（如事件记录、转让信息等）。",
         "B": '请开启"自动换行"并调整行高以查看完整内容。',
         "C": ""},
        {"A": "• 可使用 Excel 自动筛选（数据 → 筛选）按任意列排序和过滤。",
         "B": "", "C": ""},
        {"A": "• 专利号和申请号可在 Google Patents 或 USPTO 官网上搜索查看详情。",
         "B": "", "C": ""},
    ]


def _instructions_sheet_xml(lang: str) -> str:
    """Build the instructions sheet XML (no header formatting, no freeze)."""
    rows = _instructions_rows(lang)
    xml_parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<cols>'
        '<col min="1" max="1" width="42" customWidth="1"/>'
        '<col min="2" max="2" width="60" customWidth="1"/>'
        '<col min="3" max="3" width="40" customWidth="1"/>'
        '</cols>',
        '<sheetData>',
    ]
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, col_key in enumerate(["A", "B", "C"], start=1):
            val = row.get(col_key, "")
            # Bold the first row (title)
            sid = 1 if row_idx == 1 else None
            cells.append(_cell_xml(row_idx, col_idx, val, style_id=sid))
        xml_parts.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    xml_parts.append('</sheetData>')
    xml_parts.append('</worksheet>')
    return "".join(xml_parts)


# ── Main XLSX builder ─────────────────────────────────────────────────
def build_xlsx_bytes(
    columns: list[str],
    rows: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
    *,
    lang: str = "zh",
) -> bytes:
    """Build an XLSX workbook as raw bytes (hand-written OOXML, no dependencies).

    The workbook contains three sheets in order:
    1. **Instructions** (localized usage guide)
    2. **Results** (the patent data with localized headers)
    3. **Metadata** (query statistics with localized labels)

    Args:
        columns: Raw API field paths for the Results sheet.
        rows: Data rows keyed by *columns*.
        metadata: Optional dict of query metadata.
        lang: ``"zh"`` or ``"en"`` — controls header labels, instructions,
            and metadata captions.
    """
    # Normalize lang
    lang = lang.split("-")[0].lower() if lang else "zh"
    if lang not in ("zh", "en"):
        lang = "zh"

    # Localized metadata column names
    md_key_col = "字段" if lang == "zh" else "Key"
    md_val_col = "值" if lang == "zh" else "Value"

    # Localize metadata rows (dict keys must match the column names above)
    metadata_rows = []
    for key, value in (metadata or {}).items():
        m_label = _METADATA_LABELS.get(key, {}).get(lang, key)
        metadata_rows.append({md_key_col: m_label, md_val_col: _stringify_cell(value)})

    # ── Content Types ─────────────────────────────────────────────────
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet2.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet3.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )

    # ── Root relationships ────────────────────────────────────────────
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    # ── Workbook (sheets ordered: Instructions → Results → Metadata) ──
    instr_sheet_name = "使用说明" if lang == "zh" else "Instructions"
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        f'<sheet name="{instr_sheet_name}" sheetId="3" r:id="rId3"/>'
        '<sheet name="Results" sheetId="1" r:id="rId1"/>'
        '<sheet name="Metadata" sheetId="2" r:id="rId2"/>'
        '</sheets>'
        '</workbook>'
    )

    # ── Workbook relationships ────────────────────────────────────────
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet3.xml"/>'
        '<Relationship Id="rId4" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )

    # ── Assemble the ZIP ──────────────────────────────────────────────
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", _styles_xml())

        # Sheet 1 — Results (with localized headers + formatting)
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            _worksheet_xml(
                columns,
                rows,
                lang=lang,
                header_style_id=1,
                freeze_header=True,
                column_width=_DEFAULT_COL_WIDTH,
                localize_headers=True,
            ),
        )

        # Sheet 2 — Metadata (no header formatting, no localization beyond labels)
        archive.writestr(
            "xl/worksheets/sheet2.xml",
            _worksheet_xml(
                [md_key_col, md_val_col],
                metadata_rows,
                lang=lang,
                header_style_id=1,
                freeze_header=False,
                column_width=24,
                localize_headers=False,  # column names already localized
            ),
        )

        # Sheet 3 — Instructions (listed first in workbook order)
        archive.writestr("xl/worksheets/sheet3.xml", _instructions_sheet_xml(lang))

    return buffer.getvalue()


# ── Public API ────────────────────────────────────────────────────────
def build_result_artifacts(
    items: list[Any],
    *,
    source: str = "uspto",
    query_id: str | None = None,
    original_count: int | None = None,
    filter_applied: bool = False,
    generated_at: datetime | None = None,
    lang: str = "zh",
) -> list[dict[str, Any]]:
    """Build CSV, XLSX and structured JSON artifacts from result items.

    ``source`` is one of ``uspto`` / ``google_patents`` / ``uspto_documents``
    and rides along in the JSON payload so the frontend can drive per-row
    detail actions.

    Args:
        items: List of patent result dicts (raw USPTO API format).
        source: Data source identifier carried into the JSON payload.
        query_id: Optional query identifier.
        original_count: Total result count before filtering.
        filter_applied: Whether a filter was applied to the results.
        generated_at: Timestamp for the export (defaults to now).
        lang: ``"zh"`` for Chinese labels, ``"en"`` for English.
            Follows the user's query language.

    Returns:
        List of artifact dicts (CSV + XLSX + JSON) ready for SSE delivery.
        Returns an empty list when the result count is too small.
    """
    exported_count = len(items)
    source_count = original_count if original_count is not None else exported_count
    if source_count < _export_min_rows() or exported_count == 0:
        return []

    # Normalize lang
    lang = lang.split("-")[0].lower() if lang else "zh"
    if lang not in ("zh", "en"):
        lang = "zh"

    columns, rows = normalize_result_rows(items)
    generated_at = generated_at or datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    base_name = f"CopiioAI_Result_{timestamp}"
    metadata = {
        "query_id": query_id or "",
        "original_count": source_count,
        "exported_count": exported_count,
        "filter_applied": filter_applied,
        "generated_at": generated_at.isoformat(),
    }

    csv_content = build_csv_bytes(columns, rows)
    xlsx_content = build_xlsx_bytes(columns, rows, metadata, lang=lang)

    json_payload = {
        "source": source,
        "columns": [
            {
                "key": col,
                "label": uspto_field_label(col, lang),
                "role": infer_column_role(col),
            }
            for col in columns
        ],
        "rows": rows,
    }
    json_content = json.dumps(
        json_payload, ensure_ascii=False,
    ).encode("utf-8")

    common = {
        "row_count": exported_count,
        "column_count": len(columns),
    }
    return [
        {
            **common,
            "artifact_id": f"{uuid.uuid4().hex}-csv",
            "format": "csv",
            "filename": f"{base_name}.csv",
            "mime_type": CSV_MIME_TYPE,
            "content": csv_content,
        },
        {
            **common,
            "artifact_id": f"{uuid.uuid4().hex}-xlsx",
            "format": "xlsx",
            "filename": f"{base_name}.xlsx",
            "mime_type": XLSX_MIME_TYPE,
            "content": xlsx_content,
        },
        {
            **common,
            "artifact_id": f"{uuid.uuid4().hex}-json",
            "format": "json",
            "filename": f"{base_name}.json",
            "mime_type": "application/json",
            "content": json_content,
        },
    ]
