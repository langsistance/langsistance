import csv
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone
from html import escape
from typing import Any


DEFAULT_EXPORT_MIN_ROWS = 6
MAX_XLSX_CELL_CHARS = 32767

CSV_MIME_TYPE = "text/csv;charset=utf-8"
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _export_min_rows() -> int:
    try:
        return max(1, int(os.getenv("RESULT_EXPORT_MIN_ROWS", DEFAULT_EXPORT_MIN_ROWS)))
    except ValueError:
        return DEFAULT_EXPORT_MIN_ROWS


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


def _cell_xml(row_index: int, column_index: int, value: Any) -> str:
    ref = f"{_column_name(column_index)}{row_index}"
    text = escape(_xlsx_cell_text(value), quote=False)
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _worksheet_xml(columns: list[str], rows: list[dict[str, str]]) -> str:
    xml_rows = []
    header_cells = "".join(
        _cell_xml(1, column_index, column)
        for column_index, column in enumerate(columns, start=1)
    )
    xml_rows.append(f'<row r="1">{header_cells}</row>')

    for row_offset, row in enumerate(rows, start=2):
        cells = "".join(
            _cell_xml(row_offset, column_index, row.get(column, ""))
            for column_index, column in enumerate(columns, start=1)
        )
        xml_rows.append(f'<row r="{row_offset}">{cells}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def build_xlsx_bytes(
    columns: list[str],
    rows: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> bytes:
    metadata_rows = [
        {"key": key, "value": _stringify_cell(value)}
        for key, value in (metadata or {}).items()
    ]

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet2.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        '<sheet name="Results" sheetId="1" r:id="rId1"/>'
        '<sheet name="Metadata" sheetId="2" r:id="rId2"/>'
        '</sheets>'
        '</workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/>'
        '</Relationships>'
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", _worksheet_xml(columns, rows))
        archive.writestr(
            "xl/worksheets/sheet2.xml",
            _worksheet_xml(["key", "value"], metadata_rows),
        )
    return buffer.getvalue()


def build_result_artifacts(
    items: list[Any],
    *,
    query_id: str | None = None,
    original_count: int | None = None,
    filter_applied: bool = False,
    generated_at: datetime | None = None,
) -> list[dict[str, Any]]:
    exported_count = len(items)
    source_count = original_count if original_count is not None else exported_count
    if source_count < _export_min_rows() or exported_count == 0:
        return []

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
    xlsx_content = build_xlsx_bytes(columns, rows, metadata)

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
    ]
