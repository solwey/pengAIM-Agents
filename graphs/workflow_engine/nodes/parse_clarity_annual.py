"""Parse Clarity HUD Annual Performance Report node.

Mirrors `api/src/api/clarity/parseAnnualReport.ts` + `parseSheet.ts`. Input is
the `[{name, rows}]` workbook produced by `read_from_storage` (format=xlsx),
where each row is a list of cell values. Output is the AnnualReport view
shape: `{reportTitle, dateRange, sheets: [{name, title, headers, rows}]}`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import ParseClarityAnnualConfig

_TITLE_RE = re.compile(r"^Q\d")
_DATE_PAGE_RE = re.compile(r"^\d+ / \d+$")
_DAY_PREFIX_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) ")
_NEWLINE_RE = re.compile(r"\r?\n")


def _normalize_header(value: Any) -> str | None:
    if value is None:
        return None
    return _NEWLINE_RE.sub(" ", str(value)).strip()


def _extract_metadata(rows: list[list[Any]]) -> tuple[str, str]:
    """Scan the first 10 rows of the first sheet for report title + date range."""
    report_title = "HUD Annual Performance Report"
    date_range = ""
    for row in rows[:10]:
        if not row:
            continue
        for cell in row:
            if not isinstance(cell, str):
                continue
            if "HUD Annual" in cell:
                report_title = _NEWLINE_RE.sub(" ", cell).strip()
            if "Date Range:" in cell:
                date_range = cell.replace("Date Range: ", "").strip()
    return report_title, date_range


def _parse_sheet(name: str, rows: list[list[Any]]) -> dict[str, Any]:
    """Parse one sheet — find Q\\d title, header row two below, then row records."""
    title_row_idx = -1
    title = name
    for i, row in enumerate(rows[:15]):
        if not row:
            continue
        match = next(
            (c for c in row if isinstance(c, str) and _TITLE_RE.match(c.strip())),
            None,
        )
        if match is not None:
            title = match.strip()
            title_row_idx = i
            break

    header_row_idx = title_row_idx + 2 if title_row_idx >= 0 else -1
    header_row = rows[header_row_idx] if 0 <= header_row_idx < len(rows) else None

    if not header_row:
        return {"name": name, "title": title, "headers": [], "rows": []}

    headers: list[str | None] = [_normalize_header(h) for h in header_row]

    out_rows: list[dict[str, Any]] = []
    for row in rows[header_row_idx + 1 :]:
        if not row or all(c is None for c in row):
            continue

        first = row[0]
        if isinstance(first, str):
            stripped = first.strip()
            if _DATE_PAGE_RE.match(stripped) or _DAY_PREFIX_RE.match(stripped):
                break

        record: dict[str, Any] = {}
        for c, cell in enumerate(row):
            header = headers[c] if c < len(headers) else None
            key = header if header is not None else f"col_{c}"
            if header is None and cell is None:
                continue
            record[key] = cell if cell is not None else None
        if record:
            out_rows.append(record)

    return {
        "name": name,
        "title": title,
        "headers": [h for h in headers if h is not None],
        "rows": out_rows,
    }


def _parse_clarity_annual(sheets: list[dict[str, Any]]) -> dict[str, Any]:
    if not sheets:
        return {"ok": False, "error": "no_sheets"}

    first_rows = sheets[0].get("rows") or []
    report_title, date_range = _extract_metadata(first_rows)

    parsed_sheets = [_parse_sheet(s.get("name") or "", s.get("rows") or []) for s in sheets]

    return {
        "ok": True,
        "reportTitle": report_title,
        "dateRange": date_range,
        "sheets": parsed_sheets,
    }


class ParseClarityAnnualExecutor(NodeExecutor):
    """Parse a Clarity HUD Annual Performance Report from xlsx workbook data already in state."""

    @staticmethod
    def create(config: dict[str, Any]) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
        cfg = ParseClarityAnnualConfig(**config)

        async def parse_clarity_annual_node(
            state: dict[str, Any],
            config: RunnableConfig,
        ) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            value = resolve_field(data, cfg.data_key) if cfg.data_key else None
            if value is None:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "data_key_not_found",
                    "data_key": cfg.data_key,
                }
            elif not isinstance(value, list):
                result = {
                    "ok": False,
                    "error": "invalid_input",
                    "detail": "expected list of {name, rows}",
                }
            else:
                result = _parse_clarity_annual(value)

            return {"data": {**data, cfg.response_key: result}}

        return parse_clarity_annual_node
