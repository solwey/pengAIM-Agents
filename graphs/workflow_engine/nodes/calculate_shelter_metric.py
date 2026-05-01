"""Calculate Shelter Metric node.

Operates on the AnnualReport shape produced by `parse_clarity_annual`
(``{reportTitle, dateRange, sheets:[{name,title,headers,rows}]}``) and
dispatches to a metric-specific aggregator. Mirrors the HMISights
``api/src/api/metrics/shelter/*`` route logic without the file I/O —
input must already be in workflow state.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import CalculateShelterMetricConfig


def _find_sheet(sheets: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    target = name.lower()
    return next((s for s in sheets if str(s.get("name") or "").lower() == target), None)


def _num(value: Any) -> float | int:
    """Coerce a cell value to a number — None / non-numeric strings → 0."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        try:
            stripped = value.replace(",", "").strip()
            return float(stripped) if "." in stripped else int(stripped)
        except ValueError:
            return 0
    return 0


def _length_of_stay(report: dict[str, Any]) -> dict[str, Any]:
    """Q22a1 — participation buckets (leavers vs stayers)."""
    sheet = _find_sheet(report.get("sheets") or [], "Q22a1")
    if not sheet:
        return {"ok": False, "error": "sheet_not_found", "sheet": "Q22a1"}

    rows = sheet.get("rows") or []
    buckets = [
        {
            "bucket": str(r.get("col_0")),
            "total": _num(r.get("Total")),
            "leavers": _num(r.get("Leavers")),
            "stayers": _num(r.get("Stayers")),
        }
        for r in rows
        if r.get("col_0") and r.get("col_0") != "Total"
    ]
    total_row = next((r for r in rows if r.get("col_0") == "Total"), None)
    return {
        "ok": True,
        "buckets": buckets,
        "totals": {
            "total": _num(total_row.get("Total")) if total_row else 0,
            "leavers": _num(total_row.get("Leavers")) if total_row else 0,
            "stayers": _num(total_row.get("Stayers")) if total_row else 0,
        },
    }


def _persons_served(report: dict[str, Any]) -> dict[str, Any]:
    """Q5a validation summary + Q7a household breakdown."""
    sheets = report.get("sheets") or []
    q5a = _find_sheet(sheets, "Q5a")
    q7a = _find_sheet(sheets, "Q7a")

    summary: dict[str, Any] = {}
    if q5a:
        for row in q5a.get("rows") or []:
            cat = row.get("Category")
            count = row.get("Count of Clients")
            if isinstance(cat, str) and count is not None:
                summary[cat] = _num(count)

    households = []
    if q7a:
        households = [
            {
                "category": str(r.get("col_0")),
                "total": _num(r.get("Total")),
                "withoutChildren": _num(r.get("Without Children")),
                "withChildrenAndAdults": _num(r.get("With Children and Adults")),
                "withOnlyChildren": _num(r.get("With Only Children")),
            }
            for r in (q7a.get("rows") or [])
            if r.get("col_0")
        ]

    return {
        "ok": True,
        "dateRange": report.get("dateRange", ""),
        "summary": summary,
        "households": households,
    }


def _parse_exit_categories(sheet: dict[str, Any]) -> list[dict[str, Any]]:
    """Group Q23c rows into category → destinations + subtotal (header rows have null Total)."""
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for row in sheet.get("rows") or []:
        label = row.get("col_0")
        if not isinstance(label, str):
            continue

        if row.get("Total") is None:
            current = {"category": label, "destinations": [], "subtotal": 0}
            groups.append(current)
            continue

        if label == "Subtotal" and current is not None:
            current["subtotal"] = _num(row.get("Total"))
            continue

        if label == "Total":
            continue

        total = _num(row.get("Total"))
        if current is not None and total > 0:
            current["destinations"].append({"destination": label, "total": total})

    return groups


def _exit_destinations(report: dict[str, Any]) -> dict[str, Any]:
    """Q23c flat list of destinations — uses category grouping to avoid double-counting subtotals."""
    sheet = _find_sheet(report.get("sheets") or [], "Q23c")
    if not sheet:
        return {"ok": False, "error": "sheet_not_found", "sheet": "Q23c"}

    groups = _parse_exit_categories(sheet)
    destinations = [d for g in groups for d in g["destinations"]]
    total = sum(g["subtotal"] for g in groups)

    return {
        "ok": True,
        "dateRange": report.get("dateRange", ""),
        "destinations": destinations,
        "groups": groups,
        "total": total,
    }


def _bed_availability(report: dict[str, Any]) -> dict[str, Any]:
    """Q7a/Q7b/Q8a/Q8b — point-in-time persons + households snapshots."""
    sheets = report.get("sheets") or []
    q7a = _find_sheet(sheets, "Q7a")
    q7b = _find_sheet(sheets, "Q7b")
    q8a = _find_sheet(sheets, "Q8a")
    q8b = _find_sheet(sheets, "Q8b")

    total_row_q7a = next((r for r in (q7a.get("rows") or []) if r.get("col_0") == "Total"), None) if q7a else None
    persons_served = {
        "total": _num(total_row_q7a.get("Total")) if total_row_q7a else 0,
        "withoutChildren": _num(total_row_q7a.get("Without Children")) if total_row_q7a else 0,
        "withChildrenAndAdults": _num(total_row_q7a.get("With Children and Adults")) if total_row_q7a else 0,
    }

    person_snapshots = []
    if q7b:
        for r in q7b.get("rows") or []:
            if not r.get("col_0"):
                continue
            total = _num(r.get("Total"))
            if total <= 0:
                continue
            person_snapshots.append(
                {
                    "quarter": str(r.get("col_0")),
                    "total": total,
                    "withoutChildren": _num(r.get("Without Children")),
                    "withChildrenAndAdults": _num(r.get("With Children and Adults")),
                }
            )

    hh_total_row = (
        next(
            (r for r in (q8a.get("rows") or []) if isinstance(r.get("col_0"), str) and r["col_0"].startswith("Total")),
            None,
        )
        if q8a
        else None
    )
    households_served = {
        "total": _num(hh_total_row.get("Total")) if hh_total_row else 0,
        "withoutChildren": _num(hh_total_row.get("Without Children")) if hh_total_row else 0,
        "withChildrenAndAdults": _num(hh_total_row.get("With Children and Adults")) if hh_total_row else 0,
    }

    household_snapshots = []
    if q8b:
        for r in q8b.get("rows") or []:
            if not r.get("col_0"):
                continue
            total = _num(r.get("Total"))
            if total <= 0:
                continue
            household_snapshots.append(
                {
                    "quarter": str(r.get("col_0")),
                    "total": total,
                    "withoutChildren": _num(r.get("Without Children")),
                    "withChildrenAndAdults": _num(r.get("With Children and Adults")),
                }
            )

    latest_person = person_snapshots[-1] if person_snapshots else None
    latest_household_total = household_snapshots[-1]["total"] if household_snapshots else 0

    return {
        "ok": True,
        "dateRange": report.get("dateRange", ""),
        "personsServed": persons_served,
        "householdsServed": households_served,
        "personSnapshots": person_snapshots,
        "householdSnapshots": household_snapshots,
        "latestSnapshot": (
            {
                "quarter": latest_person["quarter"],
                "occupiedPersons": latest_person["total"],
                "occupiedHouseholds": latest_household_total,
            }
            if latest_person
            else None
        ),
    }


_POSITIVE_EXIT_CATEGORY = "Permanent Situations"
_NEGATIVE_EXIT_CATEGORIES = {"Homeless Situations"}

_INCOME_ROW_LABEL = "Number of Adults with Any Income"
_INCOME_HEADER = "Income Change by Income Category (Universe: Adult Leavers with Income Information at Start and Exit)"
_INCOME_TOTAL_COL = "Total Adults (including those with No Income)"
_INCOME_GAINED_COL = "Did Not Have the Income Category at Start and Gained the Income Category at Exit"
_INCOME_INCREASED_COL = "Retained Income Category and Increased $ at Exit"
_INCOME_RETAINED_COL = "Retained Income Category and Same $ at Exit as at Start"
_INCOME_DECREASED_COL = "Retained Income Category But Had Less $ at Exit Than at Start"
_INCOME_LOST_COL = "Had Income Category at Start and Did Not Have It at Exit"


def _success_rate(report: dict[str, Any]) -> dict[str, Any]:
    """Q23c exit outcomes + Q19a2 income outcomes (goals CSV not handled here)."""
    sheets = report.get("sheets") or []

    q23c = _find_sheet(sheets, "Q23c")
    exit_outcomes: dict[str, Any] | None = None
    if q23c:
        groups = _parse_exit_categories(q23c)
        total_exits = sum(g["subtotal"] for g in groups)
        positive = next((g["subtotal"] for g in groups if g["category"] == _POSITIVE_EXIT_CATEGORY), 0)
        negative = sum(g["subtotal"] for g in groups if g["category"] in _NEGATIVE_EXIT_CATEGORIES)
        exit_outcomes = {
            "totalExits": total_exits,
            "positiveExits": positive,
            "negativeExits": negative,
            "positiveRate": round(positive / total_exits * 1000) / 10 if total_exits > 0 else 0,
            "groups": groups,
        }

    q19a2 = _find_sheet(sheets, "Q19a2")
    income_outcomes: dict[str, Any] | None = None
    if q19a2:
        income_row = next(
            (
                r
                for r in (q19a2.get("rows") or [])
                if isinstance(r.get(_INCOME_HEADER), str) and _INCOME_ROW_LABEL in r[_INCOME_HEADER]
            ),
            None,
        )
        if income_row is not None:
            total_adults = _num(income_row.get(_INCOME_TOTAL_COL))
            gained = _num(income_row.get(_INCOME_GAINED_COL))
            increased = _num(income_row.get(_INCOME_INCREASED_COL))
            income_outcomes = {
                "totalAdults": total_adults,
                "gained": gained,
                "increased": increased,
                "retained": _num(income_row.get(_INCOME_RETAINED_COL)),
                "decreased": _num(income_row.get(_INCOME_DECREASED_COL)),
                "lost": _num(income_row.get(_INCOME_LOST_COL)),
                "gainedOrIncreasedRate": (
                    round((gained + increased) / total_adults * 1000) / 10 if total_adults > 0 else 0
                ),
            }

    return {
        "ok": True,
        "dateRange": report.get("dateRange", ""),
        "exitOutcomes": exit_outcomes,
        "incomeOutcomes": income_outcomes,
    }


_METRIC_FNS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "length_of_stay": _length_of_stay,
    "persons_served": _persons_served,
    "exit_destinations": _exit_destinations,
    "bed_availability": _bed_availability,
    "success_rate": _success_rate,
}


class CalculateShelterMetricExecutor(NodeExecutor):
    """Compute one shelter metric from a parsed Clarity annual report already in state."""

    @staticmethod
    def create(config: dict[str, Any]) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
        cfg = CalculateShelterMetricConfig(**config)

        async def calculate_shelter_metric_node(
            state: dict[str, Any],
            config: RunnableConfig,
        ) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            report = resolve_field(data, cfg.report_key) if cfg.report_key else None
            if not isinstance(report, dict):
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "report_not_found",
                    "report_key": cfg.report_key,
                }
            elif report.get("ok") is False:
                result = {
                    "ok": False,
                    "error": "upstream_failed",
                    "detail": report.get("error"),
                }
            else:
                metric_fn = _METRIC_FNS.get(cfg.metric)
                if metric_fn is None:
                    result = {"ok": False, "error": "unknown_metric", "metric": cfg.metric}
                else:
                    result = {**metric_fn(report), "metric": cfg.metric}

            return {"data": {**data, cfg.response_key: result}}

        return calculate_shelter_metric_node
