"""Actual-by-department SuiteQL aggregator.

Mirrors ``GET /api/metrics/budget/actual-by-department`` in the HMISights API
(see api/src/api/metrics/budget/index.ts). Aggregates Bill / Check / JournalEntry
transaction lines over [from_date, to_date] grouped by department.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Strict ISO-8601 date guard — from/to are interpolated into a SuiteQL query below.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


async def actual_by_department_metric(
    *,
    access_token: str,
    account_id: str,
    from_date: str,
    to_date: str,
) -> dict[str, Any]:
    """Run the actual-by-department SuiteQL aggregate and return per-department spend.

    Raises:
        ValueError: if either date is not YYYY-MM-DD.
        httpx.HTTPStatusError / httpx.RequestError / httpx.TimeoutException on NetSuite failures.
    """
    if not _DATE_RE.match(from_date) or not _DATE_RE.match(to_date):
        raise ValueError("from_date and to_date must be YYYY-MM-DD")

    sel = "SELECT"

    query = (
        f"{sel} d.name AS department, "
        "SUM(tl.amount) AS actual_spend, "
        "MAX(t.trandate) AS last_date, "
        "COUNT(*) AS transaction_count "
        "FROM transactionLine tl "
        "JOIN department d ON tl.department = d.id "
        "JOIN transaction t ON tl.transaction = t.id "
        "WHERE t.type IN ('Bill', 'Check', 'JournalEntry') "
        f"AND trandate BETWEEN TO_DATE('{from_date}', 'YYYY-MM-DD') "
        f"AND TO_DATE('{to_date}', 'YYYY-MM-DD') "
        "GROUP BY d.name "
        "ORDER BY actual_spend DESC"
    )  # noqa: S608

    url = f"https://{account_id}.suitetalk.api.netsuite.com/services/rest/query/v1/suiteql"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "transient",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
        resp = await client.post(url, headers=headers, json={"q": query})
        resp.raise_for_status()
        body = resp.json()

    items = body.get("items", []) if isinstance(body, dict) else []

    departments: list[dict[str, Any]] = []
    total_spend = 0.0
    for row in items:
        try:
            spend = float(row.get("actual_spend") or 0)
        except (TypeError, ValueError):
            spend = 0.0
        try:
            tx_count = int(row.get("transaction_count") or 0)
        except (TypeError, ValueError):
            tx_count = 0
        departments.append(
            {
                "department": row.get("department"),
                "actual_spend": spend,
                "last_date": row.get("last_date"),
                "transaction_count": tx_count,
            }
        )
        total_spend += spend

    logger.info(
        "actual_by_department [%s..%s]: %d departments, total_spend=%.2f",
        from_date,
        to_date,
        len(departments),
        total_spend,
    )

    return {
        "from": from_date,
        "to": to_date,
        "total_spend": round(total_spend, 2),
        "departments": departments,
    }
