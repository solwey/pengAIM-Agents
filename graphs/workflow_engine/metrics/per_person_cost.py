"""Per-person-cost SuiteQL aggregator + per-department division.

Mirrors ``GET /api/metrics/budget/per-person-cost`` in the HMISights API
(see api/src/api/metrics/budget/index.ts). Aggregates Bill / Check / JournalEntry
spend by department over [from_date, to_date], then divides each department's
spend by ``client_count`` to yield cost-per-person.

Unlike the other aggregators in this package, this one needs an external
``client_count`` (sourced from the Clarity HUD Annual Q5a row "Total number
of persons served"), so the signature differs from the standard
``(access_token, account_id, from_date, to_date)`` shape used by
``calculate_netsuite_metric``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


async def per_person_cost_metric(
    *,
    access_token: str,
    account_id: str,
    from_date: str,
    to_date: str,
    client_count: int,
) -> dict[str, Any]:
    """Return per-department spend and cost-per-person.

    Raises:
        ValueError: if either date is not YYYY-MM-DD or client_count <= 0.
        httpx.HTTPStatusError / httpx.RequestError / httpx.TimeoutException on NetSuite failures.
    """
    if not _DATE_RE.match(from_date) or not _DATE_RE.match(to_date):
        raise ValueError("from_date and to_date must be YYYY-MM-DD")
    if client_count <= 0:
        raise ValueError("client_count must be > 0")

    sel = "SELECT"
    query = (
        f"{sel} d.name AS department, "
        "SUM(tl.amount) AS actual_spend "
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
        departments.append(
            {
                "department": row.get("department"),
                "actual_spend": round(spend, 2),
                "cost_per_person": round(spend / client_count, 2),
            }
        )
        total_spend += spend

    logger.info(
        "per_person_cost [%s..%s]: %d departments, total_spend=%.2f, client_count=%d",
        from_date,
        to_date,
        len(departments),
        total_spend,
        client_count,
    )

    return {
        "from": from_date,
        "to": to_date,
        "client_count": client_count,
        "total_spend": round(total_spend, 2),
        "total_cost_per_person": round(total_spend / client_count, 2),
        "departments": departments,
    }
