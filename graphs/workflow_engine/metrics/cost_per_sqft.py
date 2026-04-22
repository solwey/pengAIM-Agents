from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Strict ISO-8601 date guard — from/to are interpolated into a SuiteQL query below.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Facility square footage, by NetSuite department name.
# Values of 1 are placeholders — update once actual square footage is measured.
_FACILITY_SQ_FT: dict[str, int] = {
    "Facilities - 2723": 1,
    "Facilities - 5217": 1,
    "Facilities - Masterhouse": 1,
}

_PLACEHOLDER_NOTE = (
    "Square footage values are placeholder (1). Update _FACILITY_SQ_FT in cost_per_sqft.py with actual values."
)


async def cost_per_sqft_metric(
    *,
    access_token: str,
    account_id: str,
    from_date: str,
    to_date: str,
) -> dict[str, Any]:
    """Run facility spend aggregation and compute cost per square foot.

    Returns one row per facility department with ``facility_spend``, ``sqft``,
    and ``cost_per_sqft`` (``None`` when ``sqft`` is still the placeholder).

    Raises:
        ValueError: if either date is not YYYY-MM-DD.
        httpx.HTTPStatusError / httpx.RequestError / httpx.TimeoutException on NetSuite failures.
    """
    if not _DATE_RE.match(from_date) or not _DATE_RE.match(to_date):
        raise ValueError("from_date and to_date must be YYYY-MM-DD")

    sel = "SELECT"

    query = (
        f"{sel} d.name AS department, SUM(tl.amount) AS facility_spend "
        "FROM transactionLine tl "
        "JOIN department d ON tl.department = d.id "
        "JOIN transaction t ON tl.transaction = t.id "
        "WHERE d.id IN (5, 6, 7) "
        "AND tl.account IN ("
        "SELECT id FROM account WHERE accttype IN ('Expense', 'OthExpense')"
        ") "
        "AND t.type IN ('Bill', 'Check', 'JournalEntry') "
        f"AND trandate BETWEEN TO_DATE('{from_date}', 'YYYY-MM-DD') "
        f"AND TO_DATE('{to_date}', 'YYYY-MM-DD') "
        "GROUP BY d.name "
        "ORDER BY facility_spend DESC"
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

    facilities: list[dict[str, Any]] = []
    has_placeholder = False
    for row in items:
        department = row.get("department")
        try:
            spend = float(row.get("facility_spend") or 0)
        except (TypeError, ValueError):
            spend = 0.0
        sqft = _FACILITY_SQ_FT.get(department, 1)
        cost_per_sqft = round(spend / sqft, 2) if sqft > 1 else None
        if cost_per_sqft is None:
            has_placeholder = True
        facilities.append(
            {
                "department": department,
                "facility_spend": spend,
                "sqft": sqft,
                "cost_per_sqft": cost_per_sqft,
            }
        )

    logger.info(
        "cost_per_sqft [%s..%s]: %d facilities, placeholder_sqft=%s",
        from_date,
        to_date,
        len(facilities),
        has_placeholder,
    )

    result: dict[str, Any] = {
        "from": from_date,
        "to": to_date,
        "facilities": facilities,
    }
    if has_placeholder:
        result["note"] = _PLACEHOLDER_NOTE
    return result
