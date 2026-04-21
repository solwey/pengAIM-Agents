"""Calculate NetSuite Metric node.

Resolves a preset period to a date range, dispatches to a metric-specific
SuiteQL aggregator (see ``_METRIC_FNS``), and writes the result to state.
Requires a prior ``activate_netsuite`` node to have written a token payload
to ``state.data[cfg.token_key]``.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import CalculateNetsuiteMetricConfig

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

    Mirrors the query in api/src/api/metrics/budget/index.ts → GET /actual-by-department.
    Aggregates Bill / Check / JournalEntry transaction lines over [from_date, to_date]
    grouped by department.

    Raises:
        ValueError: if either date is not YYYY-MM-DD.
        httpx.HTTPStatusError / httpx.RequestError / httpx.TimeoutException on NetSuite failures.
    """
    if not _DATE_RE.match(from_date) or not _DATE_RE.match(to_date):
        raise ValueError("from_date and to_date must be YYYY-MM-DD")

    sel = "SELECT"
    fr = "FROM"
    query = (
        f"{sel} d.name AS department, "
        "SUM(tl.amount) AS actual_spend, "
        "MAX(t.trandate) AS last_date, "
        "COUNT(*) AS transaction_count "
        f"{fr} transactionLine tl "
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


def _resolve_date_range(period: str, start_date: str, end_date: str) -> tuple[str, str]:
    """Translate a preset period (or custom range) into an ISO YYYY-MM-DD (from, to) pair."""

    if period == "custom":
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required when period is 'custom'")
        return start_date, end_date

    today = datetime.now(UTC).date()
    if period == "mtd":
        return today.replace(day=1).isoformat(), today.isoformat()
    if period == "qtd":
        quarter_start_month = ((today.month - 1) // 3) * 3 + 1
        return today.replace(month=quarter_start_month, day=1).isoformat(), today.isoformat()
    if period == "ytd":
        return today.replace(month=1, day=1).isoformat(), today.isoformat()
    if period == "last_7_days":
        return (today - timedelta(days=7)).isoformat(), today.isoformat()
    if period == "last_30_days":
        return (today - timedelta(days=30)).isoformat(), today.isoformat()
    if period == "last_90_days":
        return (today - timedelta(days=90)).isoformat(), today.isoformat()
    raise ValueError(f"Unknown period: {period!r}")


# metric → async (access_token, account_id, from_date, to_date) -> dict
_METRIC_FNS = {
    "actual_by_department": actual_by_department_metric,
}


class CalculateNetsuiteMetricExecutor(NodeExecutor):
    """Dispatch by ``cfg.metric`` to a SuiteQL aggregator and write the result to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = CalculateNetsuiteMetricConfig(**config)

        async def calculate_netsuite_metric_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            metric_fn = _METRIC_FNS.get(cfg.metric)
            if metric_fn is None:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Metric '{cfg.metric}' is not implemented yet",
                        },
                    }
                }

            token_payload = resolve_field(data, cfg.token_key)
            if not isinstance(token_payload, dict):
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"No NetSuite token found at state.data.{cfg.token_key} — run activate_netsuite first",
                        },
                    }
                }

            access_token = token_payload.get("access_token")
            account_id = token_payload.get("account_id")
            if not access_token or not account_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Token payload missing access_token or account_id",
                        },
                    }
                }

            try:
                from_date, to_date = _resolve_date_range(cfg.period, cfg.start_date, cfg.end_date)
            except ValueError as exc:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": str(exc)}}}

            try:
                result = await metric_fn(
                    access_token=access_token,
                    account_id=account_id,
                    from_date=from_date,
                    to_date=to_date,
                )
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "SuiteQL [%s] rejected: status=%s body=%s",
                    cfg.metric,
                    exc.response.status_code,
                    exc.response.text[:300],
                )
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "status_code": exc.response.status_code,
                            "error": exc.response.text[:500],
                        },
                    }
                }
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"SuiteQL request failed: {exc}"},
                    }
                }
            except ValueError as exc:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": str(exc)}}}

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "metric": cfg.metric,
                        "period": cfg.period,
                        **result,
                    },
                }
            }

        return calculate_netsuite_metric_node
