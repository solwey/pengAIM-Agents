"""Calculate Budget Metric node.

Mirrors HMISights ``api/src/api/metrics/budget/*`` route logic, but reads
both the parsed Clarity annual report and the NetSuite token from upstream
workflow state instead of from disk / env.

Currently dispatches to ``per_person_cost`` only — divides per-department
NetSuite spend by the Q5a "Total number of persons served" client count
(see api/src/api/metrics/budget/index.ts ``/per-person-cost``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.metrics import per_person_cost_metric
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import CalculateBudgetMetricConfig

logger = logging.getLogger(__name__)


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


def _extract_client_count(report: dict[str, Any]) -> int | None:
    """Pull "Total number of persons served" from the Q5a sheet of a parsed Clarity annual report.

    Matches the row shape produced by ``parse_clarity_annual``:
        {"Category": "Total number of persons served", "Count of Clients": 482, ...}
    """
    sheets = report.get("sheets") or []
    q5a = next((s for s in sheets if str(s.get("name") or "").lower() == "q5a"), None)
    if not q5a:
        return None
    for row in q5a.get("rows") or []:
        category = row.get("Category")
        if isinstance(category, str) and "Total number of persons served" in category:
            count = row.get("Count of Clients")
            if isinstance(count, int):
                return count
            if isinstance(count, float):
                return int(count)
            if isinstance(count, str):
                try:
                    return int(count.replace(",", "").strip())
                except ValueError:
                    return None
    return None


def _err(data: dict[str, Any], response_key: str, error: str, **extra: Any) -> dict[str, Any]:
    return {"data": {**data, response_key: {"ok": False, "error": error, **extra}}}


class CalculateBudgetMetricExecutor(NodeExecutor):
    """Compute one budget metric using upstream Clarity + NetSuite state."""

    @staticmethod
    def create(config: dict[str, Any]) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
        cfg = CalculateBudgetMetricConfig(**config)

        async def calculate_budget_metric_node(
            state: dict[str, Any],
            config: RunnableConfig,
        ) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            report = resolve_field(data, cfg.report_key) if cfg.report_key else None
            if not isinstance(report, dict):
                return _err(data, cfg.response_key, "report_not_found", report_key=cfg.report_key)
            if report.get("ok") is False:
                return _err(data, cfg.response_key, "upstream_failed", detail=report.get("error"))

            client_count = _extract_client_count(report)
            if not client_count or client_count <= 0:
                return _err(data, cfg.response_key, "client_count_unavailable")

            token_payload = resolve_field(data, cfg.token_key)
            if not isinstance(token_payload, dict):
                return _err(
                    data,
                    cfg.response_key,
                    f"No NetSuite token found at state.data.{cfg.token_key} — run activate_netsuite first",
                )
            access_token = token_payload.get("access_token")
            account_id = token_payload.get("account_id")
            if not access_token or not account_id:
                return _err(data, cfg.response_key, "Token payload missing access_token or account_id")

            try:
                from_date, to_date = _resolve_date_range(cfg.period, cfg.start_date, cfg.end_date)
            except ValueError as exc:
                return _err(data, cfg.response_key, str(exc))

            if cfg.metric != "per_person_cost":
                return _err(data, cfg.response_key, "unknown_metric", metric=cfg.metric)

            try:
                result = await per_person_cost_metric(
                    access_token=access_token,
                    account_id=account_id,
                    from_date=from_date,
                    to_date=to_date,
                    client_count=client_count,
                )
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "SuiteQL [per_person_cost] rejected: status=%s body=%s",
                    exc.response.status_code,
                    exc.response.text[:300],
                )
                return _err(
                    data,
                    cfg.response_key,
                    exc.response.text[:500],
                    status_code=exc.response.status_code,
                )
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                return _err(data, cfg.response_key, f"SuiteQL request failed: {exc}")
            except ValueError as exc:
                return _err(data, cfg.response_key, str(exc))

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

        return calculate_budget_metric_node
