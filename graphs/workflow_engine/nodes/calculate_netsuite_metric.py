"""Calculate NetSuite Metric node.

Resolves a preset period to a date range, dispatches to a metric-specific
SuiteQL aggregator (see ``_METRIC_FNS``), and writes the result to state.
Requires a prior ``activate_netsuite`` node to have written a token payload
to ``state.data[cfg.token_key]``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.metrics import (
    actual_by_department_metric,
    cost_per_sqft_metric,
)
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import CalculateNetsuiteMetricConfig

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


# metric → async (access_token, account_id, from_date, to_date) -> dict
_METRIC_FNS = {
    "actual_by_department": actual_by_department_metric,
    "cost_per_sqft": cost_per_sqft_metric,
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
