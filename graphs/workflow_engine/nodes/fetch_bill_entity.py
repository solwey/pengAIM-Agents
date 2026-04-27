"""Fetch Bill.com Entity node — lists records of a given type from the Bill.com Connect v3 API.

Expects a prior activate_billcom node to have written {session_id, dev_key, organization_id}
to state.data[token_key] (default "bill_activate_result"). Calls
GET /{endpoint}?start=0&max={limit} with the {sessionId, devKey} headers.

Mirrors the list pattern in api/src/api/bill/bill.api.ts (fetchList).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import FetchBillEntityConfig

logger = logging.getLogger(__name__)

_BILL_BASE_URL = "https://gateway.prod.bill.com/connect/v3"

# Maps semantic record types (shared with the revops UI and BillRecordType in
# schema.py) to Bill.com Connect v3 REST path segments.
_BILL_ENDPOINT: dict[str, str] = {
    "bill": "bills",
    "credit_memo": "credit-memos",
    "customer": "customers",
    "invoice": "invoices",
    "payment": "payments",
    "recurring_bill": "recurringbills",
    "recurring_invoice": "recurring-invoices",
    "vendor": "vendors",
}


def _project_fields(items: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    """Return items with only the requested keys; pass through when fields is empty."""
    if not fields:
        return items
    return [{f: item.get(f) for f in fields} for item in items]


def _extract_items(body: Any) -> list[dict[str, Any]]:
    """Pull a list of records out of a Bill.com response, tolerating minor shape drift."""
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if not isinstance(body, dict):
        return []
    for key in ("results", "data", "items"):
        value = body.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for value in body.values():
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


class FetchBillEntityExecutor(NodeExecutor):
    """List records of a Bill.com entity type and write items to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchBillEntityConfig(**config)

        async def fetch_bill_entity_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            token_payload = resolve_field(data, cfg.token_key)
            if not isinstance(token_payload, dict):
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"No Bill.com session found at state.data.{cfg.token_key} — run activate_billcom first",
                        },
                    }
                }

            session_id = token_payload.get("session_id")
            dev_key = token_payload.get("dev_key")
            if not session_id or not dev_key:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Token payload missing session_id or dev_key",
                        },
                    }
                }

            endpoint = _BILL_ENDPOINT.get(cfg.record_type)
            if endpoint is None:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Unsupported record_type '{cfg.record_type}'",
                        },
                    }
                }

            url = f"{_BILL_BASE_URL}/{endpoint}"
            headers = {
                "sessionId": session_id,
                "devKey": dev_key,
                "Content-Type": "application/json",
            }
            params = {"start": 0, "max": cfg.limit}

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    body = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Bill.com [%s] rejected: status=%s body=%s",
                    cfg.record_type,
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
                        cfg.response_key: {"ok": False, "error": f"Bill.com request failed: {exc}"},
                    }
                }

            raw_items = _extract_items(body)
            items = _project_fields(raw_items, cfg.fields)

            logger.info("Bill.com [%s] returned %d items", cfg.record_type, len(items))

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "record_type": cfg.record_type,
                        "count": len(items),
                        "items": items,
                    },
                }
            }

        return fetch_bill_entity_node
