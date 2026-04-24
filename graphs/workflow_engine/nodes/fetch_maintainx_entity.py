"""Fetch MaintainX Entity node — lists records of a given type from the MaintainX v1 API.

Fetches the per-tenant API key from pengAIM-RAG (stored via /api/v1/maintainx/connect as
provider=maintainx, name=maintainx_api_key), then calls GET /v1/{endpoint}?pageSize={limit}
with a Bearer Authorization header per the MaintainX v1 API contract.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import FetchMaintainxEntityConfig

logger = logging.getLogger(__name__)

# Must stay aligned with rag/app/routers/maintainx.py (MAINTAINX_PROVIDER,
# MAINTAINX_API_KEY_KEY, MAINTAINX_BASE_URL).
_MAINTAINX_PROVIDER = "maintainx"
_MAINTAINX_API_KEY_NAME = "maintainx_api_key"
_MAINTAINX_BASE_URL = "https://api.getmaintainx.com/v1"

# Maps semantic record types (shared with the revops UI and MaintainxRecordType
# in schema.py) to (endpoint_path, response_items_key). The response key is the
# camelCase plural that MaintainX uses for the returned list — when a response
# uses a different shape, _extract_items() falls back to common envelopes.
_MAINTAINX_ENDPOINTS: dict[str, tuple[str, str]] = {
    "workorder": ("workorders", "workOrders"),
    "asset": ("assets", "assets"),
    "location": ("locations", "locations"),
    "part": ("parts", "parts"),
    "category": ("categories", "categories"),
    "user": ("users", "users"),
    "team": ("teams", "teams"),
    "vendor": ("vendors", "vendors"),
    "purchaseorder": ("purchaseorders", "purchaseOrders"),
    "meter": ("meters", "meters"),
    "workrequest": ("workrequests", "workRequests"),
    "meterreading": ("meterreadings", "meterReadings"),
}


async def _reveal_maintainx_key(config: RunnableConfig) -> str | None:
    """Fetch the tenant's MaintainX API key from pengAIM-RAG via /keys/by-name/reveal.

    Returns None when auth context is missing or the fetch fails.
    """
    configurable = config.get("configurable", {})
    auth_token = configurable.get("auth_token", "")
    tenant_uuid = configurable.get("tenant_uuid", "")

    if not auth_token or not tenant_uuid:
        logger.warning(
            "MaintainX credentials unavailable (auth_token=%s, tenant_uuid=%s)",
            bool(auth_token),
            bool(tenant_uuid),
        )
        return None

    url = f"{settings.graphs.RAG_API_URL}/tenant/{tenant_uuid}/keys/by-name/reveal"
    headers = {"authorization": auth_token, "Accept": "text/plain"}
    params = {"provider": _MAINTAINX_PROVIDER, "name": _MAINTAINX_API_KEY_NAME}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp.text
    except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to reveal MaintainX api key: %s", exc)
        return None


def _project_fields(items: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    """Return items with only the requested keys; pass through when fields is empty."""
    if not fields:
        return items
    return [{f: item.get(f) for f in fields} for item in items]


def _extract_items(body: Any, response_key: str) -> list[dict[str, Any]]:
    """Pull a list of records out of a MaintainX response, tolerating minor shape drift."""
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if not isinstance(body, dict):
        return []
    for key in (response_key, "data", "items", "results"):
        value = body.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for value in body.values():
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


class FetchMaintainxEntityExecutor(NodeExecutor):
    """List records of a MaintainX entity type and write items to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchMaintainxEntityConfig(**config)

        async def fetch_maintainx_entity_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            api_key = await _reveal_maintainx_key(config)
            if not api_key:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Missing MaintainX API key — connect MaintainX in integrations",
                        },
                    }
                }

            endpoint_info = _MAINTAINX_ENDPOINTS.get(cfg.record_type)
            if endpoint_info is None:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Unsupported record_type '{cfg.record_type}'",
                        },
                    }
                }
            endpoint, response_key = endpoint_info

            url = f"{_MAINTAINX_BASE_URL}/{endpoint}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            params = {}

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    body = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "MaintainX [%s] rejected: status=%s body=%s",
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
                        cfg.response_key: {"ok": False, "error": f"MaintainX request failed: {exc}"},
                    }
                }

            raw_items = _extract_items(body, response_key)
            total = body.get("totalCount") if isinstance(body, dict) else None
            items = _project_fields(raw_items, cfg.fields)

            logger.info(
                "MaintainX [%s] returned %d items (total=%s)",
                cfg.record_type,
                len(items),
                total,
            )

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "record_type": cfg.record_type,
                        "count": len(items),
                        "total": total,
                        "items": items,
                    },
                }
            }

        return fetch_maintainx_entity_node
