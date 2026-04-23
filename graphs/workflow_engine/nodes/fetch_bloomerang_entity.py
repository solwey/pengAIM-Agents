"""Fetch Bloomerang Entity node — lists records of a given type from the Bloomerang v2 API.

Fetches the per-tenant API key from pengAIM-RAG (stored via /api/v1/bloomerang/connect as
provider=bloomerang, name=bloomerang_api_key), then calls GET /v2/{record_type}?skip=0&take={limit}
with the X-API-KEY header per the Bloomerang v2 API contract.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import FetchBloomerangEntityConfig

logger = logging.getLogger(__name__)

# Must stay aligned with rag/app/routers/bloomerang.py (BLOOMERANG_PROVIDER,
# BLOOMERANG_API_KEY_KEY, BLOOMERANG_BASE_URL).
_BLOOMERANG_PROVIDER = "bloomerang"
_BLOOMERANG_API_KEY_NAME = "bloomerang_api_key"
_BLOOMERANG_BASE_URL = "https://api.bloomerang.co/v2"

# Maps semantic record types (shared with the revops UI and BloomerangRecordType
# in schema.py) to Bloomerang v2 REST path segments. Kept as a dict so future
# record types whose path diverges from a simple pluralization can be added
# without breaking the enum.
_BLOOMERANG_ENDPOINT: dict[str, str] = {
    "appeal": "appeals",
    "campaign": "campaigns",
    "constituent": "constituents",
    "designation": "transactions/designations",
    "fund": "funds",
    "household": "households",
    "interaction": "interactions",
    "note": "notes",
    "relationship": "relationshiproles",
    "transaction": "transactions",
    "tribute": "tributes",
}


async def _reveal_bloomerang_key(config: RunnableConfig) -> str | None:
    """Fetch the tenant's Bloomerang API key from pengAIM-RAG via /keys/by-name/reveal.

    Returns None when auth context is missing or the fetch fails.
    """
    configurable = config.get("configurable", {})
    auth_token = configurable.get("auth_token", "")
    tenant_uuid = configurable.get("tenant_uuid", "")

    if not auth_token or not tenant_uuid:
        logger.warning(
            "Bloomerang credentials unavailable (auth_token=%s, tenant_uuid=%s)",
            bool(auth_token),
            bool(tenant_uuid),
        )
        return None

    url = f"{settings.graphs.RAG_API_URL}/tenant/{tenant_uuid}/keys/by-name/reveal"
    headers = {"authorization": auth_token, "Accept": "text/plain"}
    params = {"provider": _BLOOMERANG_PROVIDER, "name": _BLOOMERANG_API_KEY_NAME}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp.text
    except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to reveal Bloomerang api key: %s", exc)
        return None


def _project_fields(items: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    """Return items with only the requested keys; pass through when fields is empty."""
    if not fields:
        return items
    return [{f: item.get(f) for f in fields} for item in items]


class FetchBloomerangEntityExecutor(NodeExecutor):
    """List records of a Bloomerang entity type and write items to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchBloomerangEntityConfig(**config)

        async def fetch_bloomerang_entity_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            api_key = await _reveal_bloomerang_key(config)
            if not api_key:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Missing Bloomerang API key — connect Bloomerang in integrations",
                        },
                    }
                }

            endpoint = _BLOOMERANG_ENDPOINT.get(cfg.record_type)
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

            url = f"{_BLOOMERANG_BASE_URL}/{endpoint}"
            headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
            params = {"skip": 0, "take": cfg.limit}

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    body = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Bloomerang [%s] rejected: status=%s body=%s",
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
                        cfg.response_key: {"ok": False, "error": f"Bloomerang request failed: {exc}"},
                    }
                }

            raw_items = body.get("Results", []) if isinstance(body, dict) else []
            total = body.get("TotalResults") if isinstance(body, dict) else None
            items = _project_fields(raw_items, cfg.fields)

            logger.info(
                "Bloomerang [%s] returned %d items (total=%s)",
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

        return fetch_bloomerang_entity_node
