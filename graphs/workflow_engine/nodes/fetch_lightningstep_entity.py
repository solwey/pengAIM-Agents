"""Fetch Lightning Step Entity node — lists records of a given type from the
Lightning Step v1 API.

Fetches the per-tenant organization subdomain + API token from pengAIM-RAG
(stored via /api/v1/lightningstep/connect as provider=lightningstep, names
``lightningstep_organization`` and ``lightningstep_api_token``), then calls
GET https://{organization}.lightningstep.com/api/v1/{record_type}list with a
Bearer Authorization header per the Lightning Step API contract.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import FetchLightningstepEntityConfig

logger = logging.getLogger(__name__)

# Must stay aligned with rag/app/routers/lightningstep.py (LIGHTNINGSTEP_PROVIDER,
# LIGHTNINGSTEP_API_TOKEN_KEY, LIGHTNINGSTEP_ORGANIZATION_KEY).
_LIGHTNINGSTEP_PROVIDER = "lightningstep"


def _base_url(organization: str) -> str:
    return f"https://{organization}.lightningstep.com/api/v1"


async def _reveal_lightningstep_value(config: RunnableConfig, name: str) -> str | None:
    """Fetch a stored Lightning Step credential by name from pengAIM-RAG.

    Returns None when auth context is missing or the fetch fails.
    """
    configurable = config.get("configurable", {})
    auth_token = configurable.get("auth_token", "")
    tenant_uuid = configurable.get("tenant_uuid", "")

    if not auth_token or not tenant_uuid:
        logger.warning(
            "Lightning Step credentials unavailable (auth_token=%s, tenant_uuid=%s)",
            bool(auth_token),
            bool(tenant_uuid),
        )
        return None

    url = f"{settings.graphs.RAG_API_URL}/tenant/{tenant_uuid}/keys/by-name/reveal"
    headers = {"authorization": auth_token, "Accept": "text/plain"}
    params = {"provider": _LIGHTNINGSTEP_PROVIDER, "name": name}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp.text
    except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to reveal Lightning Step %s: %s", name, exc)
        return None


def _project_fields(items: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    """Return items with only the requested keys; pass through when fields is empty."""
    if not fields:
        return items
    return [{f: item.get(f) for f in fields} for item in items]


def _extract_items(body: Any) -> list[dict[str, Any]]:
    """Pull a list of records out of a Lightning Step response, tolerating shape drift."""
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if not isinstance(body, dict):
        return []
    for key in ("data", "items", "results", "records"):
        value = body.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for value in body.values():
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


class FetchLightningstepEntityExecutor(NodeExecutor):
    """List records of a Lightning Step entity type and write items to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchLightningstepEntityConfig(**config)

        async def fetch_lightningstep_entity_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            organization = await _reveal_lightningstep_value(config, "lightningstep_organization")
            api_token = await _reveal_lightningstep_value(config, "lightningstep_api_token")
            if not organization or not api_token:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Missing Lightning Step credentials — connect Lightning Step in integrations",
                        },
                    }
                }

            inquiry_id = cfg.inquiry_id.strip()
            path = f"/{cfg.record_type}list/{inquiry_id}" if inquiry_id else f"/{cfg.record_type}list"
            url = f"{_base_url(organization)}{path}"
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            }

            print(url, headers)

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    body = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Lightning Step [%s] rejected: status=%s body=%s",
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
                        cfg.response_key: {"ok": False, "error": f"Lightning Step request failed: {exc}"},
                    }
                }

            raw_items = _extract_items(body)
            items = _project_fields(raw_items, cfg.fields)

            logger.info(
                "Lightning Step [%s] returned %d items",
                cfg.record_type,
                len(items),
            )

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "record_type": cfg.record_type,
                        "inquiry_id": inquiry_id or None,
                        "count": len(items),
                        "items": items,
                    },
                }
            }

        return fetch_lightningstep_entity_node
