"""Remove from List node — removes an account or contact from a CRM list."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import RemoveFromListConfig

logger = logging.getLogger(__name__)

REVY_API_URL = os.getenv("REVY_API_URL", "http://localhost:8002")


class RemoveFromListExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = RemoveFromListConfig(**config)

        async def remove_from_list_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            entity_id = resolve_field(data, cfg.entity_id_key)
            list_id = cfg.list_id or resolve_field(data, "list_id")

            if not entity_id:
                return {"data": {**data, cfg.response_key: {
                    "ok": False, "error": f"No entity_id found at '{cfg.entity_id_key}'"
                }}}
            if not list_id:
                return {"data": {**data, cfg.response_key: {
                    "ok": False, "error": "No list_id configured"
                }}}

            entity_ids = entity_id if isinstance(entity_id, list) else [entity_id]

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.request(
                        "DELETE",
                        f"{REVY_API_URL}/api/v1/lists/{list_id}/members",
                        json={"entity_ids": entity_ids},
                        headers=headers,
                    )
                    if resp.status_code in (200, 204):
                        result = {"ok": True, "list_id": list_id, "removed": len(entity_ids)}
                        logger.info("Removed %d entities from list %s", len(entity_ids), list_id)
                    else:
                        result = {"ok": False, "error": resp.text[:300]}

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return remove_from_list_node
