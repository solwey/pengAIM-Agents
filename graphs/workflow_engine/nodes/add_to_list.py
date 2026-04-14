"""Add to List node — adds an account or contact to a CRM list."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import AddToListConfig

logger = logging.getLogger(__name__)


class AddToListExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = AddToListConfig(**config)

        async def add_to_list_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            entity_id = resolve_field(data, cfg.entity_id_key)
            list_id = cfg.list_id or resolve_field(data, "list_id")

            if not entity_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"No entity_id found at '{cfg.entity_id_key}'"},
                    }
                }
            if not list_id:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "No list_id configured"}}}

            # Support single ID or array
            entity_ids = entity_id if isinstance(entity_id, list) else [entity_id]

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        f"{settings.graphs.REVY_API_URL}/api/v1/lists/{list_id}/members",
                        json={"entity_ids": entity_ids},
                        headers=headers,
                    )
                    if resp.status_code in (200, 201, 204):
                        result = {"ok": True, "list_id": list_id, "added": len(entity_ids)}
                        logger.info("Added %d entities to list %s", len(entity_ids), list_id)
                    else:
                        result = {"ok": False, "error": resp.text[:500]}

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return add_to_list_node
