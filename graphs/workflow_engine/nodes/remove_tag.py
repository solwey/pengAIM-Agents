"""Remove Tag node — removes a tag from an account or contact."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import RemoveTagConfig

logger = logging.getLogger(__name__)


class RemoveTagExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = RemoveTagConfig(**config)

        async def remove_tag_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            entity_id = resolve_field(data, cfg.entity_id_key)
            tag_name = resolve_templates(cfg.tag_name, data)

            if not entity_id or not tag_name:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "Missing entity_id or tag_name"}}}

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    # Find tag by name
                    resp = await client.get(
                        f"{settings.graphs.REVY_API_URL}/api/v1/tags",
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        return {
                            "data": {
                                **data,
                                cfg.response_key: {"ok": False, "error": f"Failed to list tags: {resp.text[:500]}"},
                            }
                        }

                    tags = resp.json()
                    tag = next((t for t in tags if t["name"] == tag_name), None)
                    if not tag:
                        return {
                            "data": {**data, cfg.response_key: {"ok": False, "error": f"Tag '{tag_name}' not found"}}
                        }

                    # Remove tag from entity
                    entity_plural = f"{cfg.entity_type}s"
                    resp = await client.request(
                        "DELETE",
                        f"{settings.graphs.REVY_API_URL}/api/v1/{entity_plural}/{entity_id}/tags",
                        json={"tag_ids": [tag["id"]]},
                        headers=headers,
                    )
                    if resp.status_code in (200, 204):
                        result = {"ok": True, "tag_name": tag_name}
                        logger.info("Tag '%s' removed from %s %s", tag_name, cfg.entity_type, entity_id)
                    else:
                        result = {"ok": False, "error": resp.text[:500]}

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return remove_tag_node
