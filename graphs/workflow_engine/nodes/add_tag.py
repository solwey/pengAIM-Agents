"""Add Tag node — assigns a tag to an account or contact on the RevOps platform."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import AddTagConfig

logger = logging.getLogger(__name__)


class AddTagExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = AddTagConfig(**config)

        async def add_tag_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            # Resolve entity ID and tag name from state data
            entity_id = resolve_field(data, cfg.entity_id_key)
            tag_name = resolve_templates(cfg.tag_name, data)

            if not entity_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"No entity_id found at '{cfg.entity_id_key}'"},
                    }
                }

            if not tag_name:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "tag_name is empty"}}}

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    # Step 1: Ensure tag exists (create if needed)
                    resp = await client.post(
                        f"{settings.graphs.REVY_API_URL}/api/v1/tags/ensure",
                        json={"name": tag_name, "color": cfg.tag_color},
                        headers=headers,
                    )
                    if resp.status_code not in (200, 201):
                        result = {"ok": False, "error": f"Tag ensure failed: {resp.text[:500]}"}
                        return {"data": {**data, cfg.response_key: result}}

                    tag = resp.json()

                    # Step 2: Assign tag to entity
                    entity_plural = f"{cfg.entity_type}s"
                    resp = await client.post(
                        f"{settings.graphs.REVY_API_URL}/api/v1/{entity_plural}/{entity_id}/tags",
                        json={"tag_ids": [tag["id"]]},
                        headers=headers,
                    )
                    if resp.status_code in (200, 201, 204):
                        result = {"ok": True, "tag_id": tag["id"], "tag_name": tag_name}
                        logger.info("Tag '%s' assigned to %s %s", tag_name, cfg.entity_type, entity_id)
                    else:
                        result = {"ok": False, "error": f"Tag assign failed: {resp.text[:500]}"}

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return add_tag_node
