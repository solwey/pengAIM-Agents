"""Set Source node — assigns a lead source to an account or contact."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import (
    NodeExecutor,
    http_request_with_retry,
    resolve_field,
    resolve_templates,
)
from graphs.workflow_engine.schema import SetSourceConfig

logger = logging.getLogger(__name__)


class SetSourceExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = SetSourceConfig(**config)

        async def set_source_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            entity_id = resolve_field(data, cfg.entity_id_key)
            source_name = resolve_templates(cfg.source_name, data)

            if not entity_id or not source_name:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "Missing entity_id or source_name"}}}

            result: dict[str, Any]
            try:
                resp = await http_request_with_retry(
                    "GET",
                    f"{settings.graphs.REVY_API_URL}/api/v1/sources",
                    headers=headers,
                    timeout_seconds=cfg.timeout_seconds,
                    op_name="set_source.list",
                )
                if resp.status_code != 200:
                    return {
                        "data": {
                            **data,
                            cfg.response_key: {
                                "ok": False,
                                "error": f"Failed to list sources: {resp.text[:500]}",
                            },
                        }
                    }

                sources_data = resp.json()
                sources = sources_data.get("items", sources_data) if isinstance(sources_data, dict) else sources_data
                source = next((s for s in sources if s["name"] == source_name), None)
                if not source:
                    return {
                        "data": {
                            **data,
                            cfg.response_key: {"ok": False, "error": f"Source '{source_name}' not found"},
                        }
                    }

                entity_plural = f"{cfg.entity_type}s"
                resp = await http_request_with_retry(
                    "PUT",
                    f"{settings.graphs.REVY_API_URL}/api/v1/{entity_plural}/{entity_id}",
                    json={"source_id": source["id"]},
                    headers=headers,
                    timeout_seconds=cfg.timeout_seconds,
                    op_name="set_source.set",
                )
                if resp.status_code == 200:
                    result = {"ok": True, "source_name": source_name, "source_id": source["id"]}
                    logger.info("Source '%s' set on %s %s", source_name, cfg.entity_type, entity_id)
                else:
                    result = {"ok": False, "error": resp.text[:500]}

            except httpx.TimeoutException:
                result = {"ok": False, "error": f"Request timed out after {cfg.timeout_seconds}s"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return set_source_node
