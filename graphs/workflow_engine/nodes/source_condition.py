"""Source Condition node — checks if an entity has a specific source, routes yes/no."""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import SourceConditionConfig

logger = logging.getLogger(__name__)


class SourceConditionExecutor(NodeExecutor):
    """Checks if entity has the specified source. Stores result in state for routing."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = SourceConditionConfig(**config)

        async def source_condition_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            entity_id = resolve_field(data, cfg.entity_id_key)
            source_name = resolve_templates(cfg.source_name, data)

            if not entity_id or not source_name:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "match": False, "error": "Missing entity_id or source_name"},
                    }
                }

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    entity_plural = f"{cfg.entity_type}s"
                    resp = await client.get(
                        f"{settings.graphs.REVY_API_URL}/api/v1/{entity_plural}/{entity_id}",
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        return {
                            "data": {**data, cfg.response_key: {"ok": False, "match": False, "error": resp.text[:500]}}
                        }

                    entity = resp.json()
                    entity_source = entity.get("source", {})
                    entity_source_name = entity_source.get("name", "") if entity_source else ""

                    match = entity_source_name.lower() == source_name.lower()

                    result = {
                        "ok": True,
                        "match": match,
                        "entity_source": entity_source_name,
                        "expected_source": source_name,
                    }
                    logger.info(
                        "Source check %s %s: match=%s (expected=%s, actual=%s)",
                        cfg.entity_type,
                        entity_id,
                        match,
                        source_name,
                        entity_source_name,
                    )

            except httpx.TimeoutException:
                result = {"ok": False, "match": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "match": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return source_condition_node


def build_source_condition_router(config: dict[str, Any]):
    """Build a routing function for add_conditional_edges(). Routes yes/no."""
    cfg = SourceConditionConfig(**config)

    def route(state: dict[str, Any]) -> Literal["yes", "no"]:
        data = state.get("data", {})
        result = resolve_field(data, cfg.response_key)
        if isinstance(result, dict) and result.get("match"):
            return "yes"
        return "no"

    return route
