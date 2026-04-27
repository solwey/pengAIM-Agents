"""Tag Condition node — checks if an entity has specific tags, routes yes/no."""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import (
    NodeExecutor,
    http_request_with_retry,
    resolve_field,
)
from graphs.workflow_engine.schema import TagConditionConfig

logger = logging.getLogger(__name__)


class TagConditionExecutor(NodeExecutor):
    """Fetches entity tags and stores check result in state.
    The router then reads the result to route yes/no."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = TagConditionConfig(**config)

        async def tag_condition_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            entity_id = resolve_field(data, cfg.entity_id_key)
            if not entity_id or not cfg.tag_names:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "match": False, "error": "Missing entity_id or tag_names"},
                    }
                }

            try:
                entity_plural = f"{cfg.entity_type}s"
                resp = await http_request_with_retry(
                    "GET",
                    f"{settings.graphs.REVY_API_URL}/api/v1/{entity_plural}/{entity_id}",
                    headers=headers,
                    timeout_seconds=cfg.timeout_seconds,
                    op_name="tag_condition",
                )
                if resp.status_code != 200:
                    return {"data": {**data, cfg.response_key: {"ok": False, "match": False, "error": resp.text[:500]}}}

                entity = resp.json()
                entity_tags = {t["name"] for t in entity.get("tags", [])}

                if cfg.match_mode == "all":
                    match = all(tn in entity_tags for tn in cfg.tag_names)
                else:  # "any"
                    match = any(tn in entity_tags for tn in cfg.tag_names)

                result = {"ok": True, "match": match, "entity_tags": list(entity_tags)}
                logger.info(
                    "Tag check %s %s: match=%s (mode=%s, tags=%s)",
                    cfg.entity_type,
                    entity_id,
                    match,
                    cfg.match_mode,
                    cfg.tag_names,
                )

            except httpx.TimeoutException:
                result = {"ok": False, "match": False, "error": f"Request timed out after {cfg.timeout_seconds}s"}
            except httpx.RequestError as exc:
                result = {"ok": False, "match": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return tag_condition_node


def build_tag_condition_router(config: dict[str, Any]):
    """Build a routing function for add_conditional_edges().
    Reads the tag check result from state and routes yes/no."""
    cfg = TagConditionConfig(**config)

    def route(state: dict[str, Any]) -> Literal["yes", "no"]:
        data = state.get("data", {})
        result = resolve_field(data, cfg.response_key)
        if isinstance(result, dict) and result.get("match"):
            return "yes"
        return "no"

    return route
