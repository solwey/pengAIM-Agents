"""List Condition node — checks if an entity is a member of a list, routes yes/no."""

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
from graphs.workflow_engine.schema import ListConditionConfig

logger = logging.getLogger(__name__)


class ListConditionExecutor(NodeExecutor):
    """Checks if entity is in a list. Stores result in state for routing."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ListConditionConfig(**config)

        async def list_condition_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            entity_id = resolve_field(data, cfg.entity_id_key)
            list_id = cfg.list_id or resolve_field(data, "list_id")

            if not entity_id or not list_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "match": False, "error": "Missing entity_id or list_id"},
                    }
                }

            try:
                # Fetch all members page by page to check membership
                # For large lists, a dedicated endpoint would be better
                found = False
                page = 1
                while True:
                    resp = await http_request_with_retry(
                        "GET",
                        f"{settings.graphs.REVY_API_URL}/api/v1/lists/{list_id}/members",
                        params={"page": page, "page_size": 100},
                        headers=headers,
                        timeout_seconds=cfg.timeout_seconds,
                        op_name="list_condition",
                    )
                    if resp.status_code != 200:
                        return {
                            "data": {
                                **data,
                                cfg.response_key: {"ok": False, "match": False, "error": resp.text[:500]},
                            }
                        }

                    body = resp.json()
                    items = body.get("items", [])
                    for item in items:
                        if item.get("id") == entity_id:
                            found = True
                            break
                    if found or not body.get("meta", {}).get("has_next", False):
                        break
                    page += 1

                result = {"ok": True, "match": found, "list_id": list_id}
                logger.info("List check %s in list %s: match=%s", entity_id, list_id, found)

            except httpx.TimeoutException:
                result = {"ok": False, "match": False, "error": f"Request timed out after {cfg.timeout_seconds}s"}
            except httpx.RequestError as exc:
                result = {"ok": False, "match": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return list_condition_node


def build_list_condition_router(config: dict[str, Any]):
    """Build a routing function for add_conditional_edges(). Routes yes/no."""
    cfg = ListConditionConfig(**config)

    def route(state: dict[str, Any]) -> Literal["yes", "no"]:
        data = state.get("data", {})
        result = resolve_field(data, cfg.response_key)
        if isinstance(result, dict) and result.get("match"):
            return "yes"
        return "no"

    return route
