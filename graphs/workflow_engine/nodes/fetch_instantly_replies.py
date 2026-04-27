"""Fetch Instantly replies (received emails) for a campaign."""

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
)
from graphs.workflow_engine.schema import FetchInstantlyRepliesConfig

logger = logging.getLogger(__name__)


class FetchInstantlyRepliesExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchInstantlyRepliesConfig(**config)

        async def fetch_instantly_replies_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            campaign_id = cfg.campaign_id or (resolve_field(data, cfg.campaign_id_key) if cfg.campaign_id_key else "")
            if not campaign_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": "No campaign_id"},
                    }
                }

            result: dict[str, Any]
            try:
                resp = await http_request_with_retry(
                    "GET",
                    f"{settings.graphs.REVY_API_URL}/api/v1/campaigns/{campaign_id}/instantly/replies",
                    params={"limit": cfg.limit},
                    headers=headers,
                    timeout_seconds=cfg.timeout_seconds,
                    op_name="fetch_instantly_replies",
                )
                if resp.status_code == 200:
                    body = resp.json()
                    items = body.get("items", []) if isinstance(body, dict) else []
                    result = {
                        "ok": True,
                        "count": len(items),
                        "items": items,
                        "next_starting_after": body.get("next_starting_after") if isinstance(body, dict) else None,
                    }
                    logger.info(
                        "Fetched %d Instantly replies for campaign %s",
                        len(items),
                        campaign_id,
                    )
                else:
                    result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": f"Request timed out after {cfg.timeout_seconds}s"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return fetch_instantly_replies_node
