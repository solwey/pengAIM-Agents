"""Pause Instantly node — pauses a campaign in Instantly to stop sending emails."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import PauseInstantlyConfig

logger = logging.getLogger(__name__)


class PauseInstantlyExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = PauseInstantlyConfig(**config)

        async def pause_instantly_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            # Resolve campaign ID
            campaign_id = (
                cfg.campaign_id or resolve_field(data, cfg.campaign_id_key) if cfg.campaign_id_key else cfg.campaign_id
            )
            if not campaign_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": "No campaign_id configured or found in state"},
                    }
                }

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        f"{settings.graphs.REVY_API_URL}/api/v1/campaigns/{campaign_id}/instantly/pause",
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        result = {
                            "ok": True,
                            "campaign_id": campaign_id,
                            "status": "paused",
                        }
                        logger.info("Instantly campaign %s paused", campaign_id)
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return pause_instantly_node
