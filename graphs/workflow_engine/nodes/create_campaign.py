"""Create Campaign node — creates a campaign on the RevOps platform."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import CreateCampaignConfig

logger = logging.getLogger(__name__)

REVY_API_URL = os.getenv("REVY_API_URL", "http://localhost:8002")


class CreateCampaignExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = CreateCampaignConfig(**config)

        async def create_campaign_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            name = resolve_templates(cfg.name, data) if cfg.name else ""
            if not name:
                return {"data": {**data, cfg.response_key: {
                    "ok": False, "error": "Campaign name is required"
                }}}

            payload: dict[str, Any] = {
                "name": name,
                "channels": cfg.channels,
            }
            if cfg.description:
                payload["description"] = resolve_templates(cfg.description, data)
            if cfg.target_persona:
                payload["target_persona"] = resolve_templates(cfg.target_persona, data)

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        f"{REVY_API_URL}/api/v1/campaigns",
                        json=payload,
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        body = resp.json()
                        result = {
                            "ok": True,
                            "campaign_id": body.get("id"),
                            "name": body.get("name"),
                            "status": body.get("status"),
                        }
                        logger.info("Campaign created: %s (%s)", result["name"], result["campaign_id"])
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return create_campaign_node
