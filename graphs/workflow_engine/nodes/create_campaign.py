"""Create Campaign node — creates a campaign on the RevOps platform."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import (
    NodeExecutor,
    http_request_with_retry,
    resolve_templates,
)
from graphs.workflow_engine.schema import CreateCampaignConfig

logger = logging.getLogger(__name__)


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
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "Campaign name is required"}}}

            # Extract team_id from JWT for campaign creation
            team_id = configurable.get("team_id", "")
            if not team_id and auth_token:
                try:
                    token_part = auth_token.replace("Bearer ", "").split(".")[1]
                    token_part += "=" * (-len(token_part) % 4)
                    claims = json.loads(base64.b64decode(token_part))
                    team_id = claims.get("team_id", "")
                except (ValueError, KeyError, IndexError):
                    logger.debug("Could not extract team_id from auth token")

            payload: dict[str, Any] = {
                "name": name,
                "channels": cfg.channels,
            }
            if team_id:
                payload["team_id"] = team_id
            if cfg.description:
                payload["description"] = resolve_templates(cfg.description, data)
            if cfg.target_persona:
                payload["target_persona"] = resolve_templates(cfg.target_persona, data)

            result: dict[str, Any]
            try:
                resp = await http_request_with_retry(
                    "POST",
                    f"{settings.graphs.REVY_API_URL}/api/v1/campaigns",
                    json=payload,
                    headers=headers,
                    timeout_seconds=cfg.timeout_seconds,
                    op_name="create_campaign",
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
                result = {"ok": False, "error": f"Request timed out after {cfg.timeout_seconds}s"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return create_campaign_node
