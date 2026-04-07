"""Add to Campaign node — links accounts to a campaign on the RevOps platform."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import AddToCampaignConfig

logger = logging.getLogger(__name__)

REVY_API_URL = os.getenv("REVY_API_URL", "http://localhost:8002")


class AddToCampaignExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = AddToCampaignConfig(**config)

        async def add_to_campaign_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            # Resolve campaign ID
            campaign_id = cfg.campaign_id or resolve_field(data, cfg.campaign_id_key) if cfg.campaign_id_key else cfg.campaign_id
            if not campaign_id:
                return {"data": {**data, cfg.response_key: {
                    "ok": False, "error": "No campaign_id configured or found in state"
                }}}

            # Resolve account IDs
            account_id = resolve_field(data, cfg.account_id_key)
            if not account_id:
                return {"data": {**data, cfg.response_key: {
                    "ok": False, "error": f"No account_id found at '{cfg.account_id_key}'"
                }}}

            account_ids = account_id if isinstance(account_id, list) else [account_id]

            # Update campaign to link accounts
            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.put(
                        f"{REVY_API_URL}/api/v1/campaigns/{campaign_id}",
                        json={"account_ids": account_ids},
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        result = {
                            "ok": True,
                            "campaign_id": campaign_id,
                            "accounts_added": len(account_ids),
                        }
                        logger.info("Added %d accounts to campaign %s", len(account_ids), campaign_id)
                    else:
                        result = {"ok": False, "error": resp.text[:300]}

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return add_to_campaign_node
