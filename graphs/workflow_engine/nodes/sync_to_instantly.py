"""Sync to Instantly node — pushes a campaign to Instantly for email delivery."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import SyncToInstantlyConfig

logger = logging.getLogger(__name__)


class SyncToInstantlyExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = SyncToInstantlyConfig(**config)

        async def sync_to_instantly_node(state: dict, config: RunnableConfig) -> dict:
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

            payload: dict[str, Any] = {}
            if cfg.account_email:
                email = resolve_templates(cfg.account_email, data)
                if email:
                    payload["account_email"] = email

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        f"{settings.graphs.REVY_API_URL}/api/v1/campaigns/{campaign_id}/sync-to-instantly",
                        json=payload,
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        body = resp.json()
                        result = {
                            "ok": True,
                            "campaign_id": campaign_id,
                            "instantly_campaign_id": body.get("instantly_campaign_id"),
                        }
                        logger.info("Campaign %s synced to Instantly", campaign_id)
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return sync_to_instantly_node
