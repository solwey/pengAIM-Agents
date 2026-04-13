"""Add Leads to Instantly node — pushes contacts as leads to an Instantly campaign."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import AddLeadsToInstantlyConfig

logger = logging.getLogger(__name__)

REVY_API_URL = os.getenv("REVY_API_URL", "http://localhost:8002")


class AddLeadsToInstantlyExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = AddLeadsToInstantlyConfig(**config)

        async def add_leads_to_instantly_node(state: dict, config: RunnableConfig) -> dict:
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

            # Resolve contact IDs from state
            contact_ids = resolve_field(data, cfg.contact_ids_key)
            if not contact_ids:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"No contact IDs found at '{cfg.contact_ids_key}'"},
                    }
                }

            if not isinstance(contact_ids, list):
                contact_ids = [contact_ids]

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
                    resp = await client.post(
                        f"{REVY_API_URL}/api/v1/campaigns/{campaign_id}/instantly/leads",
                        json={"contact_ids": contact_ids},
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        body = resp.json()
                        result = {
                            "ok": True,
                            "campaign_id": campaign_id,
                            "leads_added": body.get("added", len(contact_ids)),
                        }
                        logger.info("Added %d leads to Instantly campaign %s", len(contact_ids), campaign_id)
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return add_leads_to_instantly_node
