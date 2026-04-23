"""Update Instantly lead interest status node."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import UpdateInstantlyLeadStatusConfig

logger = logging.getLogger(__name__)


class UpdateInstantlyLeadStatusExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = UpdateInstantlyLeadStatusConfig(**config)

        async def update_instantly_lead_status_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            campaign_id = cfg.campaign_id or (resolve_field(data, cfg.campaign_id_key) if cfg.campaign_id_key else "")
            lead_email = cfg.lead_email or (resolve_field(data, cfg.lead_email_key) if cfg.lead_email_key else "")
            interest_value = cfg.interest_value
            if interest_value is None and cfg.interest_value_key:
                resolved = resolve_field(data, cfg.interest_value_key)
                if isinstance(resolved, int):
                    interest_value = resolved

            if not campaign_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": "No campaign_id"},
                    }
                }
            if not lead_email:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": "No lead_email"},
                    }
                }

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        f"{settings.graphs.REVY_API_URL}/api/v1/campaigns/{campaign_id}/instantly/leads/update-interest",
                        json={
                            "lead_email": lead_email,
                            "interest_value": interest_value,
                        },
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        body = resp.json()
                        result = {
                            "ok": True,
                            "lead_email": lead_email,
                            "interest_value": interest_value,
                            "result": body.get("result"),
                        }
                        logger.info(
                            "Updated Instantly lead status: %s -> %s (campaign=%s)",
                            lead_email,
                            interest_value,
                            campaign_id,
                        )
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return update_instantly_lead_status_node
