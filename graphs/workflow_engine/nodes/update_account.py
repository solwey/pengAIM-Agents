"""Update Account node — updates an existing account on the RevOps platform."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import UpdateAccountConfig

logger = logging.getLogger(__name__)


class UpdateAccountExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = UpdateAccountConfig(**config)

        async def update_account_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            account_id = resolve_field(data, cfg.account_id_key)
            if not account_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"Account ID not found at '{cfg.account_id_key}'"},
                    }
                }

            # Resolve template values in updates
            updates: dict[str, Any] = {}
            for key, template in cfg.updates.items():
                resolved = resolve_templates(template, data)
                # Try to parse numbers for score
                if key == "score":
                    with contextlib.suppress(ValueError, TypeError):
                        resolved = int(resolved)
                updates[key] = resolved

            if not updates:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "No updates specified"}}}

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.put(
                        f"{settings.graphs.REVY_API_URL}/api/v1/accounts/{account_id}",
                        json=updates,
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        result = {"ok": True, "account_id": account_id}
                        logger.info("Account updated: %s", account_id)
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return update_account_node
