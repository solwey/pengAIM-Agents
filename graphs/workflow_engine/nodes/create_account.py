"""Create Account node — creates accounts on the RevOps platform."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import CreateAccountConfig

logger = logging.getLogger(__name__)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")

# Auto-mapping for common webhook fields (RB2B, etc.)
AUTO_FIELD_MAP: dict[str, str] = {
    "company_name": "name",
    "website": "website",
    "industry": "industry",
    "employee_count": "employees",
    "estimate_revenue": "revenue",
}


class CreateAccountExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = CreateAccountConfig(**config)

        async def create_account_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            # Get source data
            if cfg.data_key:
                source = resolve_field(data, cfg.data_key)
            else:
                source = data

            # Normalize to list
            items: list[dict[str, Any]]
            if isinstance(source, list):
                items = source
            elif isinstance(source, dict):
                items = [source]
            else:
                return {"data": {**data, cfg.response_key: {
                    "ok": False, "error": f"No data found at '{cfg.data_key}'"
                }}}

            # Map fields for each item
            accounts = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                account: dict[str, Any] = {}

                if cfg.field_mapping:
                    # Explicit mapping with template resolution
                    for target, template in cfg.field_mapping.items():
                        account[target] = resolve_templates(template, item)
                else:
                    # Auto-mapping
                    for src_key, dst_key in AUTO_FIELD_MAP.items():
                        if src_key in item:
                            account[dst_key] = item[src_key]
                    # Also copy name directly if present
                    if "name" in item and "name" not in account:
                        account["name"] = item["name"]
                    # Build location from city + state
                    city = item.get("city", "")
                    st = item.get("state", "")
                    if city or st:
                        account["location"] = f"{city}, {st}".strip(", ")

                if account.get("name"):
                    accounts.append(account)

            if not accounts:
                return {"data": {**data, cfg.response_key: {
                    "ok": False, "error": "No valid accounts to create (name is required)"
                }}}

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        f"{BACKEND_API_URL}/api/v1/accounts/bulk",
                        json={"items": accounts},
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        body = resp.json()
                        results_list = body.get("results", [])
                        success_ids = [
                            r["item_id"] for r in results_list
                            if r.get("success") and r.get("item_id")
                        ]
                        result = {
                            "ok": True,
                            "total": body.get("total", len(accounts)),
                            "successful": body.get("successful", 0),
                            "failed": body.get("failed", 0),
                            "results": results_list,
                            "account_id": success_ids[0] if success_ids else None,
                            "account_ids": success_ids,
                        }
                        logger.info("Accounts created: %d/%d", result["successful"], result["total"])
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return create_account_node
