"""Create Account node — creates accounts on the RevOps platform."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import CreateAccountConfig

logger = logging.getLogger(__name__)

# Auto-mapping for common webhook fields (RB2B, etc.)
# List of (source_key, target_key) — allows one source to map to multiple targets.
AUTO_FIELD_MAP: list[tuple[str, str]] = [
    ("company_name", "name"),
    ("company", "name"),
    ("organization", "name"),
    ("organization_name", "name"),
    ("account_name", "name"),
    ("website", "website"),
    ("domain", "website"),
    ("company_domain", "website"),
    ("industry", "industry"),
    ("employee_count", "employees"),
    ("employee_count", "size"),
    ("employees", "employees"),
    ("company_size", "size"),
    ("estimate_revenue", "revenue"),
    ("revenue", "revenue"),
    ("annual_revenue", "revenue"),
    ("phone", "phone"),
    ("company_phone", "phone"),
]


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
                return {
                    "data": {**data, cfg.response_key: {"ok": False, "error": f"No data found at '{cfg.data_key}'"}}
                }

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
                    # Auto-mapping: convert values to strings (backend expects string fields)
                    for src_key, dst_key in AUTO_FIELD_MAP:
                        val = item.get(src_key)
                        if val is not None and val != "":
                            account[dst_key] = str(val) if not isinstance(val, str) else val
                    # Also copy name directly if present
                    if "name" in item and "name" not in account:
                        account["name"] = item["name"]
                    # Build location from city + state
                    city = item.get("city", "")
                    st = item.get("state", "")
                    if city or st:
                        account["location"] = f"{city}, {st}".strip(", ")

                # Fallback: try to derive name from other fields
                if not account.get("name"):
                    # Try domain from email
                    email = item.get("email") or item.get("business_email") or ""
                    if "@" in email:
                        domain = email.split("@")[1].split(".")[0].capitalize()
                        account["name"] = domain
                    # Try first_name + last_name as last resort
                    elif item.get("first_name") or item.get("last_name"):
                        parts = [item.get("first_name", ""), item.get("last_name", "")]
                        account["name"] = " ".join(p for p in parts if p).strip()

                if account.get("name"):
                    accounts.append(account)

            if not accounts:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": "No valid accounts to create (name is required)"},
                    }
                }

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        f"{settings.graphs.REVY_API_URL}/api/v1/accounts/bulk",
                        json={"items": accounts, "dedup_mode": cfg.dedup_mode},
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        body = resp.json()
                        results_list = body.get("results", [])
                        success_ids = [r["item_id"] for r in results_list if r.get("success") and r.get("item_id")]
                        deduplicated_count = sum(1 for r in results_list if r.get("deduplicated"))
                        result = {
                            "ok": True,
                            "total": body.get("total", len(accounts)),
                            "successful": body.get("successful", 0),
                            "failed": body.get("failed", 0),
                            "deduplicated": deduplicated_count,
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
