"""Create Contact node — creates contacts on the RevOps platform."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, resolve_templates
from graphs.workflow_engine.schema import CreateContactConfig

logger = logging.getLogger(__name__)

# Auto-mapping for common webhook fields
AUTO_FIELD_MAP: dict[str, str] = {
    "first_name": "first_name",
    "last_name": "last_name",
    "business_email": "email",
    "email": "email",
    "title": "title",
    "linkedin_url": "linkedin_url",
    "city": "city",
    "state": "state",
}


class CreateContactExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = CreateContactConfig(**config)

        async def create_contact_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            # Resolve account ID
            account_id = resolve_field(data, cfg.account_id_key)
            if not account_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"Account ID not found at '{cfg.account_id_key}'"},
                    }
                }

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
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "No contact data found"}}}

            contacts = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                contact: dict[str, Any] = {"account_id": account_id}

                if cfg.field_mapping:
                    for target, template in cfg.field_mapping.items():
                        contact[target] = resolve_templates(template, item)
                else:
                    for src_key, dst_key in AUTO_FIELD_MAP.items():
                        if src_key in item and item[src_key]:
                            contact[dst_key] = item[src_key]

                # Ensure required fields have at least empty defaults
                if "last_name" not in contact:
                    contact["last_name"] = ""

                # Need at least first_name or email
                if contact.get("first_name") or contact.get("email"):
                    contacts.append(contact)

            if not contacts:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "No valid contacts to create"}}}

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        f"{settings.graphs.REVY_API_URL}/api/v1/contacts/bulk",
                        json={"items": contacts},
                        headers=headers,
                    )
                    if resp.status_code in (200, 201):
                        body = resp.json()
                        results_list = body.get("results", [])
                        success_ids = [r["item_id"] for r in results_list if r.get("success") and r.get("item_id")]
                        result = {
                            "ok": True,
                            "total": body.get("total", len(contacts)),
                            "successful": body.get("successful", 0),
                            "failed": body.get("failed", 0),
                            "results": results_list,
                            "contact_id": success_ids[0] if success_ids else None,
                            "contact_ids": success_ids,
                        }
                        logger.info("Contacts created: %d/%d", result["successful"], result["total"])
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return create_contact_node
