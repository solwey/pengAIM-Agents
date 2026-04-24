"""Add an email or domain (or a list of them) to the team's Instantly blocklist."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import (
    NodeExecutor,
    resolve_field,
    resolve_templates,
)
from graphs.workflow_engine.schema import AddToInstantlyBlocklistConfig

logger = logging.getLogger(__name__)


def _normalise_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        v = raw.strip()
        return [v] if v else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                v = item.strip()
                if v:
                    out.append(v)
            elif isinstance(item, dict):
                for key in ("bl_value", "value", "email", "domain"):
                    if isinstance(item.get(key), str) and item[key].strip():
                        out.append(item[key].strip())
                        break
        return out
    return []


class AddToInstantlyBlocklistExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = AddToInstantlyBlocklistConfig(**config)

        async def add_to_instantly_blocklist_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            values: list[str] = []
            if cfg.bl_values_key:
                values = _normalise_values(resolve_field(data, cfg.bl_values_key))
            if not values and cfg.bl_value_key:
                values = _normalise_values(resolve_field(data, cfg.bl_value_key))
            if not values and cfg.bl_value:
                resolved = resolve_templates(cfg.bl_value, data).strip()
                if resolved:
                    values = [resolved]

            if not values:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "No blocklist value resolved (set bl_value, bl_value_key, or bl_values_key).",
                        },
                    }
                }

            body: dict[str, Any]
            if len(values) == 1:
                body = {"bl_value": values[0]}
            else:
                if len(values) > 1000:
                    return {
                        "data": {
                            **data,
                            cfg.response_key: {
                                "ok": False,
                                "error": f"Too many values ({len(values)}); max 1000 per call.",
                            },
                        }
                    }
                body = {"bl_values": values}

            url = f"{settings.graphs.REVY_API_URL}/api/v1/instantly/blocklist"
            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(url, json=body, headers=headers)
                    if resp.status_code in (200, 201):
                        payload = resp.json() if resp.content else {}
                        result = {
                            "ok": True,
                            "added_count": len(values),
                            "values": values,
                            "result": payload,
                        }
                        logger.info(
                            "Added %d entries to Instantly blocklist",
                            len(values),
                        )
                    else:
                        result = {"ok": False, "error": resp.text[:500]}
            except httpx.TimeoutException:
                result = {"ok": False, "error": "Request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return add_to_instantly_blocklist_node
