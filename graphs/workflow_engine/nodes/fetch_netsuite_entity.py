"""Fetch NetSuite Entity node — executes a SuiteQL query using a token minted by activate_netsuite.

Expects a prior activate_netsuite node to have written {access_token, account_id} to
state.data[token_key] (default "netsuite_activate_result").
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import FetchNetsuiteEntityConfig

logger = logging.getLogger(__name__)

# SuiteQL identifiers: letters, digits, underscore. Prevents injection via record_type / fields.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str) -> bool:
    return bool(_IDENT_RE.match(value))


class FetchNetsuiteEntityExecutor(NodeExecutor):
    """Run a SuiteQL SELECT against a NetSuite record type and write rows to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchNetsuiteEntityConfig(**config)

        async def fetch_netsuite_entity_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            token_payload = resolve_field(data, cfg.token_key)
            if not isinstance(token_payload, dict):
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"No NetSuite token found at state.data.{cfg.token_key} — run activate_netsuite first",
                        },
                    }
                }

            access_token = token_payload.get("access_token")
            account_id = token_payload.get("account_id")
            if not access_token or not account_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Token payload missing access_token or account_id",
                        },
                    }
                }

            if not _validate_identifier(cfg.record_type):
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Invalid record_type '{cfg.record_type}'",
                        },
                    }
                }

            if cfg.fields:
                invalid = [f for f in cfg.fields if not _validate_identifier(f)]
                if invalid:
                    return {
                        "data": {
                            **data,
                            cfg.response_key: {
                                "ok": False,
                                "error": f"Invalid field names: {invalid}",
                            },
                        }
                    }
                select_clause = ", ".join(cfg.fields)
            else:
                select_clause = "*"

            sel = "SELECT"
            fr = "FROM"

            query = f"{sel} {select_clause} {fr} {cfg.record_type}"  # noqa: S608
            suiteql_url = f"https://{account_id}.suitetalk.api.netsuite.com/services/rest/query/v1/suiteql"

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Prefer": "transient",
            }

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.post(
                        suiteql_url,
                        headers=headers,
                        params={"limit": cfg.limit},
                        json={"q": query},
                    )
                    resp.raise_for_status()
                    body = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "SuiteQL [%s] rejected: status=%s body=%s",
                    cfg.record_type,
                    exc.response.status_code,
                    exc.response.text[:300],
                )
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "status_code": exc.response.status_code,
                            "error": exc.response.text[:500],
                        },
                    }
                }
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"SuiteQL request failed: {exc}"},
                    }
                }

            items = body.get("items", []) if isinstance(body, dict) else []
            logger.info(
                "SuiteQL [%s] returned %d rows (total=%s)",
                cfg.record_type,
                len(items),
                body.get("totalResults") if isinstance(body, dict) else "?",
            )

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "record_type": cfg.record_type,
                        "count": len(items),
                        "items": items,
                    },
                }
            }

        return fetch_netsuite_entity_node
