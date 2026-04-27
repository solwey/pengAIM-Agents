"""Activate Bill.com node — exchanges stored credentials for a session.

Bill.com requires four credentials (devKey, username, password, organizationId)
all stored encrypted in pengAIM-RAG via /api/v1/bill/connect (provider=bill,
names=bill_api_key | bill_username | bill_password | bill_organization_id).
This executor reveals all four, calls POST /login on the Bill.com Connect v3
API, and writes the resulting sessionId to state under cfg.response_key so
downstream Bill.com nodes (e.g. fetch_bill_entity) can use it without
re-authenticating per call.

Mirrors the auth pattern in api/src/api/bill/bill.api.ts (login).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import ActivateBillcomConfig

logger = logging.getLogger(__name__)

# Must stay aligned with rag/app/routers/bill.py (BILL_PROVIDER, *_KEY, BILL_BASE_URL).
_BILL_PROVIDER = "bill"
_BILL_API_KEY_NAME = "bill_api_key"
_BILL_USERNAME_NAME = "bill_username"
_BILL_PASSWORD_NAME = "bill_password"
_BILL_ORGANIZATION_ID_NAME = "bill_organization_id"
_BILL_BASE_URL = "https://gateway.prod.bill.com/connect/v3"


async def _get_bill_credentials(
    config: RunnableConfig,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Fetch the four Bill.com credentials in parallel from pengAIM-RAG.

    Hits the tenant-scoped `/keys/by-name/reveal` endpoint once per credential.
    Returns (api_key, username, password, organization_id). Any element is None
    when the credential is not connected or the request fails.
    """
    configurable = config.get("configurable", {})
    auth_token = configurable.get("auth_token", "")
    tenant_uuid = configurable.get("tenant_uuid", "")

    if not auth_token or not tenant_uuid:
        logger.warning(
            "Bill.com credentials unavailable (auth_token=%s, tenant_uuid=%s)",
            bool(auth_token),
            bool(tenant_uuid),
        )
        return None, None, None, None

    url = f"{settings.graphs.RAG_API_URL}/tenant/{tenant_uuid}/keys/by-name/reveal"
    headers = {"authorization": auth_token, "Accept": "text/plain"}

    async def _fetch(http: httpx.AsyncClient, name: str) -> str | None:
        try:
            resp = await http.get(url, headers=headers, params={"provider": _BILL_PROVIDER, "name": name})
            resp.raise_for_status()
            return resp.text
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to reveal %s/%s: %s", _BILL_PROVIDER, name, exc)
            return None

    async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
        return await asyncio.gather(
            _fetch(client, _BILL_API_KEY_NAME),
            _fetch(client, _BILL_USERNAME_NAME),
            _fetch(client, _BILL_PASSWORD_NAME),
            _fetch(client, _BILL_ORGANIZATION_ID_NAME),
        )


async def _login(
    api_key: str,
    username: str,
    password: str,
    organization_id: str,
) -> str:
    """POST /login and return the sessionId on success.

    Raises httpx.HTTPStatusError on non-2xx, httpx.RequestError /
    httpx.TimeoutException on network failures, and ValueError when the
    response body is missing a sessionId.
    """
    payload = {
        "username": username,
        "password": password,
        "organizationId": organization_id,
        "devKey": api_key,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(20)) as client:
        resp = await client.post(
            f"{_BILL_BASE_URL}/login",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        body = resp.json() if resp.content else {}

    session_id = body.get("sessionId") if isinstance(body, dict) else None
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Bill.com login returned no sessionId")
    return session_id


class ActivateBillcomExecutor(NodeExecutor):
    """Log into Bill.com and store the session in state for downstream nodes."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ActivateBillcomConfig(**config)

        async def activate_billcom_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            api_key, username, password, organization_id = await _get_bill_credentials(config)

            missing = [
                name
                for name, value in (
                    (_BILL_API_KEY_NAME, api_key),
                    (_BILL_USERNAME_NAME, username),
                    (_BILL_PASSWORD_NAME, password),
                    (_BILL_ORGANIZATION_ID_NAME, organization_id),
                )
                if not value
            ]
            if missing:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Missing Bill.com credentials: {', '.join(missing)}",
                        },
                    }
                }

            try:
                session_id = await _login(api_key, username, password, organization_id)
            except httpx.HTTPStatusError as exc:
                logger.warning("Bill.com rejected login: %s", exc.response.text[:300])
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Bill.com rejected credentials (status {exc.response.status_code})",
                        },
                    }
                }
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"Login request failed: {exc}"},
                    }
                }
            except ValueError as exc:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": str(exc)},
                    }
                }

            logger.info("Bill.com session activated for org %s", organization_id)
            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "session_id": session_id,
                        "dev_key": api_key,
                        "organization_id": organization_id,
                    },
                }
            }

        return activate_billcom_node
