"""Activate NetSuite node — mints an ES256 JWT and exchanges it for an OAuth2 access token.

Credentials (account_id, client_id, certificate_id, private_key) are fetched from
pengAIM-RAG as team-scoped integration keys (stored via /api/v1/netsuite/connect).
The resulting access token is written to state so downstream NetSuite nodes can use it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from jose import jwt as jose_jwt
from jose.exceptions import JOSEError
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import ActivateNetsuiteConfig

logger = logging.getLogger(__name__)

_NETSUITE_PROVIDER = "netsuite"
_ACCOUNT_ID_KEY = "netsuite_account_id"
_CLIENT_ID_KEY = "netsuite_client_id"
_CERTIFICATE_ID_KEY = "netsuite_certificate_id"
_PRIVATE_KEY_KEY = "netsuite_private_key"


async def _create_token(
    account_id: str,
    client_id: str,
    certificate_id: str,
    private_key: str,
) -> dict[str, Any]:
    """Mint a NetSuite OAuth2 access token via the JWT client_credentials flow.

    Returns a dict with access_token, token_type, and absolute expires_at (epoch seconds).
    Raises JOSEError on signing issues, httpx.HTTPStatusError on non-2xx, and
    httpx.RequestError / httpx.TimeoutException on network failures.
    """
    token_url = f"https://{account_id}.suitetalk.api.netsuite.com/services/rest/auth/oauth2/v1/token"

    now = int(time.time())
    assertion = jose_jwt.encode(
        {
            "iss": client_id,
            "scope": "rest_webservices",
            "aud": token_url,
            "iat": now,
            "exp": now + 3600,
        },
        private_key,
        algorithm="ES256",
        headers={"kid": certificate_id},
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(15)) as client:
        resp = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": assertion,
            },
        )
        resp.raise_for_status()
        body = resp.json()

    access_token = body.get("access_token")
    if not access_token:
        raise ValueError("Token endpoint returned no access_token")

    expires_in = int(body.get("expires_in", 3600))
    return {
        "access_token": access_token,
        "token_type": body.get("token_type", "Bearer"),
        "expires_at": now + expires_in,
    }


async def _get_netsuite_credential(
    config: RunnableConfig,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Fetch the four NetSuite credentials in parallel from pengAIM-RAG.

    Hits the tenant-scoped `/keys/by-name/reveal` endpoint once per credential.
    Returns (account_id, client_id, certificate_id, private_key). Any element
    is None when the credential is not connected or the request fails.
    """
    configurable = config.get("configurable", {})
    auth_token = configurable.get("auth_token", "")
    tenant_uuid = configurable.get("tenant_uuid", "")

    if not auth_token or not tenant_uuid:
        logger.warning(
            "NetSuite credentials unavailable (auth_token=%s, tenant_uuid=%s)",
            bool(auth_token),
            bool(tenant_uuid),
        )
        return None, None, None, None

    url = f"{settings.graphs.RAG_API_URL}/tenant/{tenant_uuid}/keys/by-name/reveal"
    headers = {"authorization": auth_token, "Accept": "text/plain"}

    async def _fetch(http: httpx.AsyncClient, name: str) -> str | None:
        try:
            resp = await http.get(url, headers=headers, params={"provider": _NETSUITE_PROVIDER, "name": name})
            resp.raise_for_status()
            return resp.text
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to reveal %s/%s: %s", _NETSUITE_PROVIDER, name, exc)
            return None

    async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
        return await asyncio.gather(
            _fetch(client, _ACCOUNT_ID_KEY),
            _fetch(client, _CLIENT_ID_KEY),
            _fetch(client, _CERTIFICATE_ID_KEY),
            _fetch(client, _PRIVATE_KEY_KEY),
        )


class ActivateNetsuiteExecutor(NodeExecutor):
    """Mint a NetSuite OAuth2 access token and store it in state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ActivateNetsuiteConfig(**config)

        async def activate_netsuite_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            account_id, client_id, certificate_id, private_key = await _get_netsuite_credential(config)

            missing = [
                name
                for name, value in (
                    (_ACCOUNT_ID_KEY, account_id),
                    (_CLIENT_ID_KEY, client_id),
                    (_CERTIFICATE_ID_KEY, certificate_id),
                    (_PRIVATE_KEY_KEY, private_key),
                )
                if not value
            ]
            if missing:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Missing NetSuite credentials: {', '.join(missing)}",
                        },
                    }
                }

            try:
                token_data = await _create_token(account_id, client_id, certificate_id, private_key)

            except JOSEError as exc:
                logger.exception("Failed to sign NetSuite JWT assertion")
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"JWT signing failed: {exc}"},
                    }
                }
            except httpx.HTTPStatusError as exc:
                logger.warning("NetSuite rejected token request: %s", exc.response.text[:300])
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"NetSuite rejected credentials (status {exc.response.status_code})",
                        },
                    }
                }
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"Token request failed: {exc}"},
                    }
                }
            except ValueError as exc:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": str(exc)},
                    }
                }

            result = {"ok": True, "account_id": account_id, **token_data}
            logger.info(
                "NetSuite token activated for account %s (expires_at=%s)",
                account_id,
                token_data["expires_at"],
            )
            return {"data": {**data, cfg.response_key: result}}

        return activate_netsuite_node
