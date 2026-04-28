"""Fetch Google Form Structure node — pulls the form definition from the Google Forms API.

Two values are stored encrypted in pengAIM-RAG via /api/v1/google-forms/connect
(provider=google_forms, names=google_forms_form_id | google_forms_service_account_json).
This executor reveals both, parses the service-account JSON, mints an OAuth2 access
token via the JWT-bearer grant against oauth2.googleapis.com/token, and calls
GET https://forms.googleapis.com/v1/forms/{form_id}.

Mirrors the auth flow in rag/app/routers/google_forms.py (_mint_access_token) and the
HMISights surface in api/src/api/google_forms/google_forms.api.ts (getFormStructure).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx
from jose import jwt as jose_jwt
from jose.exceptions import JOSEError
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import FetchGoogleFormStructureConfig

logger = logging.getLogger(__name__)

# Must stay aligned with rag/app/routers/google_forms.py.
_GOOGLE_FORMS_PROVIDER = "google_forms"
_GOOGLE_FORMS_FORM_ID_NAME = "google_forms_form_id"
_GOOGLE_FORMS_SERVICE_ACCOUNT_NAME = "google_forms_service_account_json"

_GOOGLE_FORMS_SCOPES = " ".join(
    [
        "https://www.googleapis.com/auth/forms.body.readonly",
        "https://www.googleapis.com/auth/forms.responses.readonly",
    ]
)
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_FORMS_API = "https://forms.googleapis.com/v1/forms"
_JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"


async def _get_google_forms_credentials(
    config: RunnableConfig,
) -> tuple[str | None, str | None]:
    """Fetch the form_id and service-account JSON in parallel from pengAIM-RAG."""
    configurable = config.get("configurable", {})
    auth_token = configurable.get("auth_token", "")
    tenant_uuid = configurable.get("tenant_uuid", "")

    if not auth_token or not tenant_uuid:
        logger.warning(
            "Google Forms credentials unavailable (auth_token=%s, tenant_uuid=%s)",
            bool(auth_token),
            bool(tenant_uuid),
        )
        return None, None

    url = f"{settings.graphs.RAG_API_URL}/tenant/{tenant_uuid}/keys/by-name/reveal"
    headers = {"authorization": auth_token, "Accept": "text/plain"}

    async def _fetch(http: httpx.AsyncClient, name: str) -> str | None:
        try:
            resp = await http.get(url, headers=headers, params={"provider": _GOOGLE_FORMS_PROVIDER, "name": name})
            resp.raise_for_status()
            return resp.text
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to reveal %s/%s: %s", _GOOGLE_FORMS_PROVIDER, name, exc)
            return None

    async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
        return await asyncio.gather(
            _fetch(client, _GOOGLE_FORMS_FORM_ID_NAME),
            _fetch(client, _GOOGLE_FORMS_SERVICE_ACCOUNT_NAME),
        )


def _parse_service_account(raw: str) -> dict[str, Any] | None:
    """Parse and validate the service-account JSON. Returns None on any defect."""
    try:
        sa = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Service account JSON invalid: %s", exc)
        return None
    if not isinstance(sa, dict):
        return None
    if not sa.get("private_key") or not sa.get("client_email"):
        return None
    return sa


async def _mint_access_token(sa: dict[str, Any]) -> str:
    """Sign a JWT and exchange it for a Google OAuth2 access token.

    Raises JOSEError on signing failure, httpx.HTTPStatusError on non-2xx,
    httpx.RequestError / httpx.TimeoutException on network failure, and
    ValueError if the token endpoint returns no access_token.
    """
    now = int(time.time())
    token_uri = sa.get("token_uri") or _GOOGLE_TOKEN_URL
    assertion = jose_jwt.encode(
        {
            "iss": sa["client_email"],
            "scope": _GOOGLE_FORMS_SCOPES,
            "aud": token_uri,
            "iat": now,
            "exp": now + 3600,
        },
        sa["private_key"],
        algorithm="RS256",
        headers={"kid": sa.get("private_key_id", "")},
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(15)) as client:
        resp = await client.post(
            token_uri,
            data={"grant_type": _JWT_BEARER_GRANT, "assertion": assertion},
        )
        resp.raise_for_status()
        body = resp.json()

    access_token = body.get("access_token") if isinstance(body, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("Google token endpoint returned no access_token")
    return access_token


class FetchGoogleFormStructureExecutor(NodeExecutor):
    """Fetch the Google Form definition (questions, sections, settings) and write it to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchGoogleFormStructureConfig(**config)

        async def fetch_google_form_structure_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            stored_form_id, sa_json = await _get_google_forms_credentials(config)
            if not sa_json:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Missing Google Forms service account — connect Google Forms in integrations",
                        },
                    }
                }

            form_id = (cfg.form_id or stored_form_id or "").strip()
            if not form_id:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "No form_id available — set on the node or on the integration",
                        },
                    }
                }

            sa = _parse_service_account(sa_json)
            if sa is None:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "Stored Google Forms service account JSON is invalid — reconnect the integration",
                        },
                    }
                }

            try:
                access_token = await _mint_access_token(sa)
            except JOSEError as exc:
                logger.exception("Failed to sign Google service-account JWT")
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": f"JWT signing failed: {exc}"},
                    }
                }
            except httpx.HTTPStatusError as exc:
                logger.warning("Google rejected token request: %s", exc.response.text[:300])
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Google rejected service-account credentials (status {exc.response.status_code})",
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

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    resp = await client.get(
                        f"{_GOOGLE_FORMS_API}/{form_id}",
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    resp.raise_for_status()
                    form = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Google Forms rejected request: status=%s body=%s",
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
                        cfg.response_key: {"ok": False, "error": f"Google Forms request failed: {exc}"},
                    }
                }

            items = form.get("items", []) if isinstance(form, dict) else []
            logger.info("Google Forms returned form structure (form_id=%s, items=%d)", form_id, len(items))

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "form_id": form_id,
                        "form": form,
                    },
                }
            }

        return fetch_google_form_structure_node
