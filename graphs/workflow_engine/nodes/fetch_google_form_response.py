"""Fetch Google Form Responses node — lists form submissions via the Google Forms API.

Reuses the credential reveal + token minting from fetch_google_form_structure to keep
the auth dance DRY across the two Google Forms executors. Calls
GET https://forms.googleapis.com/v1/forms/{form_id}/responses?pageSize=N&pageToken=T
and writes the response list (plus next_page_token) to state.

Mirrors the surface in api/src/api/google_forms/google_forms.api.ts (getFormResponse).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from jose.exceptions import JOSEError
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.nodes.fetch_google_form_structure import (
    _GOOGLE_FORMS_API,
    _get_google_forms_credentials,
    _mint_access_token,
    _parse_service_account,
)
from graphs.workflow_engine.schema import FetchGoogleFormResponseConfig

logger = logging.getLogger(__name__)


class FetchGoogleFormResponseExecutor(NodeExecutor):
    """List Google Form submissions and write them to state."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = FetchGoogleFormResponseConfig(**config)

        async def fetch_google_form_response_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
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

            params: dict[str, Any] = {"pageSize": cfg.page_size}
            if cfg.page_token:
                params["pageToken"] = cfg.page_token

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
                    resp = await client.get(
                        f"{_GOOGLE_FORMS_API}/{form_id}/responses",
                        headers={"Authorization": f"Bearer {access_token}"},
                        params=params,
                    )
                    resp.raise_for_status()
                    body = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Google Forms responses rejected: status=%s body=%s",
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

            responses = body.get("responses", []) if isinstance(body, dict) else []
            next_page_token = body.get("nextPageToken") if isinstance(body, dict) else None
            logger.info(
                "Google Forms returned %d responses (form_id=%s, next_page=%s)",
                len(responses),
                form_id,
                bool(next_page_token),
            )

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "form_id": form_id,
                        "count": len(responses),
                        "responses": responses,
                        "next_page_token": next_page_token,
                    },
                }
            }

        return fetch_google_form_response_node
