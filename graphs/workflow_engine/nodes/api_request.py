"""API Request node executor — makes HTTP requests and stores results in state."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import ApiRequestConfig

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}


class ApiRequestExecutor(NodeExecutor):
    """Execute an HTTP request and write the response into state["data"]."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ApiRequestConfig(**config)

        async def api_request_node(
            state: dict[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            # Resolve templates in URL and headers
            resolved_url = resolve_templates(cfg.url, data)
            resolved_headers = {
                k: resolve_templates(v, data) for k, v in cfg.headers.items()
            }

            # Validate URL scheme
            parsed = urlparse(resolved_url)
            if parsed.scheme not in _ALLOWED_SCHEMES:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "status_code": 0,
                            "body": None,
                            "headers": {},
                            "error": (
                                f"Invalid URL scheme '{parsed.scheme}'. "
                                f"Only {_ALLOWED_SCHEMES} are allowed."
                            ),
                        },
                    }
                }

            # Resolve templates in body if present
            resolved_body = None
            if cfg.body is not None:
                resolved_body = {
                    k: resolve_templates(str(v), data) if isinstance(v, str) else v
                    for k, v in cfg.body.items()
                }

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(cfg.timeout_seconds)
                ) as client:
                    response = await client.request(
                        method=cfg.method,
                        url=resolved_url,
                        headers=resolved_headers,
                        json=resolved_body,
                    )

                    # Parse response body
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        try:
                            body = response.json()
                        except Exception:
                            body = response.text
                    else:
                        body = response.text

                    result = {
                        "status_code": response.status_code,
                        "body": body,
                        "headers": dict(response.headers),
                    }

            except httpx.TimeoutException:
                result = {
                    "status_code": 0,
                    "body": None,
                    "headers": {},
                    "error": f"Request timed out after {cfg.timeout_seconds}s",
                }
            except httpx.RequestError as exc:
                result = {
                    "status_code": 0,
                    "body": None,
                    "headers": {},
                    "error": f"Request failed: {exc}",
                }

            logger.info(
                "Workflow api_request [%s %s] -> status=%s",
                cfg.method,
                resolved_url,
                result.get("status_code"),
            )

            return {"data": {**data, cfg.response_key: result}}

        return api_request_node
