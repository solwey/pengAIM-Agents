"""API Request node executor — makes HTTP requests and stores results in state."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import (
    NodeExecutor,
    http_request_with_retry,
    resolve_templates,
)
from graphs.workflow_engine.schema import ApiRequestConfig

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}


def _parse_response_body(response: httpx.Response) -> Any:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.text


def _scheme_error(data: dict[str, Any], response_key: str, scheme: str) -> dict[str, Any]:
    return {
        "data": {
            **data,
            response_key: {
                "status_code": None,
                "body": None,
                "headers": {},
                "error": f"Invalid URL scheme '{scheme}'. Only {sorted(_ALLOWED_SCHEMES)} are allowed.",
            },
        }
    }


class ApiRequestExecutor(NodeExecutor):
    """Execute an HTTP request and write the response into state["data"]."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ApiRequestConfig(**config)

        async def api_request_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            resolved_url = resolve_templates(cfg.url, data)
            resolved_headers = {k: resolve_templates(v, data) for k, v in cfg.headers.items()}
            resolved_params = {k: resolve_templates(v, data) for k, v in cfg.params.items()}

            parsed = urlparse(resolved_url)
            if parsed.scheme not in _ALLOWED_SCHEMES:
                return _scheme_error(data, cfg.response_key, parsed.scheme)

            resolved_body: dict[str, Any] | None = None
            if cfg.body is not None:
                resolved_body = {
                    k: resolve_templates(str(v), data) if isinstance(v, str) else v for k, v in cfg.body.items()
                }

            result: dict[str, Any]
            try:
                response = await http_request_with_retry(
                    cfg.method,
                    resolved_url,
                    headers=resolved_headers or None,
                    params=resolved_params or None,
                    json=resolved_body,
                    timeout_seconds=cfg.timeout_seconds,
                    max_attempts=cfg.retry_count + 1,
                    retry_on_status=tuple(cfg.retry_on_status),
                    op_name="api_request",
                )
                result = {
                    "status_code": response.status_code,
                    "body": _parse_response_body(response),
                    "headers": dict(response.headers),
                }
            except httpx.TimeoutException:
                result = {
                    "status_code": None,
                    "body": None,
                    "headers": {},
                    "error": f"Request timed out after {cfg.timeout_seconds}s",
                }
            except httpx.RequestError as exc:
                result = {
                    "status_code": None,
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
