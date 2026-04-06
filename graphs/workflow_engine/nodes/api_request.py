"""API Request node executor — makes HTTP requests and stores results in state."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import ApiRequestConfig

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}


def _parse_response_body(response: httpx.Response) -> Any:
    """Parse response body as JSON if content-type indicates it, else text."""
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            return response.json()
        except Exception:
            return response.text
    return response.text


class ApiRequestExecutor(NodeExecutor):
    """Execute an HTTP request and write the response into state["data"]."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ApiRequestConfig(**config)

        async def api_request_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            # Resolve templates in URL and headers
            resolved_url = resolve_templates(cfg.url, data)
            resolved_headers = {k: resolve_templates(v, data) for k, v in cfg.headers.items()}

            # Validate URL scheme
            parsed = urlparse(resolved_url)
            if parsed.scheme not in _ALLOWED_SCHEMES:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "status_code": None,
                            "body": None,
                            "headers": {},
                            "error": (f"Invalid URL scheme '{parsed.scheme}'. Only {_ALLOWED_SCHEMES} are allowed."),
                        },
                    }
                }

            # Resolve templates in body if present
            resolved_body = None
            if cfg.body is not None:
                resolved_body = {
                    k: resolve_templates(str(v), data) if isinstance(v, str) else v for k, v in cfg.body.items()
                }

            max_attempts = cfg.retry_count + 1
            result: dict[str, Any] = {}

            for attempt in range(max_attempts):
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout_seconds)) as client:
                        response = await client.request(
                            method=cfg.method,
                            url=resolved_url,
                            headers=resolved_headers,
                            json=resolved_body,
                        )

                        body = _parse_response_body(response)
                        result = {
                            "status_code": response.status_code,
                            "body": body,
                            "headers": dict(response.headers),
                        }

                        # Retry on specific status codes
                        if response.status_code in cfg.retry_on_status and attempt < max_attempts - 1:
                            logger.info(
                                "Workflow api_request [%s %s] retry %d/%d (status=%s)",
                                cfg.method,
                                resolved_url,
                                attempt + 1,
                                cfg.retry_count,
                                response.status_code,
                            )
                            await asyncio.sleep(cfg.retry_delay_seconds)
                            continue

                        break  # Success or non-retryable status

                except httpx.TimeoutException:
                    result = {
                        "status_code": None,
                        "body": None,
                        "headers": {},
                        "error": f"Request timed out after {cfg.timeout_seconds}s",
                    }
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(cfg.retry_delay_seconds)
                        continue
                    break

                except httpx.RequestError as exc:
                    result = {
                        "status_code": None,
                        "body": None,
                        "headers": {},
                        "error": f"Request failed: {exc}",
                    }
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(cfg.retry_delay_seconds)
                        continue
                    break

            if cfg.retry_count > 0:
                result["attempts"] = attempt + 1  # noqa: F821

            logger.info(
                "Workflow api_request [%s %s] -> status=%s%s",
                cfg.method,
                resolved_url,
                result.get("status_code"),
                f" (attempts={result['attempts']})" if "attempts" in result else "",
            )

            return {"data": {**data, cfg.response_key: result}}

        return api_request_node
