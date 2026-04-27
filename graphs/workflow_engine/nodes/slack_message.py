"""Slack Message node executor — sends a message to Slack via incoming webhook."""

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
from graphs.workflow_engine.schema import SlackMessageConfig

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS = {"hooks.slack.com"}


class SlackMessageExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = SlackMessageConfig(**config)

        async def slack_message_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})

            resolved_url = resolve_templates(cfg.webhook_url, data)
            resolved_message = resolve_templates(cfg.message, data)

            # Validate URL
            parsed = urlparse(resolved_url)
            if parsed.scheme not in {"http", "https"}:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Invalid URL scheme '{parsed.scheme}'",
                        },
                    }
                }

            payload: dict[str, Any] = {"text": resolved_message}
            if cfg.username:
                payload["username"] = resolve_templates(cfg.username, data)
            if cfg.icon_emoji:
                payload["icon_emoji"] = cfg.icon_emoji

            result: dict[str, Any]
            try:
                response = await http_request_with_retry(
                    "POST",
                    resolved_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout_seconds=cfg.timeout_seconds,
                    op_name="slack_message",
                )
                result = {
                    "ok": response.status_code == 200,
                    "status_code": response.status_code,
                    "body": response.text,
                }
                if response.status_code == 200:
                    logger.info(
                        "Slack message sent successfully to %s",
                        parsed.hostname,
                    )
                else:
                    logger.warning(
                        "Slack webhook returned %d: %s",
                        response.status_code,
                        response.text[:200],
                    )
            except httpx.TimeoutException:
                result = {
                    "ok": False,
                    "error": f"Slack webhook timed out after {cfg.timeout_seconds}s",
                }
            except httpx.RequestError as exc:
                result = {
                    "ok": False,
                    "error": f"Slack webhook request failed: {exc}",
                }

            return {"data": {**data, cfg.response_key: result}}

        return slack_message_node
