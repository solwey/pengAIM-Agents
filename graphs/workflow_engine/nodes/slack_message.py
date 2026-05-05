"""Slack Message node executor — sends a message to Slack via incoming webhook."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import (
    NodeExecutor,
    http_request_with_retry,
    resolve_field,
    resolve_templates,
)
from graphs.workflow_engine.schema import SlackMessageConfig

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS = {"hooks.slack.com"}


def _resolve_blocks(blocks: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    """Recursively resolve {{templates}} inside Block Kit JSON."""

    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            return resolve_templates(node, data)
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        return node

    return _walk(blocks)


class SlackMessageExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = SlackMessageConfig(**config)

        async def slack_message_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})

            resolved_url = resolve_templates(cfg.webhook_url, data)
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
            if parsed.hostname not in _ALLOWED_HOSTS:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": f"Slack webhook host '{parsed.hostname}' not allowed; expected one of {sorted(_ALLOWED_HOSTS)}",
                        },
                    }
                }

            payload: dict[str, Any] = {}
            if cfg.message:
                payload["text"] = resolve_templates(cfg.message, data)
            if cfg.blocks:
                payload["blocks"] = _resolve_blocks(cfg.blocks, data)
            if cfg.thread_ts:
                payload["thread_ts"] = (
                    resolve_templates(cfg.thread_ts, data) or resolve_field(data, cfg.thread_ts) or cfg.thread_ts
                )
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
                    logger.info("Slack message sent successfully to %s", parsed.hostname)
                else:
                    logger.warning(
                        "Slack webhook returned %d: %s (payload: %s)",
                        response.status_code,
                        response.text[:200],
                        json.dumps(payload)[:200],
                    )
            except httpx.TimeoutException:
                result = {"ok": False, "error": f"Slack webhook timed out after {cfg.timeout_seconds}s"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Slack webhook request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return slack_message_node
