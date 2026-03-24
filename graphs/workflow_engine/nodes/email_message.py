"""Email Message node executor — sends email via pengAIM-RAG email service."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import EmailMessageConfig

logger = logging.getLogger(__name__)


class EmailMessageExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = EmailMessageConfig(**config)

        async def email_message_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})

            resolved_to = resolve_templates(cfg.to, data)
            resolved_subject = resolve_templates(cfg.subject, data)
            resolved_html = resolve_templates(cfg.html_body, data)
            resolved_text = (
                resolve_templates(cfg.text_body, data) if cfg.text_body else None
            )

            rag_url = os.environ.get("RAG_API_URL", "")
            if not rag_url:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "RAG_API_URL not configured",
                        },
                    }
                }

            payload: dict[str, Any] = {
                "to": resolved_to,
                "subject": resolved_subject,
                "html_body": resolved_html,
            }
            if resolved_text:
                payload["text_body"] = resolved_text

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    response = await client.post(
                        f"{rag_url}/api/v1/email/send",
                        json=payload,
                    )
                    body = response.json()
                    result = {
                        "ok": body.get("ok", False),
                    }
                    if body.get("error"):
                        result["error"] = body["error"]

                    if result["ok"]:
                        logger.info("Email sent to %s via RAG", resolved_to)
                    else:
                        logger.warning("Email send failed: %s", result.get("error"))

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Email request to RAG timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Email request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return email_message_node
