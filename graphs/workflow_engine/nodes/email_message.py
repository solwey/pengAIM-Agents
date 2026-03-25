"""Email Message node executor — sends email via direct SMTP."""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

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

            # SMTP settings: node config overrides env vars
            host = (
                resolve_templates(cfg.smtp_host, data)
                if cfg.smtp_host
                else os.environ.get("SMTP_HOST", "smtp.gmail.com")
            )
            port = cfg.smtp_port or int(os.environ.get("SMTP_PORT", "587"))
            user = (
                resolve_templates(cfg.smtp_user, data)
                if cfg.smtp_user
                else os.environ.get("SMTP_USER", "")
            )
            password = (
                resolve_templates(cfg.smtp_password, data)
                if cfg.smtp_password
                else os.environ.get("SMTP_PASSWORD", "")
            )
            from_addr = (
                resolve_templates(cfg.smtp_from, data)
                if cfg.smtp_from
                else os.environ.get("SMTP_FROM", "")
            )

            if not user or not password:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {
                            "ok": False,
                            "error": "SMTP credentials not configured",
                        },
                    }
                }

            result: dict[str, Any]
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = resolved_subject
                msg["From"] = from_addr or user
                msg["To"] = resolved_to

                if resolved_text:
                    msg.attach(MIMEText(resolved_text, "plain"))
                msg.attach(MIMEText(resolved_html, "html"))

                with smtplib.SMTP(host, port, timeout=30) as server:
                    server.starttls()
                    server.login(user, password)
                    server.send_message(msg)

                result = {"ok": True}
                logger.info("Email sent to %s via %s", resolved_to, host)

            except smtplib.SMTPException as exc:
                result = {"ok": False, "error": f"SMTP error: {exc}"}
                logger.warning("Email send failed: %s", exc)
            except TimeoutError:
                result = {"ok": False, "error": "SMTP connection timed out"}
            except OSError as exc:
                result = {"ok": False, "error": f"Connection error: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return email_message_node