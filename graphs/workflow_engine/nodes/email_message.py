"""Email Message node executor — sends email via direct SMTP."""

from __future__ import annotations

import asyncio
import base64
import logging
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import (
    NodeExecutor,
    _get_http_client,
    resolve_templates,
    reveal_api_key,
)
from graphs.workflow_engine.schema import EmailAttachment, EmailMessageConfig

logger = logging.getLogger(__name__)


def _split_addrs(value: str) -> list[str]:
    return [addr.strip() for addr in value.split(",") if addr.strip()]


def _send_smtp(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    use_ssl: bool,
    msg: MIMEMultipart,
    rcpts: list[str],
) -> None:
    """Send via SMTP (blocking — call via asyncio.to_thread)."""
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=30) as server:
            server.login(user, password)
            server.send_message(msg, to_addrs=rcpts)
        return
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg, to_addrs=rcpts)


async def _fetch_attachment(att: EmailAttachment) -> bytes | None:
    if att.content_b64:
        try:
            return base64.b64decode(att.content_b64)
        except (ValueError, TypeError) as exc:
            logger.warning("Attachment '%s': bad base64 (%s)", att.filename, exc)
            return None
    if att.url:
        client = _get_http_client()
        try:
            resp = await client.get(att.url, timeout=httpx.Timeout(30))
            resp.raise_for_status()
            return resp.content
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as exc:
            logger.warning("Attachment '%s' fetch from %s failed: %s", att.filename, att.url, exc)
            return None
    return None


class EmailMessageExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = EmailMessageConfig(**config)

        async def email_message_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})

            resolved_to = resolve_templates(cfg.to, data)
            resolved_cc = resolve_templates(cfg.cc, data) if cfg.cc else ""
            resolved_bcc = resolve_templates(cfg.bcc, data) if cfg.bcc else ""
            resolved_subject = resolve_templates(cfg.subject, data)
            resolved_html = resolve_templates(cfg.html_body, data)
            resolved_text = resolve_templates(cfg.text_body, data) if cfg.text_body else None

            host: str = resolve_templates(cfg.smtp_host, data) if cfg.smtp_host else (settings.graphs.SMTP_HOST or "")
            port: int = cfg.smtp_port or settings.graphs.SMTP_PORT or 587
            user: str = resolve_templates(cfg.smtp_user, data) if cfg.smtp_user else (settings.graphs.SMTP_USER or "")

            password = ""  # nosec B105
            if cfg.smtp_password_key_id:
                password = await reveal_api_key(config, cfg.smtp_password_key_id) or ""
            if not password:
                password = settings.graphs.SMTP_PASSWORD or ""

            from_addr: str = (
                resolve_templates(cfg.smtp_from, data) if cfg.smtp_from else (settings.graphs.SMTP_FROM or "")
            )
            from_addr = from_addr or user
            from_display: str = formataddr((cfg.from_name, from_addr)) if cfg.from_name else from_addr

            if not user or not password:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": "SMTP credentials not configured"},
                    }
                }

            msg = MIMEMultipart("mixed")
            msg["Subject"] = resolved_subject
            msg["From"] = from_display
            msg["To"] = resolved_to
            if resolved_cc:
                msg["Cc"] = resolved_cc

            body_part = MIMEMultipart("alternative")
            if resolved_text:
                body_part.attach(MIMEText(resolved_text, "plain"))
            body_part.attach(MIMEText(resolved_html, "html"))
            msg.attach(body_part)

            for att in cfg.attachments:
                payload = await _fetch_attachment(att)
                if payload is None:
                    return {
                        "data": {
                            **data,
                            cfg.response_key: {
                                "ok": False,
                                "error": f"Attachment '{att.filename}' could not be loaded",
                            },
                        }
                    }
                part = MIMEApplication(payload, _subtype=att.content_type.split("/")[-1] or "octet-stream")
                part.add_header("Content-Disposition", "attachment", filename=att.filename)
                if att.content_type:
                    part.replace_header("Content-Type", att.content_type)
                msg.attach(part)

            rcpts = _split_addrs(resolved_to) + _split_addrs(resolved_cc) + _split_addrs(resolved_bcc)
            if not rcpts:
                return {
                    "data": {
                        **data,
                        cfg.response_key: {"ok": False, "error": "No recipients (to/cc/bcc all empty)"},
                    }
                }

            result: dict[str, Any]
            try:
                await asyncio.to_thread(
                    _send_smtp,
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    use_ssl=cfg.use_ssl,
                    msg=msg,
                    rcpts=rcpts,
                )
                result = {"ok": True, "recipients": len(rcpts)}
                logger.info("Email sent to %d recipient(s) via %s:%d", len(rcpts), host, port)
            except smtplib.SMTPException as exc:
                logger.warning("Email send failed: %s", exc)
                result = {"ok": False, "error": f"SMTP error: {exc}"}
            except TimeoutError:
                result = {"ok": False, "error": "SMTP connection timed out"}
            except OSError as exc:
                result = {"ok": False, "error": f"Connection error: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return email_message_node


__all__ = ["EmailMessageExecutor"]
