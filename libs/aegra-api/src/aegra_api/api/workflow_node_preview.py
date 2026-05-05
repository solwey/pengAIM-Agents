"""Workflow node preview endpoints — single-node dry-runs without WorkflowRun persistence.

Used by the workflow editor to let users test a node's config against a sample input
before saving the full workflow. Preview endpoints intentionally clamp expensive
behaviour (retries, token budgets, real email sends) so they are safe to call
interactively from the UI.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, Path, Request
from graphs.workflow_engine.nodes.base import fetch_ingestion_configurable, resolve_templates, reveal_api_key
from graphs.workflow_engine.nodes.icp_score import resolve_llm_key, score_account
from graphs.workflow_engine.schema import (
    ApiRequestConfig,
    EmailMessageConfig,
    ICPScoreConfig,
    LLMCompleteConfig,
    SlackMessageConfig,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from aegra_api.settings import settings

from ..core.auth_deps import auth_dependency, get_current_user
from ..models.auth import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workflow Node Preview"], dependencies=auth_dependency)

_ALLOWED_SCHEMES: set[str] = {"http", "https"}
_PREVIEW_BODY_LIMIT: int = 500
_PREVIEW_LLM_MAX_TOKENS: int = 300
_PREVIEW_HTTP_TIMEOUT_CAP: int = 30
_PREVIEW_SMTP_TIMEOUT: int = 15
_SLACK_PREVIEW_PREFIX: str = "🧪 [PREVIEW] "


# ── ICP Score ────────────────────────────────────────────────


class ICPScorePreviewRequest(BaseModel):
    account_data: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class ICPScorePreviewResponse(BaseModel):
    ok: bool
    score: int | None = None
    status: str | None = None
    reasoning: str | None = None
    error: str | None = None


@router.post("/workflow-nodes/icp-score/preview", response_model=ICPScorePreviewResponse)
async def preview_icp_score(
    body: ICPScorePreviewRequest,
    request: Request,
    tenant_uuid: str = Path(..., description="Tenant UUID"),
    user: User = Depends(get_current_user),
) -> ICPScorePreviewResponse:
    """Run the ICP score node on a single account without persisting a WorkflowRun."""
    if not body.account_data:
        return ICPScorePreviewResponse(ok=False, error="account_data is required")

    try:
        cfg = ICPScoreConfig(**body.config)
    except ValidationError as exc:
        return ICPScorePreviewResponse(ok=False, error=_format_validation_error(exc))

    auth_token = request.headers.get("authorization", "")

    ingestion_cfg = await fetch_ingestion_configurable(auth_token, tenant_uuid)
    synthetic_config: RunnableConfig = {
        "configurable": {"auth_token": auth_token, "tenant_uuid": tenant_uuid, **ingestion_cfg}
    }
    api_key = await resolve_llm_key(synthetic_config)
    effective_model = cfg.model or ingestion_cfg.get("llm_model") or ""

    result = await score_account(
        body.account_data,
        model=effective_model,
        hot_threshold=cfg.hot_threshold,
        warm_threshold=cfg.warm_threshold,
        custom_criteria=cfg.custom_criteria,
        auth_token=auth_token,
        api_key=api_key,
    )

    return ICPScorePreviewResponse(
        ok=bool(result.get("ok")),
        score=result.get("score"),
        status=result.get("status"),
        reasoning=result.get("reasoning"),
        error=result.get("error"),
    )


# ── API Request ──────────────────────────────────────────────


class ApiRequestPreviewRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    sample_data: dict[str, Any] = Field(default_factory=dict)


class ApiRequestPreviewResponse(BaseModel):
    ok: bool
    status_code: int | None = None
    duration_ms: int | None = None
    resolved_url: str | None = None
    body_preview: str | None = None
    body_truncated: bool = False
    headers: dict[str, str] | None = None
    error: str | None = None


@router.post("/workflow-nodes/api-request/preview", response_model=ApiRequestPreviewResponse)
async def preview_api_request(
    body: ApiRequestPreviewRequest,
    user: User = Depends(get_current_user),
) -> ApiRequestPreviewResponse:
    """Execute a single HTTP request without retries; truncate the body for preview."""
    if not body.config.get("url"):
        return ApiRequestPreviewResponse(ok=False, error="url is required")

    clamped_config: dict[str, Any] = {**body.config, "retry_count": 0}
    try:
        cfg = ApiRequestConfig(**clamped_config)
    except ValidationError as exc:
        return ApiRequestPreviewResponse(ok=False, error=_format_validation_error(exc))

    return await _run_api_request_preview(cfg, body.sample_data)


async def _run_api_request_preview(
    cfg: ApiRequestConfig,
    sample_data: dict[str, Any],
) -> ApiRequestPreviewResponse:
    resolved_url = resolve_templates(cfg.url, sample_data)
    parsed = urlparse(resolved_url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return ApiRequestPreviewResponse(
            ok=False,
            resolved_url=resolved_url,
            error=f"Invalid URL scheme '{parsed.scheme}'. Only http/https allowed.",
        )

    resolved_headers = {k: resolve_templates(v, sample_data) for k, v in cfg.headers.items()}
    resolved_body: dict[str, Any] | None = None
    if cfg.body is not None:
        resolved_body = {
            k: resolve_templates(str(v), sample_data) if isinstance(v, str) else v for k, v in cfg.body.items()
        }

    timeout_seconds = min(cfg.timeout_seconds, _PREVIEW_HTTP_TIMEOUT_CAP)
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            response = await client.request(
                method=cfg.method,
                url=resolved_url,
                headers=resolved_headers,
                json=resolved_body,
            )
    except httpx.TimeoutException:
        return ApiRequestPreviewResponse(
            ok=False,
            resolved_url=resolved_url,
            error=f"Request timed out after {timeout_seconds}s",
        )
    except httpx.RequestError as exc:
        return ApiRequestPreviewResponse(
            ok=False,
            resolved_url=resolved_url,
            error=f"Request failed: {exc}",
        )

    duration_ms = int((time.monotonic() - start) * 1000)
    body_text = response.text
    truncated = len(body_text) > _PREVIEW_BODY_LIMIT
    return ApiRequestPreviewResponse(
        ok=True,
        status_code=response.status_code,
        duration_ms=duration_ms,
        resolved_url=resolved_url,
        body_preview=body_text[:_PREVIEW_BODY_LIMIT] + ("…" if truncated else ""),
        body_truncated=truncated,
        headers=dict(response.headers),
    )


# ── Slack Message ────────────────────────────────────────────


class SlackMessagePreviewRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    sample_data: dict[str, Any] = Field(default_factory=dict)


class SlackMessagePreviewResponse(BaseModel):
    ok: bool
    status_code: int | None = None
    sent_text: str | None = None
    response_body: str | None = None
    error: str | None = None


@router.post("/workflow-nodes/slack-message/preview", response_model=SlackMessagePreviewResponse)
async def preview_slack_message(
    body: SlackMessagePreviewRequest,
    user: User = Depends(get_current_user),
) -> SlackMessagePreviewResponse:
    """Send the configured Slack message with a visible [PREVIEW] prefix so the channel knows it's a test."""
    try:
        cfg = SlackMessageConfig(**body.config)
    except ValidationError as exc:
        return SlackMessagePreviewResponse(ok=False, error=_format_validation_error(exc))

    return await _run_slack_preview(cfg, body.sample_data)


async def _run_slack_preview(
    cfg: SlackMessageConfig,
    sample_data: dict[str, Any],
) -> SlackMessagePreviewResponse:
    resolved_url = resolve_templates(cfg.webhook_url, sample_data)
    parsed = urlparse(resolved_url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return SlackMessagePreviewResponse(
            ok=False,
            error=f"Invalid URL scheme '{parsed.scheme}'",
        )

    resolved_message = resolve_templates(cfg.message, sample_data)
    sent_text = f"{_SLACK_PREVIEW_PREFIX}{resolved_message}"

    payload: dict[str, Any] = {"text": sent_text}
    if cfg.username:
        payload["username"] = resolve_templates(cfg.username, sample_data)
    if cfg.icon_emoji:
        payload["icon_emoji"] = cfg.icon_emoji

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15)) as client:
            response = await client.post(
                resolved_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
    except httpx.TimeoutException:
        return SlackMessagePreviewResponse(
            ok=False,
            sent_text=sent_text,
            error="Slack webhook request timed out",
        )
    except httpx.RequestError as exc:
        return SlackMessagePreviewResponse(
            ok=False,
            sent_text=sent_text,
            error=f"Slack webhook request failed: {exc}",
        )

    return SlackMessagePreviewResponse(
        ok=response.status_code == 200,
        status_code=response.status_code,
        sent_text=sent_text,
        response_body=response.text[:200],
        error=None if response.status_code == 200 else f"Slack returned {response.status_code}",
    )


# ── Email Message ────────────────────────────────────────────


class EmailMessagePreviewRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    sample_data: dict[str, Any] = Field(default_factory=dict)


class EmailMessagePreviewResponse(BaseModel):
    ok: bool
    smtp_reachable: bool = False
    preview_to: str | None = None
    preview_subject: str | None = None
    preview_html: str | None = None
    error: str | None = None


@router.post("/workflow-nodes/email-message/preview", response_model=EmailMessagePreviewResponse)
async def preview_email_message(
    body: EmailMessagePreviewRequest,
    request: Request,
    tenant_uuid: str = Path(..., description="Tenant UUID"),
    user: User = Depends(get_current_user),
) -> EmailMessagePreviewResponse:
    """Render the email and verify SMTP auth without actually sending anything."""
    try:
        cfg = EmailMessageConfig(**body.config)
    except ValidationError as exc:
        return EmailMessagePreviewResponse(ok=False, error=_format_validation_error(exc))

    auth_token = request.headers.get("authorization", "")
    return await _run_email_preview(cfg, body.sample_data, auth_token, tenant_uuid)


async def _run_email_preview(
    cfg: EmailMessageConfig,
    sample_data: dict[str, Any],
    auth_token: str,
    tenant_uuid: str,
) -> EmailMessagePreviewResponse:
    resolved_to = resolve_templates(cfg.to, sample_data)
    resolved_subject = resolve_templates(cfg.subject, sample_data)
    resolved_html = resolve_templates(cfg.html_body, sample_data)

    host = resolve_templates(cfg.smtp_host, sample_data) if cfg.smtp_host else settings.graphs.SMTP_HOST
    port = cfg.smtp_port or settings.graphs.SMTP_PORT
    user = resolve_templates(cfg.smtp_user, sample_data) if cfg.smtp_user else settings.graphs.SMTP_USER

    password = ""  # nosec B105
    if cfg.smtp_password_key_id:
        rc_like = cast(RunnableConfig, {"configurable": {"auth_token": auth_token, "tenant_uuid": tenant_uuid}})
        password = await reveal_api_key(rc_like, cfg.smtp_password_key_id) or ""
    if not password:
        password = settings.graphs.SMTP_PASSWORD

    render_fields: dict[str, str] = {
        "preview_to": resolved_to,
        "preview_subject": resolved_subject,
        "preview_html": resolved_html,
    }

    if not host or not port or not user or not password:
        return EmailMessagePreviewResponse(
            ok=False,
            smtp_reachable=False,
            error="SMTP credentials not configured (host/port/user/password missing)",
            **render_fields,
        )

    # Build the message once so anything the recipient would see is validated — we just never send it.
    msg = MIMEMultipart("alternative")
    msg["Subject"] = resolved_subject
    msg["From"] = (
        resolve_templates(cfg.smtp_from, sample_data) if cfg.smtp_from else settings.graphs.SMTP_FROM
    ) or user
    msg["To"] = resolved_to
    if cfg.text_body:
        msg.attach(MIMEText(resolve_templates(cfg.text_body, sample_data), "plain"))
    msg.attach(MIMEText(resolved_html, "html"))

    try:
        await asyncio.to_thread(_verify_smtp_auth, host, port, user, password)
    except smtplib.SMTPAuthenticationError as exc:
        return EmailMessagePreviewResponse(
            ok=False,
            smtp_reachable=True,
            error=f"SMTP authentication failed: {exc}",
            **render_fields,
        )
    except smtplib.SMTPException as exc:
        return EmailMessagePreviewResponse(
            ok=False,
            smtp_reachable=False,
            error=f"SMTP error: {exc}",
            **render_fields,
        )
    except TimeoutError:
        return EmailMessagePreviewResponse(
            ok=False,
            smtp_reachable=False,
            error="SMTP connection timed out",
            **render_fields,
        )
    except OSError as exc:
        return EmailMessagePreviewResponse(
            ok=False,
            smtp_reachable=False,
            error=f"Connection error: {exc}",
            **render_fields,
        )

    return EmailMessagePreviewResponse(
        ok=True,
        smtp_reachable=True,
        **render_fields,
    )


def _verify_smtp_auth(host: str, port: int, user: str, password: str) -> None:
    """Connect → STARTTLS → login → NOOP → close. No message is sent."""
    with smtplib.SMTP(host, port, timeout=_PREVIEW_SMTP_TIMEOUT) as server:
        server.starttls()
        server.login(user, password)
        server.noop()


# ── LLM Complete ─────────────────────────────────────────────


class LLMCompletePreviewRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    sample_data: dict[str, Any] = Field(default_factory=dict)


class LLMCompletePreviewResponse(BaseModel):
    ok: bool
    content: str | None = None
    resolved_prompt: str | None = None
    usage: dict[str, int] | None = None
    error: str | None = None


@router.post("/workflow-nodes/llm-complete/preview", response_model=LLMCompletePreviewResponse)
async def preview_llm_complete(
    body: LLMCompletePreviewRequest,
    user: User = Depends(get_current_user),
) -> LLMCompletePreviewResponse:
    """Invoke the configured LLM with a capped token budget for interactive preview."""
    try:
        cfg = LLMCompleteConfig(**body.config)
    except ValidationError as exc:
        return LLMCompletePreviewResponse(ok=False, error=_format_validation_error(exc))

    return await _run_llm_preview(cfg, body.sample_data)


async def _run_llm_preview(
    cfg: LLMCompleteConfig,
    sample_data: dict[str, Any],
) -> LLMCompletePreviewResponse:
    resolved_prompt = resolve_templates(cfg.prompt, sample_data)
    resolved_system = resolve_templates(cfg.system_prompt, sample_data) if cfg.system_prompt else ""

    if not resolved_prompt.strip():
        return LLMCompletePreviewResponse(ok=False, error="prompt is empty after template resolution")

    messages: list[Any] = []
    if resolved_system:
        messages.append(SystemMessage(content=resolved_system))
    messages.append(HumanMessage(content=resolved_prompt))

    llm_model = cfg.model or settings.graphs.WORKFLOW_LLM_MODEL
    max_tokens = min(cfg.max_tokens, _PREVIEW_LLM_MAX_TOKENS)

    try:
        model = init_chat_model(llm_model, temperature=0, max_tokens=max_tokens)
        response = await model.ainvoke(messages)
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:
        logger.info("LLM preview failed (%s): %s", llm_model, str(exc)[:200])
        return LLMCompletePreviewResponse(
            ok=False,
            resolved_prompt=resolved_prompt,
            error=str(exc)[:500],
        )

    content = response.content or ""
    usage: dict[str, int] | None = None
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = {
            "input_tokens": int(response.usage_metadata.get("input_tokens", 0)),
            "output_tokens": int(response.usage_metadata.get("output_tokens", 0)),
        }

    return LLMCompletePreviewResponse(
        ok=True,
        content=content,
        resolved_prompt=resolved_prompt,
        usage=usage,
    )


# ── Helpers ──────────────────────────────────────────────────


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic validation error as a single short string for the UI."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "Invalid config"
