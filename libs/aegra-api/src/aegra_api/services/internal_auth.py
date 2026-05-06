"""OAuth2 client_credentials token fetcher for service-to-service calls.

Exchanges the configured OAUTH2_CLIENT_ID/OAUTH2_CLIENT_SECRET for a short-lived
access token at pengAIM-RAG's `/tenant/{uuid}/oauth/token` endpoint so that
downstream workflow nodes (create_account, create_contact, etc.) can call
revops-backend on behalf of the workflow owner.

Used by:
- Webhook handler (api/webhooks.py) — async, for webhook-triggered runs
- Scheduled task dispatcher (tasks.py) — sync, for cron-triggered runs (Celery)
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

import httpx

from aegra_api.settings import settings

logger = logging.getLogger(__name__)

_REFRESH_BUFFER = timedelta(seconds=30)
_cache: dict[tuple[str, str, str | None], tuple[str, datetime]] = {}
_cache_lock = threading.Lock()


def _cache_key(tenant_uuid: str, team_id: str, user_id: str | None) -> tuple[str, str, str | None]:
    return (tenant_uuid, team_id, user_id)


def _get_cached(key: tuple[str, str, str | None]) -> str | None:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        token, exp = entry
        if datetime.now(UTC) + _REFRESH_BUFFER >= exp:
            _cache.pop(key, None)
            return None
        return token


def _set_cached(key: tuple[str, str, str | None], token: str, expires_in: int) -> None:
    with _cache_lock:
        _cache[key] = (token, datetime.now(UTC) + timedelta(seconds=expires_in))


def _token_url(tenant_uuid: str) -> str:
    base = settings.graphs.RAG_API_URL.rstrip("/")
    return f"{base}/tenant/{tenant_uuid}/oauth/token"


def _build_form(team_id: str, user_id: str | None) -> dict[str, str]:
    form: dict[str, str] = {
        "grant_type": "client_credentials",
        "client_id": settings.app.OAUTH2_CLIENT_ID or "",
        "client_secret": settings.app.OAUTH2_CLIENT_SECRET or "",
        "team_id": team_id,
    }
    if user_id:
        form["user_id"] = user_id
    return form


def _extract_token(payload: dict[str, object]) -> tuple[str, int] | None:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        return None
    expires_in = payload.get("expires_in")
    expires_seconds = int(expires_in) if isinstance(expires_in, int | str) else 1800
    return access_token, expires_seconds


def _credentials_configured() -> bool:
    if not settings.app.OAUTH2_CLIENT_ID or not settings.app.OAUTH2_CLIENT_SECRET:
        logger.warning("OAUTH2_CLIENT_ID/OAUTH2_CLIENT_SECRET not configured — cannot mint workflow token")
        return False
    return True


def fetch_oauth_token(*, tenant_uuid: str, team_id: str, user_id: str | None) -> str:
    """Sync OAuth2 client_credentials exchange. Empty string on failure."""
    if not _credentials_configured():
        return ""

    key = _cache_key(tenant_uuid, team_id, user_id)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        resp = httpx.post(
            _token_url(tenant_uuid),
            data=_build_form(team_id, user_id),
            timeout=10.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError:
        logger.exception("OAuth2 token exchange failed (sync)")
        return ""

    extracted = _extract_token(resp.json())
    if extracted is None:
        logger.warning("OAuth2 token response missing access_token")
        return ""
    token, expires_in = extracted
    _set_cached(key, token, expires_in)
    return token


async def fetch_oauth_token_async(*, tenant_uuid: str, team_id: str, user_id: str | None) -> str:
    """Async OAuth2 client_credentials exchange. Empty string on failure."""
    if not _credentials_configured():
        return ""

    key = _cache_key(tenant_uuid, team_id, user_id)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _token_url(tenant_uuid),
                data=_build_form(team_id, user_id),
            )
            resp.raise_for_status()
    except httpx.HTTPError:
        logger.exception("OAuth2 token exchange failed (async)")
        return ""

    extracted = _extract_token(resp.json())
    if extracted is None:
        logger.warning("OAuth2 token response missing access_token")
        return ""
    token, expires_in = extracted
    _set_cached(key, token, expires_in)
    return token
