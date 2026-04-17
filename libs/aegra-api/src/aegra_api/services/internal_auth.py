"""Internal JWT token generation for service-to-service calls.

Mints short-lived tokens on behalf of workflow owners so that downstream
nodes (create_account, create_contact, etc.) can call the revops-backend
with proper authorization.

Used by:
- Webhook handler (api/webhooks.py) — for webhook-triggered runs
- Scheduled task dispatcher (tasks.py) — for cron-triggered runs
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime, timedelta

from jose import jwt

from aegra_api.settings import settings

logger = logging.getLogger(__name__)


def _decode_jwt_key(v: str | None) -> str | None:
    """Decode base64-encoded key to PEM, or return as-is if already PEM."""
    if not v:
        return None
    v = v.strip()
    if v.startswith("-----BEGIN"):
        return v
    try:
        return base64.b64decode(v).decode("utf-8")
    except Exception:
        return v


def create_internal_token(user_id: str, team_id: str, tenant_id: str) -> str:
    """Create a short-lived JWT for internal service calls.

    Uses the same signing key / algorithm that pengAIM-RAG uses so the
    revops-backend accepts it transparently.

    Returns empty string if JWT_PRIVATE_KEY is not configured.
    """
    private_key = _decode_jwt_key(settings.app.JWT_PRIVATE_KEY)

    if not private_key:
        logger.warning("No JWT_PRIVATE_KEY — cannot mint internal token")
        return ""

    now = datetime.now(UTC)
    claims: dict[str, str | datetime] = {
        "aud": tenant_id,
        "sub": user_id,
        "team_id": team_id,
        "role": "owner",
        "typ": "access",
        "iat": now,
        "exp": now + timedelta(minutes=30),
    }
    return jwt.encode(claims, private_key, algorithm=settings.app.JWT_ALG)
