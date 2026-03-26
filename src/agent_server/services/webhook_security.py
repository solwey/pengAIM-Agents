"""Webhook security utilities for workflow webhook triggers.

Supports two authentication methods:
1. HMAC-SHA256 signature (X-Webhook-Signature + X-Webhook-Timestamp headers)
2. Bearer token (Authorization: Bearer <secret>)
"""

import hashlib
import hmac
import secrets
import time


class WebhookVerificationError(Exception):
    """Raised when webhook verification fails."""


def generate_webhook_path() -> str:
    """Generate a URL-safe random slug for the webhook path (~22 chars, ~128 bits)."""
    return secrets.token_urlsafe(16)


def generate_webhook_secret() -> str:
    """Generate a cryptographically random secret (64 hex chars, 256 bits)."""
    return secrets.token_hex(32)


MAX_TIMESTAMP_AGE_SECONDS = 300  # 5 minutes


def verify_hmac_signature(
    secret: str,
    body: bytes,
    signature: str,
    timestamp: str,
) -> None:
    """Verify HMAC-SHA256 signature of a webhook request.

    Signature format: HMAC-SHA256(secret, "{timestamp}.{body}")

    Raises WebhookVerificationError on failure.
    """
    # Validate timestamp freshness
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        raise WebhookVerificationError("Invalid timestamp format")

    age = abs(time.time() - ts)
    if age > MAX_TIMESTAMP_AGE_SECONDS:
        raise WebhookVerificationError(
            f"Timestamp too old: {int(age)}s (max {MAX_TIMESTAMP_AGE_SECONDS}s)"
        )

    # Compute expected signature
    message = f"{timestamp}.".encode() + body
    expected = hmac.new(
        secret.encode(), message, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise WebhookVerificationError("Signature mismatch")


def verify_bearer_token(secret: str, token: str) -> None:
    """Verify a Bearer token matches the webhook secret.

    Raises WebhookVerificationError on failure.
    """
    if not hmac.compare_digest(secret, token):
        raise WebhookVerificationError("Invalid bearer token")
