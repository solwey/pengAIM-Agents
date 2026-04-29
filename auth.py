import base64
import time
from typing import Any

import httpx
from jose import JWTError, jwt
from langgraph_sdk import Auth

from aegra_api.core.orm import Tenant
from aegra_api.settings import settings

# The "Auth" object is a container that LangGraph will use to mark our authentication function
auth = Auth()


def decode_jwt_key(v):
    """
    Decode base64-encoded JWT public key to PEM format.
    If the value is already in PEM format (starts with -----BEGIN), return as-is.
    Otherwise, attempt to decode from base64.
    """
    if not v or not isinstance(v, str):
        return v

    v = v.strip()

    # If already in PEM format, return as-is
    if v.startswith("-----BEGIN"):
        return v

    # Try to decode from base64
    try:
        decoded = base64.b64decode(v).decode("utf-8")
        return decoded
    except Exception:
        # If decoding fails, return original value (might be empty or invalid)
        return v


# --- JWKS cache ----------------------------------------------------------------
# Fetched on first use, refreshed on TTL expiry or `kid` miss

_jwks_cache: dict[str, Any] | None = None
_jwks_fetched_at: float = 0.0


def _fetch_jwks(*, force: bool = False) -> dict[str, Any]:
    global _jwks_cache, _jwks_fetched_at
    ttl = settings.app.JWKS_CACHE_TTL_SECONDS
    now = time.time()
    if (
        not force
        and _jwks_cache is not None
        and now - _jwks_fetched_at < ttl
    ):
        return _jwks_cache

    resp = httpx.get(f"{settings.graphs.RAG_API_URL}/.well-known/jwks.json", timeout=5.0)
    resp.raise_for_status()
    _jwks_cache = resp.json()
    _jwks_fetched_at = now
    return _jwks_cache


def _find_jwk_by_kid(jwks: dict[str, Any], kid: str | None) -> dict[str, Any] | None:
    keys = jwks.get("keys") or []
    if kid is None:
        return keys[0] if keys else None
    for key in keys:
        if key.get("kid") == kid:
            return key
    return None


def _resolve_verification_key(token: str) -> dict[str, Any]:
    try:
        header = jwt.get_unverified_header(token)
    except JWTError as e:
        raise ValueError(f"Invalid JWT header: {e}") from e
    kid = header.get("kid")

    jwks = _fetch_jwks()
    jwk = _find_jwk_by_kid(jwks, kid)
    if jwk is None and kid is not None:
        # Possible key rotation - refresh once and retry.
        jwks = _fetch_jwks(force=True)
        jwk = _find_jwk_by_kid(jwks, kid)
    if jwk is None:
        raise ValueError(f"No JWKS key matches kid={kid!r}")
    return jwk


async def verify_token_status(token: str, aud: str) -> tuple[str, str, str]:
    try:
        verification_key = _resolve_verification_key(token)
        payload = jwt.decode(
            token,
            verification_key,
            algorithms=[verification_key.get("alg")],
            audience=aud,
            options={"verify_aud": True, "require_aud": True},
        )

        # Verify token type
        if payload.get("typ") != "access":
            raise ValueError("Invalid token type. Use access token, not refresh token.")

        sub = payload.get("sub")
        team_id = payload.get("team_id")
        role = payload.get("role")
        return sub, team_id, role

    except JWTError:
        raise ValueError("Invalid or expired token")


# The `authenticate` decorator tells LangGraph to call this function as middleware
# for every request. This will determine whether the request is allowed or not
@auth.authenticate
async def get_current_user(
    headers: dict[str, str] | None,
    tenant: Tenant | None = None,
):
    """Check if the user's JWT token is valid using custom logic"""
    # Extract authorization header
    authorization = (
        headers.get("authorization")
        or headers.get("Authorization")
        or headers.get(b"authorization")
        or headers.get(b"Authorization")
    )

    # Handle bytes headers
    if isinstance(authorization, bytes):
        authorization = authorization.decode("utf-8")

    # Ensure we have the authorization header
    if not authorization:
        if headers.get("x-auth-scheme") == "langsmith":
            return {
                "identity": "system",
                "is_authenticated": True,
                "permissions": [],
            }

        raise Auth.exceptions.HTTPException(status_code=401, detail="Authorization header missing")

    # Parse the authorization header
    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer" and token
    except (ValueError, AssertionError):
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid authorization header format")

    try:
        user_id, team_id, role = await verify_token_status(token, tenant.uuid)
        if not team_id:
            raise ValueError("Team id not found in verification response")

        return {
            "identity": f"{user_id}:{team_id}",
            "id": user_id,
            "team_id": team_id,
            "is_authenticated": True,
            "authorization": authorization,
            "permissions": [
                f"role:{role}",
            ],
        }
    except Exception as e:
        raise Auth.exceptions.HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")


@auth.on
async def authorize(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, Any]:
    try:
        # Get user identity from authentication context
        user_id = ctx.user.identity

        if not user_id:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid user identity")

        # Create owner filter for resource access control
        owner_filter = {"owner": user_id}

        # Add owner information to metadata for create/update operations
        metadata = value.setdefault("metadata", {})
        metadata.update(owner_filter)

        # Return filter for database operations
        return owner_filter

    except Auth.exceptions.HTTPException:
        raise
    except Exception as e:
        raise Auth.exceptions.HTTPException(status_code=500, detail="Authorization system error") from e
