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


def decode_jwt_key(v: str | None) -> str | None:
    """
    Decode base64-encoded JWT public key to PEM format.
    If the value is already in PEM format (starts with -----BEGIN), return as-is.
    Otherwise, attempt to decode from base64.
    """
    if not v or not isinstance(v, str):
        return v

    v = v.strip()

    if v.startswith("-----BEGIN"):
        return v

    try:
        decoded = base64.b64decode(v).decode("utf-8")
        return decoded
    except Exception:
        return v


# --- JWKS cache ----------------------------------------------------------------
# Fetched on first use, refreshed on TTL expiry or `kid` miss

_jwks_cache: dict[str, Any] | None = None
_jwks_fetched_at: float = 0.0


def _fetch_jwks(*, force: bool = False) -> dict[str, Any]:
    global _jwks_cache, _jwks_fetched_at
    ttl = settings.app.JWKS_CACHE_TTL_SECONDS
    now = time.time()
    if not force and _jwks_cache is not None and now - _jwks_fetched_at < ttl:
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


async def verify_token_status(token: str, aud: str) -> tuple[str, str, list[str]]:
    """Verify a JWT and return (sub, team_id, permissions)."""
    try:
        verification_key = _resolve_verification_key(token)
        payload = jwt.decode(
            token,
            verification_key,
            algorithms=[verification_key.get("alg")],
            audience=aud,
            options={"verify_aud": True, "require_aud": True},
        )

        if payload.get("typ") != "access":
            raise ValueError("Invalid token type. Use access token, not refresh token.")

        sub = payload.get("sub")
        team_id = payload.get("team_id")
        scope = payload.get("scope") or ""
        permissions = [p for p in str(scope).split() if p]
        return sub, team_id, permissions

    except JWTError:
        raise ValueError("Invalid or expired token")


@auth.authenticate
async def get_current_user(
    headers: dict[str, str] | None,
    tenant: Tenant | None = None,
) -> dict[str, Any]:
    """Validate the request's JWT and return the user context."""
    authorization = (
        headers.get("authorization")
        or headers.get("Authorization")
        or headers.get(b"authorization")
        or headers.get(b"Authorization")
    )

    if isinstance(authorization, bytes):
        authorization = authorization.decode("utf-8")

    if not authorization:
        if headers.get("x-auth-scheme") == "langsmith":
            return {
                "identity": "system",
                "is_authenticated": True,
                "permissions": [],
            }

        raise Auth.exceptions.HTTPException(status_code=401, detail="Authorization header missing")

    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer" and token
    except (ValueError, AssertionError):
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid authorization header format")

    try:
        user_id, team_id, permissions = await verify_token_status(token, tenant.uuid)
        if not team_id:
            raise ValueError("Team id not found in verification response")

        return {
            "identity": f"{user_id}:{team_id}",
            "id": user_id,
            "team_id": team_id,
            "is_authenticated": True,
            "authorization": authorization,
            "permissions": permissions,
        }
    except Exception as e:
        raise Auth.exceptions.HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")


# --- Permission-based authorization -------------------------------------------
# Permissions in the JWT scope claim follow `<resource>.<action>[.<reach>]`.
#   <reach> omitted  → own records only
#   .team            → all records in the user's team
#   .all             → all records (any team)
# Actions: read, update (covers create+update), delete.

# SDK action → permission action.  create/update merged into "update".
_ACTION_MAP: dict[str, str] = {
    "create": "update",
    "update": "update",
    "put": "update",
    "read": "read",
    "search": "read",
    "get": "read",
    "list_namespaces": "read",
    "delete": "delete",
}


def _scope_filter(
    permissions: list[str],
    user_id: str,
    team_id: str,
    resource: str,
    action: str,
) -> dict[str, Any]:
    """Return a filter dict for the given resource/action.

    Returns:
        - {} when the user has `<resource>.<action>.all` (no scoping).
        - {"team_id": ...} when the user has `<resource>.<action>.team`.
        - {"user_id": ..., "team_id": ...} when the user has `<resource>.<action>` (own).

    Raises:
        Auth.exceptions.HTTPException(403) when no matching permission is held.
    """
    perm_action = _ACTION_MAP.get(action, action)
    base = f"{resource}.{perm_action}"
    if f"{base}.all" in permissions:
        return {}
    if f"{base}.team" in permissions:
        return {"team_id": team_id}
    if base in permissions:
        return {"user_id": user_id, "team_id": team_id}
    raise Auth.exceptions.HTTPException(
        status_code=403,
        detail=f"Missing permission for {resource}.{perm_action}",
    )


def _stamp_metadata(value: dict[str, Any], user_id: str, team_id: str) -> None:
    """Attach owner/team metadata to a mutating request payload."""
    metadata = value.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    metadata.setdefault("user_id", user_id)
    metadata.setdefault("team_id", team_id)
    metadata.setdefault("owner", f"{user_id}:{team_id}")


def _is_mutating(action: str) -> bool:
    return _ACTION_MAP.get(action, action) == "update"


def _check(
    ctx: Auth.types.AuthContext,
    value: dict[str, Any],
) -> dict[str, Any]:
    user = ctx.user
    user_id = getattr(user, "id", None) or user.identity
    team_id = getattr(user, "team_id", None) or ""

    if not user_id:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid user identity")

    resource = ctx.resource
    action = ctx.action
    if resource == "threads" and action == "create_run":
        resource = "runs"
        action = "update"

    filters = _scope_filter(
        permissions=list(ctx.permissions or []),
        user_id=str(user_id),
        team_id=str(team_id),
        resource=resource,
        action=action,
    )
    if _is_mutating(action):
        _stamp_metadata(value, str(user_id), str(team_id))
    return filters


_AEGRA_RESOURCES: list[str] = [
    "assistants",
    "threads",
    "store",
    "runs",
    "workflows",
    "workflow_runs",
    "workflow_schedules",
    "control_plane"
]

@auth.on
async def _global(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, Any]:
    if not ctx.resource in _AEGRA_RESOURCES:
        # Unknown resource: deny by default.
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail=f"No authorization rule for resource {ctx.resource!r}",
        )
    return _check(ctx, value)
