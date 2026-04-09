import base64
from typing import Any

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


async def verify_token_status(token: str, aud: str) -> tuple[str, str, str]:
    try:
        verification_key = decode_jwt_key(settings.app.JWT_PUBLIC_KEY)
        payload = jwt.decode(
            token,
            verification_key,
            algorithms=[settings.app.JWT_ALG],
            audience=aud,
            options={"verify_aud": True, "require_aud": True}
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
