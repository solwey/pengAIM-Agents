from typing import Any

from langgraph_sdk import Auth

from aegra_api.core.orm import Tenant
from aegra_api.settings import settings
from aegra_api.utils.tokens import decode_keycloak_token

# The "Auth" object is a container that LangGraph will use to mark our authentication function
auth = Auth()


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
        payload = decode_keycloak_token(token, settings.app.KEYCLOAK_URL)
        if not payload.team_id:
            raise ValueError("Team id not found in verification response")

        if not payload.role:
            raise ValueError("Account is not activated. Please contact your administrator")

        # Enforce that the token's Keycloak realm matches the tenant the
        # request is addressed to. This stops a valid token from one tenant
        # being used against another tenant's endpoints.
        if tenant is not None and payload.realm != tenant.kc_realm:
            raise ValueError(
                f"Token realm '{payload.realm}' does not match tenant realm "
                f"'{tenant.kc_realm}'"
            )

        return {
            "identity": f"{payload.user_id}:{payload.team_id}",
            "id": payload.user_id,
            "team_id": payload.team_id,
            "is_authenticated": True,
            "authorization": authorization,
            "permissions": [
                f"role:{payload.role}",
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
