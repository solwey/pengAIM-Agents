"""Authentication and user context models"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class User(BaseModel):
    """User model that accepts any auth fields.

    This model uses ConfigDict(extra="allow") to accept any additional fields
    from auth handlers (e.g., subscription_tier, team_id) while maintaining
    type hints for common fields.
    """

    model_config = ConfigDict(extra="allow")

    # Required
    id: str
    team_id: str

    # Optional with defaults
    is_authenticated: bool = True
    permissions: list[str] = []
    display_name: str | None = None

    # Common optional fields (for IDE hints)
    org_id: str | None = None
    email: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict including all extra fields."""
        return self.model_dump()

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to extra fields."""
        try:
            extra = object.__getattribute__(self, "__pydantic_extra__") or {}
        except AttributeError:
            extra = {}
        if name in extra:
            return extra[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @property
    def allows_share_new_chats_by_default(self):
        return "share_new_chats_by_default:true" in self.permissions

    @property
    def is_admin(self):
        return any(role in self.permissions for role in ("role:admin", "role:superadmin"))

    @property
    def is_superadmin(self):
        return "role:superadmin" in self.permissions


class AuthContext(BaseModel):
    """Authentication context for request processing"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user: User
    request_id: str | None = None


class TokenPayload(BaseModel):
    """JWT token payload structure"""

    sub: str  # subject (user ID)
    name: str | None = None
    scopes: list[str] = []
    org: str | None = None
    exp: int | None = None
    iat: int | None = None
