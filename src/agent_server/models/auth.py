"""Authentication and user context models"""

from pydantic import BaseModel


class User(BaseModel):
    """User context model for authentication"""

    id: str
    team_id: str
    display_name: str | None = None
    permissions: list[str] = []
    org_id: str | None = None
    is_authenticated: bool = True

    @property
    def allows_share_new_chats_by_default(self):
        return "share_new_chats_by_default:true" in self.permissions

    @property
    def is_admin(self):
        return any(
            role in self.permissions for role in ("role:admin", "role:superadmin")
        )

    @property
    def is_superadmin(self):
        return "role:superadmin" in self.permissions

class AuthContext(BaseModel):
    """Authentication context for request processing"""

    user: User
    request_id: str | None = None

    class Config:
        arbitrary_types_allowed = True


class TokenPayload(BaseModel):
    """JWT token payload structure"""

    sub: str  # subject (user ID)
    name: str | None = None
    scopes: list[str] = []
    org: str | None = None
    exp: int | None = None
    iat: int | None = None
