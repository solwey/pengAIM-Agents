"""Authentication fixtures for tests"""

from typing import Any


class DummyUser:
    """Mock user for testing"""

    def __init__(
        self,
        user_id: str = "test-user",
        team_id: str = "test_team",
        display_name: str = "Test User",
    ):
        self.id = user_id
        self.team_id = team_id
        self.display_name = display_name
        self.is_authenticated = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": f"{self.id}:{self.team_id}",
            "display_name": self.display_name,
        }
