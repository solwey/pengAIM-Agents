from .api_request import ApiRequestExecutor
from .condition import ConditionExecutor, build_condition_router
from .slack_message import SlackMessageExecutor
from .transform import TransformExecutor

NODE_REGISTRY: dict[str, type] = {
    "api_request": ApiRequestExecutor,
    "condition": ConditionExecutor,
    "transform": TransformExecutor,
    "slack_message": SlackMessageExecutor,
}

__all__ = ["NODE_REGISTRY", "build_condition_router"]
