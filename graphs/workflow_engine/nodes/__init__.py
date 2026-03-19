from .api_request import ApiRequestExecutor
from .condition import ConditionExecutor, build_condition_router
from .transform import TransformExecutor

NODE_REGISTRY: dict[str, type] = {
    "api_request": ApiRequestExecutor,
    "condition": ConditionExecutor,
    "transform": TransformExecutor,
}

__all__ = ["NODE_REGISTRY", "build_condition_router"]
