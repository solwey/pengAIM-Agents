from .api_request import ApiRequestExecutor
from .condition import ConditionExecutor, build_condition_router
from .delay import DelayExecutor
from .email_message import EmailMessageExecutor
from .generate_report import GenerateReportExecutor
from .run_agent import RunAgentExecutor
from .slack_message import SlackMessageExecutor
from .switch import SwitchExecutor, build_switch_router
from .transform import TransformExecutor

NODE_REGISTRY: dict[str, type] = {
    "api_request": ApiRequestExecutor,
    "condition": ConditionExecutor,
    "transform": TransformExecutor,
    "slack_message": SlackMessageExecutor,
    "email_message": EmailMessageExecutor,
    "switch": SwitchExecutor,
    "delay": DelayExecutor,
    "generate_report": GenerateReportExecutor,
    "run_agent": RunAgentExecutor,
}

__all__ = ["NODE_REGISTRY", "build_condition_router", "build_switch_router"]
