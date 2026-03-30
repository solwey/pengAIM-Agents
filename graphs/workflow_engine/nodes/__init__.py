from .api_request import ApiRequestExecutor
from .condition import ConditionExecutor, build_condition_router
from .create_account import CreateAccountExecutor
from .create_contact import CreateContactExecutor
from .delay import DelayExecutor
from .email_message import EmailMessageExecutor
from .generate_report import GenerateReportExecutor
from .icp_score import ICPScoreExecutor
from .llm_complete import LLMCompleteExecutor
from .read_google_sheet import ReadGoogleSheetExecutor
from .run_agent import RunAgentExecutor
from .slack_message import SlackMessageExecutor
from .switch import SwitchExecutor, build_switch_router
from .transform import TransformExecutor
from .update_account import UpdateAccountExecutor

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
    "create_account": CreateAccountExecutor,
    "create_contact": CreateContactExecutor,
    "update_account": UpdateAccountExecutor,
    "icp_score": ICPScoreExecutor,
    "read_google_sheet": ReadGoogleSheetExecutor,
    "llm_complete": LLMCompleteExecutor,
}

__all__ = ["NODE_REGISTRY", "build_condition_router", "build_switch_router"]
