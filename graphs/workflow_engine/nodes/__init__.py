from .activate_instantly import ActivateInstantlyExecutor
from .add_leads_to_instantly import AddLeadsToInstantlyExecutor
from .add_tag import AddTagExecutor
from .add_to_campaign import AddToCampaignExecutor
from .add_to_instantly_blocklist import AddToInstantlyBlocklistExecutor
from .add_to_list import AddToListExecutor
from .api_request import ApiRequestExecutor
from .condition import ConditionExecutor, build_condition_router
from .create_account import CreateAccountExecutor
from .create_campaign import CreateCampaignExecutor
from .create_contact import CreateContactExecutor
from .delay import DelayExecutor
from .email_message import EmailMessageExecutor
from .fetch_instantly_replies import FetchInstantlyRepliesExecutor
from .generate_report import GenerateReportExecutor
from .icp_score import ICPScoreExecutor
from .list_condition import ListConditionExecutor, build_list_condition_router
from .llm_complete import LLMCompleteExecutor
from .pause_instantly import PauseInstantlyExecutor
from .read_google_sheet import ReadGoogleSheetExecutor
from .remove_from_list import RemoveFromListExecutor
from .remove_tag import RemoveTagExecutor
from .run_agent import RunAgentExecutor
from .set_source import SetSourceExecutor
from .slack_message import SlackMessageExecutor
from .source_condition import SourceConditionExecutor, build_source_condition_router
from .sub_workflow import SubWorkflowExecutor
from .switch import SwitchExecutor, build_switch_router
from .sync_to_instantly import SyncToInstantlyExecutor
from .tag_condition import TagConditionExecutor, build_tag_condition_router
from .transform import TransformExecutor
from .update_account import UpdateAccountExecutor
from .update_instantly_lead_status import UpdateInstantlyLeadStatusExecutor

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
    "add_tag": AddTagExecutor,
    "remove_tag": RemoveTagExecutor,
    "add_to_list": AddToListExecutor,
    "remove_from_list": RemoveFromListExecutor,
    "tag_condition": TagConditionExecutor,
    "list_condition": ListConditionExecutor,
    "source_condition": SourceConditionExecutor,
    "set_source": SetSourceExecutor,
    "create_campaign": CreateCampaignExecutor,
    "add_to_campaign": AddToCampaignExecutor,
    "sync_to_instantly": SyncToInstantlyExecutor,
    "add_leads_to_instantly": AddLeadsToInstantlyExecutor,
    "activate_instantly": ActivateInstantlyExecutor,
    "pause_instantly": PauseInstantlyExecutor,
    "update_instantly_lead_status": UpdateInstantlyLeadStatusExecutor,
    "fetch_instantly_replies": FetchInstantlyRepliesExecutor,
    "add_to_instantly_blocklist": AddToInstantlyBlocklistExecutor,
    "sub_workflow": SubWorkflowExecutor,
}

__all__ = [
    "NODE_REGISTRY",
    "build_condition_router",
    "build_switch_router",
    "build_tag_condition_router",
    "build_list_condition_router",
    "build_source_condition_router",
]
