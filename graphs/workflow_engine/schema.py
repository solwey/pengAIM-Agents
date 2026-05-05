"""Pydantic models for validating workflow JSON definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class NodeType(StrEnum):
    API_REQUEST = "api_request"
    CONDITION = "condition"
    FOR_EACH = "for_each"
    TRANSFORM = "transform"
    SLACK_MESSAGE = "slack_message"
    EMAIL_MESSAGE = "email_message"
    SWITCH = "switch"
    DELAY = "delay"
    GENERATE_REPORT = "generate_report"
    RUN_AGENT = "run_agent"
    CREATE_ACCOUNT = "create_account"
    CREATE_CONTACT = "create_contact"
    UPDATE_ACCOUNT = "update_account"
    ICP_SCORE = "icp_score"
    READ_GOOGLE_SHEET = "read_google_sheet"
    LLM_COMPLETE = "llm_complete"
    ADD_TAG = "add_tag"
    REMOVE_TAG = "remove_tag"
    ADD_TO_LIST = "add_to_list"
    TAG_CONDITION = "tag_condition"
    SET_SOURCE = "set_source"
    REMOVE_FROM_LIST = "remove_from_list"
    LIST_CONDITION = "list_condition"
    SOURCE_CONDITION = "source_condition"
    CREATE_CAMPAIGN = "create_campaign"
    ADD_TO_CAMPAIGN = "add_to_campaign"
    SYNC_TO_INSTANTLY = "sync_to_instantly"
    ADD_LEADS_TO_INSTANTLY = "add_leads_to_instantly"
    ACTIVATE_INSTANTLY = "activate_instantly"
    PAUSE_INSTANTLY = "pause_instantly"
    UPDATE_INSTANTLY_LEAD_STATUS = "update_instantly_lead_status"
    FETCH_INSTANTLY_REPLIES = "fetch_instantly_replies"
    ADD_TO_INSTANTLY_BLOCKLIST = "add_to_instantly_blocklist"
    SUB_WORKFLOW = "sub_workflow"


class ComparisonOperator(StrEnum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"


# ── Node configs ──────────────────────────────────────────────


class ApiRequestConfig(BaseModel):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    response_key: str = "api_response"
    retry_count: int = Field(default=0, ge=0, le=5)
    retry_delay_seconds: float = Field(default=2.0, ge=0.5, le=30)
    retry_on_status: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])


class ConditionConfig(BaseModel):
    field: str  # dot-notation path, e.g. "api_response.status_code"
    operator: ComparisonOperator
    value: Any  # value to compare against


class TransformConfig(BaseModel):
    set: dict[str, Any] = Field(min_length=1)


class SlackMessageConfig(BaseModel):
    webhook_url: str
    message: str = ""
    blocks: list[dict[str, Any]] | None = None
    thread_ts: str = ""
    username: str = ""  # override bot display name
    icon_emoji: str = ""  # override bot icon (e.g. ":robot_face:")
    response_key: str = "slack_response"
    timeout_seconds: int = Field(default=15, ge=1, le=600)

    @model_validator(mode="after")
    def _check_payload(self) -> SlackMessageConfig:
        if not self.message and not self.blocks:
            raise ValueError("slack_message requires either 'message' or 'blocks'")
        return self


class EmailAttachment(BaseModel):
    filename: str
    content_b64: str = ""
    url: str = ""  # alternative: fetch attachment over HTTP(S) at send time
    content_type: str = "application/octet-stream"

    @model_validator(mode="after")
    def _check_source(self) -> EmailAttachment:
        if not self.content_b64 and not self.url:
            raise ValueError(f"Attachment '{self.filename}' must specify content_b64 or url")
        return self


class EmailMessageConfig(BaseModel):
    to: str
    cc: str = ""
    bcc: str = ""
    subject: str
    html_body: str
    text_body: str | None = None
    from_name: str = ""
    attachments: list[EmailAttachment] = Field(default_factory=list)
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_password_key_id: str | None = None
    smtp_from: str | None = None
    use_ssl: bool = False  # True -> SMTPS on port 465; False -> SMTP+STARTTLS
    response_key: str = "email_response"

    @model_validator(mode="after")
    def _check_smtp_overrides(self) -> EmailMessageConfig:
        smtp_fields = {
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_user": self.smtp_user,
            "smtp_password_key_id": self.smtp_password_key_id,
        }
        provided = {k for k, v in smtp_fields.items() if v}
        if provided and provided != set(smtp_fields):
            missing = sorted(set(smtp_fields) - provided)
            raise ValueError(
                "SMTP override is incomplete: when any of "
                f"{sorted(smtp_fields)} is set, all must be provided. "
                f"Missing: {missing}"
            )
        return self


class SwitchCase(BaseModel):
    label: str
    field: str
    operator: ComparisonOperator
    value: Any


class SwitchConfig(BaseModel):
    cases: list[SwitchCase] = Field(min_length=1)
    default_label: str = "default"


class DelayConfig(BaseModel):
    seconds: float = Field(default=5, ge=0.1, le=86400)
    until_iso: str = (
        ""  # ISO 8601 timestamp (e.g. "2026-05-10T09:00:00Z"); supports {{template}}, overrides seconds when set
    )
    max_seconds: float = Field(default=86400, ge=1, le=604800)  # safety cap when until_iso resolves far in future


class GenerateReportConfig(BaseModel):
    automation_id: str  # presentation bot ID
    report_type: str = "PIC-weekly"
    period: str = ""  # e.g. "2026-w10", supports {{template}}
    content: str = ""  # optional prompt/content, supports {{template}}
    generator: str = "manus"  # "manus" or "local"
    response_key: str = "report_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class RunAgentConfig(BaseModel):
    assistant_id: str  # LangGraph assistant/graph ID
    prompt: str  # message to send to agent, supports {{template}}
    response_key: str = "agent_result"
    timeout_seconds: int = Field(default=300, ge=10, le=600)


class CreateAccountConfig(BaseModel):
    data_key: str = ""  # dot-path to data source (e.g. "sheet_data.rows"), empty = root
    field_mapping: dict[str, str] = Field(default_factory=dict)  # {"name": "{{company_name}}"}
    response_key: str = "created_accounts"
    dedup_mode: str = "upsert"  # "upsert" | "skip" | "create"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class CreateContactConfig(BaseModel):
    account_id_key: str = "created_accounts.account_id"  # dot-path to account ID in state
    data_key: str = ""  # source data key
    field_mapping: dict[str, str] = Field(default_factory=dict)
    response_key: str = "created_contacts"
    dedup_mode: str = "upsert"  # "upsert" | "skip" | "create"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class UpdateAccountConfig(BaseModel):
    account_id_key: str = "created_accounts.account_id"  # dot-path to account ID in state
    updates: dict[str, str] = Field(default_factory=dict)  # {"score": "{{icp_score.score}}"}
    response_key: str = "update_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class ICPScoreConfig(BaseModel):
    account_data_key: str = ""  # where to find account data in state
    model: str = ""  # LLM model (e.g. "openai:gpt-4o-mini"), empty = env default
    response_key: str = "icp_score"
    hot_threshold: int = Field(default=80, ge=0, le=100)
    warm_threshold: int = Field(default=50, ge=0, le=100)
    custom_criteria: str = ""  # extra rules appended to the scoring prompt
    truncation_chars: int = Field(default=3000, ge=200, le=20000)
    max_tokens: int = Field(default=200, ge=50, le=2000)

    @model_validator(mode="after")
    def _check_thresholds(self) -> ICPScoreConfig:
        if self.warm_threshold >= self.hot_threshold:
            raise ValueError("warm_threshold must be lower than hot_threshold")
        return self


class ReadGoogleSheetConfig(BaseModel):
    spreadsheet_id: str
    sheet_name: str = "Sheet1"
    range: str = "A:Z"
    google_sa_key_id: str = ""  # api_keys ID for Google Service Account JSON
    response_key: str = "sheet_data"


class LLMCompleteConfig(BaseModel):
    prompt: str = ""  # supports {{template}} variables
    system_prompt: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
    response_format: Literal["", "json"] = ""
    model: str = ""  # LLM model (e.g. "openai:gpt-4o-mini"), empty = env default
    max_tokens: int = Field(default=1000, ge=1, le=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    response_key: str = "llm_result"

    @model_validator(mode="after")
    def _check_input(self) -> LLMCompleteConfig:
        if not self.prompt and not self.messages:
            raise ValueError("llm_complete requires either 'prompt' or 'messages'")
        return self


class AddTagConfig(BaseModel):
    tag_name: str  # supports {{template}} variables
    tag_color: str = "#6366f1"
    entity_type: str = "account"  # "account" or "contact"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "add_tag_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class RemoveTagConfig(BaseModel):
    tag_name: str  # supports {{template}} variables
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "remove_tag_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class AddToListConfig(BaseModel):
    list_id: str = ""  # direct list ID
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "add_to_list_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class TagConditionConfig(BaseModel):
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    tag_names: list[str] = Field(default_factory=list)
    match_mode: str = "any"  # "any" or "all"
    response_key: str = "tag_check_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class SetSourceConfig(BaseModel):
    source_name: str = ""  # supports {{template}} — e.g. "Website Visitor"
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "set_source_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class RemoveFromListConfig(BaseModel):
    list_id: str = ""
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "remove_from_list_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class ListConditionConfig(BaseModel):
    list_id: str = ""
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "list_check_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class SourceConditionConfig(BaseModel):
    source_name: str = ""  # supports {{template}}
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "source_check_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class CreateCampaignConfig(BaseModel):
    name: str = ""  # supports {{template}}
    channels: list[str] = Field(default_factory=lambda: ["email"])
    description: str = ""  # supports {{template}}
    target_persona: str = ""
    response_key: str = "created_campaign"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class AddToCampaignConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state
    account_id_key: str = "created_accounts.account_id"  # dot-path to account ID(s)
    response_key: str = "campaign_add_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class SyncToInstantlyConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state (e.g. "created_campaign.campaign_id")
    account_email: str = ""  # sender email, supports {{template}}
    response_key: str = "instantly_sync_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class AddLeadsToInstantlyConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state
    contact_ids_key: str = "created_contacts.contact_id"  # dot-path to contact ID(s)
    response_key: str = "instantly_leads_result"
    timeout_seconds: int = Field(default=60, ge=1, le=600)


class ActivateInstantlyConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state
    response_key: str = "instantly_activate_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class PauseInstantlyConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state
    response_key: str = "instantly_pause_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class UpdateInstantlyLeadStatusConfig(BaseModel):
    campaign_id: str = ""
    campaign_id_key: str = ""
    lead_email: str = ""
    lead_email_key: str = ""
    interest_value: int | None = None
    interest_value_key: str = ""
    response_key: str = "instantly_lead_status_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class FetchInstantlyRepliesConfig(BaseModel):
    campaign_id: str = ""
    campaign_id_key: str = ""
    limit: int = 50
    response_key: str = "instantly_replies"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class AddToInstantlyBlocklistConfig(BaseModel):
    bl_value: str = ""  # direct email or domain (supports {{template}})
    bl_value_key: str = ""  # dot-path to single email/domain in state
    bl_values_key: str = ""  # dot-path to a list of emails/domains in state
    response_key: str = "instantly_blocklist_result"
    timeout_seconds: int = Field(default=30, ge=1, le=600)


class ForEachConfig(BaseModel):
    items_key: str = Field(min_length=1)  # dot-path to a list in state["data"]
    workflow_id: str = Field(min_length=1)  # child workflow to run per item
    item_var: str = "item"  # key under which the current item is exposed in child data
    index_var: str = "index"  # key under which the loop index is exposed in child data
    concurrency: int = Field(default=1, ge=1, le=20)
    max_items: int = Field(default=500, ge=1, le=10000)  # safety cap on iterations
    fail_fast: bool = False  # if True, abort the loop on first child failure
    response_key: str = "for_each_result"
    max_depth: int = Field(default=5, ge=1, le=20)


class SubWorkflowConfig(BaseModel):
    workflow_id: str = Field(min_length=1)
    response_key: str = "subflow_result"
    max_depth: int = Field(default=5, ge=1, le=20)
    # input_mapping: child_data_key -> parent dot-path (e.g. {"lead": "leads.rows.0"})
    # When non-empty, the child only sees the mapped subset; otherwise it inherits
    # the full parent data (legacy behavior).
    input_mapping: dict[str, str] = Field(default_factory=dict)
    # output_mapping: parent_data_key -> child dot-path (e.g. {"score": "icp_score.score"})
    # When non-empty, only mapped values are written to the parent under response_key.data;
    # otherwise the full child data dict is exposed (legacy behavior).
    output_mapping: dict[str, str] = Field(default_factory=dict)


# ── Node & Edge definitions ──────────────────────────────────


class NodeDef(BaseModel):
    id: str
    type: NodeType
    config: dict[str, Any]
    enabled: bool = True
    description: str = ""

    def parsed_config(self):
        """Return a typed config object based on node type."""
        config_map = {
            NodeType.API_REQUEST: ApiRequestConfig,
            NodeType.CONDITION: ConditionConfig,
            NodeType.TRANSFORM: TransformConfig,
            NodeType.SLACK_MESSAGE: SlackMessageConfig,
            NodeType.EMAIL_MESSAGE: EmailMessageConfig,
            NodeType.SWITCH: SwitchConfig,
            NodeType.DELAY: DelayConfig,
            NodeType.GENERATE_REPORT: GenerateReportConfig,
            NodeType.RUN_AGENT: RunAgentConfig,
            NodeType.CREATE_ACCOUNT: CreateAccountConfig,
            NodeType.CREATE_CONTACT: CreateContactConfig,
            NodeType.UPDATE_ACCOUNT: UpdateAccountConfig,
            NodeType.ICP_SCORE: ICPScoreConfig,
            NodeType.READ_GOOGLE_SHEET: ReadGoogleSheetConfig,
            NodeType.LLM_COMPLETE: LLMCompleteConfig,
            NodeType.ADD_TAG: AddTagConfig,
            NodeType.REMOVE_TAG: RemoveTagConfig,
            NodeType.ADD_TO_LIST: AddToListConfig,
            NodeType.TAG_CONDITION: TagConditionConfig,
            NodeType.SET_SOURCE: SetSourceConfig,
            NodeType.REMOVE_FROM_LIST: RemoveFromListConfig,
            NodeType.LIST_CONDITION: ListConditionConfig,
            NodeType.SOURCE_CONDITION: SourceConditionConfig,
            NodeType.CREATE_CAMPAIGN: CreateCampaignConfig,
            NodeType.ADD_TO_CAMPAIGN: AddToCampaignConfig,
            NodeType.SYNC_TO_INSTANTLY: SyncToInstantlyConfig,
            NodeType.ADD_LEADS_TO_INSTANTLY: AddLeadsToInstantlyConfig,
            NodeType.ACTIVATE_INSTANTLY: ActivateInstantlyConfig,
            NodeType.PAUSE_INSTANTLY: PauseInstantlyConfig,
            NodeType.UPDATE_INSTANTLY_LEAD_STATUS: UpdateInstantlyLeadStatusConfig,
            NodeType.FETCH_INSTANTLY_REPLIES: FetchInstantlyRepliesConfig,
            NodeType.ADD_TO_INSTANTLY_BLOCKLIST: AddToInstantlyBlocklistConfig,
            NodeType.SUB_WORKFLOW: SubWorkflowConfig,
            NodeType.FOR_EACH: ForEachConfig,
        }
        cls = config_map.get(self.type)
        if cls is None:
            raise ValueError(f"Unknown node type: {self.type}")
        return cls(**self.config)


class EdgeDef(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str | None = Field(default=None, alias="to")
    type: Literal["sequential", "conditional", "switch", "on_error"] = "sequential"
    branches: dict[str, str] | None = None

    model_config = {"populate_by_name": True}


# ── Workflow definition ──────────────────────────────────────


class WorkflowDefinition(BaseModel):
    version: str = "1"
    name: str
    nodes: list[NodeDef]
    edges: list[EdgeDef]

    @model_validator(mode="after")
    def validate_workflow(self) -> WorkflowDefinition:
        node_ids = {n.id for n in self.nodes}
        sentinels = {"__start__", "__end__"}
        valid_ids = node_ids | sentinels

        # Check unique node ids
        if len(node_ids) != len(self.nodes):
            seen: set[str] = set()
            for n in self.nodes:
                if n.id in seen:
                    raise ValueError(f"Duplicate node id: '{n.id}'")
                seen.add(n.id)

        # Node ids must not clash with sentinels
        for nid in node_ids:
            if nid in sentinels:
                raise ValueError(f"Node id '{nid}' is reserved. Use a different name.")

        # Validate edges reference existing nodes
        for edge in self.edges:
            if edge.from_node not in valid_ids:
                raise ValueError(f"Edge 'from' references unknown node: '{edge.from_node}'")

            if edge.type == "sequential":
                if edge.to_node is None:
                    raise ValueError(f"Sequential edge from '{edge.from_node}' must have a 'to' field")
                if edge.to_node not in valid_ids:
                    raise ValueError(f"Edge 'to' references unknown node: '{edge.to_node}'")

            elif edge.type == "conditional":
                if not edge.branches:
                    raise ValueError(f"Conditional edge from '{edge.from_node}' must have 'branches'")
                if set(edge.branches.keys()) != {"yes", "no"}:
                    raise ValueError(
                        f"Conditional edge from '{edge.from_node}' must have exactly "
                        f"'yes' and 'no' branches, got: {set(edge.branches.keys())}"
                    )
                for label, target in edge.branches.items():
                    if target not in valid_ids:
                        raise ValueError(f"Branch '{label}' references unknown node: '{target}'")

            elif edge.type == "switch":
                if not edge.branches:
                    raise ValueError(f"Switch edge from '{edge.from_node}' must have 'branches'")
                for label, target in edge.branches.items():
                    if target not in valid_ids:
                        raise ValueError(f"Switch branch '{label}' references unknown node: '{target}'")

            elif edge.type == "on_error":
                if edge.to_node is None:
                    raise ValueError(f"on_error edge from '{edge.from_node}' must have a 'to' field")
                if edge.to_node not in valid_ids:
                    raise ValueError(f"on_error edge 'to' references unknown node: '{edge.to_node}'")

        # Conditional edges must originate from condition nodes
        for edge in self.edges:
            if edge.type == "conditional":
                source_node = next((n for n in self.nodes if n.id == edge.from_node), None)
                conditional_types = {
                    NodeType.CONDITION,
                    NodeType.TAG_CONDITION,
                    NodeType.LIST_CONDITION,
                    NodeType.SOURCE_CONDITION,
                }
                if source_node is None or source_node.type not in conditional_types:
                    raise ValueError(f"Conditional edge from '{edge.from_node}' must originate from a 'condition' node")

        # Switch edges must originate from switch nodes
        for edge in self.edges:
            if edge.type == "switch":
                source_node = next((n for n in self.nodes if n.id == edge.from_node), None)
                if source_node is None or source_node.type != NodeType.SWITCH:
                    raise ValueError(f"Switch edge from '{edge.from_node}' must originate from a 'switch' node")

        # Must have exactly one edge from __start__
        start_edges = [e for e in self.edges if e.from_node == "__start__"]
        if len(start_edges) != 1:
            raise ValueError(f"Workflow must have exactly one edge from '__start__', found {len(start_edges)}")

        # Detect cycles via DFS
        adj: dict[str, list[str]] = {}
        for edge in self.edges:
            src = edge.from_node
            if src not in adj:
                adj[src] = []
            if edge.type in ("sequential", "on_error") and edge.to_node:
                adj[src].append(edge.to_node)
            elif edge.type in ("conditional", "switch") and edge.branches:
                adj[src].extend(edge.branches.values())

        visited: set[str] = set()
        visiting: set[str] = set()

        def _has_cycle(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for neighbor in adj.get(node, []):
                if _has_cycle(neighbor):
                    return True
            visiting.discard(node)
            visited.add(node)
            return False

        for start in adj:
            if start not in visited and _has_cycle(start):
                raise ValueError("Workflow contains a cycle — remove circular connections")

        return self

    def get_node(self, node_id: str) -> NodeDef | None:
        return next((n for n in self.nodes if n.id == node_id), None)
