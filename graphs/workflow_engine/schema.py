"""Pydantic models for validating workflow JSON definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class NodeType(StrEnum):
    API_REQUEST = "api_request"
    CONDITION = "condition"
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
    ACTIVATE_NETSUITE = "activate_netsuite"
    FETCH_NETSUITE_ENTITY = "fetch_netsuite_entity"
    CALCULATE_NETSUITE_METRIC = "calculate_netsuite_metric"
    FETCH_BLOOMERANG_ENTITY = "fetch_bloomerang_entity"


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
    body: dict[str, Any] | None = None
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    response_key: str = "api_response"
    retry_count: int = Field(default=0, ge=0, le=5)
    retry_delay_seconds: float = Field(default=2.0, ge=0.5, le=30)
    retry_on_status: list[int] = Field(default_factory=lambda: [500, 502, 503, 504])


class ConditionConfig(BaseModel):
    field: str  # dot-notation path, e.g. "api_response.status_code"
    operator: ComparisonOperator
    value: Any  # value to compare against


class TransformConfig(BaseModel):
    set: dict[str, Any]  # key-value pairs to merge into state["data"]


class SlackMessageConfig(BaseModel):
    webhook_url: str
    message: str
    username: str = ""  # override bot display name
    icon_emoji: str = ""  # override bot icon (e.g. ":robot_face:")
    response_key: str = "slack_response"


class EmailMessageConfig(BaseModel):
    to: str
    subject: str
    html_body: str
    text_body: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_password_key_id: str | None = None
    smtp_from: str | None = None
    response_key: str = "email_response"


class SwitchCase(BaseModel):
    label: str
    field: str
    operator: ComparisonOperator
    value: Any


class SwitchConfig(BaseModel):
    cases: list[SwitchCase] = Field(min_length=1)
    default_label: str = "default"


class DelayConfig(BaseModel):
    seconds: float = Field(default=5, ge=0.1, le=3600)


class GenerateReportConfig(BaseModel):
    automation_id: str  # presentation bot ID
    report_type: str = "PIC-weekly"
    period: str = ""  # e.g. "2026-w10", supports {{template}}
    content: str = ""  # optional prompt/content, supports {{template}}
    generator: str = "manus"  # "manus" or "local"
    response_key: str = "report_result"


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


class CreateContactConfig(BaseModel):
    account_id_key: str = "created_accounts.account_id"  # dot-path to account ID in state
    data_key: str = ""  # source data key
    field_mapping: dict[str, str] = Field(default_factory=dict)
    response_key: str = "created_contacts"
    dedup_mode: str = "upsert"  # "upsert" | "skip" | "create"


class UpdateAccountConfig(BaseModel):
    account_id_key: str = "created_accounts.account_id"  # dot-path to account ID in state
    updates: dict[str, str] = Field(default_factory=dict)  # {"score": "{{icp_score.score}}"}
    response_key: str = "update_result"


class ICPScoreConfig(BaseModel):
    account_data_key: str = ""  # where to find account data in state
    model: str = ""  # LLM model (e.g. "openai:gpt-4o-mini"), empty = env default
    response_key: str = "icp_score"


class ReadGoogleSheetConfig(BaseModel):
    spreadsheet_id: str
    sheet_name: str = "Sheet1"
    range: str = "A:Z"
    google_sa_key_id: str = ""  # api_keys ID for Google Service Account JSON
    response_key: str = "sheet_data"


class LLMCompleteConfig(BaseModel):
    prompt: str  # supports {{template}} variables
    system_prompt: str = ""
    model: str = ""  # LLM model (e.g. "openai:gpt-4o-mini"), empty = env default
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    response_key: str = "llm_result"


class AddTagConfig(BaseModel):
    tag_name: str  # supports {{template}} variables
    tag_color: str = "#6366f1"
    entity_type: str = "account"  # "account" or "contact"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "add_tag_result"


class RemoveTagConfig(BaseModel):
    tag_name: str  # supports {{template}} variables
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "remove_tag_result"


class AddToListConfig(BaseModel):
    list_id: str = ""  # direct list ID
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "add_to_list_result"


class TagConditionConfig(BaseModel):
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    tag_names: list[str] = Field(default_factory=list)
    match_mode: str = "any"  # "any" or "all"
    response_key: str = "tag_check_result"


class SetSourceConfig(BaseModel):
    source_name: str = ""  # supports {{template}} — e.g. "Website Visitor"
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "set_source_result"


class RemoveFromListConfig(BaseModel):
    list_id: str = ""
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "remove_from_list_result"


class ListConditionConfig(BaseModel):
    list_id: str = ""
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "list_check_result"


class SourceConditionConfig(BaseModel):
    source_name: str = ""  # supports {{template}}
    entity_type: str = "account"
    entity_id_key: str = "created_accounts.account_id"
    response_key: str = "source_check_result"


class CreateCampaignConfig(BaseModel):
    name: str = ""  # supports {{template}}
    channels: list[str] = Field(default_factory=lambda: ["email"])
    description: str = ""  # supports {{template}}
    target_persona: str = ""
    response_key: str = "created_campaign"


class AddToCampaignConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state
    account_id_key: str = "created_accounts.account_id"  # dot-path to account ID(s)
    response_key: str = "campaign_add_result"


class SyncToInstantlyConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state (e.g. "created_campaign.campaign_id")
    account_email: str = ""  # sender email, supports {{template}}
    response_key: str = "instantly_sync_result"


class AddLeadsToInstantlyConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state
    contact_ids_key: str = "created_contacts.contact_id"  # dot-path to contact ID(s)
    response_key: str = "instantly_leads_result"


class ActivateInstantlyConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state
    response_key: str = "instantly_activate_result"


class PauseInstantlyConfig(BaseModel):
    campaign_id: str = ""  # direct campaign ID
    campaign_id_key: str = ""  # or dot-path to campaign ID in state
    response_key: str = "instantly_pause_result"


# Must stay aligned with revops RECORD_TYPES in
# revops/app/components/workflow-editor/config-forms/fetch-netsuite-entity-config.tsx
# and with NetSuiteEntity in api/src/api/net_suite/netsuite.api.ts.
NetsuiteRecordType = Literal[
    "account",
    "check",
    "contact",
    "creditCardCharge",
    "customer",
    "customerPayment",
    "department",
    "deposit",
    "employee",
    "expenseReport",
    "inventoryItem",
    "invoice",
    "job",
    "journalEntry",
    "location",
    "purchaseOrder",
    "salesOrder",
    "subsidiary",
    "vendor",
    "vendorBill",
    "vendorCredit",
    "vendorPayment",
]
NetsuiteFetchRecordType = NetsuiteRecordType
NetsuiteMetric = Literal[
    "actual_by_department",
    "cost_per_sqft",
]
NetsuitePeriod = Literal["mtd", "qtd", "ytd", "last_7_days", "last_30_days", "last_90_days", "custom"]


class ActivateNetsuiteConfig(BaseModel):
    response_key: str = "netsuite_activate_result"


class FetchNetsuiteEntityConfig(BaseModel):
    record_type: NetsuiteFetchRecordType = "customer"
    record_id: str = ""  # direct NetSuite internal ID
    record_id_key: str = ""  # or dot-path to record ID in state
    fields: list[str] = Field(default_factory=list)  # empty = all fields
    limit: int = 100
    token_key: str = "netsuite_activate_result"  # state.data key where activate_netsuite writes its token
    response_key: str = "netsuite_entity"


class CalculateNetsuiteMetricConfig(BaseModel):
    metric: NetsuiteMetric = "actual_by_department"
    period: NetsuitePeriod = "mtd"
    start_date: str = ""  # required when period == "custom" — enforced at runtime by the executor
    end_date: str = ""  # required when period == "custom" — enforced at runtime by the executor
    filter_key: str = ""  # optional dot-path to scope the metric to entities in state
    token_key: str = "netsuite_activate_result"  # state.data key where activate_netsuite writes its token
    response_key: str = "netsuite_metric"


# Semantic record types — must stay aligned with revops RECORD_TYPES in
# revops/app/components/workflow-editor/config-forms/fetch-bloomerang-entity-config.tsx.
# The mapping to Bloomerang v2 REST paths lives in
# graphs/workflow_engine/nodes/fetch_bloomerang_entity.py (_BLOOMERANG_ENDPOINT).
BloomerangRecordType = Literal[
    "appeal",
    "campaign",
    "constituent",
    "designation",
    "fund",
    "household",
    "interaction",
    "note",
    "relationship",
    "transaction",
    "tribute",
]


class FetchBloomerangEntityConfig(BaseModel):
    record_type: BloomerangRecordType = "constituent"
    fields: list[str] = Field(default_factory=list)  # empty = all fields (client-side projection)
    limit: int = Field(default=50, ge=1, le=50)  # maps to Bloomerang's ?take=N
    response_key: str = "bloomerang_entity"


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
            NodeType.ACTIVATE_NETSUITE: ActivateNetsuiteConfig,
            NodeType.FETCH_NETSUITE_ENTITY: FetchNetsuiteEntityConfig,
            NodeType.CALCULATE_NETSUITE_METRIC: CalculateNetsuiteMetricConfig,
            NodeType.FETCH_BLOOMERANG_ENTITY: FetchBloomerangEntityConfig,
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
