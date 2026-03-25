"""Pydantic models for validating workflow JSON definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class NodeType(str, Enum):
    API_REQUEST = "api_request"
    CONDITION = "condition"
    TRANSFORM = "transform"
    SLACK_MESSAGE = "slack_message"
    EMAIL_MESSAGE = "email_message"
    SWITCH = "switch"
    DELAY = "delay"


class ComparisonOperator(str, Enum):
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
    retry_on_status: list[int] = Field(
        default_factory=lambda: [500, 502, 503, 504]
    )


class ConditionConfig(BaseModel):
    field: str  # dot-notation path, e.g. "api_response.status_code"
    operator: ComparisonOperator
    value: Any  # value to compare against


class TransformConfig(BaseModel):
    set: dict[str, Any]  # key-value pairs to merge into state["data"]


class SlackMessageConfig(BaseModel):
    webhook_url: str
    message: str
    response_key: str = "slack_response"


class EmailMessageConfig(BaseModel):
    to: str
    subject: str
    html_body: str
    text_body: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_password: str | None = None
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


# ── Node & Edge definitions ──────────────────────────────────


class NodeDef(BaseModel):
    id: str
    type: NodeType
    config: dict[str, Any]

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
                raise ValueError(
                    f"Node id '{nid}' is reserved. Use a different name."
                )

        # Validate edges reference existing nodes
        for edge in self.edges:
            if edge.from_node not in valid_ids:
                raise ValueError(
                    f"Edge 'from' references unknown node: '{edge.from_node}'"
                )

            if edge.type == "sequential":
                if edge.to_node is None:
                    raise ValueError(
                        f"Sequential edge from '{edge.from_node}' must have a 'to' field"
                    )
                if edge.to_node not in valid_ids:
                    raise ValueError(
                        f"Edge 'to' references unknown node: '{edge.to_node}'"
                    )

            elif edge.type == "conditional":
                if not edge.branches:
                    raise ValueError(
                        f"Conditional edge from '{edge.from_node}' must have 'branches'"
                    )
                if set(edge.branches.keys()) != {"yes", "no"}:
                    raise ValueError(
                        f"Conditional edge from '{edge.from_node}' must have exactly "
                        f"'yes' and 'no' branches, got: {set(edge.branches.keys())}"
                    )
                for label, target in edge.branches.items():
                    if target not in valid_ids:
                        raise ValueError(
                            f"Branch '{label}' references unknown node: '{target}'"
                        )

            elif edge.type == "switch":
                if not edge.branches:
                    raise ValueError(
                        f"Switch edge from '{edge.from_node}' must have 'branches'"
                    )
                for label, target in edge.branches.items():
                    if target not in valid_ids:
                        raise ValueError(
                            f"Switch branch '{label}' references unknown node: '{target}'"
                        )

            elif edge.type == "on_error":
                if edge.to_node is None:
                    raise ValueError(
                        f"on_error edge from '{edge.from_node}' must have a 'to' field"
                    )
                if edge.to_node not in valid_ids:
                    raise ValueError(
                        f"on_error edge 'to' references unknown node: '{edge.to_node}'"
                    )

        # Conditional edges must originate from condition nodes
        for edge in self.edges:
            if edge.type == "conditional":
                source_node = next(
                    (n for n in self.nodes if n.id == edge.from_node), None
                )
                if source_node is None or source_node.type != NodeType.CONDITION:
                    raise ValueError(
                        f"Conditional edge from '{edge.from_node}' must originate "
                        f"from a 'condition' node"
                    )

        # Switch edges must originate from switch nodes
        for edge in self.edges:
            if edge.type == "switch":
                source_node = next(
                    (n for n in self.nodes if n.id == edge.from_node), None
                )
                if source_node is None or source_node.type != NodeType.SWITCH:
                    raise ValueError(
                        f"Switch edge from '{edge.from_node}' must originate "
                        f"from a 'switch' node"
                    )

        # Must have exactly one edge from __start__
        start_edges = [e for e in self.edges if e.from_node == "__start__"]
        if len(start_edges) != 1:
            raise ValueError(
                f"Workflow must have exactly one edge from '__start__', "
                f"found {len(start_edges)}"
            )

        return self

    def get_node(self, node_id: str) -> NodeDef | None:
        return next((n for n in self.nodes if n.id == node_id), None)
