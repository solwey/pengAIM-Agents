from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator

from graphs.react_agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_RETRIEVAL_POLICY,
    UNEDITABLE_SYSTEM_PROMPT,
)
from graphs.react_agent.rag_models import DocumentCollectionInfo, SourceDocument
from graphs.shared import (
    DEFAULT_QUESTION_CATEGORIES,
    DefaultQuestionsCategory,
    RetrievalMode,
    ToolCallsVisibility,
)


def merge_sources(
    existing: list[SourceDocument], incoming: list[SourceDocument]
) -> list[SourceDocument]:
    """Merge source documents, keeping only those with matching last_human_message_id.

    Filters out existing sources whose last_human_message_id differs from the
    incoming sources' last_human_message_id, then combines the remaining.
    """
    if not incoming:
        return existing

    # Get the last_human_message_id from incoming sources
    incoming_message_ids = {
        src.last_human_message_id for src in incoming if src.last_human_message_id
    }

    # If no incoming sources have a message ID, just append
    if not incoming_message_ids:
        return existing + incoming

    # Filter existing sources to keep only those with matching message IDs
    filtered_existing = [
        src
        for src in existing
        if src.last_human_message_id in incoming_message_ids
        or src.last_human_message_id is None
    ]

    return filtered_existing + incoming


def merge_document_collections(
    existing: list[DocumentCollectionInfo], incoming: list[DocumentCollectionInfo]
) -> list[DocumentCollectionInfo]:
    """Merge document collections, keeping only those with matching last_human_message_id.

    Filters out existing collections whose last_human_message_id differs from the
    incoming collections' last_human_message_id, then combines the remaining.
    Deduplicates by document_id, keeping the entry with the highest relevance_score.
    """
    if not incoming:
        return existing

    # Get the last_human_message_id from incoming collections
    incoming_message_ids = {
        col.last_human_message_id for col in incoming if col.last_human_message_id
    }

    # If no incoming collections have a message ID, just append
    if not incoming_message_ids:
        combined = existing + incoming
    else:
        # Filter existing collections to keep only those with matching message IDs
        filtered_existing = [
            col
            for col in existing
            if col.last_human_message_id in incoming_message_ids
            or col.last_human_message_id is None
        ]
        combined = filtered_existing + incoming

    # Deduplicate by document_id, keeping the entry with the highest relevance_score
    seen: dict[str, DocumentCollectionInfo] = {}
    for col in combined:
        doc_id = col.document_id
        if doc_id not in seen:
            seen[doc_id] = col
        else:
            existing_score = seen[doc_id].relevance_score or 0.0
            new_score = col.relevance_score or 0.0
            if new_score > existing_score:
                seen[doc_id] = col

    return list(seen.values())


class McpServerConfig(BaseModel):
    """Configuration for a single MCP server.

    MCP (Model Context Protocol) servers provide additional tools that can be
    dynamically loaded and used by the agent. Each server exposes tools via
    HTTP endpoints.
    """

    name: str = Field(..., description="Unique name for this MCP server")
    url: str = Field(..., description="HTTP URL of the MCP server endpoint")


class AgentMode(Enum):
    RAG = "rag"
    WEB_SEARCH = "web_search"
    MODEL = "model"


class SearchAPI(Enum):
    """Enumeration of available search API providers for web search mode."""

    OPENAI = "openai"
    GOOGLE = "google"
    TAVILY = "tavily"
    FIRECRAWL = "firecrawl"
    NONE = "none"


class AgentInputState(TypedDict):
    messages: list[AnyMessage]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[list[SourceDocument], merge_sources]
    document_collections: Annotated[list[DocumentCollectionInfo], merge_document_collections]


class AgentOutputState(TypedDict):
    messages: list[AnyMessage]
    sources: list[SourceDocument]
    document_collections: list[DocumentCollectionInfo]


# noinspection PyArgumentList
class Context(BaseModel):
    mode: AgentMode = Field(
        default=AgentMode.RAG,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": AgentMode.RAG.value,
                "description": (
                    "Select how the agent retrieves information: "
                    "from local RAG data or online sources."
                ),
                "options": [
                    {"label": "Rag only", "value": AgentMode.RAG.value},
                    {"label": "Online only", "value": AgentMode.WEB_SEARCH.value},
                    {"label": "Model knowledge only", "value": AgentMode.MODEL.value},
                ],
            }
        },
    )

    search_api: SearchAPI = Field(
        default=SearchAPI.OPENAI,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": SearchAPI.OPENAI.value,
                "description": (
                    "Search API to use for web search mode. "
                    "Make sure the selected search API is compatible with your model."
                ),
                "options": [
                    {
                        "label": "OpenAI Native Web Search",
                        "value": SearchAPI.OPENAI.value,
                    },
                    {
                        "label": "Google Native Web Search",
                        "value": SearchAPI.GOOGLE.value,
                    },
                    {
                        "label": "Tavily",
                        "value": SearchAPI.TAVILY.value,
                    },
                    {
                        "label": "FireCrawl",
                        "value": SearchAPI.FIRECRAWL.value,
                    },
                    {"label": "None", "value": SearchAPI.NONE.value},
                ],
            }
        },
    )

    agent_openai_api_key: dict[str, str] | None = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "required": True,
                "placeholder": "Enter your custom OpenAI API key for this agent...",
                "description": (
                    "Provide a dedicated OpenAI API key to be used only by this agent. "
                ),
                "default": "",
            }
        },
    )
    rag_openai_api_key: dict[str, str] | None = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "required": True,
                "placeholder": "Enter your OpenAI API key for RAG operations...",
                "description": (
                    "Specify a separate OpenAI API key to be used for RAG tasks "
                    "such as document search, summarization, or contextual QA. "
                ),
                "default": "",
            }
        },
    )
    agent_google_api_key: dict[str, str] = Field(
        default={},
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "required": True,
                "placeholder": "Enter your Google API key for Gemini models...",
                "description": (
                    "Provide a Google API key to be used when selecting Gemini models."
                ),
                "default": {},
            }
        },
    )

    rag_google_api_key: dict[str, str] | None = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "required": True,
                "placeholder": "Enter your Google API key for RAG operations...",
                "description": (
                    "Specify a separate Google API key to be used for RAG tasks "
                    "if using Gemini models."
                ),
                "default": {},
            }
        },
    )

    model_name: str | None = Field(
        default="openai:gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o-mini",
                "description": "The model to use in all generations",
                "options": [
                    {"label": "GPT 4o", "value": "openai:gpt-4o"},
                    {"label": "GPT 4o mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT 4.1", "value": "openai:gpt-4.1"},
                    {"label": "GPT 4.1 mini", "value": "openai:gpt-4.1-mini"},
                    {"label": "GPT 5", "value": "openai:gpt-5"},
                    {"label": "GPT 5.1", "value": "openai:gpt-5.1"},
                    {"label": "GPT 5 mini", "value": "openai:gpt-5-mini"},
                    {"label": "GPT 5.2", "value": "openai:gpt-5.2"},
                    {"label": "Gemini 2.5 Pro", "value": "google_genai:gemini-2.5-pro"},
                    {"label": "Gemini 2.5 Flash", "value": "google_genai:gemini-2.5-flash"},
                    {"label": "Gemini 2.5 Flash Lite", "value": "google_genai:gemini-2.5-flash-lite"},
                ],
            }
        },
    )
    temperature: float | None = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.7,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Controls randomness (0 = deterministic, 2 = creative)",
            }
        },
    )
    max_tokens: int | None = Field(
        default=4000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4000,
                "min": 1,
                "description": "The maximum number of tokens to generate",
            }
        },
    )

    share_new_chats_by_default: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "switch",
                "default": False,
                "description": "Share new chats created with this agent with the entire team by default.",
            }
        },
    )

    tool_calls_visibility: ToolCallsVisibility = Field(
        default=ToolCallsVisibility.ALWAYS_OFF,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": ToolCallsVisibility.ALWAYS_OFF.value,
                "description": (
                    "Controls visibility and behavior of tool call toggles for this agent."
                ),
                "options": [
                    {
                        "label": "User preference",
                        "value": ToolCallsVisibility.USER_PREFERENCE.value,
                    },
                    {
                        "label": "Always ON (forced)",
                        "value": ToolCallsVisibility.ALWAYS_ON.value,
                    },
                    {
                        "label": "Always OFF (disabled)",
                        "value": ToolCallsVisibility.ALWAYS_OFF.value,
                    },
                ],
            }
        },
    )

    rag_retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.RRF,
        description="How the agent retrieves information during RAG operations.",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "select",
                "default": RetrievalMode.RRF.value,
                "description": "Select retrieval strategy for RAG.",
                "options": [
                    {"label": "Basic", "value": RetrievalMode.BASIC.value},
                    {"label": "HyDE", "value": RetrievalMode.HYDE.value},
                    {"label": "RRF", "value": RetrievalMode.RRF.value},
                ],
            }
        },
    )

    rag_system_prompt: str | None = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a custom system prompt or leave empty to use the default one...",
                "description": (
                    "Define a custom system prompt to guide the RAG agent’s behavior and tone. "
                    "If left empty, the agent will automatically use the platform’s default prompt. "
                    "Use this to personalize responses for your specific project or domain."
                ),
                "default": "",
            }
        },
    )
    system_prompt: str | None = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a system prompt...",
                "description": (
                    "The system prompt to use in all generations. "
                    "The following prompt will always be included at the end of the system prompt:\n---"
                    f"{UNEDITABLE_SYSTEM_PROMPT}\n---"
                ),
                "default": DEFAULT_SYSTEM_PROMPT,
            }
        },
    )

    tools_policy_prompt: str | None = Field(
        default=RAG_RETRIEVAL_POLICY,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a tools policy prompt...",
                "description": ("The tools policy prompt to use in all generations."),
                "default": RAG_RETRIEVAL_POLICY,
            }
        },
    )

    default_questions: list[DefaultQuestionsCategory] = Field(
        default=DEFAULT_QUESTION_CATEGORIES,
        metadata={
            "x_oap_ui_config": {
                "type": "repeatable_group",
                "description": (
                    "Configure up to four categories of starter questions that "
                    "will be shown in the chat UI for this agent."
                ),
                "item_label": "Category",
                "fields": {
                    "icon": {
                        "type": "iconify",
                        "label": "Category icon",
                    },
                    "title": {
                        "type": "text",
                        "label": "Category title",
                        "placeholder": "Enter category title",
                        "description": "Human-friendly name of the category shown to users.",
                    },
                    "questions": {
                        "type": "repeatable",
                        "label": "Example questions",
                        "item_label": "Question",
                        "min_items": 2,
                        "max_items": 2,
                        "fields": {
                            "text": {
                                "type": "textarea",
                                "label": "Question text",
                                "description": "Example question that the user can click to ask.",
                                "placeholder": "Enter example question...",
                            }
                        },
                    },
                },
                "max_items": 4,
                "default": DEFAULT_QUESTION_CATEGORIES,
            }
        },
    )

    mcp_servers: list[McpServerConfig] = Field(
        default_factory=list,
        description="List of MCP servers to load tools from",
        metadata={
            "x_oap_ui_config": {
                "type": "json",
                "description": (
                    "Configure MCP (Model Context Protocol) servers to extend the agent's "
                    "capabilities with additional tools. Each server should expose tools "
                    "via HTTP endpoints."
                ),
                "default": [],
            }
        },
    )

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v):
        if v is None or v == "":
            return AgentMode.RAG
        return v

    @field_validator("search_api", mode="before")
    @classmethod
    def validate_search_api(cls, v):
        if v is None or v == "":
            return SearchAPI.OPENAI
        return v

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, v):
        if v is None or v == "":
            return "openai:gpt-4o-mini"
        return v

    @field_validator("temperature", mode="before")
    @classmethod
    def validate_temperature(cls, v):
        if v is None:
            return 0.7
        return v

    @field_validator("max_tokens", mode="before")
    @classmethod
    def validate_max_tokens(cls, v):
        if v is None:
            return 4000
        return v

    @field_validator("share_new_chats_by_default", mode="before")
    @classmethod
    def validate_share_flag(cls, v):
        if v is None:
            return False
        return v

    @field_validator("tool_calls_visibility", mode="before")
    @classmethod
    def validate_tool_calls_visibility(cls, v):
        if v is None or v == "":
            return ToolCallsVisibility.ALWAYS_OFF
        return v

    @field_validator("rag_retrieval_mode", mode="before")
    @classmethod
    def validate_retrieval_mode(cls, v):
        if v is None or v == "":
            return RetrievalMode.RRF
        return v

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def validate_mcp_servers(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            # Convert dict items to McpServerConfig if needed
            return [
                McpServerConfig(**item) if isinstance(item, dict) else item
                for item in v
            ]
        return v

    @field_validator("default_questions", mode="before")
    @classmethod
    def validate_default_questions(cls, v):
        if not v:
            return DEFAULT_QUESTION_CATEGORIES
        return v

    @model_validator(mode="after")
    def validate_google_api_key_for_gemini(self):
        """Ensure Google API key is provided when using Gemini models."""
        model_name = self.model_name or ""
        is_gemini_model = model_name.lower().startswith(
            "google"
        ) or model_name.lower().startswith("gemini")

        if is_gemini_model:
            google_key = self.agent_google_api_key
            if not google_key or not google_key.get("keyId"):
                raise ValueError(
                    "Google API key is required when using Gemini models. "
                    "Please provide agent_google_api_key with a valid keyId."
                )
        return self

    @model_validator(mode="after")
    def validate_openai_api_key(self):
        """Ensure OpenAI API key is provided when using OpenAI models."""
        model_name = self.model_name or ""
        # "check for model name starting with openai"
        is_openai = model_name.lower().startswith("openai")
        if is_openai:
            openai_key = self.agent_openai_api_key
            if not openai_key or not openai_key.get("keyId"):
                raise ValueError(
                    "OpenAI API key is required when using OpenAI models. "
                    "Please provide agent_openai_api_key with a valid keyId."
                )
        return self
