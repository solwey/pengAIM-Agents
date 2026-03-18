from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import ConfigDict, BaseModel, Field, field_validator, model_validator

from graphs.react_agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_RETRIEVAL_POLICY,
    UNEDITABLE_SYSTEM_PROMPT,
)
from graphs.react_agent.rag_models import DocumentCollectionInfo, SourceDocument
from graphs.shared import (
    DEFAULT_QUESTION_CATEGORIES,
    DefaultQuestionsCategory,
    McpConfigMixin,
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


class AgentMode(Enum):
    RAG = "rag"
    WEB_SEARCH = "web_search"
    MODEL = "model"


class LLMProvider(Enum):
    """Enumeration of available LLM providers."""

    OPENAI = "openai"
    GOOGLE = "google"


class SearchAPI(Enum):
    """Enumeration of available search API providers for web search mode."""

    OPENAI = "openai"
    GOOGLE = "google"
    TAVILY = "tavily"
    FIRECRAWL = "firecrawl"
    NONE = "none"


def _is_openai_gpt5_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.split(":")[-1].lower().startswith("gpt-5")


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
class Context(BaseModel, McpConfigMixin):
    mode: AgentMode = Field(
        default=AgentMode.RAG,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "required": True,
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

    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": LLMProvider.OPENAI.value,
                "description": "Select the LLM Provider. This determines which API key will be used.",
                "options": [
                    {"label": "OpenAI", "value": LLMProvider.OPENAI.value},
                    {"label": "Google Gemini", "value": LLMProvider.GOOGLE.value},
                ],
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
                    {"label": "Gemini 3 Flash Preview", "value": "google_genai:gemini-3-flash-preview"},
                    {"label": "Gemini 3.1 Pro Preview", "value": "google_genai:gemini-3.1-pro-preview"},
                ],
            }
        },
    )

    agent_openai_api_key: dict[str, str] = Field(
        default={},
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "required": True,
                "placeholder": "Enter your custom OpenAI API key for this agent...",
                "description": (
                    "Provide a dedicated OpenAI API key to be used only by this agent. "
                ),
                "default": {},
            }
        },
    )

    rag_openai_api_key: dict[str, str] = Field(
        default={},
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "required": True,
                "placeholder": "Enter your OpenAI API key for RAG operations...",
                "description": (
                    "Specify a separate OpenAI API key to be used for RAG tasks "
                    "such as document search, summarization, or contextual QA. "
                ),
                "default": {},
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

    rag_google_api_key: dict[str, str] = Field(
        default={},
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

    rag_embedding_model: str = Field(
        default="text-embedding-3-small",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "text-embedding-3-small",
                "description": "Embedding model to use for RAG vectorization.",
                "options": [
                    {"label": "Text Embedding 3 Small", "value": "text-embedding-3-small"},
                    {"label": "Text Embedding 3 Large", "value": "text-embedding-3-large"},
                    {"label": "Text Embedding Ada 002", "value": "text-embedding-ada-002"},
                    {"label": "Gemini Embedding 001", "value": "gemini-embedding-001"},
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
                "group": "retrieval_settings",
                "group_label": "Retrieval Settings",
                "group_order": 5,
                "field_order": 10,
                "options": [
                    {"label": "Basic", "value": RetrievalMode.BASIC.value},
                    {"label": "HyDE", "value": RetrievalMode.HYDE.value},
                    {"label": "RRF", "value": RetrievalMode.RRF.value},
                ],
            }
        },
    )
    rag_retrieval_context_token_budget: int = Field(
        default=128000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 128000,
                "min": 1,
                "description": "Total token budget allocated for the assembled retrieval context.",
                "group": "retrieval_budget",
                "group_label": "Retrieval Budget & Ratios",
                "group_order": 10,
                "field_order": 10,
            }
        },
    )
    rag_retrieval_text_unit_ratio: float = Field(
        default=0.25,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.25,
                "min": 0.01,
                "max": 1,
                "step": 0.01,
                "description": "Share of the context token budget reserved for text units (chunks).",
                "group": "retrieval_budget",
                "group_label": "Retrieval Budget & Ratios",
                "group_order": 10,
                "field_order": 20,
                "validation_group": "retrieval_ratio_budget",
                "validation_rule": "sum_lte",
                "validation_value": 1,
                "validation_message": "Sum of retrieval ratios must be <= 1",
            }
        },
    )
    rag_retrieval_community_ratio: float = Field(
        default=0.1,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.1,
                "min": 0.01,
                "max": 1,
                "step": 0.01,
                "description": "Share of the context token budget reserved for community reports.",
                "group": "retrieval_budget",
                "group_label": "Retrieval Budget & Ratios",
                "group_order": 10,
                "field_order": 30,
                "validation_group": "retrieval_ratio_budget",
                "validation_rule": "sum_lte",
                "validation_value": 1,
                "validation_message": "Sum of retrieval ratios must be <= 1",
            }
        },
    )
    rag_retrieval_entity_ratio: float = Field(
        default=0.35,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.35,
                "min": 0.01,
                "max": 1,
                "step": 0.01,
                "description": "Share of the context token budget reserved for entities.",
                "group": "retrieval_budget",
                "group_label": "Retrieval Budget & Ratios",
                "group_order": 10,
                "field_order": 40,
                "validation_group": "retrieval_ratio_budget",
                "validation_rule": "sum_lte",
                "validation_value": 1,
                "validation_message": "Sum of retrieval ratios must be <= 1",
            }
        },
    )
    rag_retrieval_relationship_ratio: float = Field(
        default=0.3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.3,
                "min": 0.01,
                "max": 1,
                "step": 0.01,
                "description": "Share of the context token budget reserved for relationships between entities.",
                "group": "retrieval_budget",
                "group_label": "Retrieval Budget & Ratios",
                "group_order": 10,
                "field_order": 50,
                "validation_group": "retrieval_ratio_budget",
                "validation_rule": "sum_lte",
                "validation_value": 1,
                "validation_message": "Sum of retrieval ratios must be <= 1",
            }
        },
    )
    rag_retrieval_top_k_relationships: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10,
                "min": 1,
                "description": "Maximum number of relationships to keep during retrieval selection.",
                "group": "retrieval_top_k_limits",
                "group_label": "Retrieval Top-K Limits",
                "group_order": 20,
                "field_order": 10,
            }
        },
    )
    rag_retrieval_top_k_entities: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10,
                "min": 1,
                "description": "Maximum number of entities used for retrieval and context assembly.",
                "group": "retrieval_top_k_limits",
                "group_label": "Retrieval Top-K Limits",
                "group_order": 20,
                "field_order": 20,
            }
        },
    )
    rag_retrieval_chunk_top_k_per_entity: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "description": "Number of top chunks to keep per retrieved entity.",
                "group": "retrieval_top_k_limits",
                "group_label": "Retrieval Top-K Limits",
                "group_order": 20,
                "field_order": 30,
            }
        },
    )
    rag_retrieval_chunk_ranking_overfetch: int = Field(
        default=4,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4,
                "min": 1,
                "description": "Overfetch factor used before the final chunk-ranking pass.",
                "group": "chunk_ranking",
                "group_label": "Chunk Ranking",
                "group_order": 30,
                "field_order": 10,
            }
        },
    )
    rag_retrieval_chunk_rank_weight_similarity: float = Field(
        default=0.75,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.75,
                "min": 0.01,
                "max": 1,
                "step": 0.01,
                "description": "Weight of semantic similarity in chunk ranking.",
                "group": "chunk_ranking",
                "group_label": "Chunk Ranking",
                "group_order": 30,
                "field_order": 20,
            }
        },
    )
    rag_retrieval_chunk_rank_weight_entity: float = Field(
        default=0.25,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.25,
                "min": 0.01,
                "max": 1,
                "step": 0.01,
                "description": "Weight of entity relevance in chunk ranking.",
                "group": "chunk_ranking",
                "group_label": "Chunk Ranking",
                "group_order": 30,
                "field_order": 30,
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
                "group": "retrieval_settings",
                "group_label": "Retrieval Settings",
                "group_order": 5,
                "field_order": 20,
            }
        },
    )
    rag_llm_temperature: float | None = Field(
        default=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 1.0,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Temperature for LLM calls used only in RAG operations.",
            }
        },
    )
    rag_llm_max_tokens: int | None = Field(
        default=24_000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 24_000,
                "min": 20_000,
                "description": "Maximum output tokens for LLM calls used only in RAG operations.",
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

    @field_validator("rag_embedding_model", mode="before")
    @classmethod
    def validate_rag_embedding_model(cls, v):
        if v is None or v == "":
            return "text-embedding-3-small"
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

    @field_validator("rag_llm_temperature", mode="before")
    @classmethod
    def validate_rag_llm_temperature(cls, v):
        if v is None:
            return 1.0
        return v

    @field_validator("rag_llm_max_tokens", mode="before")
    @classmethod
    def validate_rag_llm_max_tokens(cls, v):
        if v is None:
            return 20000
        return v

    @field_validator(
        "rag_retrieval_context_token_budget",
        "rag_retrieval_top_k_relationships",
        "rag_retrieval_top_k_entities",
        "rag_retrieval_chunk_top_k_per_entity",
        "rag_retrieval_chunk_ranking_overfetch",
        mode="before",
    )
    @classmethod
    def validate_rag_retrieval_positive_ints(cls, v, info):
        default_map = {
            "rag_retrieval_context_token_budget": 128000,
            "rag_retrieval_top_k_relationships": 10,
            "rag_retrieval_top_k_entities": 10,
            "rag_retrieval_chunk_top_k_per_entity": 3,
            "rag_retrieval_chunk_ranking_overfetch": 4,
        }
        default_value = default_map[info.field_name]
        if v is None:
            return default_value
        try:
            parsed = int(v)
        except (TypeError, ValueError):
            return default_value
        if parsed < 1:
            return default_value
        return parsed

    @field_validator(
        "rag_retrieval_text_unit_ratio",
        "rag_retrieval_community_ratio",
        "rag_retrieval_entity_ratio",
        "rag_retrieval_relationship_ratio",
        "rag_retrieval_chunk_rank_weight_similarity",
        "rag_retrieval_chunk_rank_weight_entity",
        mode="before",
    )
    @classmethod
    def validate_rag_retrieval_ratios(cls, v, info):
        default_map = {
            "rag_retrieval_text_unit_ratio": 0.25,
            "rag_retrieval_community_ratio": 0.1,
            "rag_retrieval_entity_ratio": 0.35,
            "rag_retrieval_relationship_ratio": 0.3,
            "rag_retrieval_chunk_rank_weight_similarity": 0.75,
            "rag_retrieval_chunk_rank_weight_entity": 0.25,
        }
        default_value = default_map[info.field_name]
        if v is None:
            return default_value
        try:
            parsed = float(v)
        except (TypeError, ValueError):
            return default_value
        if not (0 <= parsed <= 1):
            return default_value
        return parsed

    @model_validator(mode="after")
    def validate_rag_retrieval_ratio_sum(self):
        retrieval_ratio_sum = (
            self.rag_retrieval_text_unit_ratio
            + self.rag_retrieval_community_ratio
            + self.rag_retrieval_entity_ratio
            + self.rag_retrieval_relationship_ratio
        )
        if retrieval_ratio_sum > 1:
            raise ValueError("Sum of retrieval ratios must be <= 1")
        return self

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


    @field_validator("default_questions", mode="before")
    @classmethod
    def validate_default_questions(cls, v):
        if not v:
            return DEFAULT_QUESTION_CATEGORIES
        return v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_llm_provider(cls, v):
        if v is None or v == "":
            return LLMProvider.OPENAI
        return v

    @model_validator(mode="after")
    def validate_google_api_key_for_gemini(self):
        """Ensure Google API key is provided when using Gemini models."""
        if self.agent_openai_api_key == {} and self.agent_google_api_key == {}:
            return self

        if self.llm_provider == LLMProvider.GOOGLE:
            google_key = self.agent_google_api_key
            if not google_key or not google_key.get("keyId"):
                raise ValueError(
                    "Google API key is required when using Google Gemini provider. "
                    "Please provide agent_google_api_key with a valid keyId."
                )
        return self

    @model_validator(mode="after")
    def validate_openai_api_key(self):
        """Ensure OpenAI API key is provided when using OpenAI models."""
        if self.agent_openai_api_key == {} and self.agent_google_api_key == {}:
            return self

        if self.llm_provider == LLMProvider.OPENAI:
            openai_key = self.agent_openai_api_key
            if not openai_key or not openai_key.get("keyId"):
                raise ValueError(
                    "OpenAI API key is required when using OpenAI provider. "
                    "Please provide agent_openai_api_key with a valid keyId."
                )
        return self

    @model_validator(mode="after")
    def validate_embedding_model(self):
        if self.rag_embedding_model is None or self.rag_embedding_model == "":
            self.rag_embedding_model = (
                "gemini-embedding-001"
                if self.llm_provider == LLMProvider.GOOGLE
                else "text-embedding-3-small"
            )
            return self

        if self.llm_provider == LLMProvider.GOOGLE  and self.rag_embedding_model == "text-embedding-3-small":
            self.rag_embedding_model = "gemini-embedding-001"
            return self

        if self.llm_provider == LLMProvider.OPENAI and self.rag_embedding_model == "gemini-embedding-001":
            self.rag_embedding_model = "text-embedding-3-small"
            return self

        return self

    @model_validator(mode="after")
    def validate_rag_llm_overrides_for_reasoning_models(self):
        if (
            self.llm_provider != LLMProvider.OPENAI
            or not _is_openai_gpt5_model(self.model_name)
        ):
            self.rag_llm_temperature = None
            self.rag_llm_max_tokens = None
        return self
