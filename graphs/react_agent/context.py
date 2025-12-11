import operator
from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, field_validator

from graphs.react_agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_RETRIEVAL_POLICY,
    UNEDITABLE_SYSTEM_PROMPT,
)
from graphs.react_agent.rag_models import DocumentCollectionInfo, SourceDocument

DEFAULT_QUESTION_CATEGORIES: list[dict] = [
    {
        "title": "Getting started",
        "questions": [
            {"text": "What can you do for me?"},
            {"text": "How should I start working with you on my tasks?"},
        ],
    },
    {
        "title": "Project / data context",
        "questions": [
            {"text": "What do you know about my current project or data?"},
            {"text": "Summarize the key information you have about our documents."},
        ],
    },
    {
        "title": "Analysis & reasoning",
        "questions": [
            {"text": "Help me analyze the main risks and opportunities here."},
            {"text": "Can you compare the main options and suggest the best one?"},
        ],
    },
    {
        "title": "Next steps & output",
        "questions": [
            {"text": "What are the next steps I should take?"},
            {"text": "Generate a concise action plan based on our discussion."},
        ],
    },
]


class DefaultQuestion(BaseModel):
    text: str = Field(
        ..., description="Question text that will be suggested to the user."
    )


class DefaultQuestionsCategory(BaseModel):
    id: str = Field(
        ..., description="Stable identifier for the category (used internally)."
    )
    title: str = Field(..., description="Human-readable category title.")
    questions: list[DefaultQuestion] = Field(
        default_factory=list,
        description="List of example questions for this category.",
    )


class AgentMode(Enum):
    RAG = "rag"
    WEB_SEARCH = "web_search"
    MODEL = "model"


class RetrievalMode(Enum):
    BASIC = "basic"
    HYDE = "hyde"
    RRF = "rrf"


class AgentInputState(TypedDict):
    messages: list[AnyMessage]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[list[SourceDocument], operator.add]
    document_collections: Annotated[list[DocumentCollectionInfo], operator.add]


class ToolCallsVisibility(Enum):
    USER_PREFERENCE = "user_preference"
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"


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

    @field_validator("default_questions", mode="before")
    @classmethod
    def validate_default_questions(cls, v):
        if not v:
            return DEFAULT_QUESTION_CATEGORIES
        return v
