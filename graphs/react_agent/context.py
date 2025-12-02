import operator
from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from graphs.react_agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_RETRIEVAL_POLICY,
    UNEDITABLE_SYSTEM_PROMPT,
)
from graphs.react_agent.rag_models import DocumentCollectionInfo, SourceDocument


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
        optional=True,
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
        optional=True,
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
        optional=True,
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
        optional=True,
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
        optional=True,
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
