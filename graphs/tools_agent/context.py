from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from graphs.tools_agent.prompts import DEFAULT_SYSTEM_PROMPT, UNEDITABLE_SYSTEM_PROMPT


class AgentMode(Enum):
    RAG = "rag"
    ONLINE = "online"


class RetrievalMode(Enum):
    BASIC = "basic"
    HYDE = "hyde"
    RRF = "rrf"


class AgentInputState(TypedDict):
    messages: list[AnyMessage]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# noinspection PyArgumentList
class MCPConfig(BaseModel):
    url: str | None = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: list[str] | None = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: bool | None = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""


# noinspection PyArgumentList
class Context(BaseModel):
    model_name: str | None = Field(
        default="openai:gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o-mini",
                "description": "The model to use in all generations",
                "options": [
                    {
                        "label": "Claude Sonnet 4",
                        "value": "anthropic:claude-sonnet-4-0",
                    },
                    {
                        "label": "Claude 3.7 Sonnet",
                        "value": "anthropic:claude-3-7-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Sonnet",
                        "value": "anthropic:claude-3-5-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Haiku",
                        "value": "anthropic:claude-3-5-haiku-latest",
                    },
                    {"label": "o4 mini", "value": "openai:o4-mini"},
                    {"label": "o3", "value": "openai:o3"},
                    {"label": "o3 mini", "value": "openai:o3-mini"},
                    {"label": "GPT 4o", "value": "openai:gpt-4o"},
                    {"label": "GPT 4o mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT 4.1", "value": "openai:gpt-4.1"},
                    {"label": "GPT 4.1 mini", "value": "openai:gpt-4.1-mini"},
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
    mcp_config: MCPConfig | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                # Here is where you would set the default tools.
                "default": {
                    "tools": [],
                    "url": "http://localhost:4444",
                },
            }
        },
    )
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
                    {"label": "Online only", "value": AgentMode.ONLINE.value},
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
    agent_openai_api_key: dict[str, str] | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "password",
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
                "placeholder": "Enter your OpenAI API key for RAG operations...",
                "description": (
                    "Specify a separate OpenAI API key to be used for RAG tasks "
                    "such as document search, summarization, or contextual QA. "
                ),
                "default": "",
            }
        },
    )
