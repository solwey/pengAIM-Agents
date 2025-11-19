"""Configuration management for the Open Deep Research system."""

import json
import os
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, field_validator


class AgentMode(Enum):
    RAG = "rag"
    ONLINE = "online"


class RetrievalMode(Enum):
    BASIC = "basic"
    HYDE = "hyde"
    RRF = "rrf"


class SearchAPI(Enum):
    """Enumeration of available search API providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    FIRECRAWL = "firecrawl"
    NONE = "none"


class SubPromptConfig(BaseModel):
    name: str
    text: str


class StepConfig(BaseModel):
    text: str
    placeholders: list[str] = Field(default_factory=list)
    parallel_sub_prompts: list[SubPromptConfig] = Field(default_factory=list)
    sequential_sub_prompts: list[SubPromptConfig] = Field(default_factory=list)


# noinspection PyArgumentList
class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""

    mode: AgentMode = Field(
        default=AgentMode.RAG,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "required": True,
                "default": AgentMode.RAG.value,
                "description": "Select how the agent retrieves information: from local RAG data or online sources.",
                "options": [
                    {"label": "Rag only", "value": AgentMode.RAG.value},
                    {"label": "Online only", "value": AgentMode.ONLINE.value},
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

    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models",
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

    allow_clarification: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research",
            }
        },
    )

    shared_chat_history: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "switch",
                "default": False,
                "description": "Allow all members of the team to view and access this agent’s runs and threads",
            }
        },
    )

    steps: list[StepConfig] = Field(
        default_factory=list,
        metadata={
            "x_oap_ui_config": {
                "type": "json",
                "default": [],
                "description": "Ordered list of steps. Each step supports optional parallel and sequential subprompts.",
            },
        },
    )

    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits.",
            }
        },
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.FIRECRAWL,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "firecrawl",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "FireCrawl", "value": SearchAPI.FIRECRAWL.value},
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {
                        "label": "OpenAI Native Web Search",
                        "value": SearchAPI.OPENAI.value,
                    },
                    {
                        "label": "Anthropic Native Web Search",
                        "value": SearchAPI.ANTHROPIC.value,
                    },
                    {"label": "None", "value": SearchAPI.NONE.value},
                ],
            }
        },
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions.",
            }
        },
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step.",
            }
        },
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o-mini",
                "options": [
                    {"label": "GPT‑4o‑mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT‑5.0", "value": "openai:gpt-5.0"},
                    {"label": "GPT‑5.1", "value": "openai:gpt-5.1"},
                    {"label": "GPT‑5‑mini", "value": "openai:gpt-5-mini"},
                ],
                "description": "Model for summarizing research results from Tavily search results",
            }
        },
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model",
            }
        },
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization",
            }
        },
    )
    research_model: str = Field(
        default="openai:gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o-mini",
                "options": [
                    {"label": "GPT‑4o‑mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT‑5.0", "value": "openai:gpt-5.0"},
                    {"label": "GPT‑5.1", "value": "openai:gpt-5.1"},
                    {"label": "GPT‑5‑mini", "value": "openai:gpt-5-mini"},
                ],
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API.",
            }
        },
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model",
            }
        },
    )
    compression_model: str = Field(
        default="openai:gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o-mini",
                "options": [
                    {"label": "GPT‑4o‑mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT‑5.0", "value": "openai:gpt-5.0"},
                    {"label": "GPT‑5.1", "value": "openai:gpt-5.1"},
                    {"label": "GPT‑5‑mini", "value": "openai:gpt-5-mini"},
                ],
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API.",
            }
        },
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model",
            }
        },
    )
    final_report_model: str = Field(
        default="openai:gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o-mini",
                "options": [
                    {"label": "GPT‑4o‑mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT‑5.0", "value": "openai:gpt-5.0"},
                    {"label": "GPT‑5.1", "value": "openai:gpt-5.1"},
                    {"label": "GPT‑5‑mini", "value": "openai:gpt-5-mini"},
                ],
                "description": "Model for writing the final report from all research findings",
            }
        },
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model",
            }
        },
    )
    system_prompt: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "description": (
                    "High-level behavioral instructions that define the agent's role, "
                    "tone, and strategy for reasoning or decision-making. "
                    "Use this to influence how the agent plans, delegates tasks, "
                    "or interacts with tools."
                ),
            }
        },
    )

    sales_context_prompt: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "description": (
                    "Context-specific guidance that provides background information, "
                    "business domain knowledge, or situational goals relevant to the current task. "
                    "Use this to help the agent tailor its reasoning and responses "
                    "to a particular scenario (e.g., sales, research, customer support)."
                ),
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

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, v):
        if v is None:
            return v
        if isinstance(v, list):
            out = []
            for item in v:
                if isinstance(item, StepConfig):
                    out.append(item)
                elif isinstance(item, dict):
                    out.append(StepConfig(**item))
                else:
                    raise TypeError("Each item in 'steps' must be StepConfig or dict")
            return out
        return v

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        base = cls()

        configurable = (config or {}).get("configurable", {}) or {}
        overrides: dict[str, Any] = {}

        def is_meaningful(v: Any) -> bool:
            # Treat None, empty strings, and empty containers as "not provided"
            if v is None:
                return False
            if isinstance(v, str) and v.strip() == "":
                return False
            if isinstance(v, (list, dict)) and len(v) == 0:
                return False
            return True

        for field_name, field_info in cls.model_fields.items():
            # env has priority over configurable
            raw = os.environ.get(field_name.upper(), None)
            val: Any = None

            # Try to JSON-decode env values for complex fields
            if is_meaningful(raw):
                try:
                    val = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    val = raw
            else:
                # fallback to configurable
                cand = configurable.get(field_name, None)
                if is_meaningful(cand):
                    val = cand

            if val is None:
                continue

            # Prefer merging into the current instance (handles Optional[...] annotations)
            current = getattr(base, field_name, None)
            if isinstance(current, BaseModel) and isinstance(val, dict):
                cleaned = {k: v for k, v in val.items() if is_meaningful(v)}
                if cleaned:
                    overrides[field_name] = current.model_copy(update=cleaned)
                # If cleaned is empty, skip override to preserve defaults
                continue

            overrides[field_name] = val

        data = base.model_dump(mode="python")
        data.update(overrides)
        return cls.model_validate(data)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
