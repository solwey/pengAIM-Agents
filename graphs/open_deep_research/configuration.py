"""Configuration management for the Open Deep Research system."""

import json
import logging
import os
from enum import Enum
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, field_validator, model_validator

from graphs.shared import (
    DEFAULT_QUESTION_CATEGORIES,
    DefaultQuestionsCategory,
    RetrievalMode,
    ToolCallsVisibility,
)


class AgentMode(Enum):
    RAG = "rag"
    ONLINE = "online"


class LLMProvider(Enum):
    """Enumeration of available LLM providers."""

    OPENAI = "openai"
    GOOGLE = "google"


class SearchAPI(Enum):
    """Enumeration of available search API providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    TAVILY = "tavily"
    FIRECRAWL = "firecrawl"
    NONE = "none"


class SubPromptConfig(BaseModel):
    name: str
    text: str


class PlaceholderConfig(BaseModel):
    """A typed placeholder (e.g. artifact picker)."""

    name: str
    type: str = "string"
    label: str | None = None


class StepConfig(BaseModel):
    name: str = ""
    text: str
    placeholders: list[str | PlaceholderConfig] = Field(default_factory=list)
    parallel_sub_prompts: list[SubPromptConfig] = Field(default_factory=list)
    sequential_sub_prompts: list[SubPromptConfig] = Field(default_factory=list)

    @field_validator("placeholders", mode="before")
    @classmethod
    def _coerce_placeholders(cls, v: Any) -> list:
        """Accept both plain strings and dicts like {"name": "x", "type": "artifact"}."""
        if not isinstance(v, list):
            return v
        result = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                config = PlaceholderConfig(**item)
                logging.info(f"[CONFIG] Parsed typed placeholder: name='{config.name}', type='{config.type}'")
                result.append(config)
            else:
                result.append(item)
        logging.info(
            f"[CONFIG] StepConfig placeholders coerced: {[p.name if isinstance(p, PlaceholderConfig) else p for p in result]}"
        )
        return result

    @property
    def placeholder_names(self) -> list[str]:
        """Return plain placeholder name strings (works for both str and PlaceholderConfig)."""
        return [p.name if isinstance(p, PlaceholderConfig) else p for p in self.placeholders]


def _is_openai_gpt5_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.split(":")[-1].lower().startswith("gpt-5")


# noinspection PyArgumentList
class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""

    mode: AgentMode = Field(
        default=AgentMode.RAG,
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

    agent_openai_api_key: dict[str, str] = Field(
        default={},
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "required": True,
                "placeholder": "Enter your custom OpenAI API key for this agent...",
                "description": ("Provide a dedicated OpenAI API key to be used only by this agent. "),
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
                "description": ("Provide a Google API key to be used when selecting Gemini models."),
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
                    "such as document search, summarization, or contextual QA."
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

    # General Configuration
    tool_calls_visibility: ToolCallsVisibility = Field(
        default=ToolCallsVisibility.ALWAYS_OFF,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": ToolCallsVisibility.ALWAYS_OFF.value,
                "description": ("Controls visibility and behavior of tool call toggles for this agent."),
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
                    {
                        "label": "Google Native Web Search",
                        "value": SearchAPI.GOOGLE.value,
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
                    {"label": "GPT‑5.2", "value": "openai:gpt-5.2"},
                    {"label": "GPT‑5‑mini", "value": "openai:gpt-5-mini"},
                    {"label": "Gemini 2.5 Pro", "value": "google_genai:gemini-2.5-pro"},
                    {"label": "Gemini 2.5 Flash", "value": "google_genai:gemini-2.5-flash"},
                    {"label": "Gemini 2.5 Flash Lite", "value": "google_genai:gemini-2.5-flash-lite"},
                    {"label": "Gemini 3 Flash Preview", "value": "google_genai:gemini-3-flash-preview"},
                    {"label": "Gemini 3.1 Pro Preview", "value": "google_genai:gemini-3.1-pro-preview"},
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
                    {"label": "GPT‑5.2", "value": "openai:gpt-5.2"},
                    {"label": "GPT‑5‑mini", "value": "openai:gpt-5-mini"},
                    {"label": "Gemini 2.5 Pro", "value": "google_genai:gemini-2.5-pro"},
                    {"label": "Gemini 2.5 Flash", "value": "google_genai:gemini-2.5-flash"},
                    {"label": "Gemini 2.5 Flash Lite", "value": "google_genai:gemini-2.5-flash-lite"},
                    {"label": "Gemini 3 Flash Preview", "value": "google_genai:gemini-3-flash-preview"},
                    {"label": "Gemini 3.1 Pro Preview", "value": "google_genai:gemini-3.1-pro-preview"},
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
                    {"label": "GPT‑5.2", "value": "openai:gpt-5.2"},
                    {"label": "GPT‑5‑mini", "value": "openai:gpt-5-mini"},
                    {"label": "Gemini 2.5 Pro", "value": "google_genai:gemini-2.5-pro"},
                    {"label": "Gemini 2.5 Flash", "value": "google_genai:gemini-2.5-flash"},
                    {"label": "Gemini 2.5 Flash Lite", "value": "google_genai:gemini-2.5-flash-lite"},
                    {"label": "Gemini 3 Flash Preview", "value": "google_genai:gemini-3-flash-preview"},
                    {"label": "Gemini 3.1 Pro Preview", "value": "google_genai:gemini-3.1-pro-preview"},
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
                    {"label": "GPT‑5.2", "value": "openai:gpt-5.2"},
                    {"label": "GPT‑5‑mini", "value": "openai:gpt-5-mini"},
                    {"label": "Gemini 2.5 Pro", "value": "google_genai:gemini-2.5-pro"},
                    {"label": "Gemini 2.5 Flash", "value": "google_genai:gemini-2.5-flash"},
                    {"label": "Gemini 2.5 Flash Lite", "value": "google_genai:gemini-2.5-flash-lite"},
                    {"label": "Gemini 3 Flash Preview", "value": "google_genai:gemini-3-flash-preview"},
                    {"label": "Gemini 3.1 Pro Preview", "value": "google_genai:gemini-3.1-pro-preview"},
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

    @field_validator("agent_openai_api_key", "rag_openai_api_key", mode="before")
    @classmethod
    def _validate_api_keys(cls, v):
        if v is None or v == "":
            return {}
        return v

    @field_validator("max_structured_output_retries", mode="before")
    @classmethod
    def _validate_max_structured_output_retries(cls, v):
        if v is None or (isinstance(v, int) and v <= 0):
            return 3
        return v

    @field_validator("allow_clarification", mode="before")
    @classmethod
    def _validate_allow_clarification(cls, v):
        if v is None:
            return False
        return v

    @field_validator("max_concurrent_research_units", mode="before")
    @classmethod
    def _validate_max_concurrent_research_units(cls, v):
        if v is None or not isinstance(v, int) or v <= 0:
            return 5
        if v > 20:
            return 20
        return v

    @field_validator("search_api", mode="before")
    @classmethod
    def _validate_search_api(cls, v):
        if v is None or v == "":
            return SearchAPI.FIRECRAWL
        return v

    @field_validator("max_researcher_iterations", mode="before")
    @classmethod
    def _validate_max_researcher_iterations(cls, v):
        if v is None or not isinstance(v, int) or v <= 0:
            return 6
        if v > 10:
            return 10
        return v

    @field_validator("max_react_tool_calls", mode="before")
    @classmethod
    def _validate_max_react_tool_calls(cls, v):
        if v is None or not isinstance(v, int) or v <= 0:
            return 10
        if v > 30:
            return 30
        return v

    @field_validator(
        "summarization_model",
        "research_model",
        "compression_model",
        "final_report_model",
        mode="before",
    )
    @classmethod
    def _validate_model_names(cls, v):
        if v is None:
            return "openai:gpt-4o-mini"
        if isinstance(v, str) and v.strip() == "":
            return "openai:gpt-4o-mini"
        return v

    @field_validator("rag_embedding_model", mode="before")
    @classmethod
    def _validate_rag_embedding_model(cls, v):
        if v is None:
            return "text-embedding-3-small"
        if isinstance(v, str) and v.strip() == "":
            return "text-embedding-3-small"
        return v

    @field_validator(
        "summarization_model_max_tokens",
        "research_model_max_tokens",
        "compression_model_max_tokens",
        "final_report_model_max_tokens",
        mode="before",
    )
    @classmethod
    def _validate_model_max_tokens(cls, v, info):
        default_map = {
            "summarization_model_max_tokens": 8192,
            "research_model_max_tokens": 10000,
            "compression_model_max_tokens": 8192,
            "final_report_model_max_tokens": 10000,
        }
        field_name = info.field_name
        default_value = default_map.get(field_name, 1000)
        if v is None or not isinstance(v, int) or v <= 0:
            return default_value
        return v

    @field_validator("max_content_length", mode="before")
    @classmethod
    def _validate_max_content_length(cls, v):
        if v is None or not isinstance(v, int) or v <= 0:
            return 50000
        return v

    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, v):
        if v is None or v == "":
            return AgentMode.RAG
        return v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _validate_llm_provider(cls, v):
        if v is None or v == "":
            return LLMProvider.OPENAI
        return v

    @field_validator("rag_retrieval_mode", mode="before")
    @classmethod
    def _validate_rag_retrieval_mode(cls, v):
        if v is None or v == "":
            return RetrievalMode.RRF
        return v

    @field_validator("rag_llm_temperature", mode="before")
    @classmethod
    def _validate_rag_llm_temperature(cls, v):
        if v is None:
            return 1.0
        return v

    @field_validator("rag_llm_max_tokens", mode="before")
    @classmethod
    def _validate_rag_llm_max_tokens(cls, v):
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
    def _validate_rag_retrieval_positive_ints(cls, v, info):
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
    def _validate_rag_retrieval_ratios(cls, v, info):
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
    def _validate_rag_retrieval_ratio_sum(self):
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
    def _validate_share_new_chats_by_default(cls, v):
        if v is None:
            return False
        return v

    @field_validator("tool_calls_visibility", mode="before")
    @classmethod
    def _validate_tool_calls_visibility(cls, v):
        if v is None or v == "":
            return ToolCallsVisibility.ALWAYS_OFF
        return v

    @field_validator("default_questions", mode="before")
    @classmethod
    def _validate_default_questions(cls, v):
        if not v:
            return DEFAULT_QUESTION_CATEGORIES
        return v

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

    @model_validator(mode="after")
    def validate_google_api_key_for_gemini(self):
        """Ensure Google API key is provided when using Google provider."""
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
        """Ensure OpenAI API key is provided when using OpenAI provider."""
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
    def validate_rag_llm_overrides_for_reasoning_models(self):
        is_openai_gpt5_summarization = _is_openai_gpt5_model(self.summarization_model)
        if self.llm_provider != LLMProvider.OPENAI or not is_openai_gpt5_summarization:
            self.rag_llm_temperature = None
            self.rag_llm_max_tokens = None
        return self

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
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
            return not (isinstance(v, (list, dict)) and len(v) == 0)

        for field_name, _field_info in cls.model_fields.items():
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

    @model_validator(mode="after")
    def validate_embedding_model(self):
        if self.rag_embedding_model is None or self.rag_embedding_model == "":
            self.rag_embedding_model = (
                "gemini-embedding-001" if self.llm_provider == LLMProvider.GOOGLE else "text-embedding-3-small"
            )
            return self

        if self.llm_provider == LLMProvider.GOOGLE and self.rag_embedding_model == "text-embedding-3-small":
            self.rag_embedding_model = "gemini-embedding-001"
            return self

        if self.llm_provider == LLMProvider.OPENAI and self.rag_embedding_model == "gemini-embedding-001":
            self.rag_embedding_model = "text-embedding-3-small"
            return self

        return self

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
