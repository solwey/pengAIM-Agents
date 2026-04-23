import json
import logging
import re
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from aegra_api.settings import settings
from graphs.react_agent.context import (
    AgentInputState,
    AgentMode,
    AgentOutputState,
    AgentState,
    Context,
    SearchAPI,
)
from graphs.react_agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    UNEDITABLE_SYSTEM_PROMPT,
)
from graphs.react_agent.rag_models import (
    DocumentCollectionInfo,
    RagToolError,
    RagToolResponse,
    SourceDocument,
)
from graphs.react_agent.utils import (
    _build_tools,
    get_api_key_for_model,
    get_provider_from_model_name,
    get_today_str,
)

logger = logging.getLogger(__name__)


def _is_openai_reasoning_model(model_id: str) -> bool:
    return (
        model_id.startswith("gpt-5")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
        or model_id.startswith("o4")
    )


def _resolve_reasoning_params(model_name: str | None, reasoning_level: str | None) -> dict[str, Any]:
    """Map generic reasoning level to provider-specific model parameters."""
    if not model_name or not reasoning_level:
        return {}

    model_lower = model_name.lower()
    model_id = model_lower.split(":", 1)[-1]

    if model_lower.startswith("openai:") or model_lower.startswith("azure_openai:"):
        if not _is_openai_reasoning_model(model_id):
            return {}

        return {"reasoning": {"effort": reasoning_level}}

    if model_lower.startswith("google_genai:") or model_lower.startswith("google:"):
        if bool(re.match(r"^gemini-3([.-]|$)", model_id)):
            return {"model_kwargs": {"thinkingLevel": reasoning_level}}
        if bool(re.match(r"^gemini-2\.5([.-]|$)", model_id)):
            thinking_budget_by_level = {
                "minimal": 0,
                "low": 1024,
                "medium": 4096,
                "high": 8192,
            }
            budget = thinking_budget_by_level.get(reasoning_level)
            if budget is not None:
                return {"model_kwargs": {"thinkingBudget": budget}}

    return {}


async def call_model(state: AgentState, config: RunnableConfig) -> dict[str, list[AIMessage]]:
    cfg = Context(**config.get("configurable", {}))

    # Prepare tools (pass config for MCP authorization)
    tools_by_name = await _build_tools(cfg, config)
    tools = list(tools_by_name.values())

    # Resolve API key for the selected model
    api_key = await get_api_key_for_model(cfg.model_name or "", config)

    model_name = cfg.model_name
    if settings.graphs.AZURE_OPENAI_ENDPOINT:
        model_name = re.sub(r"^openai:", "azure_openai:", model_name)

    reasoning_level = cfg.reasoning_level.value if cfg.reasoning_level is not None else None
    reasoning_params = _resolve_reasoning_params(model_name, reasoning_level)
    model = init_chat_model(
        model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=api_key or "No token found",
        **reasoning_params,
    )

    all_tool_specs: list = list(tools)

    if cfg.mode == AgentMode.WEB_SEARCH and cfg.search_api != SearchAPI.NONE:
        model_provider = get_provider_from_model_name(model_name or "")

        if cfg.search_api == SearchAPI.GOOGLE:
            if model_provider == "google":
                all_tool_specs.append({"google_search": {}})
                logger.info(
                    "React Agent: Binding Google native search. model=%s, search_api=%s",
                    model_name,
                    cfg.search_api.value,
                )
            else:
                logger.warning(
                    "React Agent: search_api=GOOGLE but model provider is '%s' (model=%s). "
                    "Skipping Google native search — use a Google model or switch search_api.",
                    model_provider,
                    model_name,
                )

        elif cfg.search_api == SearchAPI.OPENAI:
            if model_provider == "openai":
                all_tool_specs.append({"type": "web_search"})
                logger.info(
                    "React Agent: Binding OpenAI native search. model=%s, search_api=%s",
                    model_name,
                    cfg.search_api.value,
                )
            else:
                logger.warning(
                    "React Agent: search_api=OPENAI but model provider is '%s' (model=%s). "
                    "Skipping OpenAI native search — use an OpenAI model or switch search_api.",
                    model_provider,
                    model_name,
                )

        elif cfg.search_api in (SearchAPI.TAVILY, SearchAPI.FIRECRAWL):
            logger.info(
                "React Agent: Using %s search tool. model=%s",
                cfg.search_api.value,
                model_name,
            )

    if all_tool_specs:
        model = model.bind_tools(all_tool_specs)
        logger.debug(
            "React Agent: Bound %d tool(s). specs=%s",
            len(all_tool_specs),
            [getattr(t, "name", None) or str(t) for t in all_tool_specs],
        )
    else:
        logger.debug(
            "React Agent: No tools to bind. model=%s, mode=%s",
            model_name,
            cfg.mode.value,
        )

    final_system_prompt = (
        (cfg.system_prompt or DEFAULT_SYSTEM_PROMPT)
        + (cfg.tools_policy_prompt or "")
        + UNEDITABLE_SYSTEM_PROMPT.format(date=get_today_str())
    )

    system_message = SystemMessage(content=final_system_prompt)

    response = await model.ainvoke([system_message, *state["messages"]])
    if not isinstance(response, AIMessage):
        raise TypeError(f"Expected AIMessage from model, got {type(response)}")

    return {"messages": [response]}


async def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            last_ai = msg
            break

    if not last_ai or not last_ai.tool_calls:
        return {"messages": []}

    # Find the last human message ID for source/collection tracking
    last_human_message_id: str | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_message_id = getattr(msg, "id", None)
            break

    cfg = Context(**config.get("configurable", {}))
    tools_by_name = await _build_tools(cfg, config)

    tool_messages: list[ToolMessage] = []
    extracted_sources: list[SourceDocument] = []
    extracted_collections: list[DocumentCollectionInfo] = []

    for tool_call in last_ai.tool_calls:
        name = tool_call.get("name")
        raw_args = tool_call.get("args") or {}

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        tool = tools_by_name.get(name or "")
        if tool is None:
            content = f"Tool '{name}' is not available in the current configuration."
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call.get("id", ""),
                    name=name or "unknown_tool",
                )
            )
            continue

        try:
            # Pass config to tool invocation for proper context propagation
            result = await tool.ainvoke(args, config=config)
        except Exception as e:
            # Create structured error response matching tool format

            error = RagToolError(
                error=f"Tool '{name}' raised an error: {str(e)}",
                error_type="tool_execution_error",
                details={"tool_name": name},
            )
            result = json.dumps(error.model_dump(), ensure_ascii=False)

        tool_content = result

        # Extract sources and document_collections from RAG tool responses
        if name == "rag_search":
            try:
                parsed_result = json.loads(result)
                # Check if this is a successful RagToolResponse (not RagToolError)
                if "context_text" in parsed_result and "sources" in parsed_result:
                    # Parse the response as RagToolResponse
                    rag_response = RagToolResponse(**parsed_result)
                    # Set last_human_message_id on each source and collection
                    for source in rag_response.sources:
                        source.last_human_message_id = last_human_message_id
                    for collection in rag_response.document_collections:
                        collection.last_human_message_id = last_human_message_id
                    extracted_sources.extend(rag_response.sources)
                    extracted_collections.extend(rag_response.document_collections)
                    tool_content = rag_response.context_text
            except (json.JSONDecodeError, Exception):
                # If parsing fails, skip extraction (could be error response or malformed)
                pass

        # Don't re-encode! The tool already returns a string
        # For RAG tool: result is a string (context_text)
        # For other tools: preserve their native return format
        tool_messages.append(
            ToolMessage(
                content=tool_content,
                tool_call_id=tool_call.get("id", ""),
                name=name or tool.name,
            )
        )

    return {
        "messages": tool_messages,
        "sources": extracted_sources,
        "document_collections": extracted_collections,
    }


def route_model_output(state: AgentState) -> Literal["tools", "__end__"]:
    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    if last_ai is None:
        return "__end__"

    if getattr(last_ai, "tool_calls", None):
        return "tools"

    return "__end__"


builder = StateGraph(
    AgentState,
    input=AgentInputState,
    output=AgentOutputState,
    config_schema=Context,
)

builder.add_node("call_model", call_model)
builder.add_node("tools", tools_node)

builder.add_edge(START, "call_model")

builder.add_conditional_edges(
    "call_model",
    route_model_output,
    {
        "tools": "tools",
        "__end__": END,
    },
)

builder.add_edge("tools", "call_model")

graph = builder.compile()
