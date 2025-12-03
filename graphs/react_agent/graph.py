import json
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from graphs.react_agent.context import (
    AgentInputState,
    AgentOutputState,
    AgentState,
    Context, AgentMode,
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
from graphs.react_agent.utils import _build_tools, get_api_key_for_model, get_today_str


async def call_model(
        state: AgentState, config: RunnableConfig
) -> dict[str, list[AIMessage]]:
    cfg = Context(**config.get("configurable", {}))

    # Prepare tools
    tools_by_name = await _build_tools(cfg)
    tools = list(tools_by_name.values())

    # Resolve API key for the selected model
    api_key = await get_api_key_for_model(config)

    model = init_chat_model(
        cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=api_key or "No token found",
    )

    if tools:
        model = model.bind_tools(tools)

    if cfg.mode == AgentMode.WEB_SEARCH:
        model = model.bind_tools([{"type": "web_search"}])

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

    cfg = Context(**config.get("configurable", {}))
    tools_by_name = await _build_tools(cfg)

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

        # Don't re-encode! The tool already returns a JSON string
        # For RAG tool: result is already JSON with {context_text, sources, retrieval_metadata, document_collections}
        # For other tools: preserve their native return format
        tool_messages.append(
            ToolMessage(
                content=result,  # Use result directly without wrapping
                tool_call_id=tool_call.get("id", ""),
                name=name or tool.name,
            )
        )

        # Extract sources and document_collections from RAG tool responses
        if name == "rag_search":
            try:
                parsed_result = json.loads(result)
                # Check if this is a successful RagToolResponse (not RagToolError)
                if "context_text" in parsed_result and "sources" in parsed_result:
                    # Parse the response as RagToolResponse
                    rag_response = RagToolResponse(**parsed_result)
                    extracted_sources.extend(rag_response.sources)
                    extracted_collections.extend(rag_response.document_collections)
            except (json.JSONDecodeError, Exception):
                # If parsing fails, skip extraction (could be error response or malformed)
                pass

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
