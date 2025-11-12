import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from graphs.tools_agent.context import AgentInputState, AgentMode, AgentState, Context
from graphs.tools_agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_ONLY_PROMPT,
    UNEDITABLE_SYSTEM_PROMPT,
)
from graphs.tools_agent.utils import _build_tools, get_api_key_for_model


async def call_model(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[AIMessage]]:
    cfg = Context(**config.get("configurable", {}))

    # Prepare tools
    tools_by_name = await _build_tools(cfg, config)
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

    rag_only_contract = ""
    if cfg.mode == AgentMode.RAG:
        rag_only_contract = RAG_ONLY_PROMPT

    final_system_prompt = (
        (cfg.system_prompt or DEFAULT_SYSTEM_PROMPT)
        + rag_only_contract
        + UNEDITABLE_SYSTEM_PROMPT
    )

    system_message = SystemMessage(content=final_system_prompt)

    response = await model.ainvoke([system_message, *state["messages"]])
    if not isinstance(response, AIMessage):
        raise TypeError(f"Expected AIMessage from model, got {type(response)}")

    return {"messages": [response]}


async def tools_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[ToolMessage]]:
    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            last_ai = msg
            break

    if not last_ai or not last_ai.tool_calls:
        return {"messages": []}

    cfg = Context(**config.get("configurable", {}))
    tools_by_name = await _build_tools(cfg, config)

    tool_messages: list[ToolMessage] = []

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
            result = await tool.ainvoke(args)
        except Exception as e:
            result = f"Tool '{name}' raised an error: {e}"

        tool_messages.append(
            ToolMessage(
                content=json.dumps({"answer": result, "question": args.get("query")}),
                tool_call_id=tool_call.get("id", ""),
                name=name or tool.name,
            )
        )

    return {"messages": tool_messages}


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
