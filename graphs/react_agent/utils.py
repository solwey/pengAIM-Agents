import os
from datetime import datetime
from typing import Any

import aiohttp
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from graphs.react_agent.context import AgentMode, AgentState, Context
from graphs.react_agent.mcp_tools import load_mcp_tools
from graphs.react_agent.tools import create_rag_tool
from graphs.shared import build_middlewares, restore_python_repr_content

RAG_URL = os.getenv("RAG_API_URL", "")

if not RAG_URL:
    raise RuntimeError("RAG-only mode is enabled but RAG_API_URL is not set.")


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a %b} {now.day}, {now:%Y}"


async def get_api_key_for_model(config: RunnableConfig) -> str | None:
    authorization = (
        config.get("configurable", {})
        .get("langgraph_auth_user", {})
        .get("permissions")[0]
        .replace("authz:", "")
    )
    key_data = config.get("configurable", {}).get("agent_openai_api_key", {})

    if not authorization or not key_data:
        return None

    search_endpoint = f"{RAG_URL}/keys/{key_data.get('keyId')}/reveal"
    headers = {"authorization": authorization, "Accept": "text/plain"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_endpoint, headers=headers) as search_response:
                search_response.raise_for_status()
                key = await search_response.text()
        return key
    except Exception as e:
        print("Key exception", e)
        return None


def _extract_authorization(config: RunnableConfig | None) -> str | None:
    """Extract authorization token from RunnableConfig.

    Args:
        config: The RunnableConfig containing auth user context

    Returns:
        Authorization token string or None if not found
    """
    if not config:
        return None

    try:
        configurable = config.get("configurable", {})
        auth_user = configurable.get("langgraph_auth_user", {})
        permissions = auth_user.get("permissions", [])
        if permissions:
            return permissions[0].replace("authz:", "")
    except (KeyError, IndexError, AttributeError):
        pass

    return None


async def build_tools(
    cfg: Context,
    config: RunnableConfig | None = None,
) -> dict[str, BaseTool]:
    """Build a mapping of tool name -> tool instance based on the config.

    This preserves the previous behavior:
    * In RAG mode, only the RAG `rag_search` tool is available.
    * In ONLINE mode, only simple agent is available.

    Additionally, MCP tools are loaded if configured.

    Args:
        cfg: The Context configuration
        config: Optional RunnableConfig for extracting authorization

    Returns:
        Dictionary mapping tool names to tool instances
    """
    tools: list[BaseTool] = []

    # Load RAG tool if in RAG mode
    if cfg.mode == AgentMode.RAG:
        rag_tool = await create_rag_tool(RAG_URL)
        tools.append(rag_tool)

    # Load MCP tools if configured
    if cfg.mcp_servers:
        authorization = _extract_authorization(config)
        mcp_tools = await load_mcp_tools(cfg.mcp_servers, authorization)
        tools.extend(mcp_tools)

    return {t.name: t for t in tools}


def _merge_state_updates(updates: dict[str, Any]) -> dict[str, Any]:
    if not updates:
        return {}

    out: dict[str, Any] = {}

    if "messages" in updates:
        out["messages"] = updates["messages"]

    if "jump_to" in updates:
        out["__mw_jump_to"] = updates["jump_to"]

    for k, v in updates.items():
        if k in {"messages", "jump_to"}:
            continue
        out[k] = v

    return out


def _apply_middleware_hook(
    middlewares: list[Any],
    hook_name: str,
    state: AgentState,
    config: RunnableConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}

    runtime: dict[str, Any] = {
        "config": config,
        "kwargs": kwargs,
    }

    for mw in middlewares:
        hook = getattr(mw, hook_name, None)
        if hook is None:
            continue

        try:
            result = hook(state, runtime, **kwargs)  # type: ignore[misc]
        except TypeError:
            continue
        except Exception:
            continue

        if isinstance(result, dict) and isinstance(result.get("messages"), list):
            raw_messages = result.get("messages", [])
            fixed_messages = []

            for m in raw_messages:
                content = getattr(m, "content", "")
                fixed_content = restore_python_repr_content(content)

                if hasattr(m, "model_copy"):
                    fixed_messages.append(
                        m.model_copy(update={"content": fixed_content})
                    )
                else:
                    m.content = fixed_content
                    fixed_messages.append(m)

            updates.update({**result, "messages": fixed_messages})

    return updates


async def before_model_middleware(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    mws = build_middlewares()
    updates = _apply_middleware_hook(mws, "before_model", state, config)
    return _merge_state_updates(updates)


async def after_model_middleware(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    mws = build_middlewares()

    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    updates = _apply_middleware_hook(mws, "after_model", state, config, last_ai=last_ai)
    return _merge_state_updates(updates)


async def before_tools_middleware(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    mws = build_middlewares()
    updates = _apply_middleware_hook(mws, "before_tools", state, config)
    return _merge_state_updates(updates)


async def after_tools_middleware(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    mws = build_middlewares()

    tool_messages: list[ToolMessage] = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_messages.append(msg)
        else:
            break
    tool_messages.reverse()

    updates = _apply_middleware_hook(
        mws, "after_tools", state, config, tool_messages=tool_messages
    )
    return _merge_state_updates(updates)
