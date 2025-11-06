import json
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, ToolException, tool
import aiohttp
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, Tool, McpError


def create_langchain_mcp_tool(
        mcp_tool: Tool, mcp_server_url: str = "", headers: dict[str, str] | None = None
) -> StructuredTool:
    """Create a LangChain tool from an MCP tool."""

    @tool(
        mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.inputSchema,
    )
    async def new_tool(**kwargs):
        """Dynamically created MCP tool."""
        async with streamablehttp_client(mcp_server_url, headers=headers) as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as tool_session:
                await tool_session.initialize()
                return await tool_session.call_tool(mcp_tool.name, arguments=kwargs)

    return new_tool


def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap the tool coroutine to handle `interaction_required` MCP error.

    Tried to obtain the URL from the error, which the LLM can use to render a link."""

    old_coroutine = tool.coroutine

    async def wrapped_mcp_coroutine(**kwargs):
        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            if isinstance(exc, McpError):
                return exc
            if isinstance(exc, ExceptionGroup):
                for sub_exc in exc.exceptions:
                    if found := _find_first_mcp_error_nested(sub_exc):
                        return found
            return None

        try:
            return await old_coroutine(**kwargs)
        except BaseException as e_orig:
            mcp_error = _find_first_mcp_error_nested(e_orig)

            if not mcp_error:
                raise e_orig

            error_details = mcp_error.error
            is_interaction_required = getattr(error_details, "code", None) == -32003
            error_data = getattr(error_details, "data", None) or {}

            if is_interaction_required:
                message_payload = error_data.get("message", {})
                error_message_text = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message_text = (
                            message_payload.get("text") or error_message_text
                    )

                if url := error_data.get("url"):
                    error_message_text = f"{error_message_text} {url}"
                raise ToolException(error_message_text) from e_orig

            raise e_orig

    tool.coroutine = wrapped_mcp_coroutine
    return tool


async def create_rag_tool(rag_url: str):
    """Create a RAG tool that calls a unified QA endpoint.

    Args:
        rag_url: The base URL for the RAG API server

    Returns:
        A structured tool that accepts a user query and returns a JSON string with
        {response: str, reasoning: str}.
    """
    if rag_url.endswith("/"):
        rag_url = rag_url[:-1]

    try:
        collection_name = "collection"
        collection_description = "Search your collection of documents for results semantically similar to the input query"

        @tool(name_or_callable=collection_name, description=collection_description)
        async def get_documents(
                query: Annotated[str, "The search query to find relevant documents"],
                config: RunnableConfig = None
        ) -> str:
            """Search for documents in the collection based on the query"""

            authorization = config.get("configurable", {}).get("langgraph_auth_user", {}).get("permissions")[0].replace("authz:", "")

            system_prompt = config.get("configurable", {}).get("rag_system_prompt") or None
            retrieval_mode = config.get("configurable", {}).get("rag_retrieval_mode") or None
            key_data = config.get("configurable", {}).get("rag_openai_api_key", {})

            search_endpoint = f"{rag_url}/query"
            body = {
                "question": query,
                "system_prompt": system_prompt,
                "retrieval_mode": retrieval_mode,
                "api_key_id": key_data.get("keyId")
            }
            headers = {"authorization": authorization}

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            search_endpoint,
                            json=body,
                            headers=headers
                    ) as search_response:
                        search_response.raise_for_status()
                        data = await search_response.json()

                out = {
                    "response": data.get("response", "") or "",
                    "reasoning": data.get("reasoning", "") or "",
                    "context": data.get("context", {}) or {},
                }

                return json.dumps(out, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": f"RAG tool call failed: {str(e)}"}, ensure_ascii=False)

        return get_documents

    except Exception as e:
        raise Exception(f"Failed to create RAG tool: {str(e)}")
