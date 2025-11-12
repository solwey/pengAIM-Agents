"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import json
import logging
import os
import warnings
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

import aiohttp
from firecrawl import AsyncFirecrawlApp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from graphs.open_deep_research.configuration import AgentMode, Configuration, SearchAPI
from graphs.open_deep_research.prompts import summarize_webpage_prompt
from graphs.open_deep_research.state import ResearchComplete, Summary

RAG_URL = os.getenv("RAG_API_URL", "")

if not RAG_URL:
    raise RuntimeError("RAG_API_URL is not set")


##########################
# Agent Mode utils (RAG = RAG-only, ONLINE = Web+MCP)
##########################


def get_agent_mode(config: RunnableConfig) -> AgentMode:
    """Resolve agent mode from Configuration or raw RunnableConfig.

    Priority:
    1) Configuration.mode if present
    2) Default to "online"
    """
    try:
        cfg = Configuration.from_runnable_config(config)
        return cfg.mode if cfg.mode else AgentMode.ONLINE
    except Exception:
        return AgentMode.ONLINE


##########################
# RAG tool
##########################


@tool(
    name_or_callable="rag_search",
    description="Search your collection of documents for results semantically similar to the input query",
)
async def rag_search(
    query: Annotated[str, "The search query to find relevant documents"],
    config: RunnableConfig = None,
) -> str:
    """Search for documents in the collection based on the query"""

    authorization = (
        config.get("configurable", {})
        .get("langgraph_auth_user", {})
        .get("permissions")[0]
        .replace("authz:", "")
    )
    system_prompt = config.get("configurable", {}).get("rag_system_prompt") or None
    retrieval_mode = config.get("configurable", {}).get("rag_retrieval_mode") or None
    key_data = config.get("configurable", {}).get("rag_openai_api_key", {})

    search_endpoint = f"{RAG_URL}/query"
    body = {
        "question": query,
        "system_prompt": system_prompt,
        "retrieval_mode": retrieval_mode,
        "api_key_id": key_data.get("keyId"),
    }
    headers = {"authorization": authorization}

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                search_endpoint, json=body, headers=headers
            ) as search_response,
        ):
            search_response.raise_for_status()
            data = await search_response.json()

        out = {
            "response": data.get("response", "") or "",
            "reasoning": data.get("reasoning", "") or "",
            "context": data.get("context", {}) or {},
        }

        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        return json.dumps(
            {"error": f"RAG tool call failed: {str(e)}"}, ensure_ascii=False
        )


##########################
# FireCrawl Search Tool Utils
##########################


@tool(description="Search the web using Firecrawl API and return summarized results")
async def firecrawl_search(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None,
) -> str:
    """Perform web searches using Firecrawl and summarize results."""
    if not queries:
        return "No valid search results found."
    api_key = await get_api_key(config, "api_key", "firecrawl", "root")
    app = AsyncFirecrawlApp(api_key=api_key)
    logging.info("firecrawl_search: queries=%d", len(queries or []))

    async def search_query(query: str):
        try:
            return await app.search(
                query,
                limit=max_results,
                scrape_options={
                    "formats": ["markdown"],
                    "only_main_content": True,
                    "remove_base64_images": True,
                },
            )
        except Exception as e:
            logging.warning(f"Search failed for query '{query}': {e}")
            return None

    search_responses = await asyncio.gather(*[search_query(q) for q in queries])

    search_results = []
    seen_urls: set[str] = set()

    for query, resp in zip(queries, search_responses, strict=False):
        if not resp:
            continue
        for item in getattr(resp, "web", []) or []:
            metadata = getattr(item, "metadata", None)
            url = getattr(item, "url", "") or getattr(metadata, "url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = (
                getattr(item, "title", "")
                or getattr(metadata, "title", "")
                or "No title"
            )
            summary_text = (
                getattr(item, "description", "")
                or getattr(metadata, "description", "")
                or "No description"
            )
            raw_md = getattr(item, "markdown", "")

            search_results.append(
                {
                    "title": title,
                    "content": summary_text,
                    "url": url,
                    "raw_content": raw_md,
                    "query": query,
                }
            )

    unique_results = {}
    for result in search_results:
        if result["url"] not in unique_results:
            unique_results[result["url"]] = result

    if not unique_results:
        return "No valid search results found."

    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = configurable.max_content_length
    model_api_key = await get_api_key_for_model(
        configurable.summarization_model, config
    )
    summarization_model = (
        init_chat_model(
            model=configurable.summarization_model,
            max_tokens=configurable.summarization_model_max_tokens,
            api_key=model_api_key,
            tags=["langsmith:nostream"],
        )
        .with_structured_output(Summary)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    async def summarize_or_none(res: dict[str, str]):
        if not res.get("raw_content"):
            return None
        return await summarize_webpage(
            summarization_model, res["raw_content"][:max_char_to_include]
        )

    summaries = await asyncio.gather(
        *[summarize_or_none(r) for r in unique_results.values()]
    )

    summarized_results = {
        url: {
            "title": result["title"],
            "content": result["content"] if summary is None else summary,
        }
        for (url, result), summary in zip(
            unique_results.items(), summaries, strict=False
        )
    }

    formatted_output = "Search results:\n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i + 1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\nSUMMARY:\n{result['content']}\n"
        formatted_output += "\n" + "-" * 80 + "\n"

    return formatted_output


##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
    config: RunnableConfig = None,
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config,
    )

    # Step 2: Deduplicate results by URL to avoid processing the same content multiple times
    unique_results = {}
    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = {**result, "query": response["query"]}

    # Step 3: Set up the summarization model with configuration
    configurable = Configuration.from_runnable_config(config)

    # Character limit to stay within model token limits (configurable)
    max_char_to_include = configurable.max_content_length

    # Initialize summarization model with retry logic
    model_api_key = await get_api_key_for_model(
        configurable.summarization_model, config
    )
    summarization_model = (
        init_chat_model(
            model=configurable.summarization_model,
            max_tokens=configurable.summarization_model_max_tokens,
            api_key=model_api_key,
            tags=["langsmith:nostream"],
        )
        .with_structured_output(Summary)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    # Step 4: Create summarization tasks (skip empty content)
    async def noop():
        """No-op function for results without raw content."""
        return None

    summarization_tasks = [
        noop()
        if not result.get("raw_content")
        else summarize_webpage(
            summarization_model, result["raw_content"][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # Step 5: Execute all summarization tasks in parallel
    summaries = await asyncio.gather(*summarization_tasks)

    # Step 6: Combine results with their summaries
    summarized_results = {
        url: {
            "title": result["title"],
            "content": result["content"] if summary is None else summary,
        }
        for url, result, summary in zip(
            unique_results.keys(), unique_results.values(), summaries, strict=False
        )
    }

    # Step 7: Format the final output
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i + 1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None,
):
    """Execute multiple Tavily search queries asynchronously.

    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access

    Returns:
        List of search result dictionaries from Tavily API
    """
    # Initialize the Tavily client with API key from config
    api_key = await get_api_key(config, "key_id", "tavily", "root")
    tavily_client = AsyncTavilyClient(api_key=api_key)

    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        for query in search_queries
    ]

    # Execute all search queries in parallel and return results
    search_results = await asyncio.gather(*search_tasks)
    return search_results


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content using AI model with timeout protection.

    Args:
        model: The chat model configured for summarization
        webpage_content: Raw webpage content to be summarized

    Returns:
        Formatted summary with key excerpts, or original content if summarization fails
    """
    try:
        # Create prompt with current date context
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, date=get_today_str()
        )

        # Execute summarization with timeout to prevent hanging
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0,  # 60 second timeout for summarization
        )

        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except TimeoutError:
        # Timeout during summarization - return original content
        logging.warning(
            "Summarization timed out after 60 seconds, returning original content"
        )
        return webpage_content
    except Exception as e:
        # Other errors during summarization - log and return original content
        logging.warning(
            f"Summarization failed with error: {str(e)}, returning original content"
        )
        return webpage_content


##########################
# Reflection Tool Utils
##########################


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


##########################
# MCP Utils
##########################


async def get_mcp_access_token(
    access_token: str,
    base_mcp_url: str,
) -> dict[str, Any] | None:
    """Exchange JWT token for MCP access token using OAuth token exchange.

    Args:
        access_token: Valid JWT authentication token
        base_mcp_url: Base URL of the MCP server

    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": access_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            async with session.post(
                token_url, headers=headers, data=form_data
            ) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")

    except Exception as e:
        logging.error(f"Error during token exchange: {e}")

    return None


async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.

    Args:
        config: Runtime configuration containing thread and user identifiers

    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()

    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    team_id = config.get("metadata", {}).get("owner")
    if not team_id or not isinstance(team_id, str):
        return None

    # Retrieve stored tokens
    tokens = await store.aget((team_id, "tokens"), "data")
    if not tokens:
        return None

    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(UTC)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        # Token expired, clean up and return None
        await store.adelete((team_id, "tokens"), "data")
        return None

    return tokens.value


async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.

    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()

    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    team_id = config.get("metadata", {}).get("owner")
    if not team_id or not isinstance(team_id, str):
        return

    # Store the tokens
    await store.aput((team_id, "tokens"), "data", tokens)


async def fetch_tokens(config: RunnableConfig) -> dict[str, Any] | None:
    """Fetch and refresh MCP tokens, getting new ones if needed.

    Args:
        config: Runtime configuration with authentication details

    Returns:
        Valid token dictionary, or None if unable to get tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    # Extract JWT token for new token exchange
    access_token = config.get("configurable", {}).get("x-jwt-access-token")
    if not access_token:
        return None

    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    # Exchange JWT token for MCP tokens
    mcp_tokens = await get_mcp_access_token(access_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens


def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.

    Args:
        tool: The MCP structured tool to wrap

    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine

    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""

        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc

            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, "exceptions"):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None

        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)

        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                raise original_error

            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}

            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"

                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message

                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"

                raise ToolException(error_message) from original_error

            # For other MCP errors, re-raise the original
            raise original_error

    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool


async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.

    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts

    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)

    tools_cfg = list(configurable.mcp_config.tools or [])
    auth_required = configurable.mcp_config.auth_required
    url_cfg = configurable.mcp_config.url

    # Step 1: Handle authentication if required
    if configurable.mcp_config and auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None

    # Step 2: Validate configuration requirements
    if not (url_cfg and (mcp_tokens or not auth_required)):
        return []

    if not tools_cfg:
        return []

    # Step 3: Set up the MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"

    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}

    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http",
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code

    # Step 4: Load tools from the MCP server
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # If the MCP server connection fails, return an empty list
        return []

    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue

        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue

        # Wrap the tool with authentication handling and add to the list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)

    return configured_tools


##########################
# Tool Utils
##########################


async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.

    Args:
        search_api: The search API provider to use (Anthropic, OpenAI, Tavily or None)

    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic's native web search with usage limits
        return [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]

    elif search_api == SearchAPI.OPENAI:
        # OpenAI's web search preview functionality
        return [{"type": "web_search_preview"}]

    elif search_api == SearchAPI.TAVILY:
        # Configure Tavily search tool with metadata
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search",
        }
        return [search_tool]
    elif search_api == SearchAPI.FIRECRAWL:
        search_tool = firecrawl_search
        search_tool.metadata = {
            **(getattr(search_tool, "metadata", {}) or {}),
            "type": "search",
            "name": "web_search",
        }
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []

    # Default fallback for unknown search API types
    return []


async def get_all_tools(config: RunnableConfig):
    """Assemble toolkit strictly based on agent mode.

    Modes:
      - RAG: Core only (ResearchComplete, think_tool). No web search. No MCP.
      - ONLINE: Core + Web search (per SearchAPI) + MCP tools.

    Note: RAG/vector tools should be mounted by the calling code in RAG mode.
    This function intentionally avoids adding *any* network-capable tools when use rag mode.
    """
    tools: list[Any] = [tool(ResearchComplete), think_tool]

    mode = get_agent_mode(config)

    if mode == AgentMode.RAG:
        # do NOT include web search or MCP tools
        tools.append(rag_search)
        return tools

    # ONLINE mode: include Web search + MCP tools when configured
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)

    existing_tool_names = {
        t.name if hasattr(t, "name") else t.get("name", "web_search") for t in tools
    }

    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)

    tools.extend(mcp_tools)

    return tools


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [
        tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")
    ]


##########################
# Model Provider Native Websearch Utils
##########################


def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.

    Args:
        response: The response object from Anthropic's API

    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        # Return True if any web search requests were made
        return web_search_requests > 0

    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        return False


def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.

    Args:
        response: The response object from OpenAI's API

    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False

    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True

    return False


##########################
# Token Limit Exceeded Utils
##########################


def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.

    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection

    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()

    # Step 1: Determine provider from model name if available
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith("openai:"):
            provider = "openai"
        elif model_str.startswith("anthropic:"):
            provider = "anthropic"
        elif model_str.startswith("gemini:") or model_str.startswith("google:"):
            provider = "gemini"

    # Step 2: Check provider-specific token limit patterns
    if provider == "openai":
        return _check_openai_token_limit(exception, error_str)
    elif provider == "anthropic":
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == "gemini":
        return _check_gemini_token_limit(exception, error_str)

    # Step 3: If provider unknown, check all providers
    return (
        _check_openai_token_limit(exception, error_str)
        or _check_anthropic_token_limit(exception, error_str)
        or _check_gemini_token_limit(exception, error_str)
    )


def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is an OpenAI exception
    is_openai_exception = (
        "openai" in exception_type.lower() or "openai" in module_name.lower()
    )

    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ["BadRequestError", "InvalidRequestError"]

    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ["token", "context", "length", "maximum context", "reduce"]
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # Check for specific OpenAI error codes
    if hasattr(exception, "code") and hasattr(exception, "type"):
        error_code = getattr(exception, "code", "")
        error_type = getattr(exception, "type", "")

        if (
            error_code == "context_length_exceeded"
            or error_type == "invalid_request_error"
        ):
            return True

    return False


def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is an Anthropic exception
    is_anthropic_exception = (
        "anthropic" in exception_type.lower() or "anthropic" in module_name.lower()
    )

    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == "BadRequestError"

    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if "prompt is too long" in error_str:
            return True

    return False


def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is a Google/Gemini exception
    is_google_exception = (
        "google" in exception_type.lower() or "google" in module_name.lower()
    )

    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in [
        "ResourceExhausted",
        "GoogleGenerativeAIFetchError",
    ]

    if is_google_exception and is_resource_exhausted:
        return True

    # Check for specific Google API resource exhaustion patterns
    if "google.api_core.exceptions.resourceexhausted" in exception_type.lower():
        return True

    return False


# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
}


def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.

    Args:
        model_string: The model identifier string to look up

    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit

    # Model not found in lookup table
    return None


def remove_up_to_last_ai_message(
    messages: list[MessageLikeRepresentation],
) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.

    This is useful for handling token limit exceeded errors by removing recent context.

    Args:
        messages: List of message objects to truncate

    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]

    # No AI messages found, return original list
    return messages


##########################
# Misc Utils
##########################


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a %b} {now.day}, {now:%Y}"


def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str) or isinstance(value, dict):
        return value
    else:
        return value.value


def _non_empty(v: str | None) -> str | None:
    if isinstance(v, str) and v.strip():
        return v
    return None


async def get_api_key_for_model(model_name: str, config: RunnableConfig):
    model_name = model_name.lower()
    model_prefixes = ["openai", "anthropic", "google"]
    key_name = next(
        (prefix for prefix in model_prefixes if model_name.startswith(prefix)), None
    )
    if not key_name:
        return None

    key_data = config.get("configurable", {}).get("agent_openai_api_key", {})
    key = await get_api_key(config, key_data.get("keyId"))
    return key


async def get_api_key(
    config: RunnableConfig,
    key_id: str,
    provider: str | None = None,
    name: str | None = None,
):
    """Get API key from environment or config by key name."""
    authorization = (
        config.get("configurable", {})
        .get("langgraph_auth_user", {})
        .get("permissions")[0]
        .replace("authz:", "")
    )
    if not authorization or not key_id:
        return None

    search_params = f"provider={provider}&name={name}" if provider and name else ""
    search_endpoint = (
        f"{RAG_URL}/keys/{key_id}/reveal{f'?{search_params}' if search_params else ''}"
    )
    headers = {"authorization": authorization, "Accept": "text/plain"}

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(search_endpoint, headers=headers) as search_response,
        ):
            search_response.raise_for_status()
            key = await search_response.text()

        return key
    except Exception as e:
        print("Key exception3", e)
        return None


def normalize_placeholders(payload):
    if isinstance(payload, list):
        out = []
        for p in payload:
            if isinstance(p, dict) and p.get("field") is not None:
                out.append({"field": p.get("field"), "value": p.get("value", "")})
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append({"field": str(p[0]), "value": str(p[1])})
        return out

    if isinstance(payload, dict):
        for key in ("provided", "placeholders"):
            if key in payload and isinstance(payload[key], list):
                return normalize_placeholders(payload[key])
        out = []
        for k, v in payload.items():
            out.append({"field": str(k), "value": "" if v is None else str(v)})
        return out

    return []


def apply_placeholders(text: str, placeholders: list) -> str:
    """Replace [field] with value from placeholders"""
    result = text
    for p in placeholders:
        try:
            if isinstance(p, dict):
                field = p.get("field")
                value = p.get("value", "")
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                field, value = p[0], p[1]
            else:
                continue
            if isinstance(field, str):
                result = result.replace(f"[{field}]", str(value))
        except Exception:
            continue
    return result


def truncate_result(text: str, limit: int = 1000) -> str:
    """Trim long text for UI messages without breaking the flow"""
    if not isinstance(text, str):
        return ""
    return text[:limit] + ("…" if len(text) > limit else "")


def normalize_branch_name(name: str | None, used: set[str]) -> str:
    """Return a deterministic, unique branch name safe for state keys"""
    base = (
        "".join(
            ch if (ch.isalnum() or ch in ("_", "-")) else "_"
            for ch in (name or "subprompt")
        ).strip("_")
        or "subprompt"
    )
    nm = base
    i = 1
    while nm in used:
        i += 1
        nm = f"{base}_{i}"
    used.add(nm)
    return nm
