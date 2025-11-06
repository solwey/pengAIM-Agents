import logging
import aiohttp
from typing import Dict, Optional, Any
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store


async def get_mcp_access_token(
    access_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """
    Exchange a JWT token for an MCP access token.

    Args:
        access_token: The JWT token to exchange
        base_mcp_url: The base URL for the MCP server

    Returns:
        The token data as a dictionary if successful, None otherwise
    """
    try:
        # Exchange JWT token for MCP access token
        form_data = {
            "client_id": "mcp_default",
            "subject_token": access_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_mcp_url.rstrip("/") + "/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as token_response:
                if token_response.status == 200:
                    token_data = await token_response.json()
                    return token_data
                else:
                    response_text = await token_response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")

    return None


async def get_tokens(config: RunnableConfig):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    team_id = config.get("metadata", {}).get("owner")
    if not team_id or not isinstance(team_id, str):
        return None

    tokens = await store.aget((team_id, "tokens"), "data")
    if not tokens:
        return None

    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation

    from datetime import datetime, timedelta, timezone

    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        # Tokens have expired, delete them
        await store.adelete((team_id, "tokens"), "data")
        return None

    return tokens.value


async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    team_id = config.get("metadata", {}).get("owner")
    if not team_id or not isinstance(team_id, str):
        return

    await store.aput((team_id, "tokens"), "data", tokens)
    return


async def fetch_tokens(config: RunnableConfig) -> Optional[dict[str, Any]]:
    """
    Fetch an MCP access token if it doesn't already exist in the store.

    Args:
        config: The runnable configuration

    Raises:
        ValueError: If the required configuration is missing
    """

    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    access_token = config.get("configurable", {}).get("x-jwt-access-token")
    if not access_token:
        return None

    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    mcp_tokens = await get_mcp_access_token(access_token, mcp_config.get("url"))

    await set_tokens(config, mcp_tokens)
    return mcp_tokens
