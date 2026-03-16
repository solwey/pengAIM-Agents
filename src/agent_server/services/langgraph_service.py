"""LangGraph integration service with official patterns"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid5
from contextlib import AsyncExitStack

import structlog
from langgraph.graph import StateGraph

from ..constants import ASSISTANT_NAMESPACE_UUID
from ..observability.base import get_tracing_callbacks, get_tracing_metadata

from graphs.shared.mcp import (
    McpConfigMixin,
    build_mcp_client,
    load_tools_with_sessions,
)

State = TypeVar("State")
logger = structlog.get_logger(__name__)


class LangGraphService:
    """Service to work with LangGraph CLI configuration and graphs"""

    def __init__(self, config_path: str = "aegra.json"):
        # Default path can be overridden via AEGRA_CONFIG or by placing aegra.json
        self.config_path = Path(config_path)
        self.config: dict[str, Any] | None = None
        self._graph_registry: dict[str, Any] = {}
        # Cache ONLY uncompiled graph definitions (StateGraph). Never cache compiled graphs,
        # because compiled graphs may hold live DB connections (checkpointer/store).
        self._graph_cache: dict[str, Any] = {}

    async def initialize(self):
        """Load configuration file and setup graph registry.

        Resolution order:
        1) AEGRA_CONFIG env var (absolute or relative path)
        2) Explicit self.config_path if it exists
        3) aegra.json in CWD
        4) langgraph.json in CWD (fallback)
        """
        # 1) Env var override
        env_path = os.getenv("AEGRA_CONFIG")
        resolved_path: Path
        if env_path:
            resolved_path = Path(env_path)
        # 2) Provided path if exists
        elif self.config_path and Path(self.config_path).exists():
            resolved_path = Path(self.config_path)
        # 3) aegra.json if present
        elif Path("aegra.json").exists():
            resolved_path = Path("aegra.json")
        # 4) fallback to langgraph.json
        else:
            resolved_path = Path("langgraph.json")

        if not resolved_path.exists():
            raise ValueError(
                "Configuration file not found. Expected one of: "
                "AEGRA_CONFIG path, ./aegra.json, or ./langgraph.json"
            )

        # Persist selected path for later reference
        self.config_path = resolved_path

        with self.config_path.open() as f:
            self.config = json.load(f)

        # Load graph registry from config
        self._load_graph_registry()

        # Pre-register assistants for each graph using deterministic UUIDs so
        # clients can pass graph_id directly.
        await self._ensure_default_assistants()

    def _load_graph_registry(self):
        """Load graph definitions from aegra.json"""
        graphs_config = self.config.get("graphs", {})

        for graph_id, graph_path in graphs_config.items():
            # Parse path format: "./graphs/weather_agent.py:graph"
            if ":" not in graph_path:
                raise ValueError(f"Invalid graph path format: {graph_path}")

            file_path, export_name = graph_path.split(":", 1)
            self._graph_registry[graph_id] = {
                "file_path": file_path,
                "export_name": export_name,
            }

    async def _ensure_default_assistants(self) -> None:
        """Create a default assistant per graph with deterministic UUID.

        Uses uuid5 with a fixed namespace so that the same graph_id maps
        to the same assistant_id across restarts. Idempotent.
        """
        from sqlalchemy import select

        from ..core.orm import Assistant as AssistantORM
        from ..core.orm import AssistantVersion as AssistantVersionORM
        from ..core.orm import get_session

        # Fixed namespace used to derive assistant IDs from graph IDs
        NS = ASSISTANT_NAMESPACE_UUID
        session_gen = get_session()
        session = await anext(session_gen)
        try:
            for graph_id in self._graph_registry:
                assistant_id = str(uuid5(NS, graph_id))
                existing = await session.scalar(
                    select(AssistantORM).where(
                        AssistantORM.assistant_id == assistant_id
                    )
                )
                if existing:
                    continue
                session.add(
                    AssistantORM(
                        assistant_id=assistant_id,
                        name=graph_id,
                        description=f"Default assistant for graph '{graph_id}'",
                        graph_id=graph_id,
                        config={},
                        team_id="system",
                        metadata_dict={"created_by": "system"},
                    )
                )
                session.add(
                    AssistantVersionORM(
                        assistant_id=assistant_id,
                        version=1,
                        name=graph_id,
                        description=f"Default assistant for graph '{graph_id}'",
                        graph_id=graph_id,
                        metadata_dict={"created_by": "system"},
                    )
                )
            await session.commit()
        finally:
            await session.close()

    async def get_graph(
            self, graph_id: str, force_reload: bool = False
    ) -> StateGraph[Any]:
        """Get a compiled graph by ID with caching and LangGraph integration"""
        if graph_id not in self._graph_registry:
            raise ValueError(f"Graph not found: {graph_id}")

        graph_info = self._graph_registry[graph_id]

        # Load graph definition.
        # IMPORTANT: We must not cache compiled graphs because they can retain stale/closed
        # DB connections via checkpointer/store. We ONLY cache uncompiled StateGraph objects.
        base_graph: Any
        if not force_reload and graph_id in self._graph_cache:
            base_graph = self._graph_cache[graph_id]
        else:
            base_graph = await self._load_graph_from_file(graph_id, graph_info)
            # Cache ONLY uncompiled graphs (StateGraph) which are safe to reuse.
            if hasattr(base_graph, "compile"):
                self._graph_cache[graph_id] = base_graph
            else:
                # Do not cache already-compiled graphs (may hold live DB connections).
                self._graph_cache.pop(graph_id, None)

        # Always ensure graphs are compiled with our Postgres checkpointer for persistence
        from ..core.database import db_manager

        if hasattr(base_graph, "compile"):
            # The module exported an *uncompiled* StateGraph – compile it now with
            # a Postgres checkpointer for durable state.
            checkpointer_cm = await db_manager.get_checkpointer()
            store_cm = await db_manager.get_store()

            logger.info(
                "🔧 Compiling graph with fresh persistence components",
                graph_id=graph_id,
                checkpointer_id=id(checkpointer_cm),
                store_id=id(store_cm),
            )

            compiled_graph = base_graph.compile(
                checkpointer=checkpointer_cm, store=store_cm
            )
        else:
            # Graph was already compiled by the module.  Create a shallow copy
            # that injects our Postgres checkpointer *unless* the author already
            # set one.
            checkpointer_cm = await db_manager.get_checkpointer()
            try:
                store_cm = await db_manager.get_store()
                compiled_graph = base_graph.copy(
                    update={"checkpointer": checkpointer_cm, "store": store_cm}
                )
            except Exception as e:
                msg = str(e).lower()
                if "connection is closed" in msg:
                    logger.warning(
                        "⚠️ LangGraph persistence connection is closed. "
                        "Attempting to reset and reconnect..."
                    )

                    try:
                        # Reset LangGraph components and retry once
                        await db_manager.reset_langgraph_components()

                        checkpointer_cm = await db_manager.get_checkpointer()
                        store_cm = await db_manager.get_store()

                        compiled_graph = base_graph.copy(
                            update={"checkpointer": checkpointer_cm, "store": store_cm}
                        )
                        logger.info(
                            "✅ LangGraph persistence successfully reconnected for graph '%s'",
                            graph_id,
                        )
                    except Exception as retry_err:
                        logger.error(
                            "❌ Failed to reinitialize LangGraph persistence for '%s'; "
                            "running without persistence. Error: %s",
                            graph_id,
                            str(retry_err),
                        )
                        compiled_graph = base_graph
                else:
                    logger.warning(
                        "⚠️ Pre-compiled graph '%s' does not support checkpointer injection; "
                        "running without persistence. Error: %s",
                        graph_id,
                        str(e),
                    )
                    compiled_graph = base_graph

        return compiled_graph

    async def _load_graph_from_file(self, graph_id: str, graph_info: dict[str, str]):
        """Load graph from filesystem"""
        file_path = Path(graph_info["file_path"])
        if not file_path.exists():
            raise ValueError(f"Graph file not found: {file_path}")

        # Dynamic import of graph module
        spec = importlib.util.spec_from_file_location(
            f"graphs.{graph_id}", str(file_path.resolve())
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load graph module: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the exported graph
        export_name = graph_info["export_name"]
        if not hasattr(module, export_name):
            raise ValueError(f"Graph export not found: {export_name} in {file_path}")

        graph = getattr(module, export_name)

        # The graph should already be compiled in the module
        # If it needs our checkpointer/store, we'll handle that during execution
        return graph

    async def get_workflow_graph(self, workflow_json: dict[str, Any]) -> Any:
        """Compile a dynamic workflow from JSON and return a compiled graph.

        Caches uncompiled StateGraph by definition hash to avoid recompilation
        of identical workflows. Always compiles with fresh checkpointer/store.
        """
        from graphs.workflow_engine.compiler import compile_workflow
        from graphs.workflow_engine.schema import WorkflowDefinition

        # Validate and parse the workflow definition
        definition = WorkflowDefinition(**workflow_json)

        # Cache key based on content hash
        def_hash = hashlib.sha256(
            json.dumps(workflow_json, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_key = f"workflow:{def_hash}"

        # Check cache for uncompiled graph
        if cache_key in self._graph_cache:
            base_graph = self._graph_cache[cache_key]
        else:
            base_graph = compile_workflow(definition)
            self._graph_cache[cache_key] = base_graph
            logger.info(
                "Compiled and cached workflow",
                workflow_name=definition.name,
                cache_key=cache_key,
            )

        # Always compile with fresh persistence (same pattern as get_graph)
        from ..core.database import db_manager

        checkpointer_cm = await db_manager.get_checkpointer()
        store_cm = await db_manager.get_store()

        compiled_graph = base_graph.compile(
            checkpointer=checkpointer_cm, store=store_cm
        )

        return compiled_graph

    def list_graphs(self) -> dict[str, str]:
        """List all available graphs"""
        return {
            graph_id: info["file_path"]
            for graph_id, info in self._graph_registry.items()
        }

    def invalidate_cache(self, graph_id: str = None):
        """Invalidate cached uncompiled graph definitions for hot-reload"""
        if graph_id:
            self._graph_cache.pop(graph_id, None)
        else:
            self._graph_cache.clear()

    def get_config(self) -> dict[str, Any] | None:
        """Get loaded configuration"""
        return self.config

    def get_dependencies(self) -> list:
        """Get dependencies from config"""
        if self.config is None:
            return []
        return self.config.get("dependencies", [])


# Global service instance
_langgraph_service = None


def get_langgraph_service() -> LangGraphService:
    """Get global LangGraph service instance"""
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphService()
    return _langgraph_service


def inject_user_context(user, base_config: dict = None) -> dict:
    """Inject user context into LangGraph configuration for user isolation"""
    config = (base_config or {}).copy()
    config["configurable"] = config.get("configurable", {})

    # All user-related data injection (only if user exists)
    if user:
        # Basic user identity for multi-tenant scoping
        config["configurable"].setdefault("user_id", user.id)
        config["configurable"].setdefault(
            "user_display_name", getattr(user, "display_name", user.id)
        )

        # Full auth payload for graph nodes
        if "langgraph_auth_user" not in config["configurable"]:
            try:
                config["configurable"]["langgraph_auth_user"] = user.to_dict()  # type: ignore[attr-defined]
            except Exception:
                # Fallback: minimal dict if to_dict unavailable
                config["configurable"]["langgraph_auth_user"] = {
                    "identity": f"{user.id}:{user.team_id}",
                    "permissions": user.permissions,
                }

    return config


def create_thread_config(thread_id: str, user, additional_config: dict = None) -> dict:
    """Create LangGraph configuration for a specific thread with user context"""
    base_config = {"configurable": {"thread_id": thread_id}}

    if additional_config:
        base_config.update(additional_config)

    return inject_user_context(user, base_config)


def create_run_config(
        run_id: str,
        thread_id: str,
        user,
        additional_config: dict = None,
        checkpoint: dict | None = None,
) -> dict:
    """Create LangGraph configuration for a specific run with full context.

    The function is *additive*: it never removes or renames anything the client
    supplied.  We simply ensure a `configurable` dict exists and then merge a
    few server-side keys so graph nodes can rely on them.
    """
    from copy import deepcopy

    cfg: dict = deepcopy(additional_config) if additional_config else {}

    # Ensure a configurable section exists
    cfg.setdefault("configurable", {})

    # Merge server-provided fields (do NOT overwrite if client already set)
    cfg["configurable"].setdefault("thread_id", thread_id)
    cfg["configurable"].setdefault("run_id", run_id)

    # Add observability callbacks from various potential sources
    tracing_callbacks = get_tracing_callbacks()
    if tracing_callbacks:
        existing_callbacks = cfg.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            # If we want to be more robust, we can log a warning here
            existing_callbacks = []

        # Combine existing callbacks with new tracing callbacks to be non-destructive
        cfg["callbacks"] = existing_callbacks + tracing_callbacks

    # Add metadata from all observability providers (independent of callbacks)
    cfg.setdefault("metadata", {})
    user_identity = f"{user.id}:{user.team_id}" if user else None
    observability_metadata = get_tracing_metadata(run_id, thread_id, user_identity)
    cfg["metadata"].update(observability_metadata)

    # Apply checkpoint parameters if provided
    if checkpoint and isinstance(checkpoint, dict):
        cfg["configurable"].update(
            {k: v for k, v in checkpoint.items() if v is not None}
        )

    # Finally inject user context via existing helper
    return inject_user_context(user, cfg)

async def inject_mcp_tools(
        graph: Any,
        run_config: dict[str, Any],
        stack: "AsyncExitStack",
) -> dict:
    """Pre-load MCP tools with persistent sessions if the graph supports them.
    """

    config_schema = graph.config_schema()
    configurable_fields = getattr(config_schema, "model_fields", {})
    configurable_field = configurable_fields.get("configurable")
    config_cls = (
        configurable_field.annotation if configurable_field else None
    )

    if config_cls is None or not issubclass(config_cls, McpConfigMixin):
        return run_config

    graph_ctx = config_cls(**run_config.get("configurable", {}))
    if not graph_ctx.mcp_servers:
        return run_config

    auth_user = run_config.get("configurable", {}).get(
        "langgraph_auth_user", {}
    )
    permissions = auth_user.get("permissions", [])
    authorization = (
        permissions[0].replace("authz:", "") if permissions else None
    )

    client = build_mcp_client(graph_ctx.mcp_servers, authorization)
    server_names = [s.name for s in graph_ctx.mcp_servers]
    mcp_tools = await load_tools_with_sessions(client, server_names, stack)
    run_config["configurable"]["mcp_tools"] = mcp_tools

    return run_config
