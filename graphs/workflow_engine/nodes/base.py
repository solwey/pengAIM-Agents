"""Base class for all workflow node executors.

Every new node type should subclass NodeExecutor and implement `create()`.
This makes adding new node types trivial — just create a new file,
subclass NodeExecutor, and register in nodes/__init__.py.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any


class NodeExecutor(ABC):
    """Abstract base for workflow node executors.

    Subclasses must implement `create(config)` which returns an async
    node function compatible with LangGraph's `StateGraph.add_node()`.
    """

    @staticmethod
    @abstractmethod
    def create(config: dict[str, Any]) -> Callable[..., Coroutine[Any, Any, dict]]:
        """Build and return an async node function from the given config.

        The returned function signature must be:
            async def node_fn(state: dict, config: RunnableConfig) -> dict
        """
        ...


# ── Shared utilities ─────────────────────────────────────────


_TEMPLATE_RE = re.compile(r"\{\{(\w+(?:\.\w+)*)\}\}")


def resolve_field(data: dict[str, Any], field_path: str) -> Any:
    """Resolve a dot-notation path against a nested dict.

    Example:
        resolve_field({"api_response": {"status_code": 200}}, "api_response.status_code")
        → 200
    """
    parts = field_path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def resolve_templates(value: str, data: dict[str, Any]) -> str:
    """Replace {{key}} or {{key.subkey}} placeholders with values from data.

    Only resolves from state data — never from environment variables.
    """
    def _replacer(match: re.Match) -> str:
        path = match.group(1)
        resolved = resolve_field(data, path)
        return str(resolved) if resolved is not None else match.group(0)

    return _TEMPLATE_RE.sub(_replacer, value)


# Comparison operators — whitelist only, no eval/exec
OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a is not None and b is not None and a > b,
    "lt": lambda a, b: a is not None and b is not None and a < b,
    "gte": lambda a, b: a is not None and b is not None and a >= b,
    "lte": lambda a, b: a is not None and b is not None and a <= b,
    "contains": lambda a, b: b in a if a is not None else False,
    "not_contains": lambda a, b: b not in a if a is not None else True,
}


def compare(actual: Any, operator: str, expected: Any) -> bool:
    """Safely compare two values using a whitelisted operator."""
    fn = OPERATORS.get(operator)
    if fn is None:
        raise ValueError(f"Unknown operator: '{operator}'")
    try:
        return fn(actual, expected)
    except TypeError:
        return False
