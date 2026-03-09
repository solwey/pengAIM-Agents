"""Merge runtime config overrides from the client into the assistant's stored config."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def merge_runtime_config(
    assistant_config: dict[str, Any] | None,
    request_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge request-level config overrides into the assistant's stored config.

    The assistant config is the base. Request config values override on a per-key
    basis inside ``configurable``.  Top-level keys outside ``configurable`` are
    also merged (request wins).

    Returns a new dict — neither input is mutated.
    """
    base = deepcopy(assistant_config) if assistant_config else {}
    if not request_config:
        return base

    override = request_config

    # Merge top-level keys (except 'configurable' which needs special handling)
    for key, value in override.items():
        if key == "configurable":
            continue
        base[key] = value

    # Deep-merge configurable
    override_configurable = override.get("configurable")
    if override_configurable and isinstance(override_configurable, dict):
        base.setdefault("configurable", {})
        base["configurable"].update(override_configurable)

    return base
