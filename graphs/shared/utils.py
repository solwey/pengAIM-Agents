import re
from typing import Any


def is_openai_reasoning_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    model_id = model_name.split(":")[-1].lower()
    return (
        model_id.startswith("gpt-5")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
        or model_id.startswith("o4")
    )


def resolve_reasoning_model_params(model_name: str | None, reasoning_level: str | None) -> dict[str, Any]:
    """Map generic reasoning level to provider-specific model kwargs."""
    if not model_name or not reasoning_level:
        return {}

    model_lower = model_name.lower()

    if model_lower.startswith("openai:") or model_lower.startswith("azure_openai:"):
        if not is_openai_reasoning_model(model_name):
            return {}

        return {"reasoning": {"effort": reasoning_level}}

    if model_lower.startswith("google_genai:") or model_lower.startswith("google:"):
        model_id = model_name.split(":", 1)[-1]

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
