"""Canonical list of LLM models supported by agents.

Single source of truth consumed by:
- aegra_api.api.workflow_models (health probe endpoint)
- graphs.react_agent.context (model_name Field options)
- graphs.open_deep_research.configuration (model Field options)
"""

DEFAULT_LLM_MODEL: str = "openai:gpt-4o-mini"

LLM_MODELS: list[tuple[str, str]] = [
    ("openai:gpt-4o", "GPT-4o"),
    ("openai:gpt-4o-mini", "GPT-4o Mini"),
    ("openai:gpt-4.1", "GPT-4.1"),
    ("openai:gpt-4.1-mini", "GPT-4.1 Mini"),
    ("openai:gpt-5", "GPT-5"),
    ("openai:gpt-5-mini", "GPT-5 Mini"),
    ("openai:gpt-5.1", "GPT-5.1"),
    ("openai:gpt-5.2", "GPT-5.2"),
    ("google_genai:gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("google_genai:gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("google_genai:gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
    ("google_genai:gemini-3-flash-preview", "Gemini 3 Flash Preview"),
    ("google_genai:gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
]


def to_ui_options(models: list[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    """Convert (value, label) tuples to [{"label": ..., "value": ...}] dicts
    for Pydantic Field metadata ``x_oap_ui_config.options``.
    """
    source = models if models is not None else LLM_MODELS
    return [{"label": label, "value": value} for value, label in source]
