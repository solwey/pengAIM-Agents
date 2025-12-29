import ast
from enum import Enum

from langchain.agents.middleware import PIIMiddleware
from pydantic import BaseModel, Field

DEFAULT_QUESTION_CATEGORIES: list[dict] = [
    {
        "icon": "lucide:mail",
        "title": "Email Tasks",
        "questions": [
            {"text": "How do I send an invoice to a customer?"},
            {"text": "How can I email the weekly inventory report?"},
        ],
    },
    {
        "icon": "lucide:file-text",
        "title": "Creating Reports",
        "questions": [
            {"text": "How do I create a monthly sales summary?"},
            {"text": "How can I generate a product performance report?"},
        ],
    },
    {
        "icon": "lucide:package",
        "title": "Order Management",
        "questions": [
            {"text": "How do I track a shipment?"},
            {"text": "How can I process a customer return?"},
        ],
    },
    {
        "icon": "lucide:users",
        "title": "Customer Information",
        "questions": [
            {"text": "How do I look up a customer's order history?"},
            {"text": "How can I update customer contact details?"},
        ],
    },
]


class DefaultQuestion(BaseModel):
    text: str = Field(
        ..., description="Question text that will be suggested to the user."
    )


class DefaultQuestionsCategory(BaseModel):
    icon: str | None = Field(
        default=None,
        description="Iconify icon name for this category (e.g., 'lucide:sparkles').",
    )
    title: str = Field(..., description="Human-readable category title.")
    questions: list[DefaultQuestion] = Field(
        default_factory=list,
        description="List of example questions for this category.",
    )


class RetrievalMode(Enum):
    BASIC = "basic"
    HYDE = "hyde"
    RRF = "rrf"


class ToolCallsVisibility(Enum):
    USER_PREFERENCE = "user_preference"
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"


def restore_python_repr_content(value):
    """Restore content if it looks like a Python repr of list[dict]."""
    if not isinstance(value, str):
        return value

    s = value.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return value

    if "{'type':" not in s and "{'text':" not in s:
        return value

    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return value

    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
        return parsed

    return value


def build_middlewares():
    return [
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        PIIMiddleware(
            "ip",
            strategy="redact",
            apply_to_input=True,
        ),
        PIIMiddleware(
            "openai_api_key",
            detector=r"\bsk-[A-Za-z0-9]{16,}\b",
            strategy="block",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "github_token",
            detector=r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
            strategy="block",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "slack_token",
            detector=r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
            strategy="block",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "private_key_pem",
            detector=r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            strategy="block",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "db_url_with_creds",
            detector=r"\b(?:postgres|postgresql|mysql|mongodb|redis)://[^\\s:@]+:[^\\s@]+@",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
    ]
