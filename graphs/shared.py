from enum import Enum

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
